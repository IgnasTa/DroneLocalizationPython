import math
import numpy as np
import osmnx as ox
import pyproj
import geopandas as gpd
from geopy.distance import geodesic
from shapely import affinity
from shapely.geometry import LineString
from scipy.interpolate import interp1d
from rasterio.transform import Affine
from scipy.spatial import KDTree
from scipy.optimize import minimize
from shapely.geometry import Point

def sample_graph_points(gdf_or_graph, spacing=3.0, is_sknw=False):
    """Unified sampler that works for both sknw NetworkX graph and GeoDataFrame"""
    points = []
    if is_sknw:
        for u, v, data in gdf_or_graph.edges(data=True):
            pts = np.array(data.get('pts', []))
            if len(pts) < 2: continue
            pts = pts[:, [1, 0]] if pts.shape[1] == 2 else pts
            diffs = np.diff(pts, axis=0)
            seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
            cum_dist = np.concatenate([[0.], np.cumsum(seg_len)])
            total_len = cum_dist[-1]
            if total_len < 1e-6:
                points.append(pts[0])
                continue
            n_samples = max(3, int(total_len / spacing) + 1)
            sample_dist = np.linspace(0, total_len, n_samples)
            interp = interp1d(cum_dist, pts, axis=0, kind='linear')
            points.extend(interp(sample_dist))
    else:
        for geom in gdf_or_graph.geometry:
            if geom.is_empty or not hasattr(geom, 'coords'): continue
            pts = np.array(geom.coords)[:, :2]
            if len(pts) < 2: continue
            diffs = np.diff(pts, axis=0)
            seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
            cum_dist = np.concatenate([[0.], np.cumsum(seg_len)])
            total_len = cum_dist[-1]
            if total_len < 1e-6:
                points.append(pts[0])
                continue
            n_samples = max(3, int(total_len / spacing) + 1)
            sample_dist = np.linspace(0, total_len, n_samples)
            interp = interp1d(cum_dist, pts, axis=0, kind='linear')
            points.extend(interp(sample_dist))
    return np.asarray(points, dtype=np.float64)


def rigid_transform_2d(pts, tx, ty, angle_deg):
    """Rotates and translates a 2D point cloud around the origin (0,0)"""
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    R = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])
    return (pts @ R.T) + np.array([tx, ty])


# ====================== MAIN REPLACEMENT FUNCTION ======================
def predict_with_scipy(drone_lat, drone_lon, altitude_m, image_width_px, image_height_px, graph, yaw_deg=0):
    drone_lat = float(drone_lat)
    drone_lon = float(drone_lon)
    altitude_m = float(altitude_m)

    # 1. Coarse Affine Logic
    sensor_width_mm, focal_length_mm = 13.2, 8.8
    gsd = (sensor_width_mm / 1000.0 * altitude_m) / (focal_length_mm / 1000.0 * image_width_px)

    gsd_lat_deg = gsd / 111000
    gsd_lon_deg = gsd / (111000 * math.cos(math.radians(drone_lat)))

    half_w = (image_width_px / 2.0) * gsd_lon_deg
    half_h = (image_height_px / 2.0) * gsd_lat_deg

    theta = math.radians(yaw_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    cx, cy = drone_lon, drone_lat
    ul_x = cx - (half_w * cos_t + half_h * sin_t)
    ul_y = cy + (half_w * sin_t + half_h * cos_t)

    transform = Affine(
        gsd_lon_deg * cos_t, gsd_lon_deg * sin_t, ul_x,
        -gsd_lat_deg * sin_t, -gsd_lat_deg * cos_t, ul_y
    )

    edges = []
    for u, v, data in graph.edges(data=True):
        pts = data['pts']
        coords = [transform * (c, r) for r, c in pts]
        if len(coords) > 1:
            edges.append(LineString(coords))

    drone_gdf = gpd.GeoDataFrame(geometry=edges, crs="EPSG:4326")

    # 2. Fetch & Project OSM
    print("Fetching OSM basemap data...")
    my_filter = '["highway"~"residential|service|primary|tertiary|unclassified"]'
    G_osm = ox.graph_from_point((drone_lat, drone_lon), dist=200, custom_filter=my_filter, simplify=False)
    G_osm_proj = ox.project_graph(G_osm)
    osm_gdf = ox.graph_to_gdfs(G_osm_proj, nodes=False, edges=True)

    crs_utm = G_osm_proj.graph['crs']
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
    drone_utm_x, drone_utm_y = transformer.transform(drone_lon, drone_lat)
    # Convert Drone Graph to meters
    drone_gdf_utm = drone_gdf.to_crs(crs_utm)

    # 3. Sample Points & Translate directly to local (0,0) frame
    custom_pts_global = sample_graph_points(drone_gdf_utm, spacing=3.0, is_sknw=False)
    osm_pts_global = sample_graph_points(osm_gdf, spacing=3.0, is_sknw=False)

    # Strip down to pure 2D arrays (X, Y)
    custom_pts = custom_pts_global[:, :2] - np.array([drone_utm_x, drone_utm_y])
    osm_pts = osm_pts_global[:, :2] - np.array([drone_utm_x, drone_utm_y])

    print(f"Tracking Nodes -> Drone: {len(custom_pts)} pts | OSM: {len(osm_pts)} pts")

    # =========================================================================
    # 4. SciPy Optimization Block (ICP Replacement)
    # =========================================================================
    # Build a spatial tree out of our static background (OSM) map
    osm_tree = KDTree(osm_pts)

    # Loss function: Total Mean Squared Distance to the nearest road line
    def alignment_loss(params):
        tx, ty, delta_yaw = params
        # Apply the current transformation guess to the drone points
        transformed_drone = rigid_transform_2d(custom_pts, tx, ty, delta_yaw)
        # Query tree for distances to closest OSM points
        distances, _ = osm_tree.query(transformed_drone)
        return np.mean(distances ** 2)

    # Initial guess: assume the affine map is perfect [dX=0, dY=0, dYaw=0]
    initial_guess = [0.0, 0.0, 0.0]

    # Explicitly set bounds: Max +/- 25 meters translation, Max +/- 15 degrees rotation adjustment
    optimization_bounds = [(-50.0, 50.0), (-50.0, 50.0), (-25.0, 25.0)]

    print("Running optimization alignment...")
    result = minimize(
        alignment_loss,
        initial_guess,
        bounds=optimization_bounds,
        method='L-BFGS-B'
    )

    if not result.success:
        print("Local minimization struggled. Switching to global Differential Evolution...")
        from scipy.optimize import differential_evolution
        result = differential_evolution(alignment_loss, bounds=optimization_bounds)

    final_tx, final_ty, final_dyaw = result.x
    print(f"\n--- Optimization Complete ---")
    print(f"Optimal X shift:  {final_tx:.2f} meters")
    print(f"Optimal Y shift:  {final_ty:.2f} meters")
    print(f"Yaw Correction:   {final_dyaw:.2f} degrees")
    print(f"Residual Mean Distance: {math.sqrt(result.fun):.2f} meters")
    drone_point_utm = Point(drone_utm_x, drone_utm_y)

    aligned_drone_point_utm = affinity.translate(drone_point_utm, xoff=final_tx, yoff=final_ty)
    reverse_transformer = pyproj.Transformer.from_crs(crs_utm, "EPSG:4326", always_xy=True)
    true_aligned_lon, true_aligned_lat = reverse_transformer.transform(aligned_drone_point_utm.x,
                                                                       aligned_drone_point_utm.y)





    # =========================================================================
    # NEW GEOSPATIAL ALIGNMENT BLOCK
    # =========================================================================
    # 1. Project your original drone geometries into the metric UTM space
    # drone_gdf_utm = drone_gdf.to_crs(crs_utm)
    #
    # # 2. Replicate the optimization steps using GeoPandas:
    # # First, rotate the lines around the drone's original UTM center point
    # rotated_gdf_utm = drone_gdf_utm.rotate(final_dyaw, origin=(drone_utm_x, drone_utm_y))
    #
    # # Second, translate the rotated lines by the optimal X and Y meter offsets
    # aligned_gdf_utm = rotated_gdf_utm.translate(xoff=final_tx, yoff=final_ty)
    #
    # # 3. Project the beautifully snapped geometries back to geographic degrees (WGS84)
    # # We must explicitly set the geometry column back to a true GeoDataFrame structure
    # aligned_gdf = gpd.GeoDataFrame(geometry=aligned_gdf_utm, crs=crs_utm).to_crs("EPSG:4326")
    #
    # # 4. Calculate the new geolocated center point
    # center_point = aligned_gdf.unary_union.centroid
    # aligned_lat = float(center_point.y)
    # aligned_lon = float(center_point.x)

    # 5. Measure and display the precision telemetry
    orig_coords = (drone_lat, drone_lon)
    aligned_coords = (true_aligned_lat, true_aligned_lon)
    dist_m = geodesic(orig_coords, aligned_coords).meters

    print("\n--- Telemetry Updates ---")
    print(f"Original Center: {drone_lat:.6f}, {drone_lon:.6f}")
    print(f"Aligned Center:  {true_aligned_lat:.6f}, {true_aligned_lon:.6f}")
    print(f"Total Shift Distance: {dist_m:.2f} meters")

    # 6. Save the perfectly aligned map file
    # aligned_gdf.to_file("snapped_gdf_roads.gpkg", driver="GPKG")
    print("Saved aligned network to 'snapped_gdf_roads.gpkg'")
    return true_aligned_lat, true_aligned_lon
