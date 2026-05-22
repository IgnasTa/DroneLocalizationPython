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
def predict_with_scipy(
    drone_lat,
    drone_lon,
    altitude_m,
    image_width_px,
    image_height_px,
    graph,
    yaw_deg=0,
    fov=None,
    sensor_width_mm=13.2,
    focal_length_mm=8.8,
    coordinate_system="EPSG:4326"

):
    drone_lat = float(drone_lat)
    drone_lon = float(drone_lon)
    altitude_m = float(altitude_m)

    # Basic input validation
    if float(image_width_px) <= 0 or float(image_height_px) <= 0:
        raise ValueError("image_width_px and image_height_px must be positive integers")
    if fov is not None:
        try:
            fov_deg = float(fov)
        except Exception:
            raise ValueError("fov must be numeric (degrees) or None")
        if not (0.0 < fov_deg < 180.0):
            print(f"Warning: fov value {fov_deg} is outside (0, 180) degrees; results may be invalid.")
        ground_width_m = 2.0 * altitude_m * math.tan(math.radians(fov_deg) / 2.0)
        gsd = ground_width_m / float(image_width_px)
        used_method = f"fov-based ({fov_deg} deg)"
    else:
        gsd = (sensor_width_mm / 1000.0 * altitude_m) / (focal_length_mm / 1000.0 * image_width_px)
        used_method = f"sensor/focal-based (sensor {sensor_width_mm} mm, focal {focal_length_mm} mm)"

    gsd_lat_deg = gsd / 111000
    gsd_lon_deg = gsd / (111000 * math.cos(math.radians(drone_lat)))
    print(f"Using GSD calculation method: {used_method}; computed GSD = {gsd:.4f} m/px")
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

    drone_gdf = gpd.GeoDataFrame(geometry=edges, crs=coordinate_system)

    # 2. Fetch & Project OSM
    print("Fetching OSM basemap data...")
    my_filter = '["highway"~"residential|service|primary|tertiary|unclassified"]'
    G_osm = ox.graph_from_point((drone_lat, drone_lon), dist=200, custom_filter=my_filter, simplify=False)
    G_osm_proj = ox.project_graph(G_osm)
    osm_gdf = ox.graph_to_gdfs(G_osm_proj, nodes=False, edges=True)

    crs_utm = G_osm_proj.graph['crs']
    transformer = pyproj.Transformer.from_crs(coordinate_system, crs_utm, always_xy=True)
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

    # 4. SciPy Optimization Block
    osm_tree = KDTree(osm_pts)

    def alignment_loss(params):
        tx, ty, delta_yaw = params
        transformed_drone = rigid_transform_2d(custom_pts, tx, ty, delta_yaw)
        distances, _ = osm_tree.query(transformed_drone)
        return np.mean(distances ** 2)

    best_result = None
    min_error = float('inf')

    print("Running optimization across 30-degree rotation steps...")

    # Iterate through rotations: 0, 30, 60... 330
    for i in range(12):
        initial_yaw_guess = i * 30.0
        initial_guess = [0.0, 0.0, initial_yaw_guess]


        bounds = [(-50.0, 50.0), (-50.0, 50.0), (initial_yaw_guess - 20.0, initial_yaw_guess + 20.0)]

        result = minimize(
            alignment_loss,
            initial_guess,
            bounds=bounds,
            method='L-BFGS-B'
        )

        current_error = result.fun
        if current_error < min_error:
            min_error = current_error
            best_result = result
            print(f"  New best found at start yaw {initial_yaw_guess}°: Error {math.sqrt(current_error):.2f}m")

    # Use the best result found
    final_tx, final_ty, final_dyaw = best_result.x

    # ... (Keep the rest of your original logic for updating telemetry and returning coordinates)
    drone_point_utm = Point(drone_utm_x, drone_utm_y)
    aligned_drone_point_utm = affinity.translate(drone_point_utm, xoff=final_tx, yoff=final_ty)

    reverse_transformer = pyproj.Transformer.from_crs(crs_utm, coordinate_system, always_xy=True)
    true_aligned_lon, true_aligned_lat = reverse_transformer.transform(
        aligned_drone_point_utm.x, aligned_drone_point_utm.y
    )

    return true_aligned_lat, true_aligned_lon
