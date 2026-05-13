from scipy.optimize import minimize
from shapely.geometry import LineString, Point
from geopy.distance import geodesic
from rasterio.transform import Affine
import geopandas as gpd
import osmnx as ox
import numpy as np
from shapely.geometry import Point
import math

def predict(drone_lat, drone_lon, altitude_m, image_width_px, image_height_px, graph, yaw_deg=0):
    drone_lat = float(drone_lat)
    drone_lon = float(drone_lon)
    altitude_m = float(altitude_m)

    # Keep everything in EPSG:4326 (WGS84)
    target_crs = "EPSG:4326"  # Changed from EPSG:3346

    sensor_width_mm, focal_length_mm = 13.2, 8.8

    # 1. Calculate GSD and Transform
    gsd = (sensor_width_mm / 1000.0 * float(altitude_m)) / (focal_length_mm / 1000.0 * image_width_px)

    # Convert GSD from meters to degrees (approximate)
    # 1 degree lat ≈ 111,000 meters, 1 degree lon ≈ 111,000 * cos(lat) meters
    gsd_lat_deg = gsd / 111000
    gsd_lon_deg = gsd / (111000 * math.cos(math.radians(drone_lat)))

    half_w = (image_width_px / 2.0) * gsd_lon_deg   # in degrees
    half_h = (image_height_px / 2.0) * gsd_lat_deg  # in degrees

    pixel_size_lat = gsd_lat_deg
    pixel_size_lon = gsd_lon_deg

    theta = math.radians(yaw_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # 2. Center point in EPSG:4326 (no conversion needed)
    cx, cy = drone_lon, drone_lat

    # 3. Compute Affine Transform (Top-Left corner) in degrees
    ul_x = cx - (half_w * cos_t + half_h * sin_t)
    ul_y = cy + (half_w * sin_t + half_h * cos_t)  # Note: Y is latitude (increases north)

    # IMPORTANT: The second row of the Affine matrix for geographic coordinates
    transform = Affine(pixel_size_lon * cos_t,  pixel_size_lon * sin_t,  ul_x,
                       -pixel_size_lat * sin_t, -pixel_size_lat * cos_t, ul_y)

    return align_by_shape_overlap(graph, drone_lat, drone_lon, transform)


def align_by_shape_overlap(graph, drone_lat, drone_lon, transform, max_dist=20, buffer_m=100):
    # Hardcoded flag to switch between implementations
    use_weighted_sampling = True  # Set to False for original, True for weighted

    edges = []
    for u, v, data in graph.edges(data=True):
        pts = data['pts']
        # Convert pixel (r, c) -> (c, r) then apply transform
        coords = [transform * (c, r) for r, c in pts]
        if len(coords) > 1:  # LineString needs at least 2 points
            edges.append(LineString(coords))

    # Keep in EPSG:4326
    drone_gdf = gpd.GeoDataFrame(geometry=edges, crs="EPSG:4326")

    # 2. Fetch OSM Data
    print(f"Fetching OSM data for snapping...")
    my_filter = '["highway"~"residential|service|primary|tertiary|unclassified"]'
    G_osm = ox.graph_from_point(
        (drone_lat, drone_lon),
        dist=buffer_m,
        network_type='all',
        custom_filter=my_filter,
        simplify=False
    )
    # Keep OSM data in EPSG:4326 (no conversion)
    osm_gdf = ox.graph_to_gdfs(G_osm, nodes=False, edges=True)

    # 1. Prepare the "Target" (OSM)
    # We use unary_union so we can calculate distance to the whole network at once
    target_shape = osm_gdf.unary_union

    # 2. Extract all vertices from the drone graph to represent its "Shape"
    if use_weighted_sampling:
        # Suggested: Weighted sampling - 2x weight for lines with > 3 points
        sample_points = []
        point_weights = []

        for line in drone_gdf.geometry:
            # Convert line length from degrees to meters for sampling
            line_length_m = geodesic((line.coords[0][1], line.coords[0][0]),
                                   (line.coords[-1][1], line.coords[-1][0])).meters

            # Sample every 5 meters, but convert to parameter along line
            num_samples = max(1, int(line_length_m / 5))
            for i in range(num_samples):
                param = i / max(1, num_samples - 1)  # 0 to 1
                point = line.interpolate(param, normalized=True)
                sample_points.append((point.x, point.y))  # lon, lat
                # Weight based on number of samples per line
                weight = 2.0 if num_samples > 3 else 1.0
                point_weights.append(weight)

        pts = np.array(sample_points)  # Shape: (N, 2) with [lon, lat]
        weights = np.array(point_weights)
    else:
        # Current: Simple sampling every 5 meters, no weights
        sample_points = []
        for line in drone_gdf.geometry:
            # Convert line length from degrees to meters for sampling
            line_length_m = geodesic((line.coords[0][1], line.coords[0][0]),
                                   (line.coords[-1][1], line.coords[-1][0])).meters

            # Sample every 5 meters
            num_samples = max(1, int(line_length_m / 5))
            for i in range(num_samples):
                param = i / max(1, num_samples - 1)  # 0 to 1
                point = line.interpolate(param, normalized=True)
                sample_points.append((point.x, point.y))  # lon, lat

        pts = np.array(sample_points)
        weights = np.ones(len(pts))  # All weights = 1

    # 3. Define the "Overlap Cost" function
    def cost_function(params):
        dx_deg, dy_deg = params  # Now in degrees

        # Apply shift to our sample points (in degrees)
        shifted_pts_lon = pts[:, 0] + dx_deg  # longitude
        shifted_pts_lat = pts[:, 1] + dy_deg  # latitude

        # Calculate weighted geodesic distance from every shifted point to the OSM shape
        total_dist = 0
        for i in range(len(shifted_pts_lon)):
            p = Point(shifted_pts_lon[i], shifted_pts_lat[i])
            # Use geodesic distance in meters
            dist_m = p.distance(target_shape)
            total_dist += weights[i] * (dist_m ** 2)  # Weight the squared distance

        return total_dist / len(pts)

    # 4. Minimize the cost (Find the best overlap)
    print(f"Optimizing shape overlap... (weighted={use_weighted_sampling})")
    # Start with (0,0) shift as initial guess
    res = minimize(cost_function, x0=[0, 0], method='Nelder-Mead')

    best_dx_deg, best_dy_deg = res.x
    print(f"Optimal Shape Match: ΔX={best_dx_deg:.6f}°, ΔY={best_dy_deg:.6f}°")

    # 5. Apply the final rigid shift to the whole graph
    aligned_gdf = drone_gdf.translate(xoff=best_dx_deg, yoff=best_dy_deg)
    center_point = aligned_gdf.unary_union.centroid

    print(f"Aligned center (EPSG:4326): {center_point.y:.6f}, {center_point.x:.6f}")

    # No conversion needed - already in WGS84
    aligned_lat = float(center_point.y)
    aligned_lon = float(center_point.x)

    orig_coords = (drone_lat, drone_lon)
    aligned_coords = (aligned_lat, aligned_lon)
    # Calculate distance in meters
    dist_m = geodesic(orig_coords, aligned_coords).meters
    print(f"Original Center: {drone_lat:.6f}, {drone_lon:.6f}")
    print(f"Aligned Center:  {aligned_lat:.6f}, {aligned_lon:.6f}")
    print(f"Total Shift Distance: {dist_m:.2f} meters")
    aligned_gdf.to_file("snapped_gdf_roads.gpkg", driver="GPKG")
    return aligned_gdf