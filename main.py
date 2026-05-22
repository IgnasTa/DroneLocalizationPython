from fastapi import FastAPI, UploadFile, File, Query
import torch
import cv2
import os
import math
import numpy as np

from DiscardSmallObjects import remove_small_objects
from ExifMetadata import extrat_lat_lon
from MatchToMap import predict
from MatchToMapWIthRotation import predict_with_rotation
from MatchToMapv3 import predict_with_scipy
from PredictLargeImage import predict_large_image
from RotateImage import correct_yaw
from SkeletonToGraph import skeleton_to_graph
from Skeletonize import skeletonize_mask
from UNetFormer import UNetFormer
from YawExtract import extract_xmp_pure_python

app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNetFormer().to(device)
state_dict = torch.load("unet_former_with_moving_and_static_cars_as_road.pth", map_location=device)
model.load_state_dict(state_dict)

# Global variables for video processing
video_metadata: dict = {
    "lat": None,
    "lon": None,
    "altitude": None
}
video_processing_lock = False

@app.post("/")
async def root(file: UploadFile = File(...)):
    """Upload an image and get aligned GPS coordinates"""
    try:
        addNoise = False

        # 1. Save uploaded file
        photo_path = f"uploaded_{file.filename}"
        contents = await file.read()
        with open(photo_path, "wb") as f:
            f.write(contents)
        print(f"✅ File saved: {photo_path}")

        # 2. Extract metadata
        yaw, altitude = extract_xmp_pure_python(photo_path)
        drone_lat, drone_lon = extrat_lat_lon(photo_path)
        drone_lat_noisy = drone_lat
        drone_lon_noisy = drone_lon
        print(f"Original GPS: ({drone_lat:.6f}, {drone_lon:.6f}), Yaw: {yaw}°, Altitude: {altitude}m")

        if addNoise:
            error_m = np.random.uniform(5, 5)  # Random error
            error_direction = np.random.uniform(0, 2 * np.pi)  # Random direction
            lat_error_deg = error_m * math.cos(error_direction) / 111000
            lon_error_deg = error_m * math.sin(error_direction) / (111000 * math.cos(math.radians(drone_lat)))
            print(f"Added GPS error: {error_m:.1f}m at {math.degrees(error_direction):.1f}°")
            print(f"Error in degrees: lat={lat_error_deg:.8f}°, lon={lon_error_deg:.8f}°")
            print(f"Error in meters: lat={lat_error_deg * 111000:.2f}m, lon={lon_error_deg * (111000 * math.cos(math.radians(drone_lat))):.2f}m")

            drone_lat_noisy = drone_lat + lat_error_deg
            drone_lon_noisy = drone_lon + lon_error_deg


        # 3. Read image and get dimensions
        img = cv2.imread(photo_path)
        X, Y, _ = img.shape
        print(f"Image dimensions: {Y}x{X} pixels")

        # 4. Predict road mask
        img, pred = predict_large_image(model, img, device)
        mask = remove_small_objects(pred)

        # 5. Rotate by yaw
        correct_yaw(mask, yaw)

        # 6. Skeletonize
        skeletonize_mask("drone_road_mask_rotated.png", "drone_road_skeleton_3346.tif")

        # 7. Convert to graph
        graph = skeleton_to_graph("drone_road_skeleton_3346.tif")

        # 8. Match to OSM and get aligned coordinates
        lat, lon = predict_with_scipy(
            drone_lat=drone_lat_noisy,
            drone_lon=drone_lon_noisy,
            altitude_m=altitude,
            image_width_px=Y,
            image_height_px=X,
            graph=graph
        )

        # 10. Clean up
        os.remove(photo_path)

        return {
            "status": "success",
            "original_gps": {"lat": float(drone_lat_noisy), "lon": float(drone_lon_noisy)},
            "aligned_gps": {"lat": lat, "lon": lon},
            "aligned_center": f"{lat:.6f}, {lon:.6f}"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/video/metadata")
def setVideoMetadata(lat: float = Query(...),
                           lon: float = Query(...),
                           altitude: float = Query(...)):
    """Set global GPS metadata for video processing"""
    global video_metadata
    video_metadata["lat"] = lat
    video_metadata["lon"] = lon
    video_metadata["altitude"] = altitude
    print(f"✅ Video metadata set - GPS: ({lat:.6f}, {lon:.6f}), Altitude: {altitude}m")
    return {
        "status": "success",
        "message": "Metadata set successfully",
        "lat": lat,
        "lon": lon,
        "altitude": altitude
    }


@app.post("/video")
async def photoFromVideo(file: UploadFile = File(...)):
    """Upload an image and get aligned GPS coordinates using previously set metadata"""
    global video_metadata, video_processing_lock

    # Check if already processing
    if video_processing_lock:
        return {"status": "error", "message": "Video processing already in progress. Please wait for the current request to complete."}

    # Set lock
    video_processing_lock = True

    try:
        # Check if metadata has been set
        if video_metadata["lat"] is None or video_metadata["lon"] is None or video_metadata["altitude"] is None:
            return {"status": "error", "message": "Metadata not set. Please call /video/metadata first."}

        addNoise = False
        # 1. Save uploaded file
        photo_path = f"uploaded_{file.filename}"
        contents = await file.read()
        with open(photo_path, "wb") as f:
            f.write(contents)
        print(f"✅ File saved: {photo_path}")

        # 2. Extract yaw from metadata, use provided lat, lon, altitude
        yaw = 0
        drone_lat = video_metadata["lat"]
        drone_lon = video_metadata["lon"]
        altitude = video_metadata["altitude"]
        drone_lat_noisy = drone_lat
        drone_lon_noisy = drone_lon
        print(f"Using GPS: ({drone_lat:.6f}, {drone_lon:.6f}), Yaw: {yaw}°, Altitude: {altitude}m")

        if addNoise:
            # 3. Add random GPS error (~10-20m)
            error_m = np.random.uniform(5, 12)  # Random error 10-20 meters
            error_direction = np.random.uniform(0, 2 * np.pi)  # Random direction

            # Convert meters to degrees
            lat_error_deg = error_m * math.cos(error_direction) / 111000  # 1° lat ≈ 111km
            lon_error_deg = error_m * math.sin(error_direction) / (111000 * math.cos(math.radians(drone_lat)))
            print(f"Added GPS error: {error_m:.1f}m at {math.degrees(error_direction):.1f}°")
            print(f"Error in degrees: lat={lat_error_deg:.8f}°, lon={lon_error_deg:.8f}°")
            print(f"Error in meters: lat={lat_error_deg * 111000:.2f}m, lon={lon_error_deg * (111000 * math.cos(math.radians(drone_lat))):.2f}m")

            # Apply error
            drone_lat_noisy = drone_lat + lat_error_deg
            drone_lon_noisy = drone_lon + lon_error_deg

        # 3. Read image and get dimensions
        img = cv2.imread(photo_path)
        X, Y, _ = img.shape
        print(f"Image dimensions: {Y}x{X} pixels")

        # 4. Predict road mask
        img, pred = predict_large_image(model, img, device)
        mask = remove_small_objects(pred)

        # 5. Rotate by yaw
        correct_yaw(mask, yaw)

        # 6. Skeletonize
        skeletonize_mask("drone_road_mask_rotated.png", "drone_road_skeleton_3346.tif")

        # 7. Convert to graph
        graph = skeleton_to_graph("drone_road_skeleton_3346.tif")

        # 8. Match to OSM and get aligned coordinates
        aligned_gdf = predict_with_rotation(
            drone_lat=drone_lat_noisy,
            drone_lon=drone_lon_noisy,
            altitude_m=altitude,
            image_width_px=Y,
            image_height_px=X,
            graph=graph
        )

        # 9. Extract aligned center coordinates
        center_point = aligned_gdf.unary_union.centroid

        aligned_lat = float(center_point.y)
        aligned_lon = float(center_point.x)

        # 10. Clean up
        os.remove(photo_path)
        print()
        return {
            "status": "success",
            "original_gps": {"lat": float(drone_lat), "lon": float(drone_lon)},
            "aligned_gps": {"lat": aligned_lat, "lon": aligned_lon},
            "aligned_center": f"{aligned_lat:.6f}, {aligned_lon:.6f}"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        video_processing_lock = False
