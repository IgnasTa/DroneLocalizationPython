import numpy as np
import tifffile
from skimage.morphology import skeletonize
import numpy as np
from skimage.morphology import medial_axis, disk, dilation
from skimage.transform import resize
from skimage import io
import cv2


def process_skeleton(input_path, output_path):
    # 1. Load the TIF image
    # We use tifffile to handle potentially large 4K images efficiently
    print(f"Loading image from {input_path}...")
    mask = tifffile.imread(input_path)

    # 2. Ensure binary format
    # The skeletonize function expects a boolean array
    # If the image is uint8 (0 and 255), we convert it to boolean
    if mask.max() > 1:
        mask = mask > 0

    # 3. Perform Skeletonization
    # 'method' can be 'lee' (default) which is generally better for
    # preserving topology in road networks
    print("Computing skeleton...")
    skeleton = skeletonize(mask.astype(bool))

    # 4. Convert back to uint8 for saving (0 and 255)
    result = (skeleton.astype(np.uint8)) * 255

    # 5. Save the result
    print(f"Saving skeleton to {output_path}...")
    h, w = skeleton.shape
    clean_skeleton = np.zeros_like(skeleton)
    # h_start, h_end = int(h * 0.10), int(h * 0.90)
    # w_start, w_end = int(w * 0.10), int(w * 0.90)
    h_start, h_end = int(h * 0.01), int(h * 0.99)
    w_start, w_end = int(w * 0.01), int(w * 0.99)
    clean_skeleton[h_start:h_end, w_start:w_end] = skeleton[h_start:h_end, w_start:w_end]
    skeleton_to_save = (clean_skeleton * 255).astype(np.uint8)
    # skeleton_to_save = (skeleton * 255).astype(np.uint8)
    tifffile.imwrite('drone_road_skeleton_3346.tif', skeleton_to_save, compression='lzw')
    print("Done! Edges have been zeroed out.")


def skeletonize_downsample_upsample(mask_path, target_size=512, return_distance=False):
    """
    Downsample → Medial Axis → Upsample skeleton
    """
    # 1. Load mask
    mask = io.imread(mask_path, as_gray=True)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask > 0  # or your proper threshold

    # 2. Downsample (preserve aspect ratio)
    h, w = mask.shape
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    mask_small = resize(mask.astype(float), (new_h, new_w),
                        order=0,  # nearest neighbor for binary
                        anti_aliasing=False,
                        preserve_range=True).astype(bool)

    print(f"Downsampled from {h}x{w} → {new_h}x{new_w}")

    # 3. Compute Medial Axis on small image
    if return_distance:
        skeleton_small, distance_small = medial_axis(mask_small, return_distance=True)
    else:
        skeleton_small = medial_axis(mask_small)
        distance_small = None
    # footprint = disk(2)                    # radius = thick_px
    # skeleton_small = dilation(skeleton_small, footprint)
    # 4. Upsample skeleton back to original size
    skeleton = resize(skeleton_small.astype(float), (h, w),
                      order=0,  # nearest neighbor - crucial for thin lines
                      anti_aliasing=False,
                      preserve_range=True).astype(bool)

    # Optional: Light morphological cleaning
    # from skimage.morphology import remove_small_objects
    # skeleton = remove_small_objects(skeleton, min_size=20)

    if return_distance:
        # Upsample distance transform too (with linear interp)
        dist_up = resize(distance_small, (h, w), order=1, preserve_range=True)
        return skeleton, dist_up
    return skeleton


def skeletonize_mask(mask_path = "drone_road_mask_rotated.png", output_file = "drone_road_skeleton_3346.tif"):
    skeleton = skeletonize_downsample_upsample(mask_path, target_size=1024)
    print(skeleton.shape)

    print(f"Saving skeleton...")
    h, w = skeleton.shape
    clean_skeleton = np.zeros_like(skeleton)
    h_start, h_end = int(h * 0.01), int(h * 0.99)
    w_start, w_end = int(w * 0.01), int(w * 0.99)
    clean_skeleton[h_start:h_end, w_start:w_end] = skeleton[h_start:h_end, w_start:w_end]
    skeleton_to_save = (clean_skeleton * 255).astype(np.uint8)
    tifffile.imwrite('drone_road_skeleton_33466.tif', skeleton_to_save, compression='lzw')
    input_file = "drone_road_skeleton_33466.tif"
    process_skeleton(input_file, output_file)
