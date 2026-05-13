from skimage import morphology
import numpy as np
import cv2

def remove_small_objects(pred):
    mask = morphology.remove_small_objects(pred.astype(bool), min_size=80000)
    h, w = mask.shape
    small_mask = cv2.resize(mask.astype(np.uint8), (w//4, h//4), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    small_mask = cv2.morphologyEx(small_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_to_save = (final_mask * 255).astype(np.uint8)

    return mask_to_save
