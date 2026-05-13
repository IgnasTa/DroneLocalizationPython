import cv2
import numpy as np

def correct_yaw(mask_to_save, angle):
    print(angle)
    h, w = mask_to_save.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -1 * float(angle), 1.0)

    # rotate with nearest neighbor
    rotated_mask = cv2.warpAffine(
        mask_to_save,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderValue=0  # background stays black
    )

    blured_mask = cv2.GaussianBlur(rotated_mask, (25,25), 0)
    cv2.imwrite("drone_road_mask_rotated.png", blured_mask)