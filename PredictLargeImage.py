import cv2
import numpy as np
import torch


def predict_large_image(model, img, device, tile_size=1024, stride=512):
    model.eval()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape

    full_pred = np.zeros((H, W), dtype=np.uint8)

    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                # Calculate slice end points
                y_end = min(y + tile_size, H)
                x_end = min(x + tile_size, W)

                tile = img[y:y_end, x:x_end]

                # Check if padding is needed
                h_pad = tile_size - tile.shape[0]
                w_pad = tile_size - tile.shape[1]

                if h_pad > 0 or w_pad > 0:
                    # Pad the tile with zeros (or reflection padding)
                    tile = cv2.copyMakeBorder(tile, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)

                # Preprocess
                tile_tensor = torch.tensor(tile).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

                # Inference
                output = model(tile_tensor)
                pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

                # Crop the padding out of the prediction before placing back
                pred_cropped = pred[:tile.shape[0] - h_pad, :tile.shape[1] - w_pad]

                # Place back into the full map
                full_pred[y:y_end, x:x_end] = pred_cropped

    return img, full_pred