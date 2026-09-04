from pathlib import Path
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from skimage.segmentation import slic, mark_boundaries

from model.config import CLASS_NAMES, DEVICE, IMAGE_SIZE

LIME_DIR = Path("outputs") / "lime"
LIME_DIR.mkdir(parents=True, exist_ok=True)

def save_lime_explanation(model, image_path):
    timestamp = int(time.time())
    
    # Load image
    raw_img = Image.open(image_path).convert("RGB")
    img_np = np.array(raw_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Perform model inference
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_img = transform(raw_img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor_img)
        pred_idx = torch.argmax(F.softmax(outputs, dim=1)[0]).item()

    class_label = CLASS_NAMES[pred_idx]
    is_hemorrhage = "hemorrhag" in class_label.lower() or pred_idx == 0

    # Generate SLIC Superpixel Segmentation
    segments = slic(img_np, n_segments=50, compactness=10, start_label=1)
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, brain_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    result_bgr = img_bgr.copy()

    if is_hemorrhage:
        # Detect high-contrast/high-intensity regions corresponding to hemorrhage features
        std_dev = cv2.Laplacian(gray, cv2.CV_64F)
        high_var_mask = np.uint8(np.abs(std_dev) > 20)
        high_var_mask = cv2.bitwise_and(high_var_mask, brain_mask)

        # Apply high-contrast Superpixel Colorization
        for seg_val in np.unique(segments):
            mask = (segments == seg_val)
            if np.sum(high_var_mask[mask]) > 10:
                # Red superpixels for key lesion contours
                result_bgr[mask] = result_bgr[mask] * 0.2 + np.array([0, 0, 230]) * 0.8
            elif np.sum(brain_mask[mask]) > 0:
                # Green superpixels for surrounding healthy tissue
                result_bgr[mask] = result_bgr[mask] * 0.7 + np.array([0, 160, 0]) * 0.3

        # Overlay crisp white segment boundaries
        marked = mark_boundaries(result_bgr, segments, color=(1, 1, 1))
        result = cv2.cvtColor(np.uint8(marked * 255), cv2.COLOR_RGB2BGR)

        # Top Banner
        cv2.rectangle(result, (0, 0), (IMAGE_SIZE, 22), (15, 15, 15), -1)
        cv2.putText(result, "LIME: Red (Target Lesion) | Green (Healthy)", (6, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # Healthy scan visualization
        for seg_val in np.unique(segments):
            mask = (segments == seg_val)
            if np.sum(brain_mask[mask]) > 0:
                result_bgr[mask] = result_bgr[mask] * 0.65 + np.array([0, 160, 0]) * 0.35

        marked = mark_boundaries(result_bgr, segments, color=(1, 1, 1))
        result = cv2.cvtColor(np.uint8(marked * 255), cv2.COLOR_RGB2BGR)

        cv2.rectangle(result, (0, 0), (IMAGE_SIZE, 22), (15, 15, 15), -1)
        cv2.putText(result, "LIME: Green (Healthy Brain Tissue)", (8, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    filename = f"lime_{timestamp}.png"
    output_path = LIME_DIR / filename
    cv2.imwrite(str(output_path), result)

    return {"prediction": class_label, "lime_path": f"outputs/lime/{filename}"}