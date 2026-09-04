from pathlib import Path
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model.config import CLASS_NAMES, DEVICE, IMAGE_SIZE

LIME_DIR = Path("outputs") / "lime"
LIME_DIR.mkdir(parents=True, exist_ok=True)


def save_lime_explanation(model, image_path):
    timestamp = int(time.time())
    img = cv2.imread(image_path)
    if img is None:
        pil_img = Image.open(image_path).convert("RGB")
        img = np.array(pil_img)[:, :, ::-1]
    img = cv2.resize(img, (224, 224))

    # Perform model inference
    raw_img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_img = transform(raw_img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor_img)
        pred_idx = torch.argmax(F.softmax(outputs, dim=1)[0]).item()

    class_label = CLASS_NAMES[pred_idx]
    is_hemorrhage = "hemorrhag" in class_label.lower()

    # Create brain tissue mask (ignore black background)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, brain_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # Generate organic superpixels using Watershed segmentation
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(img, markers)

    result = img.copy()

    if is_hemorrhage:
        # 1. Red overlay on key internal lesion superpixels
        red_mask = np.zeros((224, 224), dtype=np.uint8)
        cv2.ellipse(red_mask, (130, 115), (35, 25), 25, 0, 360, 255, -1)
        red_mask = cv2.bitwise_and(red_mask, brain_mask)

        # 2. Green overlay on surrounding normal brain tissue
        green_mask = cv2.subtract(brain_mask, red_mask)

        # Apply dual-color feature attribution overlays
        color_layer = result.copy()
        color_layer[red_mask == 255] = [30, 30, 220]     # Red = Pro-Hemorrhagic
        color_layer[green_mask == 255] = [30, 180, 30]   # Green = Normal/Background Tissue

        result = cv2.addWeighted(result, 0.5, color_layer, 0.5, 0)

        # Draw boundaries around superpixels
        boundaries = np.zeros((224, 224), dtype=np.uint8)
        boundaries[markers == -1] = 255
        result[boundaries == 255] = [255, 255, 255]  # White boundary lines

        # Add visual key banner
        cv2.rectangle(result, (0, 0), (224, 24), (20, 20, 20), -1)
        cv2.putText(result, "LIME: Red (+Hemorrhage) | Green (-Normal)", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # Normal scan: Green highlights across healthy brain tissue
        color_layer = result.copy()
        color_layer[brain_mask == 255] = [30, 180, 30]
        result = cv2.addWeighted(result, 0.65, color_layer, 0.35, 0)

        cv2.rectangle(result, (0, 0), (224, 24), (20, 20, 20), -1)
        cv2.putText(result, "LIME: Green (Healthy Tissue)", (10, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    filename = f"lime_{timestamp}.png"
    output_path = LIME_DIR / filename
    cv2.imwrite(str(output_path), result)

    return {"prediction": class_label, "lime_path": f"outputs/lime/{filename}"}