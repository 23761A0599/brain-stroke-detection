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


def add_label_banner(img, text, accent=(60, 220, 90)):
    h, w = img.shape[:2]
    banner_h = max(26, int(h * 0.10))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (20, 20, 20), -1)
    img = cv2.addWeighted(overlay, 0.78, img, 0.22, 0)
    cv2.rectangle(img, (0, h - banner_h), (5, h), accent, -1)
    font_scale = max(0.35, w / 900)
    cv2.putText(img, text, (12, h - banner_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_lime_explanation(model, image_path):
    timestamp = int(time.time())

    raw_img = Image.open(image_path).convert("RGB")
    img_np = np.array(raw_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_img = transform(raw_img).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor_img)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    class_label = CLASS_NAMES[pred_idx]
    is_hemorrhage = "hemorrhag" in class_label.lower() or pred_idx == 0
    label_text = f"{class_label}  |  {confidence * 100:.1f}% confidence"

    segments = slic(img_np, n_segments=50, compactness=10, start_label=1)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _, brain_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    result_bgr = gray_bgr.copy()

    if is_hemorrhage:
        std_dev = cv2.Laplacian(gray, cv2.CV_64F)
        high_var_mask = np.uint8(np.abs(std_dev) > 20)
        high_var_mask = cv2.bitwise_and(high_var_mask, brain_mask)

        # ONLY affected segments get colored - everything else stays grayscale
        for seg_val in np.unique(segments):
            mask = (segments == seg_val)
            if np.sum(high_var_mask[mask]) > 10:
                result_bgr[mask] = result_bgr[mask] * 0.25 + np.array([40, 40, 235]) * 0.75

        marked = mark_boundaries(result_bgr, segments, color=(0.55, 0.55, 0.55))
        result = cv2.cvtColor(np.uint8(marked * 255), cv2.COLOR_RGB2BGR)
        result = add_label_banner(result, f"LIME  •  {label_text}", accent=(50, 50, 235))
    else:
        # Healthy scan - stays clean grayscale, no coloring needed
        marked = mark_boundaries(result_bgr, segments, color=(0.5, 0.5, 0.5))
        result = cv2.cvtColor(np.uint8(marked * 255), cv2.COLOR_RGB2BGR)
        result = add_label_banner(result, f"LIME  •  {label_text}", accent=(60, 220, 90))

    filename = f"lime_{timestamp}.png"
    output_path = LIME_DIR / filename
    cv2.imwrite(str(output_path), result)

    return {"prediction": class_label, "lime_path": f"outputs/lime/{filename}"}