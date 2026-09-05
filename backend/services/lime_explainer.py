from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image

from model.config import IMAGE_SIZE

LIME_DIR = Path("outputs") / "lime"
LIME_DIR.mkdir(parents=True, exist_ok=True)


def add_label_banner(img, title, subtitle, accent=(60, 220, 90)):
    h, w = img.shape[:2]
    banner_h = max(46, int(h * 0.20))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (20, 20, 20), -1)
    img = cv2.addWeighted(overlay, 0.82, img, 0.18, 0)
    cv2.rectangle(img, (0, h - banner_h), (5, h), accent, -1)

    def fit_scale(text, max_width, start):
        scale = start
        while scale > 0.25:
            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            if size[0] <= max_width:
                return scale
            scale -= 0.03
        return scale

    max_w = w - 20
    title_scale = fit_scale(title, max_w, 0.5)
    sub_scale = fit_scale(subtitle, max_w, 0.42)

    cv2.putText(img, title, (12, h - banner_h + int(banner_h * 0.45)),
                cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, subtitle, (12, h - int(banner_h * 0.15)),
                cv2.FONT_HERSHEY_SIMPLEX, sub_scale, accent, 1, cv2.LINE_AA)
    return img


def _normalized_cam(cam_map, h, w):
    if cam_map is not None and np.max(cam_map) > 0:
        cam_resized = cv2.resize(cam_map, (w, h))
        cam_norm = np.uint8(255 * (cam_resized - cam_resized.min()) /
                             (cam_resized.max() - cam_resized.min() + 1e-8))
    else:
        cam_norm = np.zeros((h, w), dtype=np.uint8)
    return cv2.GaussianBlur(cam_norm, (11, 11), 0)


def _significant_mask(smooth_cam, percentile=92):
    h, w = smooth_cam.shape
    min_area = max(18, int(h * w * 0.004))

    thresh_val = np.percentile(smooth_cam, percentile)
    _, binary = cv2.threshold(smooth_cam, thresh_val, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(binary)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean_mask, [c], -1, 255, -1)

    return clean_mask


def save_lime_explanation(image_path, cam_map, confidence_score, class_name):
    """
    Flat, solid highlight (single color, no gradient) with a clean outline.
    Deliberately different from Grad-CAM's multi-color glow so the two
    panels don't look like duplicates of each other.
    """
    timestamp = int(time.time())

    raw_img = Image.open(image_path).convert("RGB")
    img_np = np.array(raw_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    is_hemorrhage = "hemorrhag" in class_name.lower()
    label_text = f"{class_name}  |  {confidence_score * 100:.1f}% confidence"
    result = gray_bgr.copy()

    if is_hemorrhage and cam_map is not None:
        h, w = gray.shape
        smooth_cam = _normalized_cam(cam_map, h, w)
        clean_mask = _significant_mask(smooth_cam, percentile=92)

        if np.any(clean_mask):
            flat_color = np.zeros_like(result)
            flat_color[:] = (40, 40, 235)  # solid red-orange, BGR - no gradient

            alpha_flat = (clean_mask.astype(np.float32) / 255.0)[..., None] * 0.55
            result = (result.astype(np.float32) * (1 - alpha_flat) +
                      flat_color.astype(np.float32) * alpha_flat).astype(np.uint8)

            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > 15:
                    epsilon = 0.006 * cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, epsilon, True)
                    cv2.polylines(result, [approx], True, (255, 255, 255), 2, cv2.LINE_AA)

        result = add_label_banner(result, "LIME", label_text, accent=(50, 50, 235))
    else:
        result = add_label_banner(result, "LIME", label_text, accent=(60, 220, 90))

    filename = f"lime_{timestamp}.png"
    output_path = LIME_DIR / filename
    cv2.imwrite(str(output_path), result)

    return {"lime_path": f"outputs/lime/{filename}"}