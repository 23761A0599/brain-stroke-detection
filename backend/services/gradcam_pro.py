import cv2
import numpy as np
import time
from pathlib import Path

OUTPUTS_DIR = Path("outputs")
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"
GRADCAM_PRO_DIR = OUTPUTS_DIR / "gradcam_pro"

for path in [GRADCAM_DIR, GRADCAM_PRO_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def add_label_banner(img, text, accent=(0, 200, 255)):
    """Clean dark bottom banner with an accent strip and confidence text."""
    h, w = img.shape[:2]
    banner_h = max(26, int(h * 0.10))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (20, 20, 20), -1)
    img = cv2.addWeighted(overlay, 0.78, img, 0.22, 0)
    cv2.rectangle(img, (0, h - banner_h), (5, h), accent, -1)
    font_scale = max(0.38, w / 850)
    cv2.putText(img, text, (12, h - banner_h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _normalized_cam(cam_map, h, w):
    if cam_map is not None and np.max(cam_map) > 0:
        cam_resized = cv2.resize(cam_map, (w, h))
        cam_norm = np.uint8(255 * (cam_resized - cam_resized.min()) /
                             (cam_resized.max() - cam_resized.min() + 1e-8))
    else:
        cam_norm = np.zeros((h, w), dtype=np.uint8)
    return cv2.GaussianBlur(cam_norm, (21, 21), 0)


def generate_gradcam_images(orig_bgr, cam_map, confidence_score, class_name):
    """
    Produces two visually distinct, clean explainability images:
    1. 'Heat Zone' Grad-CAM  - only the most active region is colored
    2. 'Circled Focus' Pro   - a single clean ring around the affected area
    """
    timestamp = int(time.time())

    if len(orig_bgr.shape) == 2 or orig_bgr.shape[2] == 1:
        orig_bgr = cv2.cvtColor(orig_bgr, cv2.COLOR_GRAY2BGR)

    h, w, _ = orig_bgr.shape
    smooth_cam = _normalized_cam(cam_map, h, w)
    label_text = f"{class_name}  |  {confidence_score * 100:.1f}% confidence"

    # =========================================================
    # 1. HEAT ZONE - only top ~15% activated pixels get colored,
    #    feathered edges so it reads as a soft highlight, not a blob
    # =========================================================
    thresh_val = np.percentile(smooth_cam, 85)
    mask = (smooth_cam >= thresh_val).astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (27, 27), 0)
    alpha_mask = (mask.astype(np.float32) / 255.0)[..., None]

    heat_color = cv2.applyColorMap(smooth_cam, cv2.COLORMAP_JET).astype(np.float32)
    base = orig_bgr.astype(np.float32)
    blended = base * (1 - alpha_mask * 0.7) + heat_color * (alpha_mask * 0.7)
    heat_result = blended.astype(np.uint8)
    heat_result = add_label_banner(heat_result, f"Grad-CAM  •  {label_text}", accent=(60, 60, 240))

    std_path = f"outputs/gradcam/heat_{timestamp}.png"
    cv2.imwrite(std_path, heat_result)

    # =========================================================
    # 2. CIRCLED FOCUS - single clean glowing ring around the
    #    strongest affected region, minimal fill
    # =========================================================
    _, binary = cv2.threshold(smooth_cam, 170, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pro_result = orig_bgr.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 20:
            overlay = pro_result.copy()
            if len(largest) >= 5:
                ellipse = cv2.fitEllipse(largest)
                cv2.ellipse(overlay, ellipse, (50, 50, 255), thickness=-1)
                pro_result = cv2.addWeighted(pro_result, 0.85, overlay, 0.15, 0)
                cv2.ellipse(pro_result, ellipse, (0, 215, 255), 2, cv2.LINE_AA)
            else:
                x, y, bw, bh = cv2.boundingRect(largest)
                cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (50, 50, 255), -1)
                pro_result = cv2.addWeighted(pro_result, 0.85, overlay, 0.15, 0)
                cv2.rectangle(pro_result, (x, y), (x + bw, y + bh), (0, 215, 255), 2, cv2.LINE_AA)

    pro_result = add_label_banner(pro_result, f"Focus Region  •  {label_text}", accent=(0, 215, 255))

    pro_path = f"outputs/gradcam_pro/circled_{timestamp}.png"
    cv2.imwrite(pro_path, pro_result)

    return {
        "gradcam": std_path,
        "gradcam_pro": pro_path,
    }