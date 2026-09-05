import cv2
import numpy as np
import time
from pathlib import Path

OUTPUTS_DIR = Path("outputs")
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"
GRADCAM_PRO_DIR = OUTPUTS_DIR / "gradcam_pro"

for path in [GRADCAM_DIR, GRADCAM_PRO_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def add_label_banner(img, title, subtitle, accent=(0, 200, 255)):
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
    return cv2.GaussianBlur(cam_norm, (21, 21), 0)


def generate_gradcam_images(orig_bgr, cam_map, confidence_score, class_name):
    timestamp = int(time.time())

    if len(orig_bgr.shape) == 2 or orig_bgr.shape[2] == 1:
        orig_bgr = cv2.cvtColor(orig_bgr, cv2.COLOR_GRAY2BGR)

    h, w, _ = orig_bgr.shape
    smooth_cam = _normalized_cam(cam_map, h, w)
    label_text = f"{class_name}  |  {confidence_score * 100:.1f}% confidence"

    # ---- 1. HEAT ZONE - only the top ~10% most active pixels, feathered ----
    thresh_val = np.percentile(smooth_cam, 90)
    mask = (smooth_cam >= thresh_val).astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (27, 27), 0)
    alpha_mask = (mask.astype(np.float32) / 255.0)[..., None]

    heat_color = cv2.applyColorMap(smooth_cam, cv2.COLORMAP_JET).astype(np.float32)
    base = orig_bgr.astype(np.float32)
    blended = base * (1 - alpha_mask * 0.7) + heat_color * (alpha_mask * 0.7)
    heat_result = blended.astype(np.uint8)
    heat_result = add_label_banner(heat_result, "Grad-CAM", label_text, accent=(60, 60, 240))

    std_path = f"outputs/gradcam/heat_{timestamp}.png"
    cv2.imwrite(std_path, heat_result)

    # ---- 2. CIRCLED FOCUS - single tight circle around the hottest core ----
    _, binary = cv2.threshold(smooth_cam, 190, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pro_result = orig_bgr.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 15:
            (cx, cy), radius = cv2.minEnclosingCircle(largest)
            radius = int(radius * 1.15)
            center = (int(cx), int(cy))
            overlay = pro_result.copy()
            cv2.circle(overlay, center, radius, (50, 50, 255), -1)
            pro_result = cv2.addWeighted(pro_result, 0.85, overlay, 0.15, 0)
            cv2.circle(pro_result, center, radius, (0, 215, 255), 2, cv2.LINE_AA)

    pro_result = add_label_banner(pro_result, "Focus Region", label_text, accent=(0, 215, 255))

    pro_path = f"outputs/gradcam_pro/circled_{timestamp}.png"
    cv2.imwrite(pro_path, pro_result)

    return {"gradcam": std_path, "gradcam_pro": pro_path}