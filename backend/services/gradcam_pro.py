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


def isolate_primary_hemorrhage(cam_map, h, w, percentile=92):
    """Normalize CAM map and retain ONLY the largest/strongest primary activation cluster."""
    if cam_map is None or np.max(cam_map) <= 0:
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8), None, None

    cam_resized = cv2.resize(cam_map, (w, h))
    cam_norm = np.uint8(255 * (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8))
    smooth_cam = cv2.GaussianBlur(cam_norm, (11, 11), 0)

    # Threshold top intensity region
    thresh_val = np.percentile(smooth_cam, percentile)
    _, binary = cv2.threshold(smooth_cam, thresh_val, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return smooth_cam, np.zeros((h, w), dtype=np.uint8), None, None

    # Keep strictly the largest active region (eliminates secondary/false clusters)
    primary_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(binary)
    cv2.drawContours(clean_mask, [primary_contour], -1, 255, -1)

    # Find center peak within primary region
    masked_cam = smooth_cam.copy()
    masked_cam[clean_mask == 0] = 0
    y, x = np.unravel_index(np.argmax(masked_cam), masked_cam.shape)

    return smooth_cam, clean_mask, primary_contour, (int(x), int(y))


def generate_gradcam_images(orig_bgr, cam_map, confidence_score, class_name):
    timestamp = int(time.time())

    if len(orig_bgr.shape) == 2 or orig_bgr.shape[2] == 1:
        orig_bgr = cv2.cvtColor(orig_bgr, cv2.COLOR_GRAY2BGR)

    h, w, _ = orig_bgr.shape
    smooth_cam, clean_mask, primary_contour, peak = isolate_primary_hemorrhage(cam_map, h, w, percentile=92)
    label_text = f"{class_name}  |  {confidence_score * 100:.1f}% confidence"

    # ==================== 1. STANDARD GRAD-CAM ====================
    masked_cam = smooth_cam.copy()
    masked_cam[clean_mask == 0] = 0

    glow_color = cv2.applyColorMap(masked_cam, cv2.COLORMAP_INFERNO)
    feather = cv2.GaussianBlur(clean_mask, (15, 15), 0)
    alpha = (feather.astype(np.float32) / 255.0)[..., None]

    base = orig_bgr.astype(np.float32)
    blended = base * (1 - alpha * 0.82) + glow_color.astype(np.float32) * (alpha * 0.82)
    heat_result = add_label_banner(blended.astype(np.uint8), "Grad-CAM", label_text, accent=(60, 60, 240))

    std_path = f"outputs/gradcam/heat_{timestamp}.png"
    cv2.imwrite(std_path, heat_result)

    # ==================== 2. PROFESSIONAL GRAD-CAM ====================
    pro_result = orig_bgr.copy()
    accent = (0, 215, 255)  # Bright Gold Accent

    if primary_contour is not None and peak is not None:
        bx, by, bw, bh = cv2.boundingRect(primary_contour)
        
        # Add padding around bounding box
        pad = 8
        x1, y1 = max(bx - pad, 0), max(by - pad, 0)
        x2, y2 = min(bx + bw + pad, w - 1), min(by + bh + pad, h - 1)
        
        arm = int(min(bw, bh) * 0.25) + 4
        corners = [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]

        # Tactical Corner Brackets
        for cxp, cyp, dx, dy in corners:
            cv2.line(pro_result, (cxp, cyp), (cxp + dx * arm, cyp), accent, 2, cv2.LINE_AA)
            cv2.line(pro_result, (cxp, cyp), (cxp, cyp + dy * arm), accent, 2, cv2.LINE_AA)

        # Center Precision Crosshair
        cx, cy = peak
        cv2.drawMarker(pro_result, (cx, cy), accent, cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
        cv2.circle(pro_result, (cx, cy), 3, accent, -1, cv2.LINE_AA)

    pro_result = add_label_banner(pro_result, "Focus Region", label_text, accent=accent)
    pro_path = f"outputs/gradcam_pro/target_{timestamp}.png"
    cv2.imwrite(pro_path, pro_result)

    return {"gradcam": std_path, "gradcam_pro": pro_path}