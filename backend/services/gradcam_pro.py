import cv2
import numpy as np
import time
from pathlib import Path

OUTPUTS_DIR = Path("outputs")
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"
GRADCAM_PRO_DIR = OUTPUTS_DIR / "gradcam_pro"

for path in [GRADCAM_DIR, GRADCAM_PRO_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def add_corner_badge(img, title, subtitle, accent=(0, 200, 255)):
    """
    CHANGED: replaces the old full-width BOTTOM banner (which could
    cover the actual highlighted lesion region, especially for
    hemorrhages located in the lower part of the scan) with a small,
    compact badge in the TOP-LEFT corner instead. The image content
    is never obscured.
    """
    h, w = img.shape[:2]
    pad = 8
    line1_scale, line2_scale = 0.48, 0.42

    (t_w, t_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, line1_scale, 1)
    (s_w, s_h), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, line2_scale, 1)
    badge_w = min(max(t_w, s_w) + pad * 2, w - 10)
    badge_h = t_h + s_h + pad * 3

    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (5 + badge_w, 5 + badge_h), (15, 15, 15), -1)
    img = cv2.addWeighted(overlay, 0.80, img, 0.20, 0)
    cv2.rectangle(img, (5, 5), (8, 5 + badge_h), accent, -1)  # accent tab on the left edge

    cv2.putText(img, title, (5 + pad, 5 + pad + t_h),
                cv2.FONT_HERSHEY_SIMPLEX, line1_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, subtitle, (5 + pad, 5 + pad * 2 + t_h + s_h),
                cv2.FONT_HERSHEY_SIMPLEX, line2_scale, accent, 1, cv2.LINE_AA)
    return img


def isolate_primary_hemorrhage(cam_map, h, w, percentile=92, border_margin_pct=0.06):
    """
    Normalize CAM map and retain ONLY the largest/strongest primary
    activation cluster.

    CHANGED: EfficientNet-B0's final feature map is only 7x7 - upscaling
    that to 224x224 (32x) frequently produces spurious high-intensity
    artifacts right at the image BORDERS/CORNERS from the interpolation
    itself, not from any real activation. Without guarding against this,
    the "largest contour" can end up being a border artifact instead of
    the real lesion - which is exactly what caused the arrow to point
    at the image edge/corner instead of the actual highlighted region.
    Fix: zero out a border margin before thresholding, so edge artifacts
    can never be selected as the primary region.
    """
    if cam_map is None or np.max(cam_map) <= 0:
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8), None, None

    cam_resized = cv2.resize(cam_map, (w, h))
    cam_norm = np.uint8(255 * (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8))
    smooth_cam = cv2.GaussianBlur(cam_norm, (11, 11), 0)

    # NEW: zero out a border margin to eliminate resize/interpolation
    # edge artifacts before thresholding
    margin_h = max(1, int(h * border_margin_pct))
    margin_w = max(1, int(w * border_margin_pct))
    cam_for_threshold = smooth_cam.copy()
    cam_for_threshold[:margin_h, :] = 0
    cam_for_threshold[-margin_h:, :] = 0
    cam_for_threshold[:, :margin_w] = 0
    cam_for_threshold[:, -margin_w:] = 0

    thresh_val = np.percentile(cam_for_threshold, percentile)
    _, binary = cv2.threshold(cam_for_threshold, thresh_val, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return smooth_cam, np.zeros((h, w), dtype=np.uint8), None, None

    primary_contour = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(binary)
    cv2.drawContours(clean_mask, [primary_contour], -1, 255, -1)

    masked_cam = smooth_cam.copy()
    masked_cam[clean_mask == 0] = 0
    y, x = np.unravel_index(np.argmax(masked_cam), masked_cam.shape)

    # NEW: clamp the peak point away from the very edge, so the arrow
    # marker is always fully visible inside the image bounds
    safe_margin = 12
    x = int(np.clip(x, safe_margin, w - safe_margin))
    y = int(np.clip(y, safe_margin, h - safe_margin))

    return smooth_cam, clean_mask, primary_contour, (x, y)


def draw_pointer_arrow(img, peak, accent, label="Affected Region"):
    """
    NEW: draws a clear arrow from a corner of the image directly to
    the peak activation point, with a small label - makes it
    unambiguous where the model is focusing, for both technical and
    non-technical viewers.
    """
    h, w = img.shape[:2]
    cx, cy = peak

    # Anchor the arrow's tail in whichever corner is farthest from the peak,
    # so the arrow doesn't overlap the highlighted region itself.
    corners = [(20, h - 20), (w - 20, h - 20), (20, 20), (w - 20, 20)]
    tail = max(corners, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)

    cv2.arrowedLine(img, tail, (cx, cy), accent, 2, cv2.LINE_AA, tipLength=0.08)
    cv2.circle(img, (cx, cy), 5, accent, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 8, accent, 2, cv2.LINE_AA)

    text_x = min(max(tail[0] - 40, 5), w - 130)
    text_y = tail[1] - 8 if tail[1] > h // 2 else tail[1] + 18
    cv2.putText(img, label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, accent, 1, cv2.LINE_AA)
    return img


def generate_gradcam_images(orig_bgr, cam_map, confidence_score, class_name):
    timestamp = int(time.time())

    if len(orig_bgr.shape) == 2 or orig_bgr.shape[2] == 1:
        orig_bgr = cv2.cvtColor(orig_bgr, cv2.COLOR_GRAY2BGR)

    h, w, _ = orig_bgr.shape
    smooth_cam, clean_mask, primary_contour, peak = isolate_primary_hemorrhage(cam_map, h, w, percentile=92)
    label_text = f"{class_name}  |  {confidence_score * 100:.1f}% confidence"

    # ==================== 1. STANDARD GRAD-CAM: soft glow, distinct warm palette ====================
    masked_cam = smooth_cam.copy()
    masked_cam[clean_mask == 0] = 0

    glow_color = cv2.applyColorMap(masked_cam, cv2.COLORMAP_INFERNO)
    feather = cv2.GaussianBlur(clean_mask, (21, 21), 0)
    alpha = (feather.astype(np.float32) / 255.0)[..., None]

    base = orig_bgr.astype(np.float32)
    blended = base * (1 - alpha * 0.85) + glow_color.astype(np.float32) * (alpha * 0.85)
    heat_result = blended.astype(np.uint8)

    if peak is not None:
        heat_result = draw_pointer_arrow(heat_result, peak, accent=(60, 60, 240), label="Activation Peak")

    heat_result = add_corner_badge(heat_result, "Grad-CAM", label_text, accent=(60, 60, 240))

    std_path = f"outputs/gradcam/heat_{timestamp}.png"
    cv2.imwrite(std_path, heat_result)

    # ==================== 2. PROFESSIONAL GRAD-CAM: tactical bracket + arrow, cool gold palette ====================
    pro_result = orig_bgr.copy()
    accent = (0, 215, 255)  # gold

    if primary_contour is not None and peak is not None:
        bx, by, bw, bh = cv2.boundingRect(primary_contour)

        pad = 8
        x1, y1 = max(bx - pad, 0), max(by - pad, 0)
        x2, y2 = min(bx + bw + pad, w - 1), min(by + bh + pad, h - 1)

        arm = int(min(bw, bh) * 0.25) + 4
        corners = [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]

        for cxp, cyp, dx, dy in corners:
            cv2.line(pro_result, (cxp, cyp), (cxp + dx * arm, cyp), accent, 2, cv2.LINE_AA)
            cv2.line(pro_result, (cxp, cyp), (cxp, cyp + dy * arm), accent, 2, cv2.LINE_AA)

        pro_result = draw_pointer_arrow(pro_result, peak, accent=accent, label="Suspected Hemorrhage")

    pro_result = add_corner_badge(pro_result, "Focus Region", label_text, accent=accent)
    pro_path = f"outputs/gradcam_pro/target_{timestamp}.png"
    cv2.imwrite(pro_path, pro_result)

    return {"gradcam": std_path, "gradcam_pro": pro_path}