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


def normalized_cam(cam_map, h, w):
    if cam_map is not None and np.max(cam_map) > 0:
        cam_resized = cv2.resize(cam_map, (w, h))
        cam_norm = np.uint8(255 * (cam_resized - cam_resized.min()) /
                             (cam_resized.max() - cam_resized.min() + 1e-8))
    else:
        cam_norm = np.zeros((h, w), dtype=np.uint8)
    return cv2.GaussianBlur(cam_norm, (17, 17), 0)


def significant_mask(smooth_cam, percentile=88):
    """Shared logic: keep only the strongest activation region, drop noise specks."""
    h, w = smooth_cam.shape
    min_area = max(18, int(h * w * 0.004))

    thresh_val = np.percentile(smooth_cam, percentile)
    _, binary = cv2.threshold(smooth_cam, thresh_val, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(binary)
    kept = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean_mask, [c], -1, 255, -1)
            kept.append(c)

    return clean_mask, kept


def generate_gradcam_images(orig_bgr, cam_map, confidence_score, class_name):
    timestamp = int(time.time())

    if len(orig_bgr.shape) == 2 or orig_bgr.shape[2] == 1:
        orig_bgr = cv2.cvtColor(orig_bgr, cv2.COLOR_GRAY2BGR)

    h, w, _ = orig_bgr.shape
    smooth_cam = normalized_cam(cam_map, h, w)
    clean_mask, contours = significant_mask(smooth_cam, percentile=88)
    label_text = f"{class_name}  |  {confidence_score * 100:.1f}% confidence"

    # =========================================================
    # STYLE 1: HEAT GLOW - colored gradient, ONLY inside the mask,
    # re-stretched to full vividness within that region
    # =========================================================
    masked_cam = smooth_cam.copy()
    masked_cam[clean_mask == 0] = 0
    if masked_cam.max() > 0:
        inside = masked_cam[clean_mask > 0]
        stretched = np.zeros_like(masked_cam)
        stretched[clean_mask > 0] = np.uint8(
            255 * (inside.astype(np.float32) - inside.min()) /
            (inside.max() - inside.min() + 1e-8)
        )
    else:
        stretched = masked_cam

    glow_color = cv2.applyColorMap(stretched, cv2.COLORMAP_INFERNO)
    feather = cv2.GaussianBlur(clean_mask, (21, 21), 0)
    alpha = (feather.astype(np.float32) / 255.0)[..., None]

    base = orig_bgr.astype(np.float32)
    blended = base * (1 - alpha * 0.75) + glow_color.astype(np.float32) * (alpha * 0.75)
    heat_result = blended.astype(np.uint8)
    heat_result = add_label_banner(heat_result, "Grad-CAM", label_text, accent=(60, 60, 240))

    std_path = f"outputs/gradcam/heat_{timestamp}.png"
    cv2.imwrite(std_path, heat_result)

    # =========================================================
    # STYLE 2: TARGET MARKERS - no fill, just clean corner brackets
    # around each affected region, like a camera focus reticle
    # =========================================================
    pro_result = orig_bgr.copy()
    ranked = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
    accent = (0, 215, 255)

    for cnt in ranked:
        x, y, bw, bh = cv2.boundingRect(cnt)
        pad = int(max(bw, bh) * 0.18) + 4
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + bw + pad, w - 1), min(y + bh + pad, h - 1)
        arm = int(min(x2 - x1, y2 - y1) * 0.28) + 6

        corners = [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]
        for cxp, cyp, dx, dy in corners:
            cv2.line(pro_result, (cxp, cyp), (cxp + dx * arm, cyp), accent, 2, cv2.LINE_AA)
            cv2.line(pro_result, (cxp, cyp), (cxp, cyp + dy * arm), accent, 2, cv2.LINE_AA)

        cx, cy = x + bw // 2, y + bh // 2
        cv2.circle(pro_result, (cx, cy), 3, accent, -1, cv2.LINE_AA)

    pro_result = add_label_banner(pro_result, "Focus Region", label_text, accent=accent)

    pro_path = f"outputs/gradcam_pro/target_{timestamp}.png"
    cv2.imwrite(pro_path, pro_result)

    return {"gradcam": std_path, "gradcam_pro": pro_path}