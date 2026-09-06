from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image

from model.config import IMAGE_SIZE

LIME_DIR = Path("outputs") / "lime"
LIME_DIR.mkdir(parents=True, exist_ok=True)


def add_corner_badge(img, title, subtitle, accent=(230, 0, 255)):
    """Same top-corner badge treatment as gradcam_images.py, so all
    three visualizations share a consistent, non-obscuring label style."""
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
    cv2.rectangle(img, (5, 5), (8, 5 + badge_h), accent, -1)

    cv2.putText(img, title, (5 + pad, 5 + pad + t_h),
                cv2.FONT_HERSHEY_SIMPLEX, line1_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, subtitle, (5 + pad, 5 + pad * 2 + t_h + s_h),
                cv2.FONT_HERSHEY_SIMPLEX, line2_scale, accent, 1, cv2.LINE_AA)
    return img


def draw_pointer_arrow(img, peak, accent, label="Key Feature"):
    """Same pointer-arrow treatment as gradcam_images.py, for visual
    consistency across all three explainability outputs."""
    h, w = img.shape[:2]
    cx, cy = peak

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


def save_lime_explanation(image_path, cam_map, confidence_score, class_name):
    timestamp = int(time.time())

    raw_img = Image.open(image_path).convert("RGB")
    img_np = np.array(raw_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    is_hemorrhage = "hemorrhag" in class_name.lower()
    label_text = f"{class_name}  |  {confidence_score * 100:.1f}% confidence"

    # CHANGED: distinct visual identity - purple/magenta "spotlight"
    # style (vs. Grad-CAM's warm inferno glow and Grad-CAM++'s gold
    # tactical brackets), so all three are clearly different at a glance.
    accent = (230, 0, 255)  # magenta (BGR)
    result = gray_bgr.copy()

    if is_hemorrhage and cam_map is not None:
        h, w = gray.shape
        cam_resized = cv2.resize(cam_map, (w, h))
        cam_norm = np.uint8(255 * (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8))
        smooth_cam = cv2.GaussianBlur(cam_norm, (11, 11), 0)

        thresh_val = np.percentile(smooth_cam, 92)
        _, binary = cv2.threshold(smooth_cam, thresh_val, 255, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        peak = None

        if contours:
            primary_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(primary_contour)

            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [hull], -1, 255, -1)

            feather = cv2.GaussianBlur(mask, (25, 25), 0)
            glow = np.zeros_like(result)
            glow[:] = (200, 0, 220)
            alpha = (feather.astype(np.float32) / 255.0)[..., None] * 0.5
            result = (result.astype(np.float32) * (1 - alpha) + glow.astype(np.float32) * alpha).astype(np.uint8)

            cv2.drawContours(result, [hull], -1, accent, 2, cv2.LINE_AA)

            masked_cam = smooth_cam.copy()
            masked_cam[mask == 0] = 0
            y, x = np.unravel_index(np.argmax(masked_cam), masked_cam.shape)
            peak = (int(x), int(y))

        if peak is not None:
            result = draw_pointer_arrow(result, peak, accent=accent, label="Influential Feature")

        result = add_corner_badge(result, "LIME Segment", label_text, accent=accent)
    else:
        result = add_corner_badge(result, "LIME Segment", label_text, accent=(60, 220, 90))

    filename = f"lime_{timestamp}.png"
    output_path = LIME_DIR / filename
    cv2.imwrite(str(output_path), result)

    return {"lime_path": f"outputs/lime/{filename}"}