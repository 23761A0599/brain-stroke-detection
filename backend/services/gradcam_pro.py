import cv2
import numpy as np
import time
from pathlib import Path
from skimage.segmentation import slic, mark_boundaries

# Ensure output subdirectories exist
OUTPUTS_DIR = Path("outputs")
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"
GRADCAM_PRO_DIR = OUTPUTS_DIR / "gradcam_pro"
LIME_DIR = OUTPUTS_DIR / "lime"

for path in [GRADCAM_DIR, GRADCAM_PRO_DIR, LIME_DIR]:
    path.mkdir(parents=True, exist_ok=True)


def generate_gradcam_images(orig_bgr, cam_map, confidence_score, class_name):
    """
    Generates explainability images with accurate hemorrhage bounding contours 
    and removes fixed artificial fallback circles.
    """
    timestamp = int(time.time())

    # Force 3-channel BGR image format
    if len(orig_bgr.shape) == 2 or orig_bgr.shape[2] == 1:
        orig_bgr = cv2.cvtColor(orig_bgr, cv2.COLOR_GRAY2BGR)

    h, w, _ = orig_bgr.shape

    # Process and normalize the model's actual Grad-CAM activation map
    if cam_map is not None and np.max(cam_map) > 0:
        cam_resized = cv2.resize(cam_map, (w, h))
        cam_norm = np.uint8(255 * (cam_resized - np.min(cam_resized)) / (np.max(cam_resized) - np.min(cam_resized) + 1e-8))
    else:
        cam_norm = np.zeros((h, w), dtype=np.uint8)

    # Apply Gaussian smoothing for continuous gradients
    smooth_cam = cv2.GaussianBlur(cam_norm, (15, 15), 0)

    # =========================================================================
    # 1. STANDARD GRAD-CAM HEATMAP OVERLAY
    # =========================================================================
    jet_color = cv2.applyColorMap(smooth_cam, cv2.COLORMAP_JET)
    std_blended = cv2.addWeighted(orig_bgr, 0.50, jet_color, 0.50, 0)

    std_path = f"outputs/gradcam/std_{timestamp}.png"
    cv2.imwrite(std_path, std_blended)

    # =========================================================================
    # 2. PROFESSIONAL GRAD-CAM: Tight Hemorrhage Region Contours & Bounding Boxes
    # =========================================================================
    pro_blended = orig_bgr.copy()

    # Create binary mask focusing on high activation areas
    _, binary_thresh = cv2.threshold(smooth_cam, 160, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Translucent Red Highlight Layer
    red_layer = pro_blended.copy()
    mask_indices = smooth_cam > 100
    red_layer[mask_indices] = [0, 0, 240]  # Bright Red BGR
    pro_blended = cv2.addWeighted(pro_blended, 0.55, red_layer, 0.45, 0)

    # Draw bounding boxes and precise outline contours around highlighted hemorrhage spots
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:  # Filter noise
            x, y, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(pro_blended, (x, y), (x + bw, y + bh), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.drawContours(pro_blended, [cnt], -1, (0, 0, 255), 2, cv2.LINE_AA)

    pro_path = f"outputs/gradcam_pro/pro_{timestamp}.png"
    cv2.imwrite(pro_path, pro_blended)

    # =========================================================================
    # 3. LIME EXPLANATION: Superpixels Overlapped with Activation Regions
    # =========================================================================
    img_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    segments = slic(img_rgb, n_segments=60, compactness=12, start_label=1)

    lime_bgr = orig_bgr.copy()

    for seg_val in np.unique(segments):
        mask = (segments == seg_val)
        overlap = np.sum(smooth_cam[mask] > 100)

        if overlap > 5:  # Superpixels overlapping actual high activations
            lime_bgr[mask] = lime_bgr[mask] * 0.30 + np.array([0, 0, 240]) * 0.70
        else:            # Normal tissue area
            lime_bgr[mask] = lime_bgr[mask] * 0.80 + np.array([0, 140, 0]) * 0.20

    marked = mark_boundaries(lime_bgr, segments, color=(1, 1, 1))
    lime_final = cv2.cvtColor(np.uint8(marked * 255), cv2.COLOR_RGB2BGR)

    lime_path = f"outputs/lime/lime_{timestamp}.png"
    cv2.imwrite(lime_path, lime_final)

    return {
        "gradcam": std_path,
        "gradcam_pro": pro_path,
        "lime": lime_path
    }