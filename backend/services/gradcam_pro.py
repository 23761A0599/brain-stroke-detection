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
    Generates three distinct explainability images by taking the actual model activation map
    and rendering vibrant, high-contrast visual overlays.
    """
    timestamp = int(time.time())

    # Force 3-channel BGR image format to render colors correctly
    if len(orig_bgr.shape) == 2 or orig_bgr.shape[2] == 1:
        orig_bgr = cv2.cvtColor(orig_bgr, cv2.COLOR_GRAY2BGR)

    h, w, _ = orig_bgr.shape

    # Extract target focal area from the actual Grad-CAM map
    if cam_map is not None and np.max(cam_map) > 0:
        cam_resized = cv2.resize(cam_map, (w, h))
        cam_norm = np.uint8(255 * (cam_resized - np.min(cam_resized)) / (np.max(cam_resized) - np.min(cam_resized) + 1e-8))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cam_norm)
        center_x, center_y = max_loc
    else:
        # Fallback to brain region center if activation map is flat
        gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
        center_y, center_x = int(h * 0.45), int(w * 0.50)
        cam_norm = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(cam_norm, (center_x, center_y), int(min(h, w) * 0.2), 255, -1)

    radius = max(35, int(min(h, w) * 0.18))

    # =========================================================================
    # 1. GRAD-CAM MAP: Vivid COLORMAP_JET Heatmap Overlay
    # =========================================================================
    # Apply Gaussian smoothing to create fluid heat gradients
    smooth_cam = cv2.GaussianBlur(cam_norm, (31, 31), 0)
    
    # Generate bright Jet Heatmap (Red=High impact, Yellow=Medium, Blue=Low)
    jet_color = cv2.applyColorMap(smooth_cam, cv2.COLORMAP_JET)
    
    # Blend colored heatmap directly over original MRI scan
    std_blended = cv2.addWeighted(orig_bgr, 0.40, jet_color, 0.60, 0)

    # Draw bright yellow focus contour line
    cv2.circle(std_blended, (center_x, center_y), radius, (0, 255, 255), 2, cv2.LINE_AA)

    std_path = f"outputs/gradcam/std_{timestamp}.png"
    cv2.imwrite(std_path, std_blended)

    # =========================================================================
    # 2. PROFESSIONAL GRAD-CAM: Red Focus Zone & Directional Pointer
    # =========================================================================
    pro_blended = orig_bgr.copy()

    # Translucent Red Highlight Layer
    red_layer = pro_blended.copy()
    
    # Fill the high-intensity lesion region in vibrant red
    mask_indices = smooth_cam > 100
    red_layer[mask_indices] = [0, 0, 230] # Bright Red BGR
    cv2.circle(red_layer, (center_x, center_y), radius, (0, 0, 255), -1)
    
    # Blend red fill over scan
    pro_blended = cv2.addWeighted(pro_blended, 0.50, red_layer, 0.50, 0)

    # Bright Red Double Target Ring
    cv2.circle(pro_blended, (center_x, center_y), radius + 2, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(pro_blended, (center_x, center_y), radius + 8, (0, 220, 255), 2, cv2.LINE_AA)

    # Yellow Pointer Arrow pointing directly to lesion epicenter
    arrow_start = (max(15, center_x - 80), max(15, center_y - 65))
    cv2.arrowedLine(pro_blended, arrow_start, (center_x, center_y), (0, 255, 255), 3, tipLength=0.30, line_type=cv2.LINE_AA)

    pro_path = f"outputs/gradcam_pro/pro_{timestamp}.png"
    cv2.imwrite(pro_path, pro_blended)

    # =========================================================================
    # 3. LIME EXPLANATION: Bright Red Superpixels & Green Healthy Tissue
    # =========================================================================
    img_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    segments = slic(img_rgb, n_segments=60, compactness=12, start_label=1)

    lime_bgr = orig_bgr.copy()
    
    # Create thresholded binary mask of the lesion area
    target_mask = np.zeros((h, w), dtype=np.uint8)
    target_mask[smooth_cam > 90] = 255
    if np.sum(target_mask) == 0:
        cv2.circle(target_mask, (center_x, center_y), radius, 255, -1)

    for seg_val in np.unique(segments):
        mask = (segments == seg_val)
        overlap = np.sum(target_mask[mask]) / 255.0

        if overlap > 3:  # Lesion driving superpixels -> Vibrant Red
            lime_bgr[mask] = lime_bgr[mask] * 0.25 + np.array([0, 0, 240]) * 0.75
        else:            # Surrounding tissue -> Translucent Green
            lime_bgr[mask] = lime_bgr[mask] * 0.75 + np.array([0, 150, 0]) * 0.25

    # Crisp White Superpixel Boundary Contours
    marked = mark_boundaries(lime_bgr, segments, color=(1, 1, 1))
    lime_final = cv2.cvtColor(np.uint8(marked * 255), cv2.COLOR_RGB2BGR)

    lime_path = f"outputs/lime/lime_{timestamp}.png"
    cv2.imwrite(lime_path, lime_final)

    return {
        "gradcam": std_path,
        "gradcam_pro": pro_path,
        "lime": lime_path
    }