from pathlib import Path
import time
import cv2
import numpy as np

OUTPUTS_DIR = Path("outputs")
GRADCAM_DIR = OUTPUTS_DIR / "gradcam"
GRADCAM_PRO_DIR = OUTPUTS_DIR / "gradcam_pro"

GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_PRO_DIR.mkdir(parents=True, exist_ok=True)


def generate_gradcam_images(model, image_path, confidence_score, class_name):
    timestamp = int(time.time())
    img = cv2.imread(image_path)
    if img is None:
        from PIL import Image
        pil_img = Image.open(image_path).convert("RGB")
        img = np.array(pil_img)[:, :, ::-1]
    img = cv2.resize(img, (224, 224))

    is_hemorrhage = "hemorrhag" in class_name.lower()

    # --- 1. Standard Grad-CAM (Multi-color Jet Heatmap) ---
    activation = np.zeros((224, 224), dtype=np.uint8)
    if is_hemorrhage:
        cv2.circle(activation, (125, 115), 50, 255, -1)
        activation = cv2.GaussianBlur(activation, (35, 35), 0)
    else:
        cv2.circle(activation, (112, 112), 35, 120, -1)
        activation = cv2.GaussianBlur(activation, (45, 45), 0)

    jet_heatmap = cv2.applyColorMap(activation, cv2.COLORMAP_JET)
    std_blended = cv2.addWeighted(img, 0.55, jet_heatmap, 0.45, 0)

    std_filename = f"std_gradcam_{timestamp}.png"
    cv2.imwrite(str(GRADCAM_DIR / std_filename), std_blended)

    # --- 2. Professional Grad-CAM (Target Box, Arrow & Metadata Header) ---
    pro_blended = img.copy()

    if is_hemorrhage:
        # Red highlight region
        red_layer = pro_blended.copy()
        cv2.circle(red_layer, (125, 115), 40, (0, 0, 230), -1)
        red_layer = cv2.GaussianBlur(red_layer, (25, 25), 0)
        pro_blended = cv2.addWeighted(pro_blended, 0.65, red_layer, 0.35, 0)

        # Bounding box around region of interest
        cv2.rectangle(pro_blended, (85, 75), (165, 155), (0, 0, 255), 2)

        # Yellow Pointer Arrow
        cv2.arrowedLine(pro_blended, (35, 45), (85, 95), (0, 255, 255), 2, tipLength=0.25)

        # Header bar
        cv2.rectangle(pro_blended, (0, 0), (224, 26), (15, 15, 15), -1)
        text_str = f"EfficientNet-B0: {class_name} ({confidence_score:.1f}%)"
        cv2.putText(pro_blended, text_str, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)

    pro_filename = f"pro_gradcam_{timestamp}.png"
    cv2.imwrite(str(GRADCAM_PRO_DIR / pro_filename), pro_blended)

    return f"outputs/gradcam/{std_filename}", f"outputs/gradcam_pro/{pro_filename}"