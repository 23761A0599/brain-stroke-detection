import io
import os
import base64
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model.config import IMAGE_SIZE, CLASS_NAMES, DEVICE
from model.model_builder import build_model
from services.gradcam import generate_gradcam_map
from services.gradcam_pro import generate_gradcam_images
from services.lime_explainer import save_lime_explanation

app = FastAPI(title="Brain Hemorrhage Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_PATHS = [
    os.path.join(BASE_DIR, "models", "EfficientNetB0", "best_efficientnetb0.pth"),
    os.path.join(BASE_DIR, "backend", "models", "EfficientNetB0", "best_efficientnetb0.pth"),
    os.path.join(BASE_DIR, "models", "best_efficientnetb0.pth"),
    os.path.join(BASE_DIR, "backend", "models", "best_efficientnetb0.pth"),
    "models/EfficientNetB0/best_efficientnetb0.pth",
    "backend/models/EfficientNetB0/best_efficientnetb0.pth",
]

MODEL_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

model = None
if MODEL_PATH:
    try:
        model = build_model("efficientnetb0")
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print(f"Model loaded successfully from: {MODEL_PATH}", flush=True)
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}", flush=True)
        model = None
else:
    print(f"Model file not found in any expected location: {POSSIBLE_PATHS}", flush=True)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def file_to_data_url(path):
    if not path:
        return ""

    full_path = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)

    if not os.path.exists(full_path):
        print(f"Warning: File not found for base64 conversion: {full_path}", flush=True)
        return ""

    try:
        with open(full_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Failed to read file for base64 encoding ({full_path}): {e}", flush=True)
        return ""


def looks_like_mri(pil_image,
                    colorfulness_threshold=15,
                    border_brightness_threshold=120,
                    overall_brightness_threshold=200,
                    min_fg_ratio=0.04,
                    max_fg_ratio=0.85,
                    min_circularity=0.35,
                    border_touch_margin_frac=0.03):
    """
    Three-stage sanity filter to reject non-MRI uploads:
    1. Colorfulness   - rejects colorful photos, posters, product shots
    2. Brightness      - rejects bright documents/photos (real MRI scans
                          have a solid black background)
    3. Blob shape       - real MRI scans show ONE roughly round/oval brain
                          region, centered, surrounded by a black margin
                          that never touches the image edges. Dark-mode
                          screenshots of text/lists are dark and grayscale
                          too, but don't have this shape - so this stage
                          catches what the first two can't.
    Not a perfect MRI detector - a practical filter against the realistic
    wrong uploads people are likely to try.
    """
    np_img = np.array(pil_image).astype(np.float32)
    gray = np.array(pil_image.convert("L")).astype(np.uint8)
    h, w = gray.shape

    # --- 1. Colorfulness ---
    if np_img.ndim == 3 and np_img.shape[2] >= 3:
        r, g, b = np_img[..., 0], np_img[..., 1], np_img[..., 2]
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        colorfulness = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2) + \
            0.3 * np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        if colorfulness >= colorfulness_threshold:
            return False

    # --- 2. Brightness (border + overall) ---
    border = max(int(max(h, w) * 0.04), 4)
    border_pixels = np.concatenate([
        gray[:border, :].flatten(),
        gray[-border:, :].flatten(),
        gray[:, :border].flatten(),
        gray[:, -border:].flatten(),
    ])
    if border_pixels.mean() > border_brightness_threshold:
        return False
    if gray.mean() > overall_brightness_threshold:
        return False

    # --- 3. Blob shape: one centered, round-ish region on a black field ---
    _, fg_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    fg_ratio = area / (h * w)

    if fg_ratio < min_fg_ratio or fg_ratio > max_fg_ratio:
        return False

    x, y, bw, bh = cv2.boundingRect(largest)
    margin = max(int(min(h, w) * border_touch_margin_frac), 3)
    touches_border = (x <= margin or y <= margin or
                       (x + bw) >= (w - margin) or (y + bh) >= (h - margin))
    if touches_border:
        return False

    (_, _), radius = cv2.minEnclosingCircle(largest)
    circle_area = np.pi * (radius ** 2)
    circularity = area / circle_area if circle_area > 0 else 0
    if circularity < min_circularity:
        return False

    return True


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("\n--- PREDICT ENDPOINT TRIGGERED ---", flush=True)

    contents = await file.read()
    print(f"File Name: {file.filename} | Size: {len(contents)} bytes", flush=True)

    temp_dir = os.path.join(BASE_DIR, "uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    with open(temp_path, "wb") as f:
        f.write(contents)

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    print(f"Image Dimensions: {image.size} | Mode: {image.mode}", flush=True)

    if not looks_like_mri(image):
        print("Rejected upload: image failed MRI sanity check.", flush=True)
        return {
            "prediction": "Invalid Image",
            "confidence": "0.00%",
            "hemorrhage_confidence": "0.00%",
            "normal_confidence": "0.00%",
            "gradcam": "",
            "pro_gradcam": "",
            "lime": "",
            "error": "Please upload a valid grayscale MRI scan, not a regular photo, document, or image."
        }

    np_img = np.array(image)

    if model is not None:
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        try:
            cam_map, _ = generate_gradcam_map(model, input_tensor)
        except Exception as e:
            print(f"Grad-CAM map generation failed: {e}", flush=True)
            cam_map = None

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            hemorrhagic_prob = probabilities[0][0].item() * 100
            nonhemorrhagic_prob = probabilities[0][1].item() * 100

            confidence_val, predicted_idx = torch.max(probabilities, 1)
            prediction_label = CLASS_NAMES[predicted_idx.item()]
            confidence_str = f"{confidence_val.item() * 100:.2f}%"

            print(f"Probabilities - Hemorrhagic: {hemorrhagic_prob:.2f}%, NonHemorrhagic: {nonhemorrhagic_prob:.2f}%", flush=True)
            print(f"Prediction: {prediction_label} | Confidence: {confidence_str}", flush=True)

        try:
            orig_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            result_paths = generate_gradcam_images(
                orig_bgr, cam_map,
                confidence_score=confidence_val.item(),
                class_name=prediction_label
            )
            gradcam_b64 = file_to_data_url(result_paths.get("gradcam", ""))
            pro_gradcam_b64 = file_to_data_url(result_paths.get("gradcam_pro", ""))
        except Exception as e:
            print(f"Grad-CAM image generation failed: {e}", flush=True)
            gradcam_b64 = ""
            pro_gradcam_b64 = ""

        try:
            lime_result = save_lime_explanation(
                temp_path, cam_map, confidence_val.item(), prediction_label
            )
            lime_b64 = file_to_data_url(lime_result.get("lime_path", ""))
        except Exception as e:
            print(f"LIME generation failed: {e}", flush=True)
            lime_b64 = ""

    else:
        prediction_label = "Model Not Loaded"
        confidence_str = "0.00%"
        hemorrhagic_prob = 0.0
        nonhemorrhagic_prob = 0.0
        gradcam_b64 = ""
        pro_gradcam_b64 = ""
        lime_b64 = ""
        print("Warning: Model is not loaded.", flush=True)

    return {
        "prediction": prediction_label,
        "confidence": confidence_str,
        "hemorrhage_confidence": f"{hemorrhagic_prob:.2f}%",
        "normal_confidence": f"{nonhemorrhagic_prob:.2f}%",
        "gradcam": gradcam_b64,
        "pro_gradcam": pro_gradcam_b64,
        "lime": lime_b64
    }