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

# Import the new router
from routes.predict import router as predict_router

from model.config import IMAGE_SIZE, CLASS_NAMES, DEVICE
from services.gradcam import generate_gradcam_map
from services.gradcam_pro import generate_gradcam_images
from services.lime_explainer import save_lime_explanation
from services.predictor import model  # Import shared model instance

app = FastAPI(title="Brain Hemorrhage Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the router
app.include_router(predict_router)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
                   overall_brightness_threshold=195,
                   min_midtone_ratio=0.15,
                   midtone_low=15,
                   midtone_high=235,
                   max_aspect_ratio=1.25):
    np_img = np.array(pil_image).astype(np.float32)
    gray = np.array(pil_image.convert("L")).astype(np.float32)
    h, w = gray.shape

    aspect_ratio = max(h, w) / min(h, w)
    if aspect_ratio > max_aspect_ratio:
        return False

    if np_img.ndim == 3 and np_img.shape[2] >= 3:
        r, g, b = np_img[..., 0], np_img[..., 1], np_img[..., 2]
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        colorfulness = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2) + \
            0.3 * np.sqrt(rg.std() ** 2 + yb.std() ** 2)
        if colorfulness >= colorfulness_threshold:
            return False

    if gray.mean() > overall_brightness_threshold:
        return False

    midtone_ratio = np.mean((gray >= midtone_low) & (gray <= midtone_high))
    if midtone_ratio < min_midtone_ratio:
        return False

    return True


@app.post("/predict-legacy")
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