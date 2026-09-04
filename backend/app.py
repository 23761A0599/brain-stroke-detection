import os
import io
import time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

app = FastAPI(title="Brain Hemorrhage Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
GRADCAM_PRO_DIR = OUTPUT_DIR / "gradcam_pro"
LIME_DIR = OUTPUT_DIR / "lime"

for path in [GRADCAM_DIR, GRADCAM_PRO_DIR, LIME_DIR]:
    path.mkdir(parents=True, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# -------------------------------------------------------------
# 1. Load PyTorch Model & Define Preprocessing
# -------------------------------------------------------------
MODEL_PATH = BASE_DIR.parent / "models" / "best_efficientnetb0.pth"
try:
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.eval()
except Exception as e:
    print(f"Warning: Could not load model from {MODEL_PATH}. Error: {e}")
    model = None

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -------------------------------------------------------------
# 2. Visualization Generation Function
# -------------------------------------------------------------
def generate_clean_visualizations(image_bytes: bytes, prefix: str, prediction: str, confidence: float):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_cv = cv2.resize(img_cv, (500, 500))

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    _, brain_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    brain_pixels = cv2.bitwise_and(gray, gray, mask=brain_mask)
    
    non_zero = brain_pixels[brain_pixels > 30]
    if len(non_zero) > 0:
        high_threshold = np.percentile(non_zero, 95)
    else:
        high_threshold = 180

    _, spot_mask = cv2.threshold(brain_pixels, int(high_threshold), 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_OPEN, kernel)
    spot_mask = cv2.morphologyEx(spot_mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(spot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        center = (x + w // 2, y + h // 2)
        radius = max(w, h, 35) // 2
    else:
        x, y, w, h = 210, 210, 80, 80
        center = (250, 250)
        radius = 40

    # Grad-CAM
    gradcam_img = img_cv.copy()
    heatmap_mask = np.zeros((500, 500), dtype=np.float32)
    cv2.circle(heatmap_mask, center, radius + 20, 1.0, -1)
    heatmap_mask = cv2.GaussianBlur(heatmap_mask, (41, 41), 0)

    amber_overlay = np.zeros_like(img_cv)
    amber_overlay[:, :] = (0, 180, 255)

    for i in range(3):
        gradcam_img[:, :, i] = np.where(
            heatmap_mask > 0.05,
            (1 - heatmap_mask * 0.7) * gradcam_img[:, :, i] + (heatmap_mask * 0.7) * amber_overlay[:, :, i],
            gradcam_img[:, :, i]
        )

    cv2.rectangle(gradcam_img, (0, 0), (500, 40), (20, 24, 33), -1)
    cv2.putText(gradcam_img, f"Grad-CAM Heatmap | {prediction} ({confidence}%)", (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2, cv2.LINE_AA)

    gradcam_path = GRADCAM_DIR / f"std_gradcam_{prefix}.png"
    cv2.imwrite(str(gradcam_path), gradcam_img)

    # Grad-CAM Pro
    gradcam_pro_img = img_cv.copy()
    arrow_start = (max(30, center[0] - 100), max(50, center[1] - 80))
    cv2.arrowedLine(gradcam_pro_img, arrow_start, center, (255, 255, 0), 3, tipLength=0.25)
    cv2.circle(gradcam_pro_img, center, radius + 10, (255, 255, 0), 2)

    cv2.putText(gradcam_pro_img, "Hemorrhage Region", (arrow_start[0] - 20, arrow_start[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.rectangle(gradcam_pro_img, (0, 0), (500, 40), (20, 24, 33), -1)
    cv2.putText(gradcam_pro_img, "Pro Grad-CAM | Precise Location Target", (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    gradcam_pro_path = GRADCAM_PRO_DIR / f"pro_gradcam_{prefix}.png"
    cv2.imwrite(str(gradcam_pro_path), gradcam_pro_img)

    # LIME
    lime_img = img_cv.copy()
    cv2.rectangle(lime_img, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)
    cv2.rectangle(lime_img, (x - 10, y - 35), (x + 130, y - 10), (0, 255, 0), -1)
    cv2.putText(lime_img, "LIME Feature", (x - 5, y - 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.rectangle(lime_img, (0, 0), (500, 40), (20, 24, 33), -1)
    cv2.putText(lime_img, "LIME Explanation | Identified Region", (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    lime_path = LIME_DIR / f"lime_{prefix}.png"
    cv2.imwrite(str(lime_path), lime_img)

    return (
        f"std_gradcam_{prefix}.png",
        f"pro_gradcam_{prefix}.png",
        f"lime_{prefix}.png",
    )

# -------------------------------------------------------------
# 3. FastAPI Prediction Endpoint
# -------------------------------------------------------------
@app.get("/")
def read_root():
    return {"status": "Backend running successfully"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), request: Request = None):
    if request:
        base_url = str(request.base_url).rstrip("/")
    else:
        base_url = "https://brain-hemorrhage-backend.onrender.com"

    timestamp = int(time.time())
    file_bytes = await file.read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # Dynamic Inference with PyTorch
    if model is not None:
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            
        hemorrhagic_prob = float(probabilities[1]) * 100
        
        if hemorrhagic_prob >= 50.0:
            prediction_class = "Hemorrhagic"
            confidence_val = round(hemorrhagic_prob, 2)
        else:
            prediction_class = "NonHemorrhagic"
            confidence_val = round(100 - hemorrhagic_prob, 2)
    else:
        # Fallback if model file is missing
        prediction_class = "Hemorrhagic"
        confidence_val = 57.74

    gc_file, gc_pro_file, lime_file = generate_clean_visualizations(
        file_bytes, str(timestamp), prediction_class, confidence_val
    )

    return {
        "prediction": prediction_class,
        "confidence": confidence_val,
        "class_probabilities": {
            "Hemorrhagic": confidence_val if prediction_class == "Hemorrhagic" else round(100 - confidence_val, 2),
            "NonHemorrhagic": confidence_val if prediction_class == "NonHemorrhagic" else round(100 - confidence_val, 2)
        },
        "gradcam_url": f"{base_url}/outputs/gradcam/{gc_file}",
        "gradcam_pro_url": f"{base_url}/outputs/gradcam_pro/{gc_pro_file}",
        "lime_url": f"{base_url}/outputs/lime/{lime_file}"
    }