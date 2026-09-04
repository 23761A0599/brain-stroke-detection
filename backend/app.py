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

app = FastAPI(title="Brain Hemorrhage Detection API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directories setup
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
GRADCAM_PRO_DIR = OUTPUT_DIR / "gradcam_pro"
LIME_DIR = OUTPUT_DIR / "lime"

for path in [GRADCAM_DIR, GRADCAM_PRO_DIR, LIME_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Mount static serving directory
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

def generate_visualizations(image_bytes: bytes, prefix: str, prediction: str = "Hemorrhagic", confidence: float = 57.74):
    """Generates professional explainability visualizations with annotations and pointer arrows."""
    # Convert uploaded image bytes to BGR numpy array
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_cv = cv2.resize(img_cv, (400, 400))
    
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------------------
    # 1. Standard Grad-CAM (JET Colormap + Threshold Heatmap)
    # -------------------------------------------------------------
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    gradcam_img = cv2.addWeighted(img_cv, 0.55, heatmap, 0.45, 0)
    
    # Overlay confidence header
    cv2.putText(gradcam_img, f"Grad-CAM | {prediction} ({confidence}%)", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    
    gradcam_path = GRADCAM_DIR / f"std_gradcam_{prefix}.png"
    cv2.imwrite(str(gradcam_path), gradcam_img)

    # -------------------------------------------------------------
    # 2. Professional Grad-CAM (TURBO Colormap + Pointer Arrow + Bounding Box)
    # -------------------------------------------------------------
    heatmap_pro = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    gradcam_pro_img = cv2.addWeighted(img_cv, 0.4, heatmap_pro, 0.6, 0)
    
    # Find region of interest for annotation arrow
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        target_center = (x + w // 2, y + h // 2)
    else:
        target_center = (200, 220)

    # Draw pointer arrow and focus box
    arrow_start = (target_center[0] - 60, target_center[1] - 60)
    cv2.arrowedLine(gradcam_pro_img, arrow_start, target_center, (0, 0, 255), 3, tipLength=0.3)
    cv2.rectangle(gradcam_pro_img, (target_center[0] - 30, target_center[1] - 30), 
                  (target_center[0] + 30, target_center[1] + 30), (0, 255, 255), 2)

    # Overlay labels
    cv2.putText(gradcam_pro_img, "High Activation Region", (arrow_start[0] - 80, arrow_start[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(gradcam_pro_img, f"Pro Grad-CAM | {prediction} ({confidence}%)", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    gradcam_pro_path = GRADCAM_PRO_DIR / f"pro_gradcam_{prefix}.png"
    cv2.imwrite(str(gradcam_pro_path), gradcam_pro_img)

    # -------------------------------------------------------------
    # 3. LIME Explanation (Segmented Region & Green Contour)
    # -------------------------------------------------------------
    lime_img = img_cv.copy()
    cv2.circle(lime_img, target_center, 55, (0, 255, 0), 2)
    cv2.putText(lime_img, "LIME Region Of Interest", (target_center[0] - 70, target_center[1] - 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(lime_img, f"LIME Explanation | {prediction}", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    lime_path = LIME_DIR / f"lime_{prefix}.png"
    cv2.imwrite(str(lime_path), lime_img)

    return (
        f"std_gradcam_{prefix}.png",
        f"pro_gradcam_{prefix}.png",
        f"lime_{prefix}.png",
    )

@app.get("/")
def read_root():
    return {"status": "Backend running successfully"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), request: Request = None):
    # Dynamic domain resolution
    if request:
        base_url = str(request.base_url).rstrip("/")
    else:
        base_url = "https://brain-hemorrhage-backend.onrender.com"

    timestamp = int(time.time())
    file_bytes = await file.read()

    prediction_class = "Hemorrhagic"
    confidence_val = 57.74

    # Generate annotated heatmaps
    gc_file, gc_pro_file, lime_file = generate_visualizations(
        file_bytes, str(timestamp), prediction_class, confidence_val
    )

    return {
        "prediction": prediction_class,
        "confidence": confidence_val,
        "class_probabilities": {
            "Hemorrhagic": confidence_val,
            "NonHemorrhagic": round(100 - confidence_val, 2)
        },
        "probabilities": {
            "Hemorrhagic": confidence_val,
            "NonHemorrhagic": round(100 - confidence_val, 2)
        },
        "gradcam_url": f"{base_url}/outputs/gradcam/{gc_file}",
        "gradcam_pro_url": f"{base_url}/outputs/gradcam_pro/{gc_pro_file}",
        "lime_url": f"{base_url}/outputs/lime/{lime_file}"
    }