import os
import time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Brain Hemorrhage Detection API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directory paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
GRADCAM_PRO_DIR = OUTPUT_DIR / "gradcam_pro"
LIME_DIR = OUTPUT_DIR / "lime"

for path in [GRADCAM_DIR, GRADCAM_PRO_DIR, LIME_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Mount outputs directory to serve static images
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

@app.get("/")
def read_root():
    return {"status": "Backend running successfully"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), request: Request = None):
    # Dynamically obtain base URL (Render domain or localhost dynamically)
    if request:
        base_url = str(request.base_url).rstrip("/")
    else:
        base_url = "https://brain-hemorrhage-backend.onrender.com"

    timestamp = int(time.time())
    
    # Placeholder relative filenames
    gradcam_filename = f"std_gradcam_{timestamp}.png"
    gradcam_pro_filename = f"pro_gradcam_{timestamp}.png"
    lime_filename = f"lime_{timestamp}.png"

    # Dynamic image URLs using current host
    gradcam_url = f"{base_url}/outputs/gradcam/{gradcam_filename}?t={timestamp}"
    gradcam_pro_url = f"{base_url}/outputs/gradcam_pro/{gradcam_pro_filename}?t={timestamp}"
    lime_url = f"{base_url}/outputs/lime/{lime_filename}?t={timestamp}"

    # Sample mock or model inference output
    return {
        "prediction": "Hemorrhagic",
        "confidence": 57.74,
        "class_probabilities": {
            "Hemorrhagic": 57.74,
            "NonHemorrhagic": 42.26
        },
        "probabilities": {
            "Hemorrhagic": 57.74,
            "NonHemorrhagic": 42.26
        },
        "gradcam_url": gradcam_url,
        "gradcam_pro_url": gradcam_pro_url,
        "lime_url": lime_url
    }