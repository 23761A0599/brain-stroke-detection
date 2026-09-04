import os
import time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

app = FastAPI(title="Brain Hemorrhage Detection API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up output directories
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
GRADCAM_PRO_DIR = OUTPUT_DIR / "gradcam_pro"
LIME_DIR = OUTPUT_DIR / "lime"

for path in [GRADCAM_DIR, GRADCAM_PRO_DIR, LIME_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Mount static files directory
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

def create_placeholder_image(file_path: Path, title: str):
    """Generates a real image file on disk so Render can serve it."""
    img = Image.new('RGB', (300, 300), color=(30, 41, 59))
    d = ImageDraw.Draw(img)
    d.text((20, 140), f"{title} Output", fill=(255, 255, 255))
    img.save(file_path)

@app.get("/")
def read_root():
    return {"status": "Backend running successfully"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), request: Request = None):
    # Dynamically resolve host base URL
    if request:
        base_url = str(request.base_url).rstrip("/")
    else:
        base_url = "https://brain-hemorrhage-backend.onrender.com"

    timestamp = int(time.time())
    
    # Define file names
    gc_name = f"std_gradcam_{timestamp}.png"
    gc_pro_name = f"pro_gradcam_{timestamp}.png"
    lime_name = f"lime_{timestamp}.png"

    # Save real image files to disk
    create_placeholder_image(GRADCAM_DIR / gc_name, "Grad-CAM")
    create_placeholder_image(GRADCAM_PRO_DIR / gc_pro_name, "Grad-CAM Pro")
    create_placeholder_image(LIME_DIR / lime_name, "LIME")

    # Construct dynamic URLs
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
        "gradcam_url": f"{base_url}/outputs/gradcam/{gc_name}",
        "gradcam_pro_url": f"{base_url}/outputs/gradcam_pro/{gc_pro_name}",
        "lime_url": f"{base_url}/outputs/lime/{lime_name}"
    }