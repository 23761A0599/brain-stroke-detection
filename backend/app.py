import io
import os
import base64

print("STARTUP: importing torch...", flush=True)
import torch
print("STARTUP: importing torchvision.transforms...", flush=True)
import torchvision.transforms as transforms
print("STARTUP: importing PIL...", flush=True)
from PIL import Image
print("STARTUP: importing numpy...", flush=True)
import numpy as np
print("STARTUP: importing cv2...", flush=True)
import cv2
print("STARTUP: importing fastapi...", flush=True)
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

print("STARTUP: importing model.config...", flush=True)
from model.config import IMAGE_SIZE, CLASS_NAMES, DEVICE
print("STARTUP: importing model_builder...", flush=True)
from model.model_builder import build_model
print("STARTUP: importing gradcam_pro...", flush=True)
from services.gradcam_pro import generate_gradcam_pro
print("STARTUP: importing lime_explainer...", flush=True)
from services.lime_explainer import save_lime_explanation
print("STARTUP: all imports done.", flush=True)

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

print(f"STARTUP: MODEL_PATH resolved to: {MODEL_PATH}", flush=True)

model = None
if MODEL_PATH:
    try:
        print("STARTUP: building model architecture...", flush=True)
        model = build_model("efficientnetb0")
        print("STARTUP: architecture built, loading state dict from disk...", flush=True)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        print("STARTUP: state dict loaded into memory, applying to model...", flush=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print(f"STARTUP: Model loaded successfully from: {MODEL_PATH}", flush=True)
    except Exception as e:
        print(f"STARTUP: Error loading model from {MODEL_PATH}: {e}", flush=True)
        model = None
else:
    print(f"STARTUP: Model file not found in any expected location: {POSSIBLE_PATHS}", flush=True)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("STARTUP: app.py fully initialized, ready for uvicorn to bind port.", flush=True)

# ... rest of the file (buffer_to_base64_url, file_to_data_url, /predict endpoint) unchanged