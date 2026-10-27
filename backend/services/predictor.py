import os
import torch
from pathlib import Path
from model.config import DEVICE, MODELS_DIR
from model.model_builder import build_model

BASE_DIR = Path(__file__).resolve().parent.parent
POSSIBLE_PATHS = [
    Path(MODELS_DIR) / "EfficientNetB0" / "best_efficientnetb0.pth",
    BASE_DIR / "models" / "EfficientNetB0" / "best_efficientnetb0.pth",
    BASE_DIR / "backend" / "models" / "EfficientNetB0" / "best_efficientnetb0.pth",
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
        print(f"Model successfully loaded in predictor.py from: {MODEL_PATH}", flush=True)
    except Exception as e:
        print(f"Error loading model weights in predictor.py: {e}", flush=True)
        model = None
else:
    print(f"Warning: Model checkpoint not found in paths: {POSSIBLE_PATHS}", flush=True)