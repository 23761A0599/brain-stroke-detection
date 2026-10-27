import os
import torch
from pathlib import Path
from model.config import DEVICE, MODELS_DIR
from model.model_builder import build_model

BASE_DIR = Path(__file__).resolve().parent.parent
POSSIBLE_PATHS = [
    Path(MODELS_DIR) / "EfficientNetB0" / "best_efficientnetb0.pth",
    Path(MODELS_DIR) / "best_efficientnetb0.pth",
    BASE_DIR / "models" / "EfficientNetB0" / "best_efficientnetb0.pth",
    BASE_DIR / "models" / "best_efficientnetb0.pth",
    BASE_DIR / "backend" / "models" / "EfficientNetB0" / "best_efficientnetb0.pth",
]

MODEL_PATH = next((p for p in POSSIBLE_PATHS if os.path.exists(p)), None)

model = None
if MODEL_PATH:
    try:
        # pretrained=False: the checkpoint supplies the weights (no download on Render)
        model = build_model("efficientnetb0", pretrained=False, mode="full")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"Model loaded from: {MODEL_PATH}", flush=True)
    except Exception as e:
        print(f"Error loading model weights in predictor.py: {e}", flush=True)
        model = None
else:
    print(f"Warning: checkpoint not found in: {POSSIBLE_PATHS}", flush=True)