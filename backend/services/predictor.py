import os
import torch
from pathlib import Path
from model.config import DEVICE, MODELS_DIR
from model.model_builder import build_model

# Find the checkpoint path safely
POSSIBLE_PATHS = [
    Path(MODELS_DIR) / "EfficientNetB0" / "best_efficientnetb0.pth",
    Path("models") / "EfficientNetB0" / "best_efficientnetb0.pth",
    Path("backend") / "models" / "EfficientNetB0" / "best_efficientnetb0.pth",
    "models/EfficientNetB0/best_efficientnetb0.pth",
    "backend/models/EfficientNetB0/best_efficientnetb0.pth",
]

MODEL_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

# Initialize and assign the model variable so other modules can import/access it
model = build_model("efficientnetb0")

if MODEL_PATH:
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Model successfully loaded in predictor.py from: {MODEL_PATH}", flush=True)
    except Exception as e:
        print(f"Error loading model weights from {MODEL_PATH}: {e}", flush=True)
else:
    print(f"Warning: Model checkpoint not found in paths: {POSSIBLE_PATHS}", flush=True)

model.to(DEVICE)
model.eval()