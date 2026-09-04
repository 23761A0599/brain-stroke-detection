from pathlib import Path
import torch

# ==========================================
# Project Root
# ==========================================

# backend/model/config.py
# Go up two levels:
# config.py -> model -> backend -> Brain_Stroke_Training

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==========================================
# Models Directory
# ==========================================

MODELS_DIR = PROJECT_ROOT / "models"

# ==========================================
# Device
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Classes
# ==========================================

NUM_CLASSES = 2
IMAGE_SIZE = 224

CLASS_NAMES = [
    "Hemorrhagic",
    "NonHemorrhagic"
]