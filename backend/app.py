import os
import torch
import torchvision.models as models
from flask import Flask, request, jsonify

app = Flask(__name__)

# Relative path pointing directly to your local file in the repo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_efficientnetb0.pth")

# Revert to standard torchvision model structure
model = models.efficientnet_b0(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 2)

# Load weights directly from repo folder
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully from local file.")
else:
    print(f"Error: File not found at {MODEL_PATH}")