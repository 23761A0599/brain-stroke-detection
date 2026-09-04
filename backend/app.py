import os
import io
import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import timm
import gdown

app = Flask(__name__)
CORS(app)

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_efficientnetb0.pth")

# Google Drive File ID for best_efficientnetb0.pth
GDRIVE_FILE_ID = "1Os3q78NLEakiBeTS-Oa2SFNQulNipFHH"

def ensure_model_downloaded():
    """Downloads model weights from Google Drive if missing or corrupt."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
        print("Downloading model weights from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# Auto-download model weights on server start
ensure_model_downloaded()

# Load Model
def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

try:
    model = load_model()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Transformations for MRI input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CLASSES = ["Non-Hemorrhagic", "Hemorrhagic"]

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model failed to load on server.",
            "prediction": "Error",
            "confidence": 0.0
        }), 500

    if "file" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["file"]
    
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)

        predicted_class = CLASSES[predicted_idx.item()]
        confidence_pct = round(confidence.item() * 100, 2)

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence_pct
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)