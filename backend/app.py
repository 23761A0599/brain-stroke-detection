import os
import io
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Brain Stroke Detection API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_efficientnetb0.pth")
# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

CLASS_NAMES = ["Normal", "Stroke"]

def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    try:
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
        
        if isinstance(state_dict, torch.nn.Module):
            loaded_model = state_dict
        else:
            loaded_model = models.efficientnet_b0(weights=None)
            num_ftrs = loaded_model.classifier[1].in_features
            loaded_model.classifier[1] = torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
            loaded_model.load_state_dict(state_dict)

        loaded_model.eval()
        return loaded_model
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise e

model = get_model()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Brain Stroke Detection API is running"}

@app.get("/health")
def health_check():
    return {"model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class_idx = torch.argmax(probabilities).item()

        confidence = float(probabilities[predicted_class_idx])
        prediction_label = CLASS_NAMES[predicted_class_idx]

        return {
            "prediction": prediction_label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")