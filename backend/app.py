import os
import io
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Brain Stroke Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_efficientnetb0.pth")

model = None

def get_model():
    global model
    if model is None:
        # Load weights with CPU mapping to keep memory minimal
        weights = models.EfficientNet_B0_Weights.DEFAULT
        m = models.efficientnet_b0(weights=None)
        # Update classifier head to match your training classes
        in_features = m.classifier[1].in_features
        m.classifier[1] = torch.nn.Linear(in_features, 2)
        
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        m.load_state_dict(state_dict)
        m.eval()
        model = m
    return model

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Brain Stroke Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transform(image).unsqueeze(0)
        
        net = get_model()
        with torch.no_grad():
            outputs = net(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        classes = ["Normal", "Stroke"]
        return {
            "prediction": classes[predicted.item()],
            "confidence": float(confidence.item())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))