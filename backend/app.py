import os
import io
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Brain Stroke Detection API")

# Configure CORS for frontend access
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
        try:
            # Load file weights onto CPU
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            
            # Case 1: Checkpoint is a full PyTorch model instance
            if isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
            # Case 2: Checkpoint is a state dictionary
            elif isinstance(checkpoint, dict):
                m = models.efficientnet_b0(weights=None)
                in_features = m.classifier[1].in_features
                m.classifier[1] = torch.nn.Linear(in_features, 2)
                
                # Extract state_dict if wrapped in a key
                state_dict = checkpoint.get("state_dict", checkpoint)
                m.load_state_dict(state_dict)
                model = m
            else:
                raise ValueError("Unrecognized model checkpoint format.")
                
            model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
            
    return model

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Brain Stroke Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Standard ImageNet pre-processing pipeline
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        tensor = transform(image).unsqueeze(0)
        
        net = get_model()
        with torch.no_grad():
            outputs = net(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        classes = ["Normal", "Stroke"]
        raw_confidence = float(confidence.item())
        
        return {
            "prediction": classes[predicted.item()],
            "confidence": raw_confidence,
            "confidence_percentage": f"{raw_confidence * 100:.2f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))