import asyncio
from pathlib import Path
import time
from fastapi import APIRouter, File, UploadFile
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model.config import CLASS_NAMES, DEVICE, IMAGE_SIZE
from model.model_loader import load_model
from services.gradcam_pro import generate_gradcam_images
from services.lime_explainer import save_lime_explanation

router = APIRouter()

model = load_model("efficientnetb0").to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    temp_path = UPLOAD_DIR / file.filename
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    raw_image = Image.open(temp_path).convert("RGB")
    tensor_image = transform(raw_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor_image)
        probabilities = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probabilities).item()

    class_name = CLASS_NAMES[pred_idx]
    confidence = float(probabilities[pred_idx].item()) * 100

    class_probs = {}
    for idx, name in enumerate(CLASS_NAMES):
        class_probs[name] = round(float(probabilities[idx].item()) * 100, 2)

    # Run heavy processing in non-blocking worker threads
    std_cam_path, pro_cam_path = await asyncio.to_thread(
        generate_gradcam_images, model, str(temp_path), confidence, class_name
    )
    lime_data = await asyncio.to_thread(
        save_lime_explanation, model, str(temp_path)
    )

    timestamp = int(time.time())

    def build_url(path_str):
        clean_p = str(path_str).replace("\\", "/").lstrip("/")
        return f"http://localhost:8000/{clean_p}?t={timestamp}"

    return {
        "prediction": class_name,
        "confidence": round(confidence, 2),
        "class_probabilities": class_probs,
        "probabilities": class_probs,
        "gradcam_url": build_url(std_cam_path),
        "gradcam_pro_url": build_url(pro_cam_path),
        "lime_url": build_url(lime_data["lime_path"]),
    }