import io
import os
import base64
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from fastapi import APIRouter, File, UploadFile, HTTPException

from model.config import IMAGE_SIZE, CLASS_NAMES, DEVICE
from services.predictor import model
from services.gradcam import generate_gradcam_map
from services.gradcam_pro import generate_gradcam_images
from services.lime_explainer import save_lime_explanation

router = APIRouter()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

BASE_DIR = Path(__file__).resolve().parent.parent if 'Path' in globals() else os.path.dirname(os.path.abspath(__file__))
if not isinstance(BASE_DIR, str):
    BASE_DIR = str(BASE_DIR.parent if hasattr(BASE_DIR, 'parent') else BASE_DIR)

def file_to_data_url(path):
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Base64 encoding error: {e}", flush=True)
        return ""

@router.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    print("\n--- API /predict ROUTE TRIGGERED ---", flush=True)
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Save temp file for LIME
        temp_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)
        with open(temp_path, "wb") as f:
            f.write(contents)

        if model is None:
            raise HTTPException(status_code=500, detail="Model is not loaded on the server.")

        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Grad-CAM map
        try:
            cam_map, _ = generate_gradcam_map(model, input_tensor)
        except Exception as e:
            print(f"Grad-CAM map error: {e}", flush=True)
            cam_map = None

        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            hemorrhagic_prob = probabilities[0][0].item() * 100
            nonhemorrhagic_prob = probabilities[0][1].item() * 100
            confidence_val, predicted_idx = torch.max(probabilities, 1)
            prediction_label = CLASS_NAMES[predicted_idx.item()]
            confidence_str = f"{confidence_val.item() * 100:.2f}%"

        # Generate Visualizations
        np_img = np.array(image)
        orig_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        
        gradcam_b64, pro_gradcam_b64, lime_b64 = "", "", ""
        try:
            result_paths = generate_gradcam_images(
                orig_bgr, cam_map,
                confidence_score=confidence_val.item(),
                class_name=prediction_label
            )
            gradcam_b64 = file_to_data_url(result_paths.get("gradcam", ""))
            pro_gradcam_b64 = file_to_data_url(result_paths.get("gradcam_pro", ""))
        except Exception as e:
            print(f"Grad-CAM visualization error: {e}", flush=True)

        try:
            lime_result = save_lime_explanation(
                temp_path, cam_map, confidence_val.item(), prediction_label
            )
            lime_b64 = file_to_data_url(lime_result.get("lime_path", ""))
        except Exception as e:
            print(f"LIME error: {e}", flush=True)

        return {
            "prediction": prediction_label,
            "confidence": confidence_str,
            "hemorrhage_confidence": f"{hemorrhagic_prob:.2f}%",
            "normal_confidence": f"{nonhemorrhagic_prob:.2f}%",
            "gradcam": gradcam_b64,
            "pro_gradcam": pro_gradcam_b64,
            "lime": lime_b64
        }
    except Exception as e:
        print(f"Prediction route exception: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))