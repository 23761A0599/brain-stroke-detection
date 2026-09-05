import base64
import cv2
import numpy as np
import torch
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image

from model.config import CLASS_NAMES, DEVICE, IMAGE_SIZE
from services.gradcam import generate_gradcam_map, transform
from services.gradcam_pro import generate_gradcam_images
from services.lime_explainer import save_lime_explanation
from services.predictor import model  # Imports the loaded PyTorch model instance

router = APIRouter()


def encode_image_to_base64(file_path: str) -> str:
    """Helper function to convert output image files to base64 strings for React rendering."""
    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Read input image bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)

        # Decode as a 3-channel BGR image directly using OpenCV to preserve colors
        orig_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if orig_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        # 2. Convert to PIL Image and apply transformations for model input
        pil_img = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
        image_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        # 3. Compute Grad-CAM activation map from model
        cam_map, pred_class_idx = generate_gradcam_map(model, image_tensor)

        # 4. Correct Probability Score Calculation (Binary vs Multi-Class)
        with torch.no_grad():
            outputs = model(image_tensor)
            if outputs.shape[1] == 1:
                # Binary output node (Sigmoid activation)
                prob = torch.sigmoid(outputs).item()
                pred_class_idx = 1 if prob >= 0.5 else 0
                confidence = prob if pred_class_idx == 1 else (1.0 - prob)
            else:
                # Multi-class output nodes (Softmax activation)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_class_idx = torch.argmax(probs).item()
                confidence = probs[pred_class_idx].item()

        predicted_label = CLASS_NAMES[pred_class_idx]
        is_hemorrhagic = "hemorrhag" in predicted_label.lower() or pred_class_idx == 1

        # 5. Generate visual overlays if classification is Hemorrhagic
        if is_hemorrhagic:
            img_paths = generate_gradcam_images(
                orig_bgr=orig_bgr,
                cam_map=cam_map,
                confidence_score=confidence,
                class_name=predicted_label
            )

            # Convert generated images to base64 response format
            gradcam_b64 = encode_image_to_base64(img_paths["gradcam"])
            gradcam_pro_b64 = encode_image_to_base64(img_paths["gradcam_pro"])
            lime_b64 = encode_image_to_base64(img_paths["lime"])
        else:
            gradcam_b64, gradcam_pro_b64, lime_b64 = None, None, None

        return {
            "prediction": predicted_label,
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "is_hemorrhagic": is_hemorrhagic,
            "gradcam": gradcam_b64,
            "gradcam_pro": gradcam_pro_b64,
            "lime": lime_b64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))