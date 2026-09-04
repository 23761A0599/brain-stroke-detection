import io
import base64
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Brain Hemorrhage Detection API")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def buffer_to_base64_url(image_np_or_buf):
    """Converts a PIL image, BytesIO buffer, or OpenCV image (BGR/RGB) to base64 Data URL."""
    if isinstance(image_np_or_buf, io.BytesIO):
        encoded = base64.b64encode(image_np_or_buf.getvalue()).decode('utf-8')
    elif isinstance(image_np_or_buf, np.ndarray):
        # If image is OpenCV BGR format, convert to RGB
        if len(image_np_or_buf.shape) == 3 and image_np_or_buf.shape[2] == 3:
            image_np_or_buf = cv2.cvtColor(image_np_or_buf, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_np_or_buf.astype('uint8'))
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    elif isinstance(image_np_or_buf, Image.Image):
        buf = io.BytesIO()
        image_np_or_buf.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    else:
        return ""
    
    return f"data:image/png;base64,{encoded}"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded file bytes
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Simple dummy model processing or your PyTorch model inference
    # Example generating visual overlay outputs
    np_img = np.array(image)
    
    # 1. Heatmap Generation (Grad-CAM)
    heatmap = cv2.applyColorMap(cv2.resize(np_img, (256, 256)), cv2.COLORMAP_JET)
    gradcam_overlay = cv2.addWeighted(cv2.resize(np_img, (256, 256)), 0.6, heatmap, 0.4, 0)
    
    # 2. Target Region (Pro Grad-CAM)
    pro_gradcam = gradcam_overlay.copy()
    cv2.circle(pro_gradcam, (128, 128), 50, (0, 0, 255), 2)  # Bounding ring target
    
    # 3. Superpixels (LIME)
    lime_img = cv2.resize(np_img, (256, 256))
    cv2.drawContours(lime_img, [np.array([[50,50],[200,50],[200,200],[50,200]])], -1, (0, 255, 0), 2)

    # Encode to Base64 Data URLs
    gradcam_b64 = buffer_to_base64_url(gradcam_overlay)
    pro_gradcam_b64 = buffer_to_base64_url(pro_gradcam)
    lime_b64 = buffer_to_base64_url(lime_img)

    return {
        "prediction": "Hemorrhagic",
        "confidence": "83.05%",
        "gradcam": gradcam_b64,
        "pro_gradcam": pro_gradcam_b64,
        "lime": lime_b64
    }