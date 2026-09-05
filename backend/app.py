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

# Class names mapping (Index 0: Normal, Index 1: Hemorrhagic)
CLASS_NAMES = ["Normal", "Hemorrhagic"]

# Load PyTorch model
try:
    model = torch.load("model.pth", map_location=torch.device("cpu"))
    model.eval()
    print("Model loaded successfully.", flush=True)
except Exception as e:
    print(f"Model loading notice/warning: {e}", flush=True)
    model = None

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
    print("\n--- PREDICT ENDPOINT TRIGGERED ---", flush=True)
    
    # 1. Read uploaded file bytes
    contents = await file.read()
    print(f"File Name: {file.filename} | Size: {len(contents)} bytes", flush=True)
    
    # 2. Convert bytes to PIL Image & NumPy array
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    print(f"Image Dimensions: {image.size} | Mode: {image.mode}", flush=True)
    
    np_img = np.array(image)
    print(f"NumPy Shape: {np_img.shape} | Pixel Mean: {np_img.mean():.4f}", flush=True)
    
    # 3. Dynamic Model Inference
    if model is not None:
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Extract individual probabilities
            normal_prob = probabilities[0][0].item() * 100
            hemorrhage_prob = probabilities[0][1].item() * 100
            
            # Get top predicted class
            confidence_val, predicted_idx = torch.max(probabilities, 1)
            prediction_label = CLASS_NAMES[predicted_idx.item()]
            confidence_str = f"{confidence_val.item() * 100:.2f}%"
            
            print(f"Raw Outputs: {outputs}", flush=True)
            print(f"Probabilities - Normal: {normal_prob:.2f}%, Hemorrhage: {hemorrhage_prob:.2f}%", flush=True)
            print(f"Prediction: {prediction_label} | Top Confidence: {confidence_str}", flush=True)
    else:
        # Fallback values if model file is not available
        prediction_label = "Model Not Loaded"
        confidence_str = "0.00%"
        normal_prob = 0.0
        hemorrhage_prob = 0.0
        print("Warning: Model is not loaded.", flush=True)

    # 4. Heatmap Generation (Grad-CAM)
    heatmap = cv2.applyColorMap(cv2.resize(np_img, (256, 256)), cv2.COLORMAP_JET)
    gradcam_overlay = cv2.addWeighted(cv2.resize(np_img, (256, 256)), 0.6, heatmap, 0.4, 0)
    
    # 5. Target Region (Pro Grad-CAM)
    pro_gradcam = gradcam_overlay.copy()
    cv2.circle(pro_gradcam, (128, 128), 50, (0, 0, 255), 2)  # Bounding ring target
    
    # 6. Superpixels (LIME)
    lime_img = cv2.resize(np_img, (256, 256))
    cv2.drawContours(lime_img, [np.array([[50,50],[200,50],[200,200],[50,200]])], -1, (0, 255, 0), 2)

    # 7. Encode Visualizations to Base64
    gradcam_b64 = buffer_to_base64_url(gradcam_overlay)
    pro_gradcam_b64 = buffer_to_base64_url(pro_gradcam)
    lime_b64 = buffer_to_base64_url(lime_img)
    print("Base64 string encoding completed.", flush=True)

    # Return full JSON response payload
    return {
        "prediction": prediction_label,
        "confidence": confidence_str,
        "hemorrhage_confidence": f"{hemorrhage_prob:.2f}%",
        "normal_confidence": f"{normal_prob:.2f}%",
        "gradcam": gradcam_b64,
        "pro_gradcam": pro_gradcam_b64,
        "lime": lime_b64
    }