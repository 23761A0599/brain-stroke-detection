import io
import os
import base64
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model.config import IMAGE_SIZE, CLASS_NAMES, DEVICE
from model.model_builder import build_model
from services.gradcam_pro import generate_gradcam_pro
from services.lime_explainer import save_lime_explanation

app = FastAPI(title="Brain Hemorrhage Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CHANGED: CLASS_NAMES now comes from model.config (single source of
# truth) instead of a hardcoded, WRONGLY-ORDERED local list. The
# actual trained model's order is {'Hemorrhagic': 0, 'NonHemorrhagic': 1} -
# the old hardcoded ["Normal", "Hemorrhagic"] had both the wrong order
# AND the wrong class name ("Normal" vs "NonHemorrhagic").

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_PATHS = [
    os.path.join(BASE_DIR, "models", "EfficientNetB0", "best_efficientnetb0.pth"),
    os.path.join(BASE_DIR, "backend", "models", "EfficientNetB0", "best_efficientnetb0.pth"),
    os.path.join(BASE_DIR, "models", "best_efficientnetb0.pth"),
    os.path.join(BASE_DIR, "backend", "models", "best_efficientnetb0.pth"),
    "models/EfficientNetB0/best_efficientnetb0.pth",
    "backend/models/EfficientNetB0/best_efficientnetb0.pth",
]

MODEL_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

model = None
if MODEL_PATH:
    try:
        # CHANGED: this is the actual fix. The checkpoint is a STATE
        # DICT (just weights), not a full pickled model object. You
        # must first build the correct architecture with build_model(),
        # then load the weights into it with load_state_dict() - never
        # torch.load() directly as if it were a complete model.
        model = build_model("efficientnetb0")
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print(f"Model loaded successfully from: {MODEL_PATH}", flush=True)
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}", flush=True)
        model = None
else:
    print(f"Model file not found in any expected location: {POSSIBLE_PATHS}", flush=True)

# CHANGED: IMAGE_SIZE now comes from model.config (224), matching what
# the model was actually trained on. The old hardcoded 256x256 would
# silently degrade predictions even once the model loads correctly.
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def buffer_to_base64_url(image_np_or_buf):
    if isinstance(image_np_or_buf, io.BytesIO):
        encoded = base64.b64encode(image_np_or_buf.getvalue()).decode('utf-8')
    elif isinstance(image_np_or_buf, np.ndarray):
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


def file_to_data_url(path):
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("\n--- PREDICT ENDPOINT TRIGGERED ---", flush=True)

    contents = await file.read()
    print(f"File Name: {file.filename} | Size: {len(contents)} bytes", flush=True)

    # Save to a temp path - needed for the real gradcam_pro/lime
    # functions, which take a file path, not raw bytes
    temp_dir = os.path.join(BASE_DIR, "uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    with open(temp_path, "wb") as f:
        f.write(contents)

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    print(f"Image Dimensions: {image.size} | Mode: {image.mode}", flush=True)

    np_img = np.array(image)

    if model is not None:
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            hemorrhagic_prob = probabilities[0][0].item() * 100
            nonhemorrhagic_prob = probabilities[0][1].item() * 100

            confidence_val, predicted_idx = torch.max(probabilities, 1)
            prediction_label = CLASS_NAMES[predicted_idx.item()]
            confidence_str = f"{confidence_val.item() * 100:.2f}%"

            print(f"Probabilities - Hemorrhagic: {hemorrhagic_prob:.2f}%, "
                  f"NonHemorrhagic: {nonhemorrhagic_prob:.2f}%", flush=True)
            print(f"Prediction: {prediction_label} | Confidence: {confidence_str}", flush=True)

        # CHANGED: use the REAL gradcam_pro and lime functions instead
        # of the fake placeholder circle/rectangle drawings
        try:
            gradcam_pro_url = generate_gradcam_pro(
                model, temp_path,
                confidence_score=confidence_val.item(),
                class_name=prediction_label
            )
            gradcam_pro_path = os.path.join(BASE_DIR, gradcam_pro_url.lstrip("/"))
            pro_gradcam_b64 = file_to_data_url(gradcam_pro_path)
        except Exception as e:
            print(f"Grad-CAM++ generation failed: {e}", flush=True)
            pro_gradcam_b64 = ""

        try:
            lime_result = save_lime_explanation(model, temp_path)
            lime_b64 = file_to_data_url(lime_result["lime_path"])
        except Exception as e:
            print(f"LIME generation failed: {e}", flush=True)
            lime_b64 = ""

        gradcam_b64 = pro_gradcam_b64  # basic Grad-CAM reuses the pro visualization if not separately wired

    else:
        prediction_label = "Model Not Loaded"
        confidence_str = "0.00%"
        hemorrhagic_prob = 0.0
        nonhemorrhagic_prob = 0.0
        gradcam_b64 = ""
        pro_gradcam_b64 = ""
        lime_b64 = ""
        print("Warning: Model is not loaded.", flush=True)

    return {
        "prediction": prediction_label,
        "confidence": confidence_str,
        "hemorrhage_confidence": f"{hemorrhagic_prob:.2f}%",
        "normal_confidence": f"{nonhemorrhagic_prob:.2f}%",
        "gradcam": gradcam_b64,
        "pro_gradcam": pro_gradcam_b64,
        "lime": lime_b64
    }