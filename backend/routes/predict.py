import base64
import io
import os
import threading
import traceback

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from model.config import CLASS_NAMES, DEVICE, IMAGE_SIZE
from services.predictor import model
from services.explainers import (
    HEM_IDX, compute_cam, make_predict_fn, lime_image, overlay_heatmap,
    draw_regions, overlay_lime, deletion_curve, deletion_auc,
    background_attention_fraction,
)

router = APIRouter()

# 800 matches the training figure; lower it on Render's free tier if requests time out
LIME_SAMPLES = int(os.getenv("LIME_SAMPLES", "400"))
LIME_SEGMENTS = int(os.getenv("LIME_SEGMENTS", "40"))

resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
to_input = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# the explainers hook a shared model layer, so only one request may run at a time
_lock = threading.Lock()


def to_data_url(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


@router.post("/predict")
def predict_endpoint(file: UploadFile = File(...)):   # plain def: FastAPI runs it in a thread
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    resized = resize(image)
    img_rgb = np.array(resized)                        # uint8 RGB, what LIME perturbs
    x = to_input(resized).unsqueeze(0).to(DEVICE)

    result = {"gradcam": "", "pro_gradcam": "", "lime": ""}
    metrics, errors = {}, {}

    with _lock:
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        pred = int(probs.argmax())
        conf = float(probs[pred])
        predict_fn = make_predict_fn(model, DEVICE)

        try:   # 1. Grad-CAM: full continuous map
            cam_g, _, _ = compute_cam(model, x, "gradcam", target_class=pred, size=IMAGE_SIZE)
            result["gradcam"] = to_data_url(overlay_heatmap(img_rgb, cam_g))
        except Exception as e:
            errors["gradcam"] = str(e); traceback.print_exc()

        try:   # 2. Grad-CAM++: heatmap + box (>= 50% of peak), boxes only for hemorrhagic
            cam_pp, _, _ = compute_cam(model, x, "gradcam++", target_class=pred, size=IMAGE_SIZE)
            img_pp = overlay_heatmap(img_rgb, cam_pp)
            if pred == HEM_IDX:
                img_pp, _ = draw_regions(img_pp, cam_pp, thr=0.5)
            result["pro_gradcam"] = to_data_url(img_pp)
            metrics["background_attention"] = background_attention_fraction(cam_pp, img_rgb)
            fr, p = deletion_curve(predict_fn, img_rgb, cam_pp, pred)
            metrics["deletion_auc"] = deletion_auc(fr, p)
        except Exception as e:
            errors["gradcam_pp"] = str(e); traceback.print_exc()

        try:   # 3. Real LIME: superpixels, perturbations, weighted ridge surrogate
            res = lime_image(predict_fn, img_rgb, pred,
                             n_samples=LIME_SAMPLES, n_segments=LIME_SEGMENTS)
            result["lime"] = to_data_url(overlay_lime(img_rgb, res["segments"], res["coef"]))
            metrics["lime_r2"] = res["r2"]
        except Exception as e:
            errors["lime"] = str(e); traceback.print_exc()

    return {
        "prediction": CLASS_NAMES[pred],
        "confidence": f"{conf * 100:.2f}%",
        "hemorrhage_confidence": f"{float(probs[0]) * 100:.2f}%",
        "normal_confidence": f"{float(probs[1]) * 100:.2f}%",
        **result,
        "explanation_metrics": metrics,
        "explanation_errors": errors,
    }