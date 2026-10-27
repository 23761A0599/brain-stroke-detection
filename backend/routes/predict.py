"""
Corrected /predict route, version 2 (robust for deployment).
"""
import base64
import time

import cv2
import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

import hashlib
from pathlib import Path

from model.config import CLASS_NAMES, DEVICE, IMAGE_SIZE, MODELS_DIR
from services import explainers as E
from services import predictor # Imported safely as a module to prevent circular loops

router = APIRouter()
HEM_IDX = CLASS_NAMES.index("Hemorrhagic")
NON_IDX = CLASS_NAMES.index("NonHemorrhagic")
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
LIME_SAMPLES_DEFAULT = 150


def _to_tensor(img_rgb):
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return ((t - _MEAN) / _STD).unsqueeze(0).to(DEVICE)


def _data_uri(img_rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def _model_fingerprint():
    p = Path(MODELS_DIR) / "EfficientNetB0" / "best_efficientnetb0.pth"
    try:
        return {"path": str(p.name), "size_bytes": p.stat().st_size, "sha256_12": hashlib.sha256(p.read_bytes()).hexdigest()[:12]}
    except Exception as e:
        return {"error": repr(e)}


@router.get("/explain-check")
def explain_check():
    info = {"device": str(DEVICE), "torch": torch.__version__, "classes": CLASS_NAMES, "checkpoint": _model_fingerprint()}
    for name in ("cv2", "sklearn", "skimage", "numpy", "PIL"):
        try:
            m = __import__(name)
            info[name] = getattr(m, "__version__", "ok")
        except Exception as e:
            info[name] = f"MISSING: {e}"
    return info


@router.post("/predict")
def predict(file: UploadFile = File(...), lime_samples: int = LIME_SAMPLES_DEFAULT):
    t0 = time.time()
    data = np.frombuffer(file.file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = np.array(Image.fromarray(rgb).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
        x = _to_tensor(img)
        
        # Access model safely from the predictor module
        active_model = predictor.model
        active_model.eval()
        
        with torch.no_grad():
            probs = torch.softmax(active_model(x), dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())
        conf = float(probs[pred])
        result = {
            "prediction": CLASS_NAMES[pred],
            "confidence": round(conf * 100, 2),
            "confidence_percentage": f"{conf * 100:.2f}%",
            "hemorrhage_confidence": round(float(probs[HEM_IDX]) * 100, 2),
            "normal_confidence": round(float(probs[NON_IDX]) * 100, 2),
            "class_probabilities": {c: round(float(p) * 100, 2) for c, p in zip(CLASS_NAMES, probs)},
            "is_hemorrhagic": bool(pred == HEM_IDX),
            "explained_class": CLASS_NAMES[pred],
            "original": _data_uri(img),
            "gradcam": None, "pro_gradcam": None, "gradcam_pro": None, "lime": None, "lime_r2": None,
            "notice": "Research prototype - not a medical device. Explanations show what the model used, not clinical findings.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction failed: {e}")

    errors = {}
    try:
        cam_g, _, _ = E.compute_cam(active_model, x, "gradcam")
        result["gradcam"] = _data_uri(E.overlay_heatmap(img, cam_g))
    except Exception as e:
        errors["gradcam"] = repr(e)
    try:
        cam_pp, _, _ = E.compute_cam(active_model, x, "gradcam++")
        over = E.overlay_heatmap(img, cam_pp)
        if pred == HEM_IDX:
            over, _ = E.draw_regions(over, cam_pp, thr=0.5)
        result["pro_gradcam"] = result["gradcam_pro"] = _data_uri(over)
    except Exception as e:
        errors["gradcam_pro"] = repr(e)
    try:
        n = max(50, min(int(lime_samples), 1000))
        lime = E.lime_image(E.make_predict_fn(active_model, DEVICE, batch=16), img, pred, n_samples=n)
        result["lime"] = _data_uri(E.overlay_lime(img, lime["segments"], lime["coef"], top_k=5))
        result["lime_r2"] = round(float(lime["r2"]), 3)
    except Exception as e:
        errors["lime"] = repr(e)
    result["explanation_errors"] = errors
    result["explanations_ok"] = not errors
    result["seconds"] = round(time.time() - t0, 1)
    return result