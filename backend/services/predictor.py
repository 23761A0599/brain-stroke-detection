"""
Corrected /predict route, version 2 (robust for deployment).

Differences from version 1
  * The prediction is computed FIRST and always returned; each explanation is computed separately, so one
    failing method (missing package, out-of-memory, timeout) no longer hides the prediction or the other
    explanations. A failed method returns null plus a message in "explanation_errors".
  * LIME is light by default (150 perturbations) so it fits a small CPU server; raise lime_samples on a GPU/large plan.
  * Returns "original" (the 224x224 input) so the front end can always show the MRI next to the overlays.
  * GET /explain-check lists which packages the server can import and which model file it loaded:
    open it in the browser on the deployed site to see why explanations are missing.

Class logic: the class is taken only from the class index (never from a substring of the label).

Version 3: the response now contains every key the React component (App.jsx) reads:
  prediction, confidence (PERCENT, 0-100), hemorrhage_confidence, normal_confidence (percent),
  gradcam, pro_gradcam (+ alias gradcam_pro), lime.
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
from services import predictor          # loaded EfficientNet-B0 (eval mode)

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
    """First 12 hex chars of the SHA-256 of the checkpoint the server loaded (compare with Get-FileHash locally)."""
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
        except Exception as e:                 # pragma: no cover
            info[name] = f"MISSING: {e}"
    return info


@router.post("/predict")
def predict(file: UploadFile = File(...), lime_samples: int = LIME_SAMPLES_DEFAULT):
    # plain `def`: FastAPI runs it in a worker thread, so slow explanations do not block other requests
    t0 = time.time()
    data = np.frombuffer(file.file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")
    try:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = np.array(Image.fromarray(rgb).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
        x = _to_tensor(img)
        predictor.model.eval()
        with torch.no_grad():
            probs = torch.softmax(predictor.model(x), dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())
        conf = float(probs[pred])
        result = {
            "prediction": CLASS_NAMES[pred],
            "confidence": round(conf * 100, 2),                    # percent, as App.jsx expects
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
    cam_pp = None
    try:
        cam_g, _, _ = E.compute_cam(predictor.model, x, "gradcam")
        result["gradcam"] = _data_uri(E.overlay_heatmap(img, cam_g))
    except Exception as e:
        errors["gradcam"] = repr(e)
    try:
        cam_pp, _, _ = E.compute_cam(predictor.model, x, "gradcam++")
        over = E.overlay_heatmap(img, cam_pp)
        if pred == HEM_IDX:                          # region boxes only for a hemorrhage prediction
            over, _ = E.draw_regions(over, cam_pp, thr=0.5)
        result["pro_gradcam"] = result["gradcam_pro"] = _data_uri(over)   # Grad-CAM++ (key names used by App.jsx and by the API)
    except Exception as e:
        errors["gradcam_pro"] = repr(e)
    try:
        n = max(50, min(int(lime_samples), 1000))
        lime = E.lime_image(E.make_predict_fn(predictor.model, DEVICE, batch=16), img, pred, n_samples=n)
        result["lime"] = _data_uri(E.overlay_lime(img, lime["segments"], lime["coef"], top_k=5))
        result["lime_r2"] = round(float(lime["r2"]), 3)
    except Exception as e:
        errors["lime"] = repr(e)
    result["explanation_errors"] = errors
    result["explanations_ok"] = not errors
    result["seconds"] = round(time.time() - t0, 1)
    return result