"""
Corrected /predict route (replaces the old route file that calls gradcam_pro.generate_gradcam_images).

What was wrong with the old route
  * is_hemorrhagic = "hemorrhag" in label.lower() or idx == 1  ->  "nonhemorrhagic" contains "hemorrhag",
    so EVERY prediction was reported as hemorrhagic and received a "Suspected Hemorrhage" overlay.
  * it imported `transform` from services.gradcam, which that file does not define.
  * it read img_paths["lime"], which generate_gradcam_images() never returns (KeyError -> HTTP 500).
  * the "LIME" overlay was a re-coloured Grad-CAM contour, not LIME.

This version uses services/explainers.py (genuine Grad-CAM, Grad-CAM++ and LIME) and decides the class ONLY
from the class index. Explanations are returned for the predicted class; boxes are drawn only for a
hemorrhagic prediction. The response keeps the keys the React front end already uses.
"""
import base64

import cv2
import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from model.config import CLASS_NAMES, DEVICE, IMAGE_SIZE
from services import explainers as E
from services.predictor import model          # loaded EfficientNet-B0 (eval mode)

router = APIRouter()
HEM_IDX = CLASS_NAMES.index("Hemorrhagic")
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return ((t - _MEAN) / _STD).unsqueeze(0).to(DEVICE)


def _data_uri(img_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


@router.post("/predict")
def predict(file: UploadFile = File(...), lime_samples: int = 500):
    # plain `def`: FastAPI runs it in a worker thread, so the heavy LIME step does not block the event loop
    try:
        data = np.frombuffer(file.file.read(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = np.array(Image.fromarray(rgb).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
        x = _to_tensor(img)

        cam_g, pred, probs = E.compute_cam(model, x, "gradcam")
        cam_pp, _, _ = E.compute_cam(model, x, "gradcam++")
        lime = E.lime_image(E.make_predict_fn(model, DEVICE), img, pred, n_samples=max(100, min(lime_samples, 1500)))

        over_g = E.overlay_heatmap(img, cam_g)
        over_pp = E.overlay_heatmap(img, cam_pp)
        if pred == HEM_IDX:                                  # region boxes only for a hemorrhage prediction
            over_pp, _ = E.draw_regions(over_pp, cam_pp, thr=0.5)
        over_lime = E.overlay_lime(img, lime["segments"], lime["coef"], top_k=5)

        confidence = float(probs[pred])
        return {
            "prediction": CLASS_NAMES[pred],
            "confidence": confidence,
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "is_hemorrhagic": bool(pred == HEM_IDX),
            "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
            "explained_class": CLASS_NAMES[pred],
            "gradcam": _data_uri(over_g),
            "gradcam_pro": _data_uri(over_pp),               # Grad-CAM++ (true second/third-order weights)
            "lime": _data_uri(over_lime),
            "lime_r2": round(float(lime["r2"]), 3),          # local fidelity of the LIME surrogate
            "notice": "Research prototype - not a medical device. Explanations show what the model used, not clinical findings.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))