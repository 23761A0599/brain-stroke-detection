"""
09_generate_explanations_v3.py

Paper figure generator with GENUINE explanations (see backend/services/explainers.py):
    [ Input | Grad-CAM | Grad-CAM++ (+ regions) | LIME ]
for SAMPLES_PER_CLASS images per class, using the FINAL deployment model.

Figures contain no file names (a patient name may be part of a file name). The real
file names are only written to outputs/explanations_v3/explanations_index.csv - keep it private.
The index also stores the LIME surrogate R^2 and the share of Grad-CAM energy that lies on the
black background (a shortcut-learning sanity check).

Note for the caption: the final model was trained on all 257 images, so these are training
images. The explanations show what the model uses, not how well it generalises.

Run from project root, inside the backend venv:
    python pipeline/09_generate_explanations_v3.py
    python pipeline/09_generate_explanations_v3.py --per-class 6 --seed 1
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model.config import DEVICE, CLASS_NAMES, MODELS_DIR, IMAGE_SIZE   # noqa: E402
from model.model_builder import build_model                              # noqa: E402
from services import explainers as E                                     # noqa: E402

ALL_DATA_DIR = PROJECT_ROOT / "data_processed" / "02_near_dedup"
MODEL_PATH = MODELS_DIR / "EfficientNetB0" / "best_efficientnetb0.pth"
OUT_DIR = PROJECT_ROOT / "outputs" / "explanations_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def to_tensor(img_rgb):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return ((t - mean) / std).unsqueeze(0).to(DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lime-samples", type=int, default=1000)
    args = ap.parse_args()

    model = build_model("efficientnetb0", pretrained=False, mode="full")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    predict = E.make_predict_fn(model, DEVICE)
    random.seed(args.seed)
    rows = []

    for cls in CLASS_NAMES:
        files = sorted(p for p in (ALL_DATA_DIR / cls).iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        for k, path in enumerate(random.sample(files, min(args.per_class, len(files))), start=1):
            img = np.array(Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
            x = to_tensor(img)

            cam_g, pred, probs = E.compute_cam(model, x, "gradcam")
            cam_pp, _, _ = E.compute_cam(model, x, "gradcam++")
            lime = E.lime_image(predict, img, pred, n_samples=args.lime_samples, seed=0)
            conf = 100 * probs[pred]

            p_grad = E.overlay_heatmap(img, cam_g)
            p_pp = E.overlay_heatmap(img, cam_pp)
            boxed = None
            if pred == E.HEM_IDX:                     # regions are only drawn for a hemorrhage prediction
                p_pp, boxes = E.draw_regions(p_pp, cam_pp, thr=0.5)
            p_lime = E.overlay_lime(img, lime["segments"], lime["coef"], top_k=5)

            fig, ax = plt.subplots(1, 4, figsize=(12.5, 3.6))
            for a, im, t in zip(ax, [img, p_grad, p_pp, p_lime],
                                ["Input MRI", "Grad-CAM", "Grad-CAM++ (regions >= 50% of peak)" if pred == 0 else "Grad-CAM++",
                                 f"LIME (surrogate R$^2$={lime['r2']:.2f})"]):
                a.imshow(im); a.set_title(t, fontsize=9); a.axis("off")
            fig.suptitle(f"True: {cls}   |   Predicted: {CLASS_NAMES[pred]} ({conf:.0f}%)   |   "
                         f"explained class: {CLASS_NAMES[pred]}", fontsize=10)
            plt.tight_layout()
            out = OUT_DIR / f"{cls}_{k}.png"
            plt.savefig(out, dpi=200); plt.close()

            rows.append({"figure": out.name, "true_class": cls, "pred_class": CLASS_NAMES[pred],
                         "confidence_pct": round(conf, 1),
                         "lime_r2": round(lime["r2"], 3),
                         "gradcam_bg_fraction": round(E.background_attention_fraction(cam_g, img), 3),
                         "gradcampp_bg_fraction": round(E.background_attention_fraction(cam_pp, img), 3),
                         "source_file": path.name})
            print(f"{cls:15} {k}: pred={CLASS_NAMES[pred]:15} conf={conf:5.1f}%  "
                  f"LIME R2={lime['r2']:.2f}  bg-attention={rows[-1]['gradcam_bg_fraction']:.2f}  -> {out.name}")

    with open(OUT_DIR / "explanations_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()