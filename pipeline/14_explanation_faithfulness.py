"""
14_explanation_faithfulness.py

Quantitative check that the explanations are faithful to the model (deletion test).
For every image the most "important" pixels (per method) are removed first and the probability
of the predicted class is tracked while 0 -> 50% of the pixels are deleted. A LOWER area under
that curve means the map really marks what the model relies on. A random ranking is the baseline.

Methods: Grad-CAM, Grad-CAM++, LIME (superpixel weights), random.
Faithfulness describes the MODEL, not generalisation, so using the final model on all 257
images is legitimate.

Outputs
    outputs/reports_pretrained/explanation_faithfulness.csv     (per image; contains file names -> private)
    printed summary (use these numbers in the paper)

Run from project root, inside the backend venv (about 5-10 min on a GPU):
    python pipeline/14_explanation_faithfulness.py
    python pipeline/14_explanation_faithfulness.py --lime-samples 500 --limit 40    # quick test
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model.config import DEVICE, CLASS_NAMES, MODELS_DIR, IMAGE_SIZE   # noqa: E402
from model.model_builder import build_model                              # noqa: E402
from services import explainers as E                                     # noqa: E402

ALL_DATA_DIR = PROJECT_ROOT / "data_processed" / "02_near_dedup"
MODEL_PATH = MODELS_DIR / "EfficientNetB0" / "best_efficientnetb0.pth"
OUT_CSV = PROJECT_ROOT / "outputs" / "reports_pretrained" / "explanation_faithfulness.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def to_tensor(img_rgb):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    return ((t - mean) / std).unsqueeze(0).to(DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lime-samples", type=int, default=600)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images per class (test run)")
    args = ap.parse_args()

    model = build_model("efficientnetb0", pretrained=False, mode="full")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    predict = E.make_predict_fn(model, DEVICE)
    rng = np.random.default_rng(0)

    rows = []
    for cls in CLASS_NAMES:
        files = sorted(p for p in (ALL_DATA_DIR / cls).iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if args.limit:
            files = files[:args.limit]
        for i, path in enumerate(files, 1):
            img = np.array(Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))
            x = to_tensor(img)
            cam_g, pred, probs = E.compute_cam(model, x, "gradcam")
            cam_pp, _, _ = E.compute_cam(model, x, "gradcam++")
            lime = E.lime_image(predict, img, pred, n_samples=args.lime_samples, seed=0)
            maps = {"gradcam": cam_g, "gradcampp": cam_pp,
                    "lime": E.lime_importance_map(lime["segments"], lime["coef"])}
            row = {"true_class": cls, "pred_class": CLASS_NAMES[pred], "p_pred": round(float(probs[pred]), 4),
                   "lime_r2": round(lime["r2"], 3), "source_file": path.name}
            for name, m in maps.items():
                f, p = E.deletion_curve(predict, img, m, pred)
                row[f"auc_{name}"] = round(E.deletion_auc(f, p), 4)
            rand_aucs = []
            for _ in range(3):
                f, p = E.deletion_curve(predict, img, rng.random((IMAGE_SIZE, IMAGE_SIZE)), pred)
                rand_aucs.append(E.deletion_auc(f, p))
            row["auc_random"] = round(float(np.mean(rand_aucs)), 4)
            rows.append(row)
            if i % 20 == 0:
                print(f"  {cls}: {i}/{len(files)} done")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 78)
    print(f"DELETION TEST  (mean predicted-class probability while deleting 0-50% of pixels; LOWER = more faithful)")
    print(f"images: {len(rows)}")
    print("=" * 78)
    rnd = np.array([r["auc_random"] for r in rows])
    for name in ("gradcam", "gradcampp", "lime", "random"):
        a = np.array([r[f"auc_{name}"] for r in rows])
        extra = "" if name == "random" else f"   better than random on {100 * np.mean(a < rnd):.0f}% of images"
        print(f"  {name:10} AUC = {a.mean():.3f} +/- {a.std():.3f}{extra}")
    r2 = np.array([r["lime_r2"] for r in rows])
    print(f"\n  LIME surrogate R^2: mean {r2.mean():.2f}, median {np.median(r2):.2f}, share with R^2 < 0.5: {100*np.mean(r2<0.5):.0f}%")
    print(f"\nPer-image results (private, contains file names): {OUT_CSV}")


if __name__ == "__main__":
    main()