"""
09_generate_explanations_v2.py

Grad-CAM example figures for the paper, using the FINAL deployment model
(models/EfficientNetB0/best_efficientnetb0.pth) and the function that actually
exists in backend/services/gradcam.py (generate_gradcam_map).

For each class it draws SAMPLES_PER_CLASS images (fixed seed) and saves a
2-panel figure (original | Grad-CAM overlay) per image. The title shows only
an anonymous index, the true class, the prediction and the confidence, so no
file name / patient name appears in a figure. The real file names are written
to outputs/gradcam_v2/explanations_index.csv (keep that file private).

NOTE for the paper: the final model was trained on all 257 images, so these
examples are training images. Say so in the Fig. 8 caption.

Run from project root, inside the backend venv:
    python pipeline/09_generate_explanations_v2.py
"""
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model.config import DEVICE, CLASS_NAMES, MODELS_DIR, IMAGE_SIZE  # noqa: E402
from model.model_builder import build_model                            # noqa: E402
from services.gradcam import generate_gradcam_map                      # noqa: E402

ALL_DATA_DIR = PROJECT_ROOT / "data_processed" / "02_near_dedup"
FINAL_MODEL_PATH = MODELS_DIR / "EfficientNetB0" / "best_efficientnetb0.pth"
OUT_DIR = PROJECT_ROOT / "outputs" / "gradcam_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_CLASS = 4
RANDOM_SEED = 42

prep = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_final_model():
    # mode="full": every parameter keeps requires_grad=True. Grad-CAM needs gradients
    # to flow back to the last feature block, which a frozen backbone would block.
    model = build_model("efficientnetb0", pretrained=False, mode="full")
    model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()


def main():
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Final model not found: {FINAL_MODEL_PATH} (run step 7 first)")
    model = load_final_model()
    random.seed(RANDOM_SEED)
    index_rows = []

    for cls in CLASS_NAMES:
        files = sorted(p for p in (ALL_DATA_DIR / cls).iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        for k, path in enumerate(random.sample(files, min(SAMPLES_PER_CLASS, len(files))), start=1):
            img = Image.open(path).convert("RGB")
            tensor = prep(img).unsqueeze(0).to(DEVICE)

            cam, pred_idx = generate_gradcam_map(model, tensor)
            with torch.no_grad():
                prob = torch.softmax(model(tensor), dim=1)[0]
            conf = 100 * prob[pred_idx].item()

            base = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))
            heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            overlay = np.uint8(0.55 * base + 0.45 * heat)

            fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
            ax[0].imshow(base); ax[0].set_title("Input MRI"); ax[0].axis("off")
            ax[1].imshow(overlay); ax[1].axis("off")
            ax[1].set_title(f"Grad-CAM: {CLASS_NAMES[pred_idx]} ({conf:.0f}%)")
            fig.suptitle(f"True class: {cls}", fontsize=10)
            plt.tight_layout()
            out = OUT_DIR / f"{cls}_{k}.png"
            plt.savefig(out, dpi=200); plt.close()

            index_rows.append({"figure": out.name, "true_class": cls,
                               "pred_class": CLASS_NAMES[pred_idx], "confidence_pct": round(conf, 1),
                               "source_file": path.name})
            print(f"{cls:15} {k}: pred={CLASS_NAMES[pred_idx]:15} conf={conf:5.1f}%  -> {out.name}")

    with open(OUT_DIR / "explanations_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        w.writeheader(); w.writerows(index_rows)
    print(f"\nFigures and index saved in {OUT_DIR}")


if __name__ == "__main__":
    main()