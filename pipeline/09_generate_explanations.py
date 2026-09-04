"""
09_generate_explanations.py

Batch-generates Grad-CAM, Grad-CAM++, and LIME visualizations for a
representative SAMPLE of images from each class (not all 257 - that
would be excessive for paper figures), using the FINAL deployment
model (models/EfficientNetB0/best_efficientnetb0.pth).

Output:
    outputs/gradcam/sample_<class>_<filename>.jpg
    outputs/gradcam_pro/sample_<class>_<filename>.jpg
    outputs/lime/sample_<class>_<filename>.png

Run from project root, inside the backend's venv:
    python pipeline/09_generate_explanations.py
"""

import sys
import random
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from model.config import DEVICE, CLASS_NAMES, MODELS_DIR
from model.model_builder import build_model
from services.gradcam import save_gradcam
from services.gradcam_pro import save_gradcam_pro
from services.lime_explainer import save_lime_explanation

ALL_DATA_DIR = PROJECT_ROOT / "data_processed" / "02_near_dedup"
FINAL_MODEL_PATH = MODELS_DIR / "EfficientNetB0" / "best_efficientnetb0.pth"

SAMPLES_PER_CLASS = 3  # how many example images per class to generate figures for
RANDOM_SEED = 42


def load_final_model():
    model = build_model("efficientnetb0")
    state = torch.load(FINAL_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def main():
    print("=" * 70)
    print("STEP 9: GENERATE EXPLAINABILITY FIGURES (Grad-CAM, Grad-CAM++, LIME)")
    print("=" * 70)

    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found at {FINAL_MODEL_PATH}. Run 07_train_final_model.py first."
        )

    model = load_final_model()
    random.seed(RANDOM_SEED)

    for cls in CLASS_NAMES:
        cls_dir = ALL_DATA_DIR / cls
        images = sorted(cls_dir.iterdir())
        sample = random.sample(images, min(SAMPLES_PER_CLASS, len(images)))

        print(f"\n{cls}: generating explanations for {len(sample)} sample images")

        for img_path in sample:
            print(f"  {img_path.name}")

            gradcam_result = save_gradcam(model, str(img_path))
            print(f"    Grad-CAM      -> {gradcam_result['gradcam_path']} "
                  f"(pred: {gradcam_result['prediction']})")

            gradcam_pro_result = save_gradcam_pro(model, str(img_path))
            print(f"    Grad-CAM++    -> {gradcam_pro_result['gradcam_pro_path']} "
                  f"(pred: {gradcam_pro_result['prediction']}, "
                  f"conf: {gradcam_pro_result['confidence']}%)")

            lime_result = save_lime_explanation(model, str(img_path))
            print(f"    LIME          -> {lime_result['lime_path']} "
                  f"(pred: {lime_result['prediction']})")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print("Note: gradcam.py and gradcam_pro.py currently save to a FIXED filename")
    print("(gradcam_result.jpg / gradcam_professional.jpg), overwriting each run.")
    print("If you want a separate file per sample image (recommended for picking")
    print("figures later), apply the same per-image-filename fix used in")
    print("lime_explainer.py to gradcam.py and gradcam_pro.py, then re-run this step.")
    print("\nNext: run 10_generate_final_summary.py")


if __name__ == "__main__":
    main()