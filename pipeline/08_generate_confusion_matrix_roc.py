"""
08_generate_confusion_matrix_roc.py

Generates a confusion matrix, ROC curve, and per-class metrics table
for the IEEE paper - using OUT-OF-FOLD predictions from the 5 models
saved in step 6, NOT the final deployment model.

Why: the final deployment model (step 7) trained on 100% of the data
with no held-out set, so generating a confusion matrix from it would
just be training-set accuracy dressed up as evaluation - the same
mistake that caused the original 99% problem. Instead, each of the 5
fold models is only run on the validation images IT never trained on
(its held-out patients). Every one of your 257 images gets exactly
one out-of-fold prediction, combined into one honest confusion matrix
covering the whole dataset with zero leakage.

Output:
    outputs/confusion_matrix/confusion_matrix.png
    outputs/roc_curve/roc_curve.png
    outputs/tables/classification_report.csv   (paper-ready table)
    outputs/tables/per_fold_metrics.csv

Run from project root, inside the backend's venv:
    python pipeline/08_generate_confusion_matrix_roc.py
"""

import sys
import csv
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from model.config import CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, DEVICE
from model.model_builder import build_model

FOLDS_DIR = PROJECT_ROOT / "data_processed" / "03_folds"
FOLD_MODELS_DIR = PROJECT_ROOT / "models" / "kfold"
N_FOLDS = 5

CM_DIR = PROJECT_ROOT / "outputs" / "confusion_matrix"
ROC_DIR = PROJECT_ROOT / "outputs" / "roc_curve"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
for d in (CM_DIR, ROC_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_frozen_model():
    model = build_model("efficientnetb0")
    return model.to(DEVICE)


def main():
    print("=" * 70)
    print("STEP 8: HONEST CONFUSION MATRIX + ROC (out-of-fold predictions)")
    print("=" * 70)

    all_true, all_pred, all_score = [], [], []
    idx_to_class_ref = None

    for fold_idx in range(N_FOLDS):
        fold_dir = FOLDS_DIR / f"fold_{fold_idx}"
        valid_dataset = datasets.ImageFolder(fold_dir / "valid", transform=eval_transform)
        idx_to_class_ref = {v: k for k, v in valid_dataset.class_to_idx.items()}
        valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=0)

        model = build_frozen_model()
        state = torch.load(FOLD_MODELS_DIR / f"fold_{fold_idx}_best.pth", map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(DEVICE)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)

                all_true.extend(labels.tolist())
                all_pred.extend(preds.cpu().tolist())
                # score for ROC = probability of the positive class (index 0 = Hemorrhagic)
                all_score.extend(probs[:, 0].cpu().tolist())

        print(f"  Fold {fold_idx}: {len(valid_dataset)} out-of-fold predictions collected")

    print(f"\nTotal out-of-fold predictions: {len(all_true)} (should equal total dataset size)")

    # ---- Confusion Matrix ----
    class_order = [idx_to_class_ref[i] for i in range(NUM_CLASSES)]
    cm = confusion_matrix(all_true, all_pred, labels=list(range(NUM_CLASSES)))

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (out-of-fold, all 5 folds combined)")
    plt.colorbar()
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, class_order)
    plt.yticks(tick_marks, class_order)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(CM_DIR / "confusion_matrix.png", dpi=200)
    plt.close()
    print(f"Confusion matrix saved: {CM_DIR / 'confusion_matrix.png'}")

    # ---- ROC Curve ----
    # Positive class = index 0 (Hemorrhagic) - true label 0 means Hemorrhagic
    binary_true = [1 if t == 0 else 0 for t in all_true]  # 1 = Hemorrhagic (positive)
    fpr, tpr, _ = roc_curve(binary_true, all_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Hemorrhagic vs NonHemorrhagic (out-of-fold)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ROC_DIR / "roc_curve.png", dpi=200)
    plt.close()
    print(f"ROC curve saved: {ROC_DIR / 'roc_curve.png'} (AUC = {roc_auc:.4f})")

    # ---- Classification report table (paper-ready) ----
    report = classification_report(
        all_true, all_pred, target_names=class_order, output_dict=True, zero_division=0
    )

    report_csv_path = TABLES_DIR / "classification_report.csv"
    with open(report_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for cls in class_order:
            r = report[cls]
            writer.writerow([cls, round(r["precision"], 4), round(r["recall"], 4),
                              round(r["f1-score"], 4), int(r["support"])])
        writer.writerow(["accuracy", "", "", round(report["accuracy"], 4), len(all_true)])
        for avg_key in ["macro avg", "weighted avg"]:
            r = report[avg_key]
            writer.writerow([avg_key, round(r["precision"], 4), round(r["recall"], 4),
                              round(r["f1-score"], 4), int(r["support"])])

    print(f"Classification report table saved: {report_csv_path}")

    print("\n" + "=" * 70)
    print("SUMMARY (this matches / should closely match step 6's aggregated result)")
    print("=" * 70)
    print(f"Overall accuracy: {report['accuracy']*100:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")
    for cls in class_order:
        r = report[cls]
        print(f"  {cls:15} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}")

    print("\nNext: run 09_generate_explanations.py for Grad-CAM/Grad-CAM++/LIME on test images")


if __name__ == "__main__":
    main()