"""
11_generate_paper_charts.py

Generates all requested IEEE paper charts from REAL pipeline data:
  1. Training vs Validation Loss curve (averaged across 5 folds)
  2. Training vs Validation Accuracy curve (averaged across 5 folds)
  3. Confusion Matrix heatmap (regenerated, out-of-fold)
  4. ROC Curve (regenerated, out-of-fold)
  5. Precision-Recall Curve (NEW, out-of-fold)
  6. Class Distribution histogram (train/valid/test, from manifest)
  7. Learning Rate schedule (averaged across 5 folds)

NOTE on "Model Performance Comparison": only ONE architecture
(EfficientNet-B0) was actually trained in this project. Generating a
bar chart comparing it against ResNet/VGG/etc. would require either
(a) actually training those baselines, or (b) using numbers from
someone else's paper - which must be clearly cited as literature
comparison, not presented as a controlled experiment. This script does
NOT fabricate that chart. If you want it, either train baseline
models with the same pipeline, or build the chart manually citing your
sources.

Run from project root, inside the backend's venv:
    python pipeline/11_generate_paper_charts.py
"""

import sys
import csv
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from model.config import CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, DEVICE
from model.model_builder import build_model

FOLDS_DIR = PROJECT_ROOT / "data_processed" / "03_folds"
FOLD_MODELS_DIR = PROJECT_ROOT / "models" / "kfold"
HISTORY_DIR = PROJECT_ROOT / "outputs" / "reports" / "fold_histories"
MANIFEST_PATH = PROJECT_ROOT / "data_processed" / "02_near_dedup" / "manifest.csv"
KFOLD_ASSIGNMENT = PROJECT_ROOT / "outputs" / "reports" / "kfold_assignment.csv"

GRAPHS_DIR = PROJECT_ROOT / "outputs" / "graphs"
CM_DIR = PROJECT_ROOT / "outputs" / "confusion_matrix"
ROC_DIR = PROJECT_ROOT / "outputs" / "roc_curve"
PR_DIR = PROJECT_ROOT / "outputs" / "precision_recall_curve"
for d in (GRAPHS_DIR, CM_DIR, ROC_DIR, PR_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============================================================
# 1 & 2 & 7: Loss / Accuracy / LR curves (averaged across folds)
# ============================================================

def generate_training_curves():
    print("\n[1/7] Loss curve, [2/7] Accuracy curve, [7/7] LR schedule...")

    all_histories = []
    for fold_idx in range(N_FOLDS):
        path = HISTORY_DIR / f"fold_{fold_idx}_history.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found - re-run 06_train_kfold.py (updated version) first.")
            return
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
            all_histories.append(rows)

    max_epochs = max(len(h) for h in all_histories)

    def get_series(key):
        # Average across folds at each epoch, using only folds that reached that epoch
        series = []
        for epoch_idx in range(max_epochs):
            values = [float(h[epoch_idx][key]) for h in all_histories if epoch_idx < len(h)]
            series.append(np.mean(values))
        return series

    epochs = list(range(1, max_epochs + 1))
    train_loss = get_series("train_loss")
    valid_loss = get_series("valid_loss")
    train_acc = get_series("train_accuracy")
    valid_acc = get_series("valid_accuracy")
    lr = get_series("learning_rate")

    # --- Loss curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, valid_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss (mean across 5 folds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "loss_curve.png", dpi=200)
    plt.close()

    # --- Accuracy curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, label="Training Accuracy")
    plt.plot(epochs, valid_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy (mean across 5 folds)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "accuracy_curve.png", dpi=200)
    plt.close()

    # --- LR schedule ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lr)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule (mean across 5 folds, ReduceLROnPlateau)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "lr_schedule.png", dpi=200)
    plt.close()

    print(f"  Saved: {GRAPHS_DIR / 'loss_curve.png'}")
    print(f"  Saved: {GRAPHS_DIR / 'accuracy_curve.png'}")
    print(f"  Saved: {GRAPHS_DIR / 'lr_schedule.png'}")


# ============================================================
# Collect out-of-fold predictions (reused for CM, ROC, PR)
# ============================================================

def build_frozen_model():
    return build_model("efficientnetb0").to(DEVICE)


def collect_out_of_fold_predictions():
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
                all_score.extend(probs[:, 0].cpu().tolist())  # P(Hemorrhagic)

    return all_true, all_pred, all_score, idx_to_class_ref


# ============================================================
# 3: Confusion Matrix
# ============================================================

def generate_confusion_matrix(all_true, all_pred, idx_to_class):
    print("\n[3/7] Confusion matrix...")
    class_order = [idx_to_class[i] for i in range(NUM_CLASSES)]
    cm = confusion_matrix(all_true, all_pred, labels=list(range(NUM_CLASSES)))

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix (out-of-fold predictions)")
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
    print(f"  Saved: {CM_DIR / 'confusion_matrix.png'}")


# ============================================================
# 4: ROC Curve
# ============================================================

def generate_roc_curve(all_true, all_score):
    print("\n[4/7] ROC curve...")
    binary_true = [1 if t == 0 else 0 for t in all_true]  # 1 = Hemorrhagic
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
    print(f"  Saved: {ROC_DIR / 'roc_curve.png'} (AUC={roc_auc:.4f})")


# ============================================================
# 5: Precision-Recall Curve
# ============================================================

def generate_pr_curve(all_true, all_score):
    print("\n[5/7] Precision-Recall curve...")
    binary_true = [1 if t == 0 else 0 for t in all_true]
    precision, recall, _ = precision_recall_curve(binary_true, all_score)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Hemorrhagic (out-of-fold)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PR_DIR / "precision_recall_curve.png", dpi=200)
    plt.close()
    print(f"  Saved: {PR_DIR / 'precision_recall_curve.png'}")


# ============================================================
# 6: Class Distribution Histogram
# ============================================================

def generate_class_distribution():
    print("\n[6/7] Class distribution histogram...")

    if not KFOLD_ASSIGNMENT.exists():
        print(f"  WARNING: {KFOLD_ASSIGNMENT} not found, skipping.")
        return

    counts = defaultdict(lambda: defaultdict(int))  # split -> class -> count
    # Use fold_0 as representative train/valid split for this chart,
    # since folds vary; test set doesn't exist separately in this
    # k-fold design (all data cycles through valid across folds).
    with open(KFOLD_ASSIGNMENT, newline="") as f:
        for row in csv.DictReader(f):
            if row["fold"] == "0":
                counts[row["split"]][row["class"]] += 1

    splits = ["train", "valid"]
    classes = CLASS_NAMES
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, split in enumerate(splits):
        values = [counts[split][cls] for cls in classes]
        ax.bar(x + i * width, values, width, label=split.capitalize())

    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution (Fold 0, representative of all folds)")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(classes)
    ax.legend()
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "class_distribution.png", dpi=200)
    plt.close()
    print(f"  Saved: {GRAPHS_DIR / 'class_distribution.png'}")


def main():
    print("=" * 70)
    print("STEP 11: GENERATE ALL PAPER CHARTS")
    print("=" * 70)

    generate_training_curves()

    all_true, all_pred, all_score, idx_to_class = collect_out_of_fold_predictions()
    generate_confusion_matrix(all_true, all_pred, idx_to_class)
    generate_roc_curve(all_true, all_score)
    generate_pr_curve(all_true, all_score)

    generate_class_distribution()

    print("\n" + "=" * 70)
    print("DONE - all charts saved under outputs/")
    print("=" * 70)
    print("NOT generated: Model Performance Comparison chart (would need")
    print("either training baseline architectures, or citing external")
    print("literature numbers explicitly - see docstring at top of this file).")


if __name__ == "__main__":
    main()