"""
06_train_kfold.py  (updated: now logs per-epoch history per fold)

Same k-fold training as before, but now saves per-epoch train/valid
loss, accuracy, and learning rate to CSV for each fold - needed to
plot honest loss/accuracy/LR-schedule curves.

Run from project root, inside the backend's venv:
    python pipeline/06_train_kfold.py
"""

import sys
import time
import copy
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from model.config import CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, DEVICE
from model.model_builder import build_model

FOLDS_DIR = PROJECT_ROOT / "data_processed" / "03_folds"
MODELS_OUT_DIR = PROJECT_ROOT / "models" / "kfold"
MODELS_OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_DIR = PROJECT_ROOT / "outputs" / "reports" / "fold_histories"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 8
MIN_DELTA = 0.001


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_frozen_model():
    model = build_model("efficientnetb0")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model.to(DEVICE)


def build_dataloaders(fold_dir):
    train_dataset = datasets.ImageFolder(fold_dir / "train", transform=train_transform)
    valid_dataset = datasets.ImageFolder(fold_dir / "valid", transform=eval_transform)

    assert train_dataset.class_to_idx == valid_dataset.class_to_idx

    targets = [label for _, label in train_dataset.samples]
    class_counts = torch.bincount(torch.tensor(targets), minlength=NUM_CLASSES)
    class_weights = 1.0 / class_counts.float().clamp(min=1)
    sample_weights = [class_weights[t] for t in targets]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, valid_loader, train_dataset.class_to_idx


def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    running_loss, total = 0.0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    macro_f1 = f1.mean()

    return {
        "loss": running_loss / total, "accuracy": accuracy, "macro_f1": macro_f1,
        "precision": precision, "recall": recall, "f1": f1, "support": support,
    }


def train_one_fold(fold_idx):
    fold_dir = FOLDS_DIR / f"fold_{fold_idx}"
    print("\n" + "=" * 70)
    print(f"FOLD {fold_idx}")
    print("=" * 70)

    train_loader, valid_loader, class_to_idx = build_dataloaders(fold_dir)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_frozen_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_macro_f1 = 0.0
    best_state = None
    best_metrics = None
    epochs_without_improvement = 0

    epoch_history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        valid_metrics = evaluate(model, valid_loader, criterion)

        current_lr_before_step = optimizer.param_groups[0]["lr"]
        scheduler.step(valid_metrics["macro_f1"])

        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_accuracy * 100, 2),
            "valid_loss": round(valid_metrics["loss"], 4),
            "valid_accuracy": round(valid_metrics["accuracy"] * 100, 2),
            "valid_macro_f1": round(valid_metrics["macro_f1"], 4),
            "learning_rate": current_lr_before_step,
        })

        if valid_metrics["macro_f1"] > best_macro_f1 + MIN_DELTA:
            best_macro_f1 = valid_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = valid_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0 or epochs_without_improvement == 0:
            print(f"  Epoch {epoch+1:3}/{EPOCHS} | valid loss={valid_metrics['loss']:.4f} "
                  f"acc={valid_metrics['accuracy']*100:.1f}% macro-F1={valid_metrics['macro_f1']:.4f} "
                  f"{'<- best' if epochs_without_improvement == 0 else ''}")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    torch.save(best_state, MODELS_OUT_DIR / f"fold_{fold_idx}_best.pth")

    history_csv_path = HISTORY_DIR / f"fold_{fold_idx}_history.csv"
    with open(history_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_history[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_history)

    print(f"\n  Fold {fold_idx} BEST: accuracy={best_metrics['accuracy']*100:.2f}% "
          f"macro-F1={best_metrics['macro_f1']:.4f}")
    for i in range(NUM_CLASSES):
        cls_name = idx_to_class[i]
        print(f"    {cls_name:15} P={best_metrics['precision'][i]:.3f} "
              f"R={best_metrics['recall'][i]:.3f} F1={best_metrics['f1'][i]:.3f} "
              f"(n={best_metrics['support'][i]})")

    return best_metrics, idx_to_class, epoch_history


def main():
    print("=" * 70)
    print("STEP 6: K-FOLD TRAINING (classifier head only, backbone frozen)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASS_NAMES}")

    fold_results = []
    idx_to_class_ref = None

    start_time = time.time()

    for fold_idx in range(N_FOLDS):
        metrics, idx_to_class, epoch_history = train_one_fold(fold_idx)
        fold_results.append(metrics)
        idx_to_class_ref = idx_to_class

    total_time = time.time() - start_time

    accuracies = [m["accuracy"] for m in fold_results]
    macro_f1s = [m["macro_f1"] for m in fold_results]
    per_class_precision = np.array([m["precision"] for m in fold_results])
    per_class_recall = np.array([m["recall"] for m in fold_results])
    per_class_f1 = np.array([m["f1"] for m in fold_results])

    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS ACROSS ALL 5 FOLDS (report THIS in the paper)")
    print("=" * 70)
    print(f"Accuracy : {np.mean(accuracies)*100:.2f}% +/- {np.std(accuracies)*100:.2f}%")
    print(f"Macro-F1 : {np.mean(macro_f1s):.4f} +/- {np.std(macro_f1s):.4f}")

    summary_rows = []
    for i in range(NUM_CLASSES):
        cls_name = idx_to_class_ref[i]
        p_mean, p_std = per_class_precision[:, i].mean(), per_class_precision[:, i].std()
        r_mean, r_std = per_class_recall[:, i].mean(), per_class_recall[:, i].std()
        f_mean, f_std = per_class_f1[:, i].mean(), per_class_f1[:, i].std()
        print(f"  {cls_name:15} P={p_mean:.3f}+/-{p_std:.3f} "
              f"R={r_mean:.3f}+/-{r_std:.3f} F1={f_mean:.3f}+/-{f_std:.3f}")
        summary_rows.append({
            "class": cls_name, "precision_mean": p_mean, "precision_std": p_std,
            "recall_mean": r_mean, "recall_std": r_std, "f1_mean": f_mean, "f1_std": f_std,
        })

    with open(REPORT_DIR / "kfold_per_class_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(REPORT_DIR / "kfold_per_fold_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "accuracy", "macro_f1"])
        for i, m in enumerate(fold_results):
            writer.writerow([i, m["accuracy"], m["macro_f1"]])

    print(f"\nTraining time: {total_time/60:.2f} minutes")
    print(f"Per-fold models saved to: {MODELS_OUT_DIR}")
    print(f"Per-fold epoch histories saved to: {HISTORY_DIR}")
    print(f"Summary CSVs saved to: {REPORT_DIR}")
    print("\nNext: run 11_generate_paper_charts.py for all paper figures")


if __name__ == "__main__":
    main()