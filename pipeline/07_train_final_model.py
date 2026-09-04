"""
07_train_final_model.py  (updated: now saves training history to CSV)

Trains the FINAL deployment model using ALL 257 images (no held-out
fold). Saves the model to the exact path the backend expects, AND
saves per-epoch training history to a CSV table for the paper.

Run from project root, inside the backend's venv:
    python pipeline/07_train_final_model.py
"""

import sys
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from model.config import CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, DEVICE, MODELS_DIR
from model.model_builder import build_model

ALL_DATA_DIR = PROJECT_ROOT / "data_processed" / "02_near_dedup"

FINAL_MODEL_DIR = MODELS_DIR / "EfficientNetB0"
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_PATH = FINAL_MODEL_DIR / "best_efficientnetb0.pth"

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_CSV = REPORT_DIR / "final_model_training_history.csv"

FINAL_EPOCHS = 15
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.05


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
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


def main():
    print("=" * 70)
    print("STEP 7: TRAIN FINAL DEPLOYMENT MODEL (all data, no held-out fold)")
    print("=" * 70)
    print(f"Training data: {ALL_DATA_DIR}")
    print(f"Deployment path: {FINAL_MODEL_PATH}")
    print(f"Epochs: {FINAL_EPOCHS}\n")

    train_dataset = datasets.ImageFolder(ALL_DATA_DIR, transform=train_transform)
    print(f"Class mapping: {train_dataset.class_to_idx}")
    print(f"Total training images: {len(train_dataset)}")

    targets = [label for _, label in train_dataset.samples]
    class_counts = torch.bincount(torch.tensor(targets), minlength=NUM_CLASSES)
    class_weights = 1.0 / class_counts.float().clamp(min=1)
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)

    model = build_frozen_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    history_rows = []

    print("\nTraining...")
    for epoch in range(FINAL_EPOCHS):
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

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"  Epoch {epoch+1:2}/{FINAL_EPOCHS} | loss={epoch_loss:.4f} | train_acc={epoch_acc*100:.1f}%")

        history_rows.append({
            "epoch": epoch + 1,
            "train_loss": round(epoch_loss, 4),
            "train_accuracy_pct": round(epoch_acc * 100, 2),
        })

    with open(HISTORY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_accuracy_pct"])
        writer.writeheader()
        writer.writerows(history_rows)

    torch.save(model.state_dict(), FINAL_MODEL_PATH)

    print("\n" + "=" * 70)
    print("FINAL MODEL SAVED")
    print("=" * 70)
    print(f"Model path:   {FINAL_MODEL_PATH}")
    print(f"History CSV:  {HISTORY_CSV}  <- use this table in your paper")
    print("\nIMPORTANT: this history is TRAINING accuracy only (no held-out set).")
    print("Do not report it as your model's performance - report the k-fold")
    print("aggregated results from step 6 for that.")
    print("\nNext: run 08_generate_confusion_matrix_roc.py")


if __name__ == "__main__":
    main()