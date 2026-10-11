"""
07_train_final_model_v2.py

Trains the FINAL deployment model on ALL 257 images, using exactly the same
settings as the cross-validation runs in 06_train_kfold_v2.py (ImageNet weights,
fixed epochs, cosine LR, frozen BatchNorm kept in eval mode, seeded).
Saves to the path backend/model/model_loader.py expects:
    models/EfficientNetB0/best_<model>.pth
An existing file at that path is backed up first (once) as *_randominit_backup.pth.

Pick --model / --mode to match the configuration you report in the paper.

Run from project root, inside the backend venv:
    python pipeline/07_train_final_model_v2.py --model efficientnetb0 --mode finetune
    python pipeline/07_train_final_model_v2.py --model efficientnetb0 --mode frozen
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model.config import CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, DEVICE, MODELS_DIR  # noqa: E402
from model.model_builder import build_model, set_train_mode  # noqa: E402

ALL_DATA_DIR = PROJECT_ROOT / "data_processed" / "02_near_dedup"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports_pretrained"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR = MODELS_DIR / "EfficientNetB0"      # folder name expected by model_loader.py
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---- keep IDENTICAL to 06_train_kfold_v2.py ----
EPOCHS = 15
BATCH_SIZE = 8
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.05
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4

train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def head_module(model, name):
    return model.fc if name in ("resnet50", "resnet101") else model.classifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnetb0",
                    choices=["efficientnetb0", "resnet50", "resnet101", "densenet121"])
    ap.add_argument("--mode", default="finetune", choices=["frozen", "finetune"])
    ap.add_argument("--init", default="imagenet", choices=["imagenet", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    gen = torch.Generator(); gen.manual_seed(args.seed)

    dataset = datasets.ImageFolder(ALL_DATA_DIR, transform=train_tf)
    assert dataset.classes == CLASS_NAMES, f"class order {dataset.classes} != {CLASS_NAMES}"
    targets = torch.tensor([y for _, y in dataset.samples])
    counts = torch.bincount(targets, minlength=NUM_CLASSES).float().clamp(min=1)
    weights = (1.0 / counts)[targets]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=gen)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)

    print("=" * 70)
    print(f"FINAL MODEL | {args.model} | {args.mode} | {args.init} | seed {args.seed} | {args.epochs} epochs")
    print(f"images: {len(dataset)} | class counts: {counts.int().tolist()} | device: {DEVICE}")
    print("=" * 70)

    model = build_model(args.model, pretrained=(args.init == "imagenet"), mode=args.mode).to(DEVICE)
    head_ids = {id(p) for p in head_module(model, args.model).parameters()}
    head_p = [p for p in model.parameters() if id(p) in head_ids]
    back_p = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head_p, "lr": LR_HEAD}]
    if back_p:
        groups.append({"params": back_p, "lr": LR_BACKBONE})

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    for epoch in range(1, args.epochs + 1):
        set_train_mode(model, args.mode)
        run_loss, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()
        print(f"  epoch {epoch:2}/{args.epochs} | loss={run_loss/total:.4f} | train_acc={100*correct/total:.1f}%")
        history.append({"epoch": epoch, "train_loss": round(run_loss / total, 4),
                        "train_accuracy_pct": round(100 * correct / total, 2)})

    tag = f"{args.model}_{args.mode}_{args.init}"
    hist_path = REPORT_DIR / f"final_model_training_history_{tag}.csv"
    with open(hist_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    final_path = FINAL_MODEL_DIR / f"best_{args.model}.pth"
    backup = FINAL_MODEL_DIR / f"best_{args.model}_randominit_backup.pth"
    if final_path.exists() and not backup.exists():
        shutil.copy2(final_path, backup)
        print(f"Existing model backed up to: {backup}")
    torch.save(model.state_dict(), final_path)

    print("\nSaved deployment model:", final_path)
    print("History CSV:", hist_path)
    print("NOTE: training accuracy only (no held-out data) - do not report it as performance.")
    print("Report the cross-validation results from 06_train_kfold_v2.py instead.")


if __name__ == "__main__":
    main()