"""
12_generate_epoch_table.py (updated: all 4 models, no "Folds Averaged" column)

Reads each model's per-fold epoch histories (from 06 / 06b / 06c / 06d)
and produces one paper-ready table per model: Epoch | Training Accuracy |
Validation Accuracy | Training Loss | Validation Loss.

Output:
    outputs/tables/epoch_by_epoch_table.csv                (EfficientNet-B0)
    outputs/tables/epoch_by_epoch_table_resnet50.csv
    outputs/tables/epoch_by_epoch_table_resnet101.csv
    outputs/tables/epoch_by_epoch_table_densenet121.csv

Run from project root, inside the backend's venv:
    python pipeline/12_generate_epoch_table.py
"""
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5

MODELS = [
    ("EfficientNet-B0", "fold_histories", "epoch_by_epoch_table.csv"),
    ("ResNet-50", "fold_histories_resnet50", "epoch_by_epoch_table_resnet50.csv"),
    ("ResNet-101", "fold_histories_resnet101", "epoch_by_epoch_table_resnet101.csv"),
    ("DenseNet-121", "fold_histories_densenet121", "epoch_by_epoch_table_densenet121.csv"),
]


def build_table(history_dir_name):
    history_dir = REPORTS_DIR / history_dir_name
    per_epoch = defaultdict(lambda: {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []})

    for fold_idx in range(N_FOLDS):
        path = history_dir / f"fold_{fold_idx}_history.csv"
        if not path.exists():
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                epoch = int(row["epoch"])
                per_epoch[epoch]["train_acc"].append(float(row["train_accuracy"]))
                per_epoch[epoch]["val_acc"].append(float(row["valid_accuracy"]))
                per_epoch[epoch]["train_loss"].append(float(row["train_loss"]))
                per_epoch[epoch]["val_loss"].append(float(row["valid_loss"]))

    rows = []
    for epoch in sorted(per_epoch.keys()):
        d = per_epoch[epoch]
        n = len(d["train_acc"])
        if n == 0:
            continue
        rows.append({
            "Epoch": epoch,
            "Training Accuracy": round(sum(d["train_acc"]) / n / 100, 4),
            "Validation Accuracy": round(sum(d["val_acc"]) / n / 100, 4),
            "Training Loss": round(sum(d["train_loss"]) / n, 4),
            "Validation Loss": round(sum(d["val_loss"]) / n, 4),
        })
    return rows


def main():
    for model_name, history_dir_name, out_filename in MODELS:
        history_dir = REPORTS_DIR / history_dir_name
        if not history_dir.exists():
            print(f"SKIPPING {model_name}: {history_dir} not found yet")
            continue

        rows = build_table(history_dir_name)
        if not rows:
            print(f"SKIPPING {model_name}: no history rows found in {history_dir}")
            continue

        out_path = TABLES_DIR / out_filename
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved: {out_path} ({len(rows)} epoch rows) - {model_name}")


if __name__ == "__main__":
    main()