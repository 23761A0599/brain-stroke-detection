"""
12_generate_epoch_table_v2.py

Epoch-by-epoch table (Table I) for the new fixed-protocol runs.
Reads outputs/reports_pretrained/<tag>_seed<k>_history.csv written by
06_train_kfold_v2.py and averages every epoch over ALL folds and seeds.
Because every fold now trains for the same number of epochs, every row
averages the same number of runs (no "folds still training" artifact).

Output: outputs/tables_pretrained/epoch_table_<tag>.csv   (one per tag)

Run from project root:
    python pipeline/12_generate_epoch_table_v2.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_DIR = PROJECT_ROOT / "outputs" / "reports_pretrained"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables_pretrained"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def build_table(tag):
    per_epoch = defaultdict(lambda: defaultdict(list))
    files = sorted(IN_DIR.glob(f"{tag}_seed*_history.csv"))
    for p in files:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                e = int(r["epoch"])
                per_epoch[e]["train_acc"].append(float(r["train_acc"]))
                per_epoch[e]["valid_acc"].append(float(r["valid_acc"]))
                per_epoch[e]["train_loss"].append(float(r["train_loss"]))
                per_epoch[e]["valid_loss"].append(float(r["valid_loss"]))
                per_epoch[e]["lr"].append(float(r["lr"]))
    rows = []
    for e in sorted(per_epoch):
        d = per_epoch[e]
        rows.append({
            "Epoch": e,
            "Training Accuracy": round(np.mean(d["train_acc"]) / 100, 4),
            "Validation Accuracy": round(np.mean(d["valid_acc"]) / 100, 4),
            "Training Loss": round(np.mean(d["train_loss"]), 4),
            "Validation Loss": round(np.mean(d["valid_loss"]), 4),
            "Learning Rate": f"{np.mean(d['lr']):.2e}",
            "Runs Averaged": len(d["train_acc"]),
        })
    return rows, len(files)


def main():
    tags = sorted({p.name.split("_seed")[0] for p in IN_DIR.glob("*_seed*_history.csv")})
    if not tags:
        print(f"No history files in {IN_DIR} - run 06_train_kfold_v2.py first.")
        return
    for tag in tags:
        rows, n_seeds = build_table(tag)
        out = TABLES_DIR / f"epoch_table_{tag}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved: {out}  ({len(rows)} epochs, {n_seeds} seeds x 5 folds per row)")


if __name__ == "__main__":
    main()