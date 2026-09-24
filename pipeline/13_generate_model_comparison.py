"""
13_generate_model_comparison.py (updated: 4-model comparison)

Output: outputs/tables/model_comparison_table.csv
"""
import csv
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = TABLES_DIR / "model_comparison_table.csv"

MODELS = [
    ("EfficientNet-B0 (ours)", "", "5.3M"),
    ("ResNet-50 (baseline)", "_resnet50", "25.6M"),
    ("ResNet-101 (baseline)", "_resnet101", "44.5M"),
    ("DenseNet-121 (baseline)", "_densenet121", "8.0M"),
]


def load_fold_results(path):
    accuracies, macro_f1s = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            accuracies.append(float(row["accuracy"]))
            macro_f1s.append(float(row["macro_f1"]))
    return accuracies, macro_f1s


def load_per_class(path):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["class"]] = row
    return rows


def main():
    rows = []
    for name, suffix, params in MODELS:
        fold_path = REPORT_DIR / f"kfold_per_fold_results{suffix}.csv"
        class_path = REPORT_DIR / f"kfold_per_class_summary{suffix}.csv"
        if not fold_path.exists():
            print(f"SKIPPING {name}: {fold_path} not found yet")
            continue

        acc, f1 = load_fold_results(fold_path)
        cls = load_per_class(class_path)
        hem = cls.get("Hemorrhagic", {})
        nonhem = cls.get("NonHemorrhagic", {})

        rows.append({
            "Model": name,
            "Parameters": params,
            "Mean Accuracy": f"{np.mean(acc)*100:.2f}% \u00b1 {np.std(acc)*100:.2f}%",
            "Mean Macro-F1": f"{np.mean(f1):.4f} \u00b1 {np.std(f1):.4f}",
            "Hemorrhagic P/R/F1": f"{float(hem.get('precision_mean',0)):.3f} / {float(hem.get('recall_mean',0)):.3f} / {float(hem.get('f1_mean',0)):.3f}",
            "Non-Hemorrhagic P/R/F1": f"{float(nonhem.get('precision_mean',0)):.3f} / {float(nonhem.get('recall_mean',0)):.3f} / {float(nonhem.get('f1_mean',0)):.3f}",
        })

    if not rows:
        print("No model results found yet - run the 06/06b/06c/06d scripts first.")
        return

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {OUT_PATH} ({len(rows)} models)")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()