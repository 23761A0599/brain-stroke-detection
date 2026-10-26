"""
13_generate_model_comparison_v2.py

Model-comparison table (Table II) from the new fixed-protocol runs.
Reads outputs/reports_pretrained/summary.csv (written by 06_train_kfold_v2.py)
and the per-fold CSVs. Parameter counts are measured from the actual models
(2-class head), not copied from ImageNet-1000 reference figures.

Output: outputs/tables_pretrained/model_comparison_table.csv

Run from project root, inside the backend venv:
    python pipeline/13_generate_model_comparison_v2.py
"""
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
IN_DIR = PROJECT_ROOT / "outputs" / "reports_pretrained"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables_pretrained"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY = {"efficientnetb0": "EfficientNet-B0", "resnet50": "ResNet-50",
           "resnet101": "ResNet-101", "densenet121": "DenseNet-121"}


def read_csv(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def param_counts(model_name, mode):
    try:
        from model.model_builder import build_model
        m = build_model(model_name, pretrained=False, mode=mode)
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return f"{total/1e6:.2f}M", f"{trainable/1e3:.1f}k" if trainable < 1e6 else f"{trainable/1e6:.2f}M"
    except Exception as e:  # torch not available / unknown name
        return "n/a", "n/a"


def pm(vals, scale=1.0, nd=2):
    v = np.array(vals, dtype=float) * scale
    if len(v) < 2:
        return f"{v.mean():.{nd}f}"
    return f"{v.mean():.{nd}f} \u00b1 {v.std(ddof=1):.{nd}f}"


def main():
    summ = {r["tag"]: r for r in read_csv(IN_DIR / "summary.csv")} if (IN_DIR / "summary.csv").exists() else {}
    tags = sorted({p.name.split("_seed")[0] for p in IN_DIR.glob("*_seed*_folds.csv")})
    if not tags:
        print("No finished runs found - run 06_train_kfold_v2.py first.")
        return
    rows = []
    for tag in tags:
        parts = tag.split("_")
        model_name, mode, init = (parts + ["", "", ""])[:3]
        acc, accsd, f1, hp, hr, hf, np_, nr, nf = ([] for _ in range(9))
        for p in sorted(IN_DIR.glob(f"{tag}_seed*_folds.csv")):
            fr = read_csv(p)
            col = lambda k: np.array([float(r[k]) for r in fr])
            acc.append(col("accuracy").mean()); accsd.append(col("accuracy").std())
            f1.append(col("macro_f1").mean())
            hp.append(col("hem_p").mean()); hr.append(col("hem_r").mean()); hf.append(col("hem_f1").mean())
            np_.append(col("non_p").mean()); nr.append(col("non_r").mean()); nf.append(col("non_f1").mean())
        total, trainable = param_counts(model_name, mode if mode in ("frozen", "finetune") else "frozen")
        s = summ.get(tag, {})
        g = lambda k: float(s[k]) if k in s and s[k] not in ("", "nan") else float("nan")
        rows.append({
            "Model": DISPLAY.get(model_name, model_name),
            "Mode": mode, "Init": init, "Seeds": len(acc),
            "Total Params": total, "Trainable Params": trainable,
            "Mean Accuracy (%) [fold SD]": f"{100*np.mean(acc):.2f} \u00b1 {100*np.mean(accsd):.2f}",
            "SD across seeds (%)": f"{100*np.std(acc, ddof=1):.2f}" if len(acc) > 1 else "n/a",
            "Mean Macro-F1": pm(f1, 1, 4),
            "Hemorrhagic P/R/F1": f"{np.mean(hp):.3f} / {np.mean(hr):.3f} / {np.mean(hf):.3f}",
            "Non-Hemorrhagic P/R/F1": f"{np.mean(np_):.3f} / {np.mean(nr):.3f} / {np.mean(nf):.3f}",
            "Pooled Accuracy (%)": f"{100*g('pooled_acc_mean'):.2f}",
            "Pooled AUC": f"{g('pooled_auc_mean'):.4f}",
            "Seed-ensemble Acc (%) [95% CI]": f"{100*g('ens_acc'):.1f} [{100*g('ens_acc_ci_lo'):.1f}-{100*g('ens_acc_ci_hi'):.1f}]",
            "Seed-ensemble AUC [95% CI]": f"{g('ens_auc'):.3f} [{g('ens_auc_ci_lo'):.3f}-{g('ens_auc_ci_hi'):.3f}]",
        })
    out = TABLES_DIR / "model_comparison_table.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {out} ({len(rows)} configurations)")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()