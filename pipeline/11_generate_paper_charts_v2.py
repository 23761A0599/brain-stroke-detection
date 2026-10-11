"""
11_generate_paper_charts_v2.py

Figures for the fixed-protocol runs. Needs NO PyTorch and NO checkpoints: everything
is drawn from the CSV files written by 06_train_kfold_v2.py.

  loss_curve.png / accuracy_curve.png / lr_schedule.png  - mean over folds and seeds
                                                           (band = +/-1 SD across runs)
  confusion_matrix.png / roc_curve.png / pr_curve.png    - pooled out-of-fold predictions,
                                                           seed-ensemble (mean of p_hem over seeds)
  per_patient_accuracy.png                               - accuracy of each of the 11 patients
  class_distribution.png                                 - fold-0 train/valid class counts

Output: outputs/graphs_pretrained/<tag>/

Run from project root:
    python pipeline/11_generate_paper_charts_v2.py --tag efficientnetb0_finetune_imagenet
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_DIR = PROJECT_ROOT / "outputs" / "reports_pretrained"
ASSIGNMENT_CSV = PROJECT_ROOT / "outputs" / "reports" / "kfold_assignment.csv"
CLASS_NAMES = ["Hemorrhagic", "NonHemorrhagic"]          # index 0 = Hemorrhagic (positive class)


def read_csv(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def pick_default_tag():
    tags = sorted({p.name.split("_seed")[0] for p in IN_DIR.glob("*_seed*_predictions.csv")})
    for t in ("efficientnetb0_finetune_imagenet", "efficientnetb0_frozen_imagenet"):
        if t in tags:
            return t
    return tags[0] if tags else None


# ---------------- curves ----------------
def curves(tag, out):
    per_epoch = defaultdict(lambda: defaultdict(list))
    for p in sorted(IN_DIR.glob(f"{tag}_seed*_history.csv")):
        for r in read_csv(p):
            e = int(r["epoch"])
            for k in ("train_loss", "valid_loss", "train_acc", "valid_acc", "lr"):
                per_epoch[e][k].append(float(r[k]))
    epochs = sorted(per_epoch)

    def series(k):
        m = np.array([np.mean(per_epoch[e][k]) for e in epochs])
        s = np.array([np.std(per_epoch[e][k]) for e in epochs])
        return m, s

    def plot(keys, labels, ylabel, title, fname):
        plt.figure(figsize=(8, 5))
        for k, lab in zip(keys, labels):
            m, s = series(k)
            plt.plot(epochs, m, label=lab)
            plt.fill_between(epochs, m - s, m + s, alpha=0.15)
        plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title)
        if len(keys) > 1:
            plt.legend()
        plt.grid(True); plt.tight_layout()
        plt.savefig(out / fname, dpi=200); plt.close()

    n = len(per_epoch[epochs[0]]["lr"])
    sub = f"mean of {n} runs (folds x seeds), band = 1 SD"
    plot(["train_loss", "valid_loss"], ["Training Loss", "Validation Loss"], "Loss",
         f"Training vs Validation Loss ({sub})", "loss_curve.png")
    plot(["train_acc", "valid_acc"], ["Training Accuracy", "Validation Accuracy"], "Accuracy (%)",
         f"Training vs Validation Accuracy ({sub})", "accuracy_curve.png")
    plot(["lr"], ["Learning rate"], "Learning Rate",
         "Learning-Rate Schedule (cosine annealing, head group)", "lr_schedule.png")


# ---------------- pooled out-of-fold figures ----------------
def load_ensemble(tag):
    d = {}
    for p in sorted(IN_DIR.glob(f"{tag}_seed*_predictions.csv")):
        for r in read_csv(p):
            k = (r["class"], r["filename"])
            e = d.setdefault(k, {"label": int(r["label"]), "patient": f'{r["class"]}:{r["patient_id"]}', "p": []})
            e["p"].append(float(r["p_hem"]))
    keys = list(d)
    y = np.array([d[k]["label"] for k in keys])
    ph = np.array([np.mean(d[k]["p"]) for k in keys])
    pat = np.array([d[k]["patient"] for k in keys])
    return y, ph, pat


def pooled_figures(tag, out):
    y, ph, pat = load_ensemble(tag)
    pred = np.where(ph >= 0.5, 0, 1)
    pos = (y == 0).astype(int)

    cm = confusion_matrix(y, pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues"); plt.colorbar()
    plt.title("Confusion Matrix (pooled out-of-fold)", fontsize=11)
    plt.xticks([0, 1], CLASS_NAMES); plt.yticks([0, 1], CLASS_NAMES)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=200); plt.close()

    fpr, tpr, _ = roc_curve(pos, ph); a = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {a:.4f}"); plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Hemorrhagic vs NonHemorrhagic (out-of-fold)")
    plt.legend(loc="lower right"); plt.grid(True); plt.tight_layout()
    plt.savefig(out / "roc_curve.png", dpi=200); plt.close()

    pr, rc, _ = precision_recall_curve(pos, ph)
    plt.figure(figsize=(6, 6)); plt.plot(rc, pr)
    plt.axhline(pos.mean(), ls="--", color="gray", label=f"prevalence = {pos.mean():.2f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve - Hemorrhagic (out-of-fold)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out / "pr_curve.png", dpi=200); plt.close()

    # per-patient accuracy, patients shown ONLY as H1..Hn / N1..Nm (real names stay in a private file)
    pats = sorted(set(pat), key=lambda s_: (s_.split(":")[0] != "Hemorrhagic", s_))
    alias, ch, cn = {}, 0, 0
    for p_ in pats:
        if p_.startswith("Hemorrhagic"):
            ch += 1; alias[p_] = f"H{ch}"
        else:
            cn += 1; alias[p_] = f"N{cn}"
    with open(IN_DIR / "patient_alias_PRIVATE.csv", "w", newline="", encoding="utf-8-sig") as f:
        w_ = csv.writer(f); w_.writerow(["alias", "patient"])
        for p_ in pats:
            w_.writerow([alias[p_], p_.split(":", 1)[1]])
    accs = [float((pred[pat == p_] == y[pat == p_]).mean()) for p_ in pats]
    ns = [int((pat == p_).sum()) for p_ in pats]
    colors = ["tab:red" if p_.startswith("Hemorrhagic") else "tab:blue" for p_ in pats]
    plt.figure(figsize=(8, 5))
    plt.barh([f"{alias[p_]} (n={n_})" for p_, n_ in zip(pats, ns)], [100 * a_ for a_ in accs], color=colors)
    plt.gca().invert_yaxis(); plt.xlim(0, 100)
    plt.xlabel("Accuracy (%)"); plt.title("Per-patient accuracy (red = hemorrhagic, blue = non-hemorrhagic)")
    plt.grid(True, axis="x"); plt.tight_layout()
    plt.savefig(out / "per_patient_accuracy.png", dpi=200); plt.close()

    print(f"  confusion matrix: {cm.tolist()}  AUC={a:.4f}  ({len(set(pat))} patients, {len(y)} images)")


def class_distribution(out):
    if not ASSIGNMENT_CSV.exists():
        print("  (class distribution skipped: kfold_assignment.csv not found)")
        return
    counts = defaultdict(lambda: defaultdict(int))
    for r in read_csv(ASSIGNMENT_CSV):
        if r["fold"] == "0":
            counts[r["split"]][r["class"]] += 1
    x = np.arange(2); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, split in enumerate(["train", "valid"]):
        ax.bar(x + i * w, [counts[split][c] for c in CLASS_NAMES], w, label=split.capitalize())
    ax.set_xticks(x + w / 2); ax.set_xticklabels(CLASS_NAMES)
    ax.set_xlabel("Class"); ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution (Fold 0, representative of all folds)"); ax.legend()
    plt.tight_layout(); plt.savefig(out / "class_distribution.png", dpi=200); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="e.g. efficientnetb0_finetune_imagenet")
    tag = ap.parse_args().tag or pick_default_tag()
    if tag is None:
        print(f"No finished runs in {IN_DIR} - run 06_train_kfold_v2.py first.")
        return
    out = PROJECT_ROOT / "outputs" / "graphs_pretrained" / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"Charts for [{tag}] -> {out}")
    curves(tag, out)
    pooled_figures(tag, out)
    class_distribution(out)
    print("Done.")


if __name__ == "__main__":
    main()