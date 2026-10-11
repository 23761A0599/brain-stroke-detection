"""
06_train_kfold_v2.py

Patient-level k-fold training with ImageNet-pretrained weights, a protocol
that never uses the held-out fold to make a training decision, multiple seeds,
and all four architectures / two training modes in one script.
Replaces 06, 06b, 06c, 06d.

PROTOCOL (what changed vs. 06_train_kfold.py)
  * Fixed number of epochs for every fold (no early stopping).
  * LR schedule is cosine (does not look at validation data).
  * The model evaluated is the LAST-epoch model (no best-checkpoint selection
    on the held-out fold).
  * Per-epoch validation metrics are still logged, for the curves only; they
    are never used to decide anything.
  * Frozen BatchNorm layers stay in eval() mode (running stats not updated).
  * Every run is seeded; several seeds are run and averaged.
  * Out-of-fold predictions are saved per image with patient IDs, so pooled
    metrics and patient-level (cluster) bootstrap CIs can be computed.

USAGE (from project root, inside the backend venv)
  # main experiment: ImageNet weights, both modes, EfficientNet-B0, 3 seeds
  python pipeline/06_train_kfold_v2.py --models efficientnetb0 --modes frozen finetune --seeds 0 1 2

  # baselines, same protocol
  python pipeline/06_train_kfold_v2.py --models resnet50 resnet101 densenet121 --modes frozen finetune --seeds 0 1 2

  # random-init ablation (the old setup, but with the fixed protocol)
  python pipeline/06_train_kfold_v2.py --models efficientnetb0 --modes frozen --init random --seeds 0 1 2

  # fair random-weight baseline: random init, BatchNorm allowed to adapt
  python pipeline/06_train_kfold_v2.py --models efficientnetb0 --modes frozen --init random --bn-adapt --seeds 0 1 2

  # just re-print the summary table
  python pipeline/06_train_kfold_v2.py --summarize_only

Finished (model, mode, init, seed) runs are skipped on re-run, so a crashed or
disconnected session can simply be restarted. Use --force to redo them.
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from model.config import CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, DEVICE  # noqa: E402
from model.model_builder import build_model, set_train_mode  # noqa: E402

FOLDS_DIR = PROJECT_ROOT / "data_processed" / "03_folds"
ASSIGNMENT_CSV = PROJECT_ROOT / "outputs" / "reports" / "kfold_assignment.csv"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "reports_pretrained"

# ---- hyper-parameters: fixed in advance, NOT tuned on the held-out folds ----
N_FOLDS = 5
BATCH_SIZE = 8
EPOCHS = 15
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.05
LR_HEAD = 1e-3        # new classification head
LR_BACKBONE = 1e-4    # unfrozen backbone blocks (finetune mode only)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def head_module(model: nn.Module, name: str) -> nn.Module:
    return model.fc if name in ("resnet50", "resnet101") else model.classifier


def load_patient_map():
    """(fold, class, filename) -> patient_id for every validation image."""
    mapping = {}
    with open(ASSIGNMENT_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] == "valid":
                mapping[(int(r["fold"]), r["class"], r["filename"])] = r["patient_id"]
    return mapping


def build_loaders(fold_dir: Path, gen: torch.Generator):
    train_ds = datasets.ImageFolder(fold_dir / "train", transform=train_tf)
    valid_ds = datasets.ImageFolder(fold_dir / "valid", transform=eval_tf)
    assert train_ds.classes == CLASS_NAMES, f"class order {train_ds.classes} != {CLASS_NAMES}"
    assert valid_ds.classes == CLASS_NAMES

    targets = torch.tensor([y for _, y in train_ds.samples])
    class_counts = torch.bincount(targets, minlength=NUM_CLASSES).float().clamp(min=1)
    sample_w = (1.0 / class_counts)[targets]          # w_c = 1 / N_c  (Eq. 5)
    sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w),
                                    replacement=True, generator=gen)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, valid_loader, valid_ds


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, n = 0.0, 0
    probs, labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        n += x.size(0)
        probs.append(torch.softmax(out, dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    preds = probs.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, labels=list(range(NUM_CLASSES)), zero_division=0)
    return {
        "loss": loss_sum / n,
        "acc": accuracy_score(labels, preds),
        "macro_f1": float(f1.mean()),
        "p": p, "r": r, "f1": f1,
        "probs": probs, "labels": labels, "preds": preds,
    }


def run_fold(model_name, mode, init, seed, fold, epochs, patient_map, bn_adapt=False):
    set_seed(seed * 1000 + fold)
    gen = torch.Generator()
    gen.manual_seed(seed * 1000 + fold)

    train_loader, valid_loader, valid_ds = build_loaders(FOLDS_DIR / f"fold_{fold}", gen)

    model = build_model(model_name, pretrained=(init == "imagenet"), mode=mode).to(DEVICE)

    head_ids = {id(p) for p in head_module(model, model_name).parameters()}
    head_params = [p for p in model.parameters() if id(p) in head_ids]
    back_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head_params, "lr": LR_HEAD}]
    if back_params:
        groups.append({"params": back_params, "lr": LR_BACKBONE})

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    last = None
    for epoch in range(1, epochs + 1):
        if bn_adapt:
            model.train()                    # BatchNorm statistics adapt (old behaviour; needed for a fair random-init baseline)
        else:
            set_train_mode(model, mode)      # keeps frozen BatchNorm in eval()
        run_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        last = evaluate(model, valid_loader, criterion)   # logged only, never used to decide
        history.append({
            "fold": fold, "epoch": epoch,
            "train_loss": round(run_loss / total, 4),
            "train_acc": round(100 * correct / total, 2),
            "valid_loss": round(last["loss"], 4),
            "valid_acc": round(100 * last["acc"], 2),
            "valid_macro_f1": round(last["macro_f1"], 4),
            "lr": lr_now,
        })

    # the model that is reported = the LAST-epoch model
    pred_rows = []
    for (path, label), prob, pred in zip(valid_ds.samples, last["probs"], last["preds"]):
        p = Path(path)
        cls, fname = p.parent.name, p.name
        pred_rows.append({
            "fold": fold,
            "patient_id": patient_map[(fold, cls, fname)],
            "class": cls, "filename": fname,
            "label": int(label), "pred": int(pred),
            "p_hem": float(prob[0]),          # class index 0 = Hemorrhagic
        })
    fold_row = {
        "fold": fold, "n_valid": len(pred_rows),
        "accuracy": last["acc"], "macro_f1": last["macro_f1"],
        "hem_p": last["p"][0], "hem_r": last["r"][0], "hem_f1": last["f1"][0],
        "non_p": last["p"][1], "non_r": last["r"][1], "non_f1": last["f1"][1],
    }
    return history, pred_rows, fold_row


def write_csv(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
def read_csv(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _auc(y, p_hem):
    yy = (np.asarray(y) == 0).astype(int)            # hemorrhagic = positive class
    return roc_auc_score(yy, p_hem) if len(set(yy)) == 2 else float("nan")


def summarize(out_dir: Path, n_boot: int = 2000):
    tags = sorted({p.name.split("_seed")[0] for p in out_dir.glob("*_seed*_predictions.csv")})
    if not tags:
        print("No finished runs found in", out_dir)
        return
    summary_rows = []
    print("\n" + "=" * 100)
    print("SUMMARY  (mean +/- SD over seeds; fold SD is the population SD, as in Eq. 14)")
    print("=" * 100)
    for tag in tags:
        files = sorted(out_dir.glob(f"{tag}_seed*_predictions.csv"))
        per_seed, key_to = [], {}
        for pf in files:
            seed = pf.name.split("_seed")[1].split("_")[0]
            prow = read_csv(pf)
            frow = read_csv(out_dir / f"{tag}_seed{seed}_folds.csv")
            accs = np.array([float(r["accuracy"]) for r in frow])
            f1s = np.array([float(r["macro_f1"]) for r in frow])
            y = np.array([int(r["label"]) for r in prow])
            pr = np.array([int(r["pred"]) for r in prow])
            ph = np.array([float(r["p_hem"]) for r in prow])
            per_seed.append({
                "fold_acc": accs.mean(), "fold_acc_sd": accs.std(),
                "fold_f1": f1s.mean(), "fold_f1_sd": f1s.std(),
                "pooled_acc": float((y == pr).mean()), "auc": _auc(y, ph),
                "hem_recall": float(np.mean([float(r["hem_r"]) for r in frow])),
                "hem_f1": float(np.mean([float(r["hem_f1"]) for r in frow])),
            })
            for r in prow:
                k = (r["class"], r["filename"])
                d = key_to.setdefault(k, {"label": int(r["label"]),
                                          "patient": f'{r["class"]}:{r["patient_id"]}',
                                          "p": []})
                d["p"].append(float(r["p_hem"]))

        def ms(name):
            v = np.array([s[name] for s in per_seed])
            return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))

        # seed-ensemble pooled predictions + patient-level cluster bootstrap
        keys = list(key_to)
        y = np.array([key_to[k]["label"] for k in keys])
        ph = np.array([np.mean(key_to[k]["p"]) for k in keys])
        pat = np.array([key_to[k]["patient"] for k in keys])
        pred = (ph < 0.5).astype(int)                 # p_hem >= 0.5 -> class 0 (hemorrhagic)
        ens_acc, ens_auc = float((y == pred).mean()), _auc(y, ph)
        rng = np.random.default_rng(0)
        pats = sorted(set(pat))
        idx = {p: np.where(pat == p)[0] for p in pats}
        b_acc, b_auc = [], []
        for _ in range(n_boot):
            ii = np.concatenate([idx[p] for p in rng.choice(pats, len(pats), replace=True)])
            b_acc.append((y[ii] == pred[ii]).mean())
            a = _auc(y[ii], ph[ii])
            if not np.isnan(a):
                b_auc.append(a)
        lo_a, hi_a = np.percentile(b_acc, [2.5, 97.5])
        lo_u, hi_u = np.percentile(b_auc, [2.5, 97.5])

        fa, fa_sd = ms("fold_acc")
        ff, ff_sd = ms("fold_f1")
        pa, pa_sd = ms("pooled_acc")
        au, au_sd = ms("auc")
        hr, _ = ms("hem_recall")
        hf, _ = ms("hem_f1")
        print(f"\n[{tag}]  seeds={len(per_seed)}  images={len(keys)}  patients={len(pats)}")
        print(f"  fold-mean accuracy : {100*fa:.2f}% +/- {100*fa_sd:.2f}% (SD over seeds); "
              f"avg within-run fold SD {100*np.mean([s['fold_acc_sd'] for s in per_seed]):.2f}%")
        print(f"  fold-mean macro-F1 : {ff:.4f} +/- {ff_sd:.4f}")
        print(f"  hemorrhagic        : mean-fold recall {hr:.3f}, mean-fold F1 {hf:.3f}")
        print(f"  pooled accuracy    : {100*pa:.2f}% +/- {100*pa_sd:.2f}%   pooled AUC: {au:.4f} +/- {au_sd:.4f}")
        print(f"  seed-ensemble      : acc {100*ens_acc:.2f}% (95% patient-bootstrap CI {100*lo_a:.1f}-{100*hi_a:.1f}), "
              f"AUC {ens_auc:.4f} (CI {lo_u:.3f}-{hi_u:.3f})")
        summary_rows.append({
            "tag": tag, "n_seeds": len(per_seed),
            "fold_acc_mean": fa, "fold_acc_sd_seeds": fa_sd,
            "fold_macro_f1_mean": ff, "fold_macro_f1_sd_seeds": ff_sd,
            "hem_recall_mean": hr, "hem_f1_mean": hf,
            "pooled_acc_mean": pa, "pooled_acc_sd": pa_sd,
            "pooled_auc_mean": au, "pooled_auc_sd": au_sd,
            "ens_acc": ens_acc, "ens_acc_ci_lo": lo_a, "ens_acc_ci_hi": hi_a,
            "ens_auc": ens_auc, "ens_auc_ci_lo": lo_u, "ens_auc_ci_hi": hi_u,
        })
    write_csv(out_dir / "summary.csv", summary_rows)
    print(f"\nSummary written to {out_dir / 'summary.csv'}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["efficientnetb0"],
                    choices=["efficientnetb0", "resnet50", "resnet101", "densenet121"])
    ap.add_argument("--modes", nargs="+", default=["frozen"], choices=["frozen", "finetune"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--init", default="imagenet", choices=["imagenet", "random"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--bn-adapt", action="store_true",
                    help="let BatchNorm statistics adapt (use with --init random for a fair random-weight baseline)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summarize_only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        patient_map = load_patient_map()
        print(f"Device: {DEVICE} | init: {args.init} | epochs: {args.epochs}")
        t0 = time.time()
        for model_name in args.models:
            for mode in args.modes:
                tag = f"{model_name}_{mode}_{args.init}" + ("-bn" if args.bn_adapt else "")
                for seed in args.seeds:
                    stem = out_dir / f"{tag}_seed{seed}"
                    if (Path(str(stem) + "_predictions.csv").exists() and not args.force):
                        print(f"skip {tag} seed {seed} (already done)")
                        continue
                    print("\n" + "=" * 70 + f"\n{tag}  seed {seed}\n" + "=" * 70)
                    all_hist, all_pred, all_fold = [], [], []
                    for fold in range(N_FOLDS):
                        h, p, fr = run_fold(model_name, mode, args.init, seed, fold,
                                            args.epochs, patient_map, args.bn_adapt)
                        all_hist += h
                        all_pred += p
                        all_fold.append(fr)
                        print(f"  fold {fold}: acc={100*fr['accuracy']:.2f}% "
                              f"macro-F1={fr['macro_f1']:.4f} "
                              f"hem R={fr['hem_r']:.3f} (n={fr['n_valid']})")
                    accs = np.array([r["accuracy"] for r in all_fold])
                    print(f"  -> mean acc {100*accs.mean():.2f}% +/- {100*accs.std():.2f}%")
                    write_csv(Path(str(stem) + "_history.csv"), all_hist)
                    write_csv(Path(str(stem) + "_folds.csv"), all_fold)
                    write_csv(Path(str(stem) + "_predictions.csv"), all_pred)   # written last = "done" marker
        print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min")

    summarize(out_dir)


if __name__ == "__main__":
    main()