"""
15_paired_comparison.py

Answers "is the main model really better than the others?" with only 11 patients.
 1. PAIRED patient-level bootstrap: the same resampled patients are used for the main model
    and each competitor, on seed-ensemble predictions (mean p_hem over seeds).
    Reports the difference (main - other) in AUC, accuracy, macro-F1 and hemorrhagic recall
    with a 95% interval and P(main better). If the interval contains 0, the difference is NOT
    reliable and the paper must not claim superiority.
 2. Sequence breakdown (DWI / GRE / SWI parsed from the image file names) of the main model:
    checks that performance is not driven by one MRI sequence. Output contains no patient names.

Outputs (outputs/reports_pretrained/)
    paired_comparison_<main>.csv
    sequence_breakdown_<main>.csv

Run from project root:
    python pipeline/15_paired_comparison.py --main efficientnetb0_frozen_imagenet
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_DIR = PROJECT_ROOT / "outputs" / "reports_pretrained"
SEQ_RE = re.compile(r"(?<![A-Za-z])(DWI|GRE|SWI|FLAIR|T2|T1|ADC)(?![A-Za-z])", re.I)


def read_csv(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def load_ensemble(tag):
    d = {}
    files = sorted(IN_DIR.glob(f"{tag}_seed*_predictions.csv"))
    if not files:
        raise SystemExit(f"No prediction files for tag '{tag}' in {IN_DIR}")
    for p in files:
        for r in read_csv(p):
            k = (r["class"], r["filename"])
            e = d.setdefault(k, {"label": int(r["label"]), "patient": f'{r["class"]}:{r["patient_id"]}', "p": []})
            e["p"].append(float(r["p_hem"]))
    return d, len(files)


def metrics(y, ph):
    pred = np.where(ph >= 0.5, 0, 1)                     # class 0 = Hemorrhagic
    pos = (y == 0)
    return {
        "AUC": roc_auc_score(pos.astype(int), ph),
        "Accuracy": float((pred == y).mean()),
        "Macro-F1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "Hemorrhagic recall": float(((pred == 0) & pos).sum() / max(pos.sum(), 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="efficientnetb0_frozen_imagenet")
    ap.add_argument("--boot", type=int, default=3000)
    args = ap.parse_args()

    tags = sorted({p.name.split("_seed")[0] for p in IN_DIR.glob("*_seed*_predictions.csv")})
    if args.main not in tags:
        raise SystemExit(f"main tag '{args.main}' not found. Available: {tags}")
    ens = {t: load_ensemble(t)[0] for t in tags}
    keys = sorted(set.intersection(*[set(e) for e in ens.values()]))
    main_e = ens[args.main]
    y = np.array([main_e[k]["label"] for k in keys])
    pat = np.array([main_e[k]["patient"] for k in keys])
    P = {t: np.array([np.mean(ens[t][k]["p"]) for k in keys]) for t in tags}
    print(f"{len(keys)} images, {len(set(pat))} patients, {len(tags)} configurations")

    base = metrics(y, P[args.main])
    print(f"\nMAIN = {args.main}:  " + "  ".join(f"{k}={v:.3f}" for k, v in base.items()))

    rng = np.random.default_rng(0)
    pats = sorted(set(pat))
    idx = {p: np.where(pat == p)[0] for p in pats}
    names = list(base)
    diffs = {t: {m: [] for m in names} for t in tags if t != args.main}
    used = 0
    for _ in range(args.boot):
        ii = np.concatenate([idx[p] for p in rng.choice(pats, len(pats), replace=True)])
        if len(set(y[ii])) < 2:
            continue
        used += 1
        mm = metrics(y[ii], P[args.main][ii])
        for t in diffs:
            mo = metrics(y[ii], P[t][ii])
            for m in names:
                diffs[t][m].append(mm[m] - mo[m])
    print(f"valid bootstrap draws: {used}/{args.boot}")

    rows = []
    print("\nDifference main - other (95% patient-bootstrap interval) | P(main better)")
    for t in diffs:
        other = metrics(y, P[t])
        line = f"\n[{t}]"
        row = {"other": t}
        for m in names:
            d = np.array(diffs[t][m])
            lo, hi = np.percentile(d, [2.5, 97.5])
            p_better = float((d > 0).mean())
            flag = "  <- reliable" if (lo > 0 or hi < 0) else ""
            line += f"\n   {m:19} {base[m]-other[m]:+.3f} [{lo:+.3f}, {hi:+.3f}]  P={p_better:.2f}{flag}"
            row[f"{m} diff"] = round(base[m] - other[m], 4)
            row[f"{m} CI"] = f"[{lo:+.3f}, {hi:+.3f}]"
            row[f"{m} P(main>other)"] = round(p_better, 3)
        print(line)
        rows.append(row)
    out = IN_DIR / f"paired_comparison_{args.main}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- sequence breakdown (main model) ----
    pred = np.where(P[args.main] >= 0.5, 0, 1)
    cell = defaultdict(lambda: [0, 0])
    for k, yy, pp in zip(keys, y, pred):
        m = SEQ_RE.search(k[1])
        seq = m.group(1).upper() if m else "UNKNOWN"
        cname = "Hemorrhagic" if yy == 0 else "NonHemorrhagic"
        for key in ((seq, cname), (seq, "ALL")):
            cell[key][0] += 1
            cell[key][1] += int(yy == pp)
    seq_rows = []
    print(f"\nSEQUENCE BREAKDOWN ({args.main}, seed-ensemble, threshold 0.5)")
    print(f"  {'sequence':9} {'class':15} {'n':>4} {'accuracy':>9}")
    for (seq, cname) in sorted(cell, key=lambda t: (t[0], t[1] != "ALL", t[1])):
        n, c = cell[(seq, cname)]
        print(f"  {seq:9} {cname:15} {n:4d} {100*c/n:8.1f}%")
        seq_rows.append({"sequence": seq, "class": cname, "n": n, "accuracy_pct": round(100 * c / n, 1)})
    with open(IN_DIR / f"sequence_breakdown_{args.main}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(seq_rows[0].keys()))
        w.writeheader(); w.writerows(seq_rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()