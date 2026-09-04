"""
04_build_patient_kfold_splits.py

Builds PATIENT-LEVEL k-fold cross-validation splits. With only 11
total patients (5 Hemorrhagic, 6 NonHemorrhagic), a single fixed
train/valid/test split would leave the test set with just 1-2
patients - too fragile to trust. K-fold cross-validation at the
patient level is the defensible approach: every patient is held out
exactly once, across K folds, and final performance is reported as
the average (with std dev) across folds.

Uses Leave-One-Patient-Out-style grouping via GroupKFold: patients are
never split across train/valid within a fold, so slice-level leakage
is structurally impossible.

Input:  data_processed/02_near_dedup/manifest.csv
Output: data_processed/03_folds/fold_{i}/train/{class}/*.png
        data_processed/03_folds/fold_{i}/valid/{class}/*.png
        outputs/reports/kfold_assignment.csv

Run from project root:
    python pipeline/04_build_patient_kfold_splits.py
"""

import csv
import shutil
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "data_processed" / "02_near_dedup"
DEST_DIR = PROJECT_ROOT / "data_processed" / "03_folds"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = SOURCE_DIR / "manifest.csv"
ASSIGNMENT_REPORT = REPORT_DIR / "kfold_assignment.csv"

CLASSES = ["Hemorrhagic", "NonHemorrhagic"]

# With only 5-6 patients per class, K must be small enough that each
# fold's validation set has at least 1 patient per class.
N_FOLDS = 5


def main():
    print("=" * 70)
    print("STEP 4: BUILD PATIENT-LEVEL K-FOLD SPLITS")
    print("=" * 70)

    rows = []
    with open(MANIFEST_PATH, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Sanity check: report unique patients per class before splitting
    patients_by_class = defaultdict(set)
    for row in rows:
        patients_by_class[row["class"]].add(row["patient_id"])

    print("\nPatients per class (before fold assignment):")
    for cls in CLASSES:
        print(f"  {cls}: {len(patients_by_class[cls])} patients "
              f"-> {sorted(patients_by_class[cls])}")

    effective_folds = min(N_FOLDS, min(len(patients_by_class[c]) for c in CLASSES))
    if effective_folds < N_FOLDS:
        print(f"\nWARNING: requested {N_FOLDS} folds, but the smallest class only "
              f"has {effective_folds} patients. Reducing to {effective_folds} folds "
              f"so every fold has at least 1 validation patient per class.")

    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)
    DEST_DIR.mkdir(parents=True)

    assignment_rows = []

    # Build folds SEPARATELY per class (so each fold has patients from
    # both classes in its validation set), then merge fold indices.
    fold_assignments = {cls: {} for cls in CLASSES}  # patient_id -> fold_number

    for cls in CLASSES:
        cls_rows = [r for r in rows if r["class"] == cls]
        patient_ids = sorted(patients_by_class[cls])

        # GroupKFold needs X and groups; we use a dummy X of the right length
        gkf = GroupKFold(n_splits=effective_folds)
        dummy_X = list(range(len(patient_ids)))
        # groups must align 1:1 with dummy_X here, so we fold over patients directly
        for fold_idx, (_, valid_patient_idx) in enumerate(gkf.split(dummy_X, groups=patient_ids)):
            for pi in valid_patient_idx:
                fold_assignments[cls][patient_ids[pi]] = fold_idx

    # Now copy each image into every fold's train or valid folder,
    # based on whether its patient is the held-out patient for that fold.
    for fold_idx in range(effective_folds):
        for split in ["train", "valid"]:
            for cls in CLASSES:
                (DEST_DIR / f"fold_{fold_idx}" / split / cls).mkdir(parents=True, exist_ok=True)

        for row in rows:
            cls = row["class"]
            patient = row["patient_id"]
            patient_fold = fold_assignments[cls][patient]

            split = "valid" if patient_fold == fold_idx else "train"

            src = SOURCE_DIR / cls / row["filename"]
            dst = DEST_DIR / f"fold_{fold_idx}" / split / cls / row["filename"]
            shutil.copy2(src, dst)

            assignment_rows.append({
                "fold": fold_idx,
                "split": split,
                "class": cls,
                "patient_id": patient,
                "filename": row["filename"],
            })

    with open(ASSIGNMENT_REPORT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "split", "class", "patient_id", "filename"])
        writer.writeheader()
        writer.writerows(assignment_rows)

    print("\n" + "=" * 70)
    print("FOLD SUMMARY")
    print("=" * 70)
    for fold_idx in range(effective_folds):
        print(f"\nFold {fold_idx}:")
        for split in ["train", "valid"]:
            for cls in CLASSES:
                folder = DEST_DIR / f"fold_{fold_idx}" / split / cls
                count = len(list(folder.iterdir()))
                held_out_patients = sorted(
                    p for p, f in fold_assignments[cls].items() if f == fold_idx
                ) if split == "valid" else None
                extra = f"  (held-out patients: {held_out_patients})" if held_out_patients else ""
                print(f"  {split:6} {cls:15}: {count} images{extra}")

    print(f"\nAssignment report: {ASSIGNMENT_REPORT}")
    print(f"Folds saved under: {DEST_DIR}")
    print("\nNext: run 05_preflight_leak_check.py")


if __name__ == "__main__":
    main()