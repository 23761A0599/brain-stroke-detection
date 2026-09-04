"""
05_preflight_leak_check.py

Hard safety gate before training. For EVERY fold, checks two
independent things and ABORTS if either fails:

  1. CONTENT LEAK: any identical file content (MD5 hash) appearing in
     both train and valid within the same fold, same class.
  2. PATIENT LEAK: any patient_id appearing in both train and valid
     within the same fold (reads back from kfold_assignment.csv).

This mirrors the exact class of bug that caused false "99% accuracy"
results in the earlier CT pipeline - checking twice, independently,
means a bug in one check is unlikely to be missed by the other.

Run from project root:
    python pipeline/05_preflight_leak_check.py
"""

import csv
import hashlib
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FOLDS_DIR = PROJECT_ROOT / "data_processed" / "03_folds"
ASSIGNMENT_CSV = PROJECT_ROOT / "outputs" / "reports" / "kfold_assignment.csv"

CLASSES = ["Hemorrhagic", "NonHemorrhagic"]
N_FOLDS = 5  # will auto-detect actual count below


def file_md5(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def check_patient_leak():
    print("-" * 70)
    print("CHECK 1: PATIENT-LEVEL OVERLAP (train vs valid, per fold)")
    print("-" * 70)

    fold_split_patients = defaultdict(lambda: defaultdict(set))  # fold -> split -> set(patient_id)

    with open(ASSIGNMENT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            fold_split_patients[row["fold"]][row["split"]].add(row["patient_id"])

    leaks_found = False
    for fold, splits in fold_split_patients.items():
        overlap = splits["train"] & splits["valid"]
        if overlap:
            leaks_found = True
            print(f"  LEAK [fold {fold}]: patients in BOTH train and valid: {sorted(overlap)}")
        else:
            print(f"  Fold {fold}: OK - no patient overlap "
                  f"(train: {len(splits['train'])} patients, valid: {len(splits['valid'])} patients)")

    return leaks_found


def check_content_leak():
    print("\n" + "-" * 70)
    print("CHECK 2: CONTENT-LEVEL LEAK (MD5, within class, per fold)")
    print("-" * 70)

    fold_dirs = sorted(FOLDS_DIR.glob("fold_*"))
    leaks_found = False

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        for cls in CLASSES:
            train_dir = fold_dir / "train" / cls
            valid_dir = fold_dir / "valid" / cls

            train_hashes = {file_md5(f): f.name for f in train_dir.iterdir() if f.is_file()}
            valid_hashes = {file_md5(f): f.name for f in valid_dir.iterdir() if f.is_file()}

            overlap = set(train_hashes) & set(valid_hashes)
            if overlap:
                leaks_found = True
                for h in overlap:
                    print(f"  LEAK [{fold_name}/{cls}]: identical content - "
                          f"train/{train_hashes[h]} == valid/{valid_hashes[h]}")
            else:
                print(f"  {fold_name}/{cls}: OK - no content overlap "
                      f"(train: {len(train_hashes)}, valid: {len(valid_hashes)})")

    return leaks_found


def main():
    print("=" * 70)
    print("STEP 5: PRE-FLIGHT LEAK CHECK (all folds)")
    print("=" * 70)

    patient_leak = check_patient_leak()
    content_leak = check_content_leak()

    print("\n" + "=" * 70)
    if patient_leak or content_leak:
        print("RESULT: LEAK(S) DETECTED - DO NOT PROCEED TO TRAINING")
        print("=" * 70)
        raise RuntimeError(
            "Pre-flight check failed. Fix the leak(s) listed above before "
            "running 06_train_kfold.py."
        )
    else:
        print("RESULT: ALL CLEAR - no patient or content leaks in any fold")
        print("=" * 70)
        print("\nNext: run 06_train_kfold.py")


if __name__ == "__main__":
    main()