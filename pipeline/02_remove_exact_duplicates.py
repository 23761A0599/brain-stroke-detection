"""
02_remove_exact_duplicates.py

Removes EXACT duplicate images (identical file content, MD5 hash match)
within each class. Never compares across classes. Keeps one copy per
duplicate group (first encountered, alphabetically), logs the rest.

Input:  data_raw/{Hemorrhagic,NonHemorrhagic}/*.png + data_raw/manifest.csv
Output: data_processed/01_deduplicated/{Hemorrhagic,NonHemorrhagic}/*.png
        data_processed/01_deduplicated/manifest.csv
        outputs/reports/exact_duplicates_removed.csv

Run from project root:
    python pipeline/02_remove_exact_duplicates.py
"""

import hashlib
import shutil
import csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "data_raw"
DEST_DIR = PROJECT_ROOT / "data_processed" / "01_deduplicated"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_MANIFEST = SOURCE_DIR / "manifest.csv"
DEST_MANIFEST = DEST_DIR / "manifest.csv"
REMOVED_LOG = REPORT_DIR / "exact_duplicates_removed.csv"

CLASSES = ["Hemorrhagic", "NonHemorrhagic"]


def file_md5(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("STEP 2: REMOVE EXACT DUPLICATES (within class, MD5)")
    print("=" * 70)

    # Load manifest for patient_id lookup
    manifest_by_filename = {}
    with open(SOURCE_MANIFEST, newline="") as f:
        for row in csv.DictReader(f):
            manifest_by_filename[(row["class"], row["filename"])] = row["patient_id"]

    for cls in CLASSES:
        (DEST_DIR / cls).mkdir(parents=True, exist_ok=True)

    dest_manifest_rows = []
    removed_rows = []

    for cls in CLASSES:
        source_folder = SOURCE_DIR / cls
        files = sorted(f for f in source_folder.iterdir() if f.is_file())

        print(f"\n{cls}: {len(files)} images")

        hash_to_first_file = {}
        kept = 0
        removed = 0

        for f in files:
            file_hash = file_md5(f)

            if file_hash in hash_to_first_file:
                removed += 1
                removed_rows.append({
                    "class": cls,
                    "removed_filename": f.name,
                    "duplicate_of": hash_to_first_file[file_hash].name,
                    "patient_id": manifest_by_filename.get((cls, f.name), "UNKNOWN"),
                })
                continue

            hash_to_first_file[file_hash] = f
            dest_path = DEST_DIR / cls / f.name
            shutil.copy2(f, dest_path)
            kept += 1

            dest_manifest_rows.append({
                "filename": f.name,
                "class": cls,
                "patient_id": manifest_by_filename.get((cls, f.name), "UNKNOWN"),
            })

        print(f"  Kept: {kept} | Exact duplicates removed: {removed}")

    with open(DEST_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "patient_id"])
        writer.writeheader()
        writer.writerows(dest_manifest_rows)

    with open(REMOVED_LOG, "w", newline="") as f:
        if removed_rows:
            writer = csv.DictWriter(f, fieldnames=["class", "removed_filename", "duplicate_of", "patient_id"])
            writer.writeheader()
            writer.writerows(removed_rows)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total kept: {len(dest_manifest_rows)}")
    print(f"Total exact duplicates removed: {len(removed_rows)}")
    print(f"Deduplicated data: {DEST_DIR}")
    print(f"Manifest: {DEST_MANIFEST}")
    print(f"Removed-files log: {REMOVED_LOG}")
    print("\nNext: run 03_remove_near_duplicates.py")


if __name__ == "__main__":
    main()