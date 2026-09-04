"""
01_copy_raw_dataset.py

Builds data_raw/ from the Kaggle MRI stroke dataset
(kaggle_source/dataset/Stroke_classification/).

Binary mapping (confirmed):
  Haemorrhagic -> Hemorrhagic       (186 images)
  Normal       -> NonHemorrhagic    (399 images)
  Ischemic     -> EXCLUDED entirely (30 images, left untouched in source)

For every copied image, extracts the PATIENT ID from the filename
(e.g. "Kuppusamy DWI-4.jpg_Haemorrhagic_1.png" -> patient "Kuppusamy",
"Prabhakar Rao DWI-10.jpg_Haemorrhagic_156.png" -> patient "Prabhakar Rao")
and writes it to a manifest CSV. This is CRITICAL: later steps must keep
every image from the same patient in the same split (train/valid/test),
or the same leakage problem from the CT project will happen again.

Output:
  data_raw/Hemorrhagic/*.png
  data_raw/NonHemorrhagic/*.png
  data_raw/manifest.csv   (filename, class, patient_id, original_path)

Run from project root:
    python pipeline/01_copy_raw_dataset.py
"""

import re
import shutil
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "kaggle_source" / "dataset" / "Stroke_classification"

DEST_DIR = PROJECT_ROOT / "data_raw"

# Confirmed binary mapping
CLASS_MAPPING = {
    "Haemorrhagic": "Hemorrhagic",
    "Normal": "NonHemorrhagic",
}
EXCLUDED_CLASSES = ["Ischemic"]

MANIFEST_PATH = DEST_DIR / "manifest.csv"

# Sequence-type tokens that separate "patient name" from "sequence-slice"
# in filenames. Longest/most-specific patterns first.
SEQUENCE_TOKENS = ["T2 Flair", "DWI", "GRE", "SWI", "T2"]

PATIENT_PATTERN = re.compile(
    r"^(?P<patient>.+?)\s+(?:" + "|".join(re.escape(t) for t in SEQUENCE_TOKENS) + r")-"
)


def extract_patient_id(filename):
    match = PATIENT_PATTERN.match(filename)
    if match:
        return match.group("patient").strip()
    # Fallback: if pattern doesn't match, use the first token as a
    # best-effort patient id and flag it for manual review.
    return filename.split(" ")[0] + " [UNPARSED]"


def main():
    print("=" * 70)
    print("STEP 1: COPY RAW DATASET (Kaggle MRI -> data_raw)")
    print("=" * 70)
    print(f"Source: {SOURCE_DIR}")
    print(f"Dest:   {DEST_DIR}\n")

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source not found: {SOURCE_DIR}")

    for dest_class in CLASS_MAPPING.values():
        (DEST_DIR / dest_class).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    unparsed_count = 0

    for source_class, dest_class in CLASS_MAPPING.items():
        source_folder = SOURCE_DIR / source_class
        if not source_folder.exists():
            print(f"  WARNING: {source_folder} not found, skipping")
            continue

        files = sorted(
            f for f in source_folder.iterdir()
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg")
        )

        print(f"{source_class} -> {dest_class}: {len(files)} images")

        for f in files:
            patient_id = extract_patient_id(f.name)
            if "[UNPARSED]" in patient_id:
                unparsed_count += 1
                print(f"  WARNING: could not parse patient id from '{f.name}' "
                      f"-> using fallback '{patient_id}'")

            dest_path = DEST_DIR / dest_class / f.name
            shutil.copy2(f, dest_path)

            manifest_rows.append({
                "filename": f.name,
                "class": dest_class,
                "patient_id": patient_id,
                "original_path": str(f),
            })

    with open(MANIFEST_PATH, "w", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=["filename", "class", "patient_id", "original_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    unique_patients = {}
    for row in manifest_rows:
        unique_patients.setdefault(row["class"], set()).add(row["patient_id"])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images copied: {len(manifest_rows)}")
    for cls, patients in unique_patients.items():
        n_images = sum(1 for r in manifest_rows if r["class"] == cls)
        print(f"  {cls:15} : {n_images} images across {len(patients)} patients")
        print(f"    Patients: {sorted(patients)}")

    if unparsed_count:
        print(f"\nWARNING: {unparsed_count} filenames had unparseable patient IDs - "
              f"review the manifest and fix manually before splitting.")

    print(f"\nExcluded classes (not copied): {EXCLUDED_CLASSES}")
    print(f"Manifest saved: {MANIFEST_PATH}")
    print("\nNext: run 02_remove_exact_duplicates.py")


if __name__ == "__main__":
    main()