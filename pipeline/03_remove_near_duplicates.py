"""
03_remove_near_duplicates.py

Finds near-duplicate images WITHIN each class (never across classes)
using perceptual hash (phash) + SSIM, comparing ALL pairs within each
class. Groups near-duplicates using union-find, keeps ONE representative
per group.

This matters even with patient-level splitting: many images within one
patient's own scan are near-identical adjacent slices. Keeping all of
them lets the model see near-duplicate content many times, which can
still inflate apparent performance and waste training signal.

Input:  data_processed/01_deduplicated/{Hemorrhagic,NonHemorrhagic}/*.png
Output: data_processed/02_near_dedup/{Hemorrhagic,NonHemorrhagic}/*.png
        data_processed/02_near_dedup/manifest.csv
        outputs/reports/near_duplicate_groups.csv

Run from project root:
    python pipeline/03_remove_near_duplicates.py
"""

import csv
from pathlib import Path
from itertools import combinations
from collections import defaultdict

from PIL import Image
import imagehash
import cv2
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "data_processed" / "01_deduplicated"
DEST_DIR = PROJECT_ROOT / "data_processed" / "02_near_dedup"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_MANIFEST = SOURCE_DIR / "manifest.csv"
DEST_MANIFEST = DEST_DIR / "manifest.csv"
GROUPS_REPORT = REPORT_DIR / "near_duplicate_groups.csv"

CLASSES = ["Hemorrhagic", "NonHemorrhagic"]

PHASH_THRESHOLD = 5
SSIM_THRESHOLD = 0.97  # slightly relaxed vs CT project - MRI slices vary more naturally


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def main():
    print("=" * 70)
    print("STEP 3: REMOVE NEAR-DUPLICATES (within class, all-pairs)")
    print("=" * 70)

    manifest_by_filename = {}
    with open(SOURCE_MANIFEST, newline="") as f:
        for row in csv.DictReader(f):
            manifest_by_filename[(row["class"], row["filename"])] = row["patient_id"]

    for cls in CLASSES:
        (DEST_DIR / cls).mkdir(parents=True, exist_ok=True)

    dest_manifest_rows = []
    group_report_rows = []

    for cls in CLASSES:
        source_folder = SOURCE_DIR / cls
        files = sorted(f for f in source_folder.iterdir() if f.is_file())
        print(f"\n{cls}: {len(files)} images")

        hashes = {}
        gray_arrays = {}
        for f in tqdm(files, desc="Hashing"):
            pil_img = Image.open(f).convert("L")
            hashes[f.name] = imagehash.phash(pil_img)
            gray_arrays[f.name] = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)

        names = [f.name for f in files]
        uf = UnionFind(names)

        total_pairs = len(names) * (len(names) - 1) // 2
        near_dup_pairs = 0

        for name1, name2 in tqdm(combinations(names, 2), total=total_pairs, desc="Comparing"):
            hash_distance = hashes[name1] - hashes[name2]

            if hash_distance <= PHASH_THRESHOLD * 3:
                img1, img2 = gray_arrays[name1], gray_arrays[name2]
                if img1.shape != img2.shape:
                    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                similarity = ssim(img1, img2)
            else:
                similarity = 0.0

            if hash_distance <= PHASH_THRESHOLD or similarity >= SSIM_THRESHOLD:
                uf.union(name1, name2)
                near_dup_pairs += 1

        # Group by root; keep first (alphabetically) representative per group
        groups = defaultdict(list)
        for name in names:
            groups[uf.find(name)].append(name)

        kept = 0
        for root, members in groups.items():
            members_sorted = sorted(members)
            representative = members_sorted[0]

            src_path = source_folder / representative
            dest_path = DEST_DIR / cls / representative
            import shutil
            shutil.copy2(src_path, dest_path)
            kept += 1

            dest_manifest_rows.append({
                "filename": representative,
                "class": cls,
                "patient_id": manifest_by_filename.get((cls, representative), "UNKNOWN"),
            })

            if len(members_sorted) > 1:
                group_report_rows.append({
                    "class": cls,
                    "representative_kept": representative,
                    "group_members": "; ".join(members_sorted),
                    "group_size": len(members_sorted),
                })

        print(f"  {len(names)} images -> {kept} groups "
              f"({len(names) - kept} near-duplicates merged, {near_dup_pairs} pairs matched)")

    with open(DEST_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "patient_id"])
        writer.writeheader()
        writer.writerows(dest_manifest_rows)

    with open(GROUPS_REPORT, "w", newline="") as f:
        if group_report_rows:
            writer = csv.DictWriter(f, fieldnames=["class", "representative_kept", "group_members", "group_size"])
            writer.writeheader()
            writer.writerows(group_report_rows)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total images after near-dedup: {len(dest_manifest_rows)}")
    print(f"Groups with >1 member (redundancy found): {len(group_report_rows)}")
    print(f"Manifest: {DEST_MANIFEST}")
    print(f"Groups report: {GROUPS_REPORT}")
    print("\nNext: run 04_build_patient_kfold_splits.py")


if __name__ == "__main__":
    main()