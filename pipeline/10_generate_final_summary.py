"""
10_generate_final_summary.py

Reads every report/table generated across the pipeline (steps 1-9)
and compiles them into ONE human-readable summary document -
a single reference while writing the IEEE paper, listing every
number, table, and figure path in one place.

Output:
    outputs/PAPER_SUMMARY.md

Run from project root:
    python pipeline/10_generate_final_summary.py
"""

import csv
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUTS_DIR / "reports"
TABLES_DIR = OUTPUTS_DIR / "tables"

SUMMARY_PATH = OUTPUTS_DIR / "PAPER_SUMMARY.md"


def read_csv_as_markdown_table(path):
    if not path.exists():
        return f"*(file not found: {path.name})*\n"

    with open(path, newline="") as f:
        rows = list(csv.reader(f))

    if not rows:
        return "*(empty)*\n"

    header, data_rows = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in data_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main():
    print("=" * 70)
    print("STEP 10: GENERATE FINAL PAPER SUMMARY")
    print("=" * 70)

    sections = []

    sections.append(f"# Brain Hemorrhage Detection - Paper Asset Summary\n")
    sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    sections.append("## Dataset\n")
    sections.append(
        "- Source: Kaggle \"Brain Stroke MRI Images\" "
        "(mitangshu11/brain-stroke-mri-images)\n"
        "- Modality: Multi-sequence MRI (DWI, GRE, SWI, T2/T2-Flair)\n"
        "- Classes used: Hemorrhagic (from Haemorrhagic), NonHemorrhagic (from Normal)\n"
        "- Ischemic class excluded (scope: binary hemorrhage detection)\n"
        "- Raw: 186 Hemorrhagic / 399 NonHemorrhagic (585 total) across 11 patients\n"
        "- After exact-duplicate removal: 186 / 387 (573 total)\n"
        "- After near-duplicate removal: 69 / 188 (257 total)\n"
        "- Patient-level 5-fold cross-validation used (no image-level or "
        "patient-level leakage between train/valid, verified via MD5 content "
        "hash + patient ID checks)\n"
    )

    sections.append("## Step 6: K-Fold Cross-Validation Results (PRIMARY result to report)\n")
    sections.append(read_csv_as_markdown_table(REPORT_DIR / "kfold_per_class_summary.csv"))
    sections.append("\nPer-fold accuracy/macro-F1:\n")
    sections.append(read_csv_as_markdown_table(REPORT_DIR / "kfold_per_fold_results.csv"))

    sections.append("\n## Step 7: Final Deployment Model Training History\n")
    sections.append(
        "*(Training accuracy only - this model trained on 100% of data with "
        "no held-out set. Do NOT report this as model performance - it is "
        "provided only to document the training process.)*\n"
    )
    sections.append(read_csv_as_markdown_table(REPORT_DIR / "final_model_training_history.csv"))

    sections.append("\n## Step 8: Out-of-Fold Confusion Matrix / ROC / Classification Report\n")
    sections.append(
        "*(Cross-checks step 6's result using an independent aggregation method - "
        "both should be reported together as corroborating evidence.)*\n"
    )
    sections.append(read_csv_as_markdown_table(TABLES_DIR / "classification_report.csv"))
    sections.append(f"\n- Confusion matrix image: `outputs/confusion_matrix/confusion_matrix.png`\n")
    sections.append(f"- ROC curve image: `outputs/roc_curve/roc_curve.png`\n")

    sections.append("\n## Explainability Figures (Step 9)\n")
    sections.append(
        "- Grad-CAM samples: `outputs/gradcam/`\n"
        "- Grad-CAM++ samples: `outputs/gradcam_pro/`\n"
        "- LIME samples: `outputs/lime/`\n"
    )

    sections.append("\n## Known Limitations (recommended to state explicitly in the paper)\n")
    sections.append(
        "- Small patient count (5 Hemorrhagic, 6 NonHemorrhagic patients) - "
        "k-fold std deviation (see table above) reflects this; results should "
        "be read as a proof-of-concept, not a clinically validated model.\n"
        "- Ischemic stroke cases excluded from this binary formulation.\n"
        "- Backbone frozen during training (only classifier head trained) due "
        "to limited data - full fine-tuning was not attempted at this scale.\n"
    )

    sections.append("\n## Model & Deployment\n")
    sections.append(
        "- Architecture: EfficientNet-B0 (ImageNet-pretrained backbone, frozen; "
        "classifier head fine-tuned)\n"
        "- Deployment model path: `models/EfficientNetB0/best_efficientnetb0.pth`\n"
        "- Explainability: Grad-CAM, Grad-CAM++ (custom region-highlighting variant), "
        "LIME (smoothed single-region overlay)\n"
    )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))

    print(f"\nSummary written to: {SUMMARY_PATH}")
    print("Open this file for a single reference covering every number/table/figure")
    print("path needed while writing the IEEE paper.")


if __name__ == "__main__":
    main()