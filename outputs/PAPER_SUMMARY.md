# Brain Hemorrhage Detection - Paper Asset Summary

Generated: 2026-09-04 11:44

## Dataset

- Source: Kaggle "Brain Stroke MRI Images" (mitangshu11/brain-stroke-mri-images)
- Modality: Multi-sequence MRI (DWI, GRE, SWI, T2/T2-Flair)
- Classes used: Hemorrhagic (from Haemorrhagic), NonHemorrhagic (from Normal)
- Ischemic class excluded (scope: binary hemorrhage detection)
- Raw: 186 Hemorrhagic / 399 NonHemorrhagic (585 total) across 11 patients
- After exact-duplicate removal: 186 / 387 (573 total)
- After near-duplicate removal: 69 / 188 (257 total)
- Patient-level 5-fold cross-validation used (no image-level or patient-level leakage between train/valid, verified via MD5 content hash + patient ID checks)

## Step 6: K-Fold Cross-Validation Results (PRIMARY result to report)

| class | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std |
|---|---|---|---|---|---|---|
| Hemorrhagic | 0.69406162464986 | 0.23192265096520007 | 0.6366378719319896 | 0.14984957836492463 | 0.6503737973967176 | 0.1609651626053441 |
| NonHemorrhagic | 0.863953823953824 | 0.048072388391603126 | 0.8668103448275861 | 0.1057453304445187 | 0.8630831426815255 | 0.06872214434967718 |


Per-fold accuracy/macro-F1:

| fold | accuracy | macro_f1 |
|---|---|---|
| 0 | 0.8765432098765432 | 0.8052884615384616 |
| 1 | 0.8055555555555556 | 0.7855319148936171 |
| 2 | 0.8809523809523809 | 0.8299595141700404 |
| 3 | 0.6226415094339622 | 0.5350877192982456 |
| 4 | 0.8444444444444444 | 0.8277747402952433 |


## Step 7: Final Deployment Model Training History

*(Training accuracy only - this model trained on 100% of data with no held-out set. Do NOT report this as model performance - it is provided only to document the training process.)*

| epoch | train_loss | train_accuracy_pct |
|---|---|---|
| 1 | 0.6572 | 62.26 |
| 2 | 0.6517 | 61.09 |
| 3 | 0.5425 | 76.26 |
| 4 | 0.5246 | 77.43 |
| 5 | 0.5096 | 75.1 |
| 6 | 0.4879 | 78.99 |
| 7 | 0.4569 | 79.77 |
| 8 | 0.488 | 80.16 |
| 9 | 0.4882 | 79.38 |
| 10 | 0.4811 | 80.16 |
| 11 | 0.509 | 75.88 |
| 12 | 0.4658 | 79.38 |
| 13 | 0.4595 | 79.38 |
| 14 | 0.4807 | 78.99 |
| 15 | 0.4598 | 80.93 |


## Step 8: Out-of-Fold Confusion Matrix / ROC / Classification Report

*(Cross-checks step 6's result using an independent aggregation method - both should be reported together as corroborating evidence.)*

| class | precision | recall | f1-score | support |
|---|---|---|---|---|
| Hemorrhagic | 0.6471 | 0.6377 | 0.6423 | 69 |
| NonHemorrhagic | 0.8677 | 0.8723 | 0.87 | 188 |
| accuracy |  |  | 0.8093 | 257 |
| macro avg | 0.7574 | 0.755 | 0.7562 | 257 |
| weighted avg | 0.8085 | 0.8093 | 0.8089 | 257 |


- Confusion matrix image: `outputs/confusion_matrix/confusion_matrix.png`

- ROC curve image: `outputs/roc_curve/roc_curve.png`


## Explainability Figures (Step 9)

- Grad-CAM samples: `outputs/gradcam/`
- Grad-CAM++ samples: `outputs/gradcam_pro/`
- LIME samples: `outputs/lime/`


## Known Limitations (recommended to state explicitly in the paper)

- Small patient count (5 Hemorrhagic, 6 NonHemorrhagic patients) - k-fold std deviation (see table above) reflects this; results should be read as a proof-of-concept, not a clinically validated model.
- Ischemic stroke cases excluded from this binary formulation.
- Backbone frozen during training (only classifier head trained) due to limited data - full fine-tuning was not attempted at this scale.


## Model & Deployment

- Architecture: EfficientNet-B0 (ImageNet-pretrained backbone, frozen; classifier head fine-tuned)
- Deployment model path: `models/EfficientNetB0/best_efficientnetb0.pth`
- Explainability: Grad-CAM, Grad-CAM++ (custom region-highlighting variant), LIME (smoothed single-region overlay)
