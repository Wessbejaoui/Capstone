# Capstone — EDA & Baselines
Comparing Classifiers on a Diabetes Risk Dataset

## Overview
We compared three classifiers: Logistic Regression (LR), Decision Tree (DT), and Support Vector Machine (SVM, linear with probability calibration).
The goal is to predict diabetes (binary target) from demographics and routine clinical variables (e.g., hbA1c_level, blood_glucose_level, bmi, age, comorbidities). The dataset contains 100,000 rows × 17 columns after basic cleaning.

Because the positive class rate is ~8.5%, the task is imbalanced. We report AUPRC (area under the precision–recall curve) as the primary metric. We also show AUC-ROC, and we compare models at (a) the default 0.5 threshold and (b) a policy threshold chosen to achieve ≥ 0.60 precision, then report the resulting recall and confusion matrices.

## Business Understanding
Healthcare teams need to prioritize screening and intervention for patients at higher risk of diabetes. The costs of mistakes are asymmetric:
- False negative (miss a true diabetic): delayed diagnosis, downstream complications, higher costs.
- False positive (flag a non-diabetic): unnecessary follow-up/labs, but less harmful than an FN.

## Implications:
- Focus on precision–recall trade-offs instead of accuracy.
- Select thresholds aligned to operational goals (e.g., ≥ 0.60 precision so follow-ups aren’t overwhelmed, while keeping recall high enough to catch true cases).
- Monitor fairness across demographics to avoid systematic disparities.

## Primary Metric: AUPRC (Area Under the Precision–Recall Curve)
AUPRC is used as the primary metric because:
- The dataset is imbalanced (~8.5% positives).
- Accuracy can be misleading (trivially high if you predict “no” for everyone).
- AUPRC reflects how well the model finds true positives without flooding with false positives, which matches the clinical screening objective.
We still report AUC-ROC and show ROC/PR curves for completeness.

## Dataset & Sanity Checks
Shape: 100,000 rows × 17 columns.
Target prevalence: ~8.5% diabetes (imbalanced).

## Sanity Checks & Why They Matter
- Work on df_clean (not df) — preserves raw data, makes the notebook restartable/reproducible.
- Drop duplicates — prevents inflated performance from repeated rows.
- Quantify class balance — with ~8–9% positives, we optimize/report precision/recall and AUPRC (not plain accuracy).

Below is a snapshot of the distribution:
  <img width="1384" height="983" alt="image" src="https://github.com/user-attachments/assets/768b9161-25d3-4e65-bceb-e7768125b6da" />


## Data Cleaning (minimal but meaningful)
- Smoking “No Info” → NaN and add smoking_missing flag (keeps signal that data was absent).
- Age validity: drop age < 1 year (nonsense for this context).
- BMI outliers: winsorize at 1st/99th percentiles to stabilize models/plots.
- Race one-hots already provided; we used them as-is (could collapse to a single categorical if needed).

## EDA Highlights (with plot takeaways)
The takeaway under each plot states: mean/median/min/max and whether diabetics trend higher or lower on that feature.

## HbA1c and Blood Glucose (most predictive)
- Diabetics show much higher hbA1c_level and blood_glucose_level.
- Clear class separation in the histograms and boxplots.

## BMI & Age
- BMI: diabetics trend higher on average; we clipped extremes to avoid undue influence.
- Age: diabetics skew older on average; remove infants (age < 1) as invalid for this study’s scope.

<img width="1384" height="983" alt="image" src="https://github.com/user-attachments/assets/60707e4a-48df-4eb3-b1c5-77e26583ecce" />


## Smoking History
Large “No Info” bucket (now NaN + smoking_missing flag). Missingness itself carries some signal; handled explicitly.

## Correlation Snapshot
- Highest numeric correlation with the target: hbA1c_level, blood_glucose_level; smaller positive associations for BMI and age.
- Correlation is a clue, not proof; the model comparison is the real test.

<img width="905" height="533" alt="image" src="https://github.com/user-attachments/assets/32a3f193-c624-4935-a348-cac876f145d3" />


#### Age by diabetes
- **Older skew for diabetics:** The diabetic curve stays low at young ages and **rises sharply after ~50**, with a broad hump across the **60–70** range.
- **Younger patients rarely diabetic:** Very little diabetic density below ~25.
- **Non-diabetics are broader:** Highest density in the **30–50** range, then tapering.
- **Edge effect at ~80:** A visible spike at the upper bound suggests **top-coding/censoring** (many ages recorded at the max). This is a data quirk, not biology.

<img width="684" height="484" alt="image" src="https://github.com/user-attachments/assets/55ec00d8-5109-4f91-b095-f15d9a2670a3" />

---

#### BMI by diabetes
- **Right-shift for diabetics:** The diabetic distribution is shifted toward **higher BMI**, with a **heavier tail** from **30–45+** (obese range).
- **Shared peak around ~27–28:** Both classes spike near 27–28 (possible **heaping/rounding**). Non-diabetics are **more concentrated** around this peak.
- **Upper-end artifacts:** A small bump near ~49–50 hints at an upper cap; we clipped extreme BMI values to stabilize training.

<img width="683" height="484" alt="image" src="https://github.com/user-attachments/assets/561c5c9a-9650-451d-875c-d0e1f4d61a78" />


### Boxplots: Age, BMI, HbA1c, and Blood Glucose vs. Diabetes

#### Age vs diabetes
- **Higher median & older spread for diabetics:** Median age is much higher for the diabetes group; IQR sits roughly in the 50s–70s.
- **Non-diabetics skew younger:** Median near the 30s–40s with a wider lower range.
- **Low-age outliers in diabetics:** A few very young diabetic cases exist, but they’re rare.
- **Upper bound artifact:** Many values touch the top whisker (~80), suggesting age may be capped/top-coded.

<img width="584" height="384" alt="image" src="https://github.com/user-attachments/assets/c159c1c6-13cf-4fd2-9505-6c7a28c0b397" />

---

#### BMI vs diabetes
- **Diabetics have higher BMI:** Median and IQR shift upward vs. non-diabetics; heavier upper tail (well into the 30s and 40s).
- **Non-diabetics are tighter around the high-20s:** Fewer extreme high-BMI points.

<img width="584" height="384" alt="image" src="https://github.com/user-attachments/assets/a6481202-6325-4508-a4b6-a52289091001" />

---

#### HbA1c level vs diabetes
- **Clear separation:** Diabetics show a markedly higher median HbA1c and wider IQR; non-diabetics cluster lower.
- **Upper whisker extends high (~9%),** reinforcing HbA1c as a top predictor.

<img width="584" height="384" alt="image" src="https://github.com/user-attachments/assets/089be7e2-829a-42f5-9555-d8b4933341f8" />

---

#### Blood glucose level vs diabetes
- **Higher median & broader spread for diabetics:** The diabetic group sits higher with a long upper whisker (into ~300).
- **Non-diabetics center lower with a narrower box.**

<img width="584" height="384" alt="image" src="https://github.com/user-attachments/assets/cc960721-21c3-43ef-bbba-075f1eb9be59" />


## Modeling
  ## Features & Split
  - Numeric: age, bmi, hbA1c_level, blood_glucose_level, hypertension, heart_disease, year, plus engineered smoking_missing.
  - Categorical: gender, smoking_history (one-hot via OneHotEncoder).
  - Split: 75/25 train/test with stratify=y.
  - Imbalance handling: class_weight='balanced' for LR, SVM, DT.
  - Preprocessing:
     - LR/DT: StandardScaler (numeric) + OHE (categorical).
     - SVM: same, but sparse-friendly scaling (with_mean=False), then LinearSVC wrapped in CalibratedClassifierCV (method="sigmoid") to get calibrated probabilities.

  ## Threshold Policy
  - Besides the default 0.5 threshold, we pick a policy threshold that achieves ≥ 0.60 precision, then report the resulting recall and confusion matrix.
  - This mirrors how a clinic would tune alerts to control follow-up load.

# Results
  ## Metrics at default 0.5 threshold
| Model         |  AUC-ROC  |   AUPRC   |  Accuracy | Precision |   Recall  |
| ------------- | :-------: | :-------: | :-------: | :-------: | :-------: |
| Logistic      |   0.960   | **0.807** |   0.886   |   0.422   | **0.881** |
| SVM (linear)  | **0.961** |   0.806   | **0.957** | **0.838** |   0.624   |
| Decision Tree | **0.965** |   0.804   |   0.823   |   0.320   | **0.952** |

Read: LR and SVM are effectively tied on ranking metrics (AUC/AUPRC). At 0.5, SVM trades recall for precision; LR does the opposite. DT maximizes recall but at low precision (too many false positives).

  ## Metrics at a policy threshold (target precision ≥ 0.60)
| Model         | Chosen thr | Precision |   Recall  | AUC-ROC |   AUPRC   |
| ------------- | :--------: | :-------: | :-------: | :-----: | :-------: |
| SVM (linear)  |    0.211   |    0.60   | **0.764** |  0.961  |   0.806   |
| Logistic      |    0.741   |    0.60   |   0.762   |  0.960  | **0.807** |
| Decision Tree |    1.000   |  **1.00** |   0.666   |  0.965  |   0.804   |

## Confusion matrices at chosen thresholds
- Logistic: [[21564, 1080], [505, 1620]]

<img width="607" height="509" alt="image" src="https://github.com/user-attachments/assets/25aa33c4-e213-404d-a575-3bc58307f22d" />

- Decision Tree: [[22644, 0], [709, 1416]]

<img width="607" height="509" alt="image" src="https://github.com/user-attachments/assets/7cec0abf-bd0d-4bac-9ef4-a465fe6ebee4" />

- SVM: [[21562, 1082], [502, 1623]]

<img width="607" height="509" alt="image" src="https://github.com/user-attachments/assets/dd67751f-f0a9-4f71-b48c-70762269845b" />


## Takeaways
- Primary metric (AUPRC): LR is marginally best; SVM is essentially tied.
- Operational point (≥ 0.60 precision): LR and SVM both deliver ~0.76 recall; DT achieves 1.00 precision only by severely cutting recall (predicts very few positives).
- Recommendation: Use Logistic Regression as the primary baseline (slightly better AUPRC, simplest to explain). Keep SVM as a strong secondary comparator. Use Decision Tree for interpretability, not for deployment at this threshold policy.



## Interpreting Model Drivers
- Logistic coefficients / SVM weights: largest positive contributions from hbA1c_level, blood_glucose_level, then age and bmi — aligns with clinical intuition.
- Decision Tree splits: emphasize glucose/HbA1c bands and related risk signals; interpretable but underperform on precision at the required policy point.

# Findings, Business Interpretation, and Next Steps

## Findings
- The dataset contains strong, clinically plausible signals (hbA1c, glucose, BMI, age).

- With precision ≥ 0.60, LR/SVM provide ~0.76 recall, enabling a practical screening workflow with manageable follow-ups.

- DT is interpretable, but not competitive under the chosen precision policy.

## Business Interpretation
- A thresholded LR/SVM can prioritize follow-ups for the top-risk segment, improving yield of true diabetes cases without overwhelming clinic capacity.
- Monitoring fairness and subgroup error rates is necessary before deployment.

## Next Steps
- Probability calibration (keep Platt; consider isotonic) and threshold tuning by capacity (e.g., fix expected daily follow-ups).
- Add feature interactions (e.g., age×BMI) and explore gradient-boosted trees (XGBoost/LightGBM) for potential AUPRC gains.
- Carefully evaluate clinical_notes (TF-IDF or embeddings) only if they add value beyond templated text.
- Expand fairness audits (gender/race/location) and report subgroup PR curves at the chosen threshold.
- Post-deployment monitoring: drift checks and quarterly recalibration.
