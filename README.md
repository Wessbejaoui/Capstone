# Capstone Project — Predicting Diabetes Risk

Comparing Classifiers on a Clinical Screening Dataset

# Overview

This project develops and compares several machine learning models to predict diabetes risk based on demographics and common clinical indicators. The dataset contains approximately 100,000 patient records × 17 features after cleaning.

  We evaluated the following classifiers:
* Logistic Regression (LR)
* Decision Tree (DT)
* Support Vector Machine (SVM) with probability calibration
* XGBoost (Gradient-Boosted Trees)

Because only about 8.5% of records are positive (diabetic), this is an imbalanced classification problem. We use AUPRC (Area Under the Precision–Recall Curve) as the primary metric, since it’s more informative than accuracy for rare events.
We also report AUC-ROC, and evaluate models at both:
* the default 0.5 threshold, and
* a policy threshold achieving ≥ 0.60 precision, mirroring real-world screening capacity constraints.

# Business Context

Healthcare systems must prioritize screening and follow-up for individuals most at risk of diabetes. The cost of errors is asymmetric:
* False negatives (missed diabetics) → delayed diagnosis, higher long-term costs.
* False positives → unnecessary follow-ups but relatively low clinical risk.

# Dataset Summary at a glance:

Shape: 100,000 rows × 17 columns
Target prevalence: ~8.5% positive (diabetes)
Here are the key Features and description that would be considered detremental for this study.

| **Feature**                   | **Description**                                                                 |
|-------------------------------|---------------------------------------------------------------------------------|
| `age`                         | Patient age (top-coded near ~80 years)                                         |
| `gender`                      | Recorded biological sex                                                        |
| `bmi`                         | Body Mass Index (BMI); >30 = obese                                             |
| `hbA1c_level`                 | Average 3-month blood sugar (≥6.5% = diabetic range)                           |
| `blood_glucose_level`         | Instant glucose reading                                                        |
| `hypertension`, `heart_disease` | Comorbidity flags (0/1)                                                      |
| `smoking_history`             | Categorical; large “No Info” bucket handled explicitly                         |
| `year`                        | Record year, used for drift checks                                             |
| `smoking_missing`             | Engineered flag marking missing smoking data                                   |

# Data Cleaning & Sanity Checks

* Duplicates removed to prevent inflated performance.
* Invalid ages (<1 year) dropped.
* BMI winsorized (1st–99th percentiles) to reduce outlier impact.
* Smoking “No Info” → NaN + smoking_missing flag (missingness carries signal).
* Imbalance noted (~8.5% positives); accuracy avoided in favor of AUPRC and recall metrics.

All processing steps are modular and reproducible using **df_clean** as the working dataset.

# Operational Implications

Focus on precision–recall trade-offs, not raw accuracy.
Choose thresholds that balance precision and recall according to screening capacity (≥ 0.60 precision target).
Evaluate fairness across demographics to avoid biased screening recommendations.

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

Key fields:

* age — patient age in years (top-coded around ~80 in this data).
* gender — recorded sex (categorical)
* bmi — body-mass index; >30 indicates obesity.
* hbA1c_level — three-month blood-sugar average; ≥6.5% is diabetic range.
* blood_glucose_level — single-time glucose measurement; very high values are red flags.
* hypertension, heart_disease — comorbidity flags (0/1).
* smoking_history — categorical status; a large “No Info” bucket existed and was handled explicitly.
* year — encounter year (helps catch drift/recording artifacts).

*Engineered flag: smoking_missing to preserve signal from missing smoking data.*

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
- Duplicates removed to avoid optimistic metrics from repeated patients.
- Invalid ages (<1) dropped; unlikely for the screening population.
- Imbalance: only ~8.5% positives. Accuracy is misleading, so we evaluate with AUPRC and thresholded precision/recall.

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

# Modeling Approach

Feature Engineering & Preprocessing
* Numerical: age, bmi, hbA1c_level, blood_glucose_level, hypertension, heart_disease, year, smoking_missing.
* Categorical: gender, smoking_history (encoded via OneHotEncoder).
* Split: 75/25 train–test split (stratified by outcome).
* Class imbalance: handled using class_weight='balanced' for LR, SVM, and DT.

Each model was evaluated on both ranking metrics (AUPRC, AUC) and classification thresholds.

# Model Implementations

| **Model**            | **Notes**                                         |
|----------------------|---------------------------------------------------|
| `Logistic Regression` | Baseline interpretable model                      |
| `SVM (Linear)`        | Calibrated probabilities using Platt scaling      |
| `Decision Tree`       | High interpretability, lower precision            |
| `XGBoost`             | Nonlinear, high-performance ensemble              |

  ## Threshold Policy
  - Besides the default 0.5 threshold, we pick a policy threshold that achieves ≥ 0.60 precision, then report the resulting recall and confusion matrix.
  - This mirrors how a clinic would tune alerts to control follow-up load.

## Confusion matrices at chosen thresholds

<img width="603" height="491" alt="image" src="https://github.com/user-attachments/assets/769caeca-56b1-4564-a091-e35245269a9a" />

**Logistic Regression: Counts**

The logistic model correctly identified most non-diabetic cases (21,564 TN) and a majority of diabetics (1,620 TP), achieving solid recall (0.762) and balanced precision (0.600).
This shows it’s effective for screening while maintaining a manageable number of false positives (1,080), fitting well within a clinical workflow.

<img width="607" height="509" alt="image" src="https://github.com/user-attachments/assets/d9e483c2-3c13-461f-bc53-e0c74e77132c" />

**Logistic Regression: Row-normalized**

About 95% of non-diabetic patients are correctly classified, while 76% of diabetics are successfully identified.
This reflects strong generalization, with minor leakage (24% missed diabetics), making it suitable for early-stage population screening.

<img width="603" height="491" alt="image" src="https://github.com/user-attachments/assets/14de2d93-7cce-46ca-9e2a-336a2befc433" />

**Decision Tree: Counts**

The decision tree predicts almost no false positives (0 FP), achieving perfect precision (1.0) but reduced recall (0.666), missing over 700 diabetics.
While its results look clean, it’s overly conservative — flagging too few diabetics, which could lead to under-detection in practice.

<img width="607" height="509" alt="image" src="https://github.com/user-attachments/assets/3e5da7f4-cb17-4013-838f-41c02bf1e8d0" />

**Decision Tree: Row-normalized**

All non-diabetics (100%) are predicted correctly, but only two-thirds of diabetics are caught.
This pattern indicates overfitting to the majority class, prioritizing precision at the cost of recall — an imbalance unsuitable for medical screening contexts.

<img width="603" height="491" alt="image" src="https://github.com/user-attachments/assets/165e13c5-5430-4689-8f0c-348b7795f01d" />

**SVM: Counts**

The linear SVM produces results nearly identical to logistic regression, correctly detecting 1,623 diabetics with comparable precision (0.600) and recall (0.764).
This consistency demonstrates that linear models converge on similar discriminative boundaries, confirming reliability and stability of findings.

<img width="607" height="509" alt="image" src="https://github.com/user-attachments/assets/2a12a6c1-24ac-48f1-837a-5edb8b191430" />

**SVM: Row-normalized**

Roughly 95% of non-diabetics are classified correctly, while 76% of diabetics are detected — a mirror of logistic regression’s balance.
It offers a robust and interpretable trade-off between over-flagging and under-detection, making it a dependable baseline for production use.

# Results Summary

### Default Threshold (0.5)

| **Model**            | **AUC-ROC** | **AUPRC** | **Precision** | **Recall** | **Accuracy** |
|----------------------|:-----------:|:----------:|:--------------:|:-----------:|:-------------:|
| `Logistic Regression` | 0.960 | **0.807** | 0.422 | **0.881** | 0.886 |
| `SVM (Linear)`        | **0.961** | 0.806 | **0.838** | 0.624 | **0.957** |
| `Decision Tree`       | **0.965** | 0.804 | 0.320 | **0.952** | 0.823 |

At 0.5, **SVM** favors precision while **LR** favors recall. **DT** achieves high recall but poor precision.

---

### Policy Threshold (≥ 0.60 Precision)

| **Model**            | **Threshold** | **Precision** | **Recall** | **AUC-ROC** | **AUPRC** |
|----------------------|:-------------:|:--------------:|:-----------:|:-----------:|:----------:|
| `Logistic Regression` | 0.741 | 0.60 | 0.762 | 0.960 | **0.807** |
| `SVM (Linear)`        | 0.211 | 0.60 | **0.764** | **0.961** | 0.806 |
| `Decision Tree`       | 1.000 | **1.00** | 0.666 | 0.965 | 0.804 |

At the operational policy level, **LR** and **SVM** perform nearly identically; **DT** overfits and predicts very few positives.


## XGBoost Results

| **Metric** | **Value** |
|-------------|-----------|
| **Best Params** | `{'subsample': 0.7, 'reg_lambda': 1.0, 'n_estimators': 300, 'min_child_weight': 5, 'max_depth': 6, 'learning_rate': 0.03, 'gamma': 1.0, 'colsample_bytree': 0.7}` |
| **AUPRC** | **0.881** |
| **AUC-ROC** | **0.978** |
| **Precision (0.5)** | 0.454 |
| **Recall (0.5)** | 0.922 |
| **Precision (≥0.60 Policy)** | 0.600 |
| **Recall (≥0.60 Policy)** | 0.857 |

XGBoost outperformed all prior models on both **AUPRC** and **AUC**, identifying more true diabetics while keeping false positives manageable.

# Key Insights

* HbA1c and glucose remain the most predictive features.
* XGBoost delivers the best overall performance (AUPRC=0.881), capturing non-linear relationships missed by linear models.
* Logistic Regression remains the most interpretable model, well-suited as a transparent baseline.
* Decision Tree provides explainability but struggles at required precision levels.
* A threshold policy around ≥0.60 precision yields the best operational trade-off for clinics managing limited screening resources.

# Business Interpretation

* A thresholded LR or XGBoost system can flag high-risk patients with precision-balanced recall.
* This allows clinics to focus follow-ups on the top-risk decile, improving yield without overwhelming staff.
* Before deployment, fairness and demographic bias checks are essential.
* Continuous monitoring for data drift, class prevalence, and model calibration is recommended.


## Summary

| **Model**             | **Strength**                                         | **Recommended Use**                                      |
|------------------------|------------------------------------------------------|-----------------------------------------------------------|
| `Logistic Regression`  | Most interpretable; robust baseline                  | Clinical explainability                                   |
| `SVM (Linear)`         | Strong precision; balanced trade-offs                | Secondary comparator                                      |
| `Decision Tree`        | Interpretable structure                              | Visualization & education                                 |
| `XGBoost`              | Best predictive power (**AUPRC = 0.881**)            | Production-grade model with SHAP explainability           |

**XGBoost** ultimately demonstrates the **highest predictive value** and scalability, while **Logistic Regression** provides a strong interpretable baseline. Together, they form a **complementary pair** suitable for operational deployment in a diabetes risk screening pipeline.




