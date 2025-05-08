---

# 📊 Social Media Misinformation Analysis

This project analyzes user engagement with misinformation across two experimental waves using a dataset called `combined.csv`. It applies Generalized Linear Models (GLMs) with clustered standard errors, hypothesis testing, and data visualization to examine how variables like content veracity, political affiliation, and engagement types (e.g., accuracy rating, sharing, liking, commenting) interact.

---

## 📁 Contents

* `combined.csv`: Input dataset (not included here).
* `social_media_interference_analysis.py`: Python script for loading data, preprocessing, modeling, and generating results.
* Output tables (e.g., Table S1 to S14): Summary of model results printed to the console.
* Visualizations: Matplotlib bar plots comparing perceived accuracy across conditions.

---

## ⚙️ Requirements

Install dependencies:

```bash
pip install pandas numpy matplotlib statsmodels patsy
```

---

## 🚀 How to Run

Ensure `combined.csv` is in the same directory as the script, then run:

```bash
python social_media_interference_analysis.py
```

---

## 📌 Key Functionalities

### 1. **Data Processing**

* Filters by wave (`wave == 1` and `wave == 2`)
* Computes item-level metrics (mean response, standard errors)
* Merges engagement types (`Accuracy`, `Sharing`)

### 2. **Model Fitting**

* Fits GLMs using formulas such as:

  ```python
  'response ~ veracity*scale(wave)*(both+order)'
  ```
* Supports:

  * Clustered standard errors (`ResponseId`, `item`)
  * Custom variable scaling
  * Binary interaction terms (`republican_binary`, `DemRep_C`)

### 3. **Statistical Tables**

* Tables S1–S14 summarize model estimates, standard errors, z-values, and p-values.
* Tables are printed with coefficient summaries using `summary2()` and custom formatting.

### 4. **Hypothesis Testing**

* `run_linear_hypothesis()` tests constraints on coefficients.

  * Example:

    ```python
    veracity:both + 0.5*veracity:order = 0
    ```

### 5. **Plotting**

* `fig2top()` creates side-by-side bar plots of perceived accuracy by condition (`Accuracy`, `Accuracy→Sharing`, `Sharing→Accuracy`) for real vs. fake content across waves.

---

## 📊 Highlighted Results

### 🧠 Wave Effects (Table S1)

* **Positive effect** of `veracity` (real > fake) on perceived accuracy: `β ≈ 0.27`
* Interaction `veracity:wave` is **negative**: people become less accurate over time.

### 🔁 Engagement Type Interaction (Table S5)

* **Liking**: Decreases perceived accuracy significantly (`β ≈ -0.0236`, p < 0.001)
* **Commenting**: Slightly increases perceived accuracy (`β ≈ 0.013`, p < 0.01)

### 🧭 Political Alignment (Tables S8–S14)

* **Republican** alignment moderates perceived accuracy and sharing likelihood.
* Binary encoding and continuous scaling both explored (`republican_binary`, `scale(republican)`).

### 📐 Descriptive Stats

* Accuracy ratings improve slightly from wave 1 to wave 2.
* Sharing rates remain stable (\~0.325).
* Republican scores slightly increase from Wave 1 to Wave 2.

---

## ⚠️ Warnings & Errors

* Mixed types detected in column 5 (`DtypeWarning`).
* Some linear hypothesis constraints failed due to syntax mismatch (e.g., incorrect use of `:` or coefficient names).
* Ensure all variables referenced in formulas exist and match naming conventions.

---

## 📌 Notes

* The script is translation-aware and adapts a previously R-based analysis pipeline.
* Code uses idiomatic `patsy`/`statsmodels` integration for replicating complex formula behavior.

---

## 📬 Contact

For questions, please reach out to the script author or contributor responsible for the analysis.

---
