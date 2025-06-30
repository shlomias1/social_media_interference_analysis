---

# Social Media Interference Analysis

This project analyzes user engagement with misinformation across two experimental waves using a dataset called `combined.csv`. It applies Generalized Linear Models (GLMs), classification models, hypothesis testing, and rich exploratory data analysis (EDA) to examine how content veracity, political alignment, digital literacy, demographic factors, and engagement types (e.g., accuracy rating, sharing, liking, commenting) affect accuracy perception.

---

## Contents

* `combined.csv`: Input dataset (not included here).
* `social_media_interference_analysis.py`: Python script for loading data, preprocessing, modeling, and generating results.
* Output tables (e.g., Table S1 to S14): Summary of model results printed to the console.
* Visualizations: Exploratory graphs and model-based plots.

---

## ⚙ Requirements

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn statsmodels patsy scikit-learn
```

---

## How to Run

Ensure `combined.csv` is in the same directory as the script, then run:

```bash
python social_media_interference_analysis.py
```

---

## Key Functionalities

### 1. **Data Processing**

* Missing values imputed using:

  * Mode for categorical/boolean (e.g., `education`, `college`)
  * Mean for numeric (e.g., `DemRep_C`, `diglitagg`, `concord`, `republican`)
* Converted `ResponseId` to index, renamed `Unnamed: 0` to `ID`

### 2. **Exploratory Data Analysis (EDA)**

**Engagement Accuracy**

* Users engaging via "accuracy rating" more likely to respond correctly than those liking/sharing/commenting

**Political Orientation**

* U-shaped relationship between `DemRep_C` and accuracy — moderates less accurate than extreme left/right
* Similar pattern with `republican` score — low and high support more accurate than mid-range

**Digital Literacy**

* Higher digital literacy (diglitagg) correlates with more accurate responses
* Boxplot confirms this trend across response groups

**Age and Education**

* Middle-aged (31–50) respondents were the most accurate
* Males slightly more accurate than females
* Higher education correlates with increased response accuracy

**Condition Effects**

* Four experimental conditions tested — Condition 1 had highest accuracy, 3 the lowest

**Cross-variable Analysis**

* Scatter plot showed combined effect of `education` + `diglitagg` on `response`

---

### 3. **Model Fitting**

* **GLM**:

  * Predictors: `veracity`, `wave`, `both`, `order`, `republican`, `DemRep_C`, etc.
  * Clustered standard errors (by `ResponseId`, `item`)

* **Logistic Regression**:

  * Binary target: `response`
  * Predictors: `engagement_type`, `veracity`, `age`, `education`, `diglitagg`, `DemRep_C`
  * Visualizations: coefficient table, confusion matrix, ROC curve (AUC ≈ 0.71)

* **Random Forest**:

  * Improved performance (AUC ≈ 0.74)
  * Feature importance ranked: `age`, `education`, `diglitagg`, etc.

* **Decision Tree**:

  * Visualized tree structure
  * Feature importance plotted with percentage labels

---

### 4. **Statistical Tables**

* Tables S1–S14 summarize model estimates, standard errors, z-values, and p-values.
* Tables are printed using `summary2()` with custom formatting.

---

### 5. **Hypothesis Testing**

* `run_linear_hypothesis()` tests constraints on coefficients.

Example:

```python
veracity:both + 0.5*veracity:order = 0
```

---

### 6. **Visualization Highlights**

* Response accuracy by:

  * Engagement type
  * Age group
  * Gender
  * Political scale (`DemRep_C`, `republican`) — overall and per wave
  * Experimental condition
  * Digital literacy (boxplots)
  * Education distribution
* Feature importances (bar plot with percent annotations)
* Model metrics (confusion matrix, ROC + AUC)

---

## Highlighted Results

* **Accuracy highest** for those rating content (vs. liking/sharing)
* **Political extremes** more accurate than moderates
* **Education, age, and digital literacy** key predictors of accurate discernment
* **Males** more accurate than females
* **Wave effect**: Accuracy drops in wave 2 for some conditions

---

## Warnings & Errors

* Mixed types in column 5 (`DtypeWarning`)
* Linear hypothesis syntax must match model specification

---

## Notes

* This project translates R-based analytical procedures to Python
* Modeling done using `statsmodels`, `sklearn`, and `seaborn` visualizations

---

## Contact

For questions, please reach out to the script author or contributor responsible for the analysis.

---
