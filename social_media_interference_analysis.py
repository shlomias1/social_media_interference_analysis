import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.anova import AnovaRM
import matplotlib.pyplot as plt
import patsy
import re # For grepl equivalent
import sys # For error handling

def error_bar(ax, x, y, upper, lower=None, length=0.05, **kwargs):
    if lower is None:
        lower = upper
    if not (len(x) == len(y) == len(lower) == len(upper)):
        raise ValueError("vectors must be same length")
    capsize = kwargs.pop('capsize', 5) # Use a default capsize, allow override
    ax.errorbar(x, y, yerr=upper, fmt='none', capsize=capsize, **kwargs)
    ax.plot(x, y, **kwargs)
    ax.errorbar(x, y, yerr=lower, fmt='none', capsize=capsize, **kwargs)
    ax.plot(x, y, **kwargs)
    return ax

# Data loading
try:
    combined = pd.read_csv("combined.csv")
except FileNotFoundError:
    print("Error: combined.csv not found at the specified path.")
    print("Please ensure 'combined.csv' exists.")
    print("Exiting script.")
    sys.exit(1) # Exit if the data file is not found

# Data filtering
s1_long = combined.query('wave == 1').copy() 
s2_long = combined.query('wave == 2').copy()

# Item-level data processing for s1
share_dat_s1 = s1_long.query("Condition == 2 and engagement_type == 'Sharing'").groupby('item').agg(
    share_rate=('response', 'mean'),
    v=('real', 'mean'),
    response_std=('response', 'std'), # Get std separately
    response_n=('response', 'size') # Get n separately
).reset_index() # Convert grouped output back to DataFrame
# Calculate se manually
# Handle case where response_n is 0 or 1 (std is NaN or 0)
share_dat_s1['se'] = 1.96 * share_dat_s1['response_std'] / np.sqrt(share_dat_s1['response_n'].replace(0, np.nan))
share_dat_s1['se'] = share_dat_s1['se'].fillna(0) # Replace NaN se with 0 where n < 2

acc_dat_s1 = s1_long.query("Condition == 1 and engagement_type == 'Accuracy'").groupby('item').agg(
    baseline_acc=('response', 'mean'),
    response_std=('response', 'std'),
    response_n=('response', 'size')
).reset_index()
acc_dat_s1['se'] = 1.96 * acc_dat_s1['response_std'] / np.sqrt(acc_dat_s1['response_n'].replace(0, np.nan))
acc_dat_s1['se'] = acc_dat_s1['se'].fillna(0)

sacc_dat_s1 = s1_long.query("Condition == 4 and engagement_type == 'Accuracy'").groupby('item').agg(
    sacc=('response', 'mean'),
    response_std=('response', 'std'),
    response_n=('response', 'size')
).reset_index()
sacc_dat_s1['se'] = 1.96 * sacc_dat_s1['response_std'] / np.sqrt(sacc_dat_s1['response_n'].replace(0, np.nan))
sacc_dat_s1['se'] = sacc_dat_s1['se'].fillna(0)

share_dat_s1 = share_dat_s1.drop(columns=['response_std', 'response_n', 'se'])
acc_dat_s1 = acc_dat_s1.drop(columns=['response_std', 'response_n', 'se'])
sacc_dat_s1 = sacc_dat_s1.drop(columns=['response_std', 'response_n', 'se'])

item_level_s1 = pd.merge(share_dat_s1, acc_dat_s1, on='item', how='left')
item_level_s1 = pd.merge(item_level_s1, sacc_dat_s1, on='item', how='left')
item_level_s1['delta'] = item_level_s1['sacc'] - item_level_s1['baseline_acc']

# Item-level data processing for s2 (repetitive, translate exactly)
share_dat_s2 = s2_long.query("Condition == 2 and engagement_type == 'Sharing'").groupby('item').agg(
    share_rate=('response', 'mean'),
    v=('real', 'mean'),
    response_std=('response', 'std'),
    response_n=('response', 'size')
).reset_index()
share_dat_s2['se'] = 1.96 * share_dat_s2['response_std'] / np.sqrt(share_dat_s2['response_n'].replace(0, np.nan))
share_dat_s2['se'] = share_dat_s2['se'].fillna(0)

acc_dat_s2 = s2_long.query("Condition == 1 and engagement_type == 'Accuracy'").groupby('item').agg(
    baseline_acc=('response', 'mean'),
    response_std=('response', 'std'),
    response_n=('response', 'size')
).reset_index()
acc_dat_s2['se'] = 1.96 * acc_dat_s2['response_std'] / np.sqrt(acc_dat_s2['response_n'].replace(0, np.nan))
acc_dat_s2['se'] = acc_dat_s2['se'].fillna(0)

sacc_dat_s2 = s2_long.query("Condition == 4 and engagement_type == 'Accuracy'").groupby('item').agg(
    sacc=('response', 'mean'),
    response_std=('response', 'std'),
    response_n=('response', 'size')
).reset_index()
sacc_dat_s2['se'] = 1.96 * sacc_dat_s2['response_std'] / np.sqrt(sacc_dat_s2['response_n'].replace(0, np.nan))
sacc_dat_s2['se'] = sacc_dat_s2['se'].fillna(0)

share_dat_s2 = share_dat_s2.drop(columns=['response_std', 'response_n', 'se'])
acc_dat_s2 = acc_dat_s2.drop(columns=['response_std', 'response_n', 'se'])
sacc_dat_s2 = sacc_dat_s2.drop(columns=['response_std', 'response_n', 'se'])

item_level_s2 = pd.merge(share_dat_s2, acc_dat_s2, on='item', how='left')
item_level_s2 = pd.merge(item_level_s2, sacc_dat_s2, on='item', how='left')
item_level_s2['delta'] = item_level_s2['sacc'] - item_level_s2['baseline_acc']

# Combine item-level data
item_level = pd.concat([item_level_s1, item_level_s2], ignore_index=True)

# Create wave variable in item_level
item_level['wave'] = item_level['item'].str.contains('w1', na=False).astype(float) - 0.5 # Handle potential NA in item

# --- Statistical Modeling ---
# Helper function to fit GLM with clustered standard errors and return summary
def fit_glm_clustered(formula, data, clusters_cols, family=sm.families.Gaussian(), hc_type='HC3'):
    # Ensure clusters_cols are in data
    if not all(col in data.columns for col in clusters_cols):
        raise ValueError(f"Cluster columns {clusters_cols} not found in data.")
    # Create cluster groups
    # Combine cluster columns into a single Series of tuples
    # Ensure the index of clusters matches the index of the data used for fitting
    data[clusters_cols] = data[clusters_cols].astype(str)  # Force string for all cluster cols
    clusters = pd.Series([tuple(x) for x in data[clusters_cols].values], index=data.index)
    # Handle scaling in the data before patsy
    data_scaled = data.copy()
    scaled_vars = re.findall(r'scale\((.*?)\)', formula)
    for var in scaled_vars:
        if var in data_scaled.columns:
            mean = data_scaled[var].mean()
            std = data_scaled[var].std()
            if std == 0:
                 # Handle case where std is 0 (all values are the same)
                 # Patsy might handle this as a constant, but scaling to 0 is also an option.
                 # Let's scale to 0 if std is 0.
                 data_scaled[var] = 0
                 print(f"Warning: Standard deviation of '{var}' is 0. Scaling resulted in 0.")
            else:
                data_scaled[var] = (data_scaled[var] - mean) / std
            # Replace scale(var) with var in formula for patsy
            formula = formula.replace(f'scale({var})', var)
        else:
             print(f"Warning: Variable '{var}' in scale() not found in data.")
    try:
        y, X = patsy.dmatrices(formula, data=data_scaled, return_type='dataframe')
        clusters = clusters.loc[y.index]
    except patsy.PatsyError as e:
        print(f"Error parsing formula '{formula}': {e}")
        raise e

    # Fit GLM
    # Ensure data types are float for statsmodels
    y = y.astype(float)
    X = X.astype(float)

    model = sm.GLM(y, X, family=family)
    fit = model.fit(cov_type='cluster', cov_kwds={'groups': clusters, 'use_correction': True})
    return fit

# Helper function to fit GLM with clustered standard errors using a separate cluster dataframe
def fit_glm_clustered_with_clusters_df(formula, data, clusters_df, family=sm.families.Gaussian(), hc_type='HC3'):
    if not data.index.equals(clusters_df.index):
         raise ValueError("Index of data and clusters_df must match.")
    clusters = pd.Series([tuple(x) for x in clusters_df.values], index=data.index)
    data_scaled = data.copy()
    scaled_vars = re.findall(r'scale\((.*?)\)', formula)
    for var in scaled_vars:
        if var in data_scaled.columns:
            mean = data_scaled[var].mean()
            std = data_scaled[var].std()
            if std == 0:
                 data_scaled[var] = 0
                 print(f"Warning: Standard deviation of '{var}' is 0. Scaling resulted in 0.")
            else:
                data_scaled[var] = (data_scaled[var] - mean) / std
            formula = formula.replace(f'scale({var})', var)
        else:
             print(f"Warning: Variable '{var}' in scale() not found in data.")
    try:
        y, X = patsy.dmatrices(formula, data=data_scaled, return_type='dataframe')
        clusters = clusters.loc[y.index]
    except patsy.PatsyError as e:
        print(f"Error parsing formula '{formula}': {e}")
        raise e
    # Fit GLM
    y = y.astype(float)
    X = X.astype(float)
    model = sm.GLM(y, X, family=family)
    fit = model.fit()
    robust_cov = cov_cluster(fit, clusters)
    fit.cov_params_default = robust_cov
    return fit

# Helper function for linear hypothesis testing
def run_linear_hypothesis(results, hypothesis):
    try:
        test_result = results.wald_test(hypothesis)
        print(f"\nLinear hypothesis test: {hypothesis}")
        print(test_result)
        return test_result
    except Exception as e:
        print(f"\nError running linear hypothesis test '{hypothesis}': {e}")
        print("Available coefficients in the model:")
        print(results.params.index.tolist())
        return None


# Helper function to print xtable equivalent
def print_xtable(results, digits=3):
    summary_obj = results.summary2()
    if summary_obj is None:
        print("\n--- xtable output (LaTeX) ---")
        print("Could not generate model summary.")
        print("-----------------------------")
        return

    summary_df = summary_obj.tables[1] # Get the coefficients table

    # Rename columns to match R's coeftest output structure (Estimate, Std. Error, t/z value, Pr(>|t/z|))
    # Ensure columns exist before selecting/renaming
    col_map = {
        'Coef.': 'Estimate',
        'Std.Err.': 'Std. Error',
        'z': 'z value', # For GLM
        't': 't value', # For OLS/other models
        'P>|z|': 'Pr(>|z|)', # For GLM
        'P>|t|': 'Pr(>|t|)'  # For OLS/other models
    }
    summary_df = summary_df.rename(columns=col_map)

    # Select and reorder columns to match R's coeftest output structure
    # Prioritize z-test columns for GLM, fall back to t-test columns
    cols_order_glm = ['Estimate', 'Std. Error', 'z value', 'Pr(>|z|)']
    cols_order_ols = ['Estimate', 'Std. Error', 't value', 'Pr(>|t|)']

    cols_to_print = [col for col in cols_order_glm if col in summary_df.columns]
    if not cols_to_print: # If z-test columns not found, try t-test columns
         cols_to_print = [col for col in cols_order_ols if col in summary_df.columns]

    if not cols_to_print:
         print("\n--- xtable output (LaTeX) ---")
         print("Could not find standard coefficient columns in summary table.")
         print("Available columns:", summary_df.columns.tolist())
         print("-----------------------------")
         return

    summary_df_ordered = summary_df[cols_to_print]

    # Use pandas to_latex for xtable equivalent
    # float_format ensures correct number of digits
    # na_rep handles potential NaN values
    # column_format helps align columns in LaTeX
    latex_table = summary_df_ordered.to_latex(
        float_format=f'%.{digits}f',
        na_rep='',
        column_format='l' + 'r' * len(cols_to_print)
    )
    print("\n--- xtable output (LaTeX) ---")
    print(latex_table)
    print("-----------------------------")


# Helper function to get bar positions for grouped bar plot (matplotlib)
def get_grouped_bar_positions(n_groups, n_bars_per_group, bar_width, group_spacing=0.2):
    # Calculate the total width of each group of bars plus spacing
    total_group_width_with_spacing = n_bars_per_group * bar_width + group_spacing
    # Calculate the x-coordinates for the left edge of each group
    group_left_edges = np.arange(n_groups) * total_group_width_with_spacing
    bar_positions = group_left_edges[:, np.newaxis] + np.arange(n_bars_per_group) * bar_width + bar_width / 2
    return bar_positions.flatten() # Return as a flat array of positions

def print_table(model):
    df = model.summary2().tables[1]
    df = df.rename(columns={
        'Coef.': 'Estimate',
        'Std.Err.': 'Std. Error',
        'z': 'z value',
        't': 't value',
        'P>|z|': 'Pr(>|z|)',
        'P>|t|': 'Pr(>|t|)'
    })
    print(df)
    return df

# table s1
print("\n--- Table S1 ---")
acc_data_s1_model = combined.query('engagement_type == "Accuracy"').copy()
# Need to manually scale 'wave' before passing to the formula/patsy
# The fit_glm_clustered helper handles scaling now.
formula_s1 = 'response ~ veracity*scale(wave)*(both+order)'
clusters_cols_s1 = ['ResponseId', 'item']
model_acc_s1 = fit_glm_clustered(formula_s1, acc_data_s1_model, clusters_cols_s1)
print_table(model_acc_s1) 

# table s2
print("\n--- Table S2 ---")
acc_data_wave1 = combined.query('engagement_type == "Accuracy" and wave == 1').copy()
formula_s2 = 'response ~ veracity*(both+order)'
clusters_cols_s2 = ['ResponseId', 'item']
clusters_data_s2 = acc_data_wave1[['ResponseId', 'item']].copy()
# Fit the model for the table (with clustered SE)
model_acc_w1 = fit_glm_clustered_with_clusters_df(formula_s2, acc_data_wave1, clusters_data_s2)
print_table(model_acc_w1) 

# Fit the UNCLUSTERED model for linearHypothesis, as done in R code
# Need to use patsy to get the design matrix correctly
y_s2_unclustered, X_s2_unclustered = patsy.dmatrices(formula_s2, data=acc_data_wave1, return_type='dataframe')
model_s2_unclustered = sm.GLM(y_s2_unclustered.astype(float), X_s2_unclustered.astype(float), family=sm.families.Gaussian()).fit()

# Run linear hypotheses using the UNCLUSTERED model results as in R code
# Coefficient names should be checked if errors occur.
run_linear_hypothesis(model_s2_unclustered, "veracity:both + 0.5*veracity:order = 0")
run_linear_hypothesis(model_s2_unclustered, "veracity:both - 0.5*veracity:order = 0")

# table s3
print("\n--- Table S3 ---")
acc_data_wave2 = combined.query('engagement_type == "Accuracy" and wave == 2').copy()
# Need to manually scale 'DemRep_C' for the unclustered model used in linearHypothesis
acc_data_wave2_scaled_for_unclustered = acc_data_wave2.copy()
mean_demrep = acc_data_wave2_scaled_for_unclustered['DemRep_C'].mean()
std_demrep = acc_data_wave2_scaled_for_unclustered['DemRep_C'].std()
if std_demrep != 0:
    acc_data_wave2_scaled_for_unclustered['DemRep_C_scaled'] = (acc_data_wave2_scaled_for_unclustered['DemRep_C'] - mean_demrep) / std_demrep
else:
    acc_data_wave2_scaled_for_unclustered['DemRep_C_scaled'] = 0
    print("Warning: Standard deviation of 'DemRep_C' is 0 in Table S3 data.")

formula_s3_patsy_unclustered = 'response ~ (both+order)*veracity*DemRep_C_scaled*concord'

# Fit unclustered model for linearHypothesis
y_s3_unclustered, X_s3_unclustered = patsy.dmatrices(formula_s3_patsy_unclustered, data=acc_data_wave2_scaled_for_unclustered, return_type='dataframe')
model_s3_unclustered = sm.GLM(y_s3_unclustered.astype(float), X_s3_unclustered.astype(float), family=sm.families.Gaussian()).fit()

# Fit and get clustered results for the table
# The helper function handles scaling internally now, so use original data and formula
formula_s3 = 'response ~ (both+order)*veracity*scale(DemRep_C)*concord'
clusters_cols_s3 = ['ResponseId', 'item']
required_columns = ['response', 'both', 'order', 'veracity', 'DemRep_C', 'concord'] + clusters_cols_s3
acc_data_wave2_cleaned = acc_data_wave2.dropna(subset=required_columns).copy()
model_acc_w2 = fit_glm_clustered(formula_s3, acc_data_wave2_cleaned, clusters_cols_s3)
print_table(model_acc_w2) 

# Run linear hypotheses using the UNCLUSTERED model results
# Coefficient names should be checked.
run_linear_hypothesis(model_s3_unclustered, "both:veracity + 0.5*order:veracity = 0")
run_linear_hypothesis(model_s3_unclustered, "both:veracity - 0.5*order:veracity = 0")

# table s4
print("\n--- Table S4 ---")
share_data_s4 = combined.query('engagement_type == "Sharing"').copy()

# Need to manually scale 'wave' for the unclustered model used in linearHypothesis
share_data_s4_scaled_for_unclustered = share_data_s4.copy()
mean_wave = share_data_s4_scaled_for_unclustered['wave'].mean()
std_wave = share_data_s4_scaled_for_unclustered['wave'].std()
if std_wave != 0:
    share_data_s4_scaled_for_unclustered['wave_scaled'] = (share_data_s4_scaled_for_unclustered['wave'] - mean_wave) / std_wave
else:
    share_data_s4_scaled_for_unclustered['wave_scaled'] = 0
    print("Warning: Standard deviation of 'wave' is 0 in Table S4 data.")

formula_s4_patsy_unclustered = 'response ~ (both+order)*veracity*wave_scaled'

# Fit unclustered model for linearHypothesis
y_s4_unclustered, X_s4_unclustered = patsy.dmatrices(formula_s4_patsy_unclustered, data=share_data_s4_scaled_for_unclustered, return_type='dataframe')
model_s4_unclustered = sm.GLM(y_s4_unclustered.astype(float), X_s4_unclustered.astype(float), family=sm.families.Gaussian()).fit()

# Fit and get clustered results for the table
# Helper handles scaling
formula_s4 = 'response ~ (both+order)*veracity*scale(wave)'
clusters_cols_s4 = ['ResponseId', 'item']
model_share_s4 = fit_glm_clustered(formula_s4, share_data_s4, clusters_cols_s4)
print_table(model_share_s4) 
run_linear_hypothesis(model_s4_unclustered, "veracity:order = 0")

# table s5
print("\n--- Table S5 ---")
# Data manipulation within the R code block
engage_data = s2_long.query('engagement_type != "Accuracy"').copy()
engage_data['liking'] = (engage_data['engagement_type'] == 'Liking').astype(int)
engage_data['commenting'] = (engage_data['engagement_type'] == 'Commenting').astype(int)
engage_data['both'] = (engage_data['Condition'] > 2).astype(int)
engage_data['veracity'] = engage_data['real']
# Use np.select for case_when
conditions = [
    engage_data['Condition'] == 3,
    engage_data['Condition'] == 4
]
choices = [-0.5, 0.5]
engage_data['order'] = np.select(conditions, choices, default=0)

formula_s5 = 'response ~ (both+order)*veracity*(liking + commenting)'
clusters_cols_s5 = ['ResponseId', 'item']

engage_model = fit_glm_clustered(formula_s5, engage_data, clusters_cols_s5)
print_table(engage_model)

# table s6
print("\n--- Table S6 ---")
formula_s6 = 'delta ~ share_rate + C(v) + wave'

# Need to handle potential NA values in item_level for these columns
item_level_s6 = item_level[['delta', 'share_rate', 'v', 'wave']].dropna().copy()

# Fit standard GLM
y_s6, X_s6 = patsy.dmatrices(formula_s6, data=item_level_s6, return_type='dataframe')
model_s6 = sm.GLM(y_s6.astype(float), X_s6.astype(float), family=sm.families.Gaussian()).fit() # Default Gaussian family matches R glm default

# coeftest in R just prints the standard coefficient table for a standard fit
# statsmodels summary() provides this.
print_table(model_s6)

# table s7
print("\n--- Table S7 ---")
# Standard GLM, no clustering
formula_s7 = 'delta ~ baseline_acc + C(v) + wave'

# Need to handle potential NA values
item_level_s7 = item_level[['delta', 'baseline_acc', 'v', 'wave']].dropna().copy()

# Fit standard GLM
y_s7, X_s7 = patsy.dmatrices(formula_s7, data=item_level_s7, return_type='dataframe')
model_s7 = sm.GLM(y_s7.astype(float), X_s7.astype(float), family=sm.families.Gaussian()).fit()
print_table(model_s7)

# table s8
print("\n--- Table S8 ---")
acc_data_s8 = combined.query('engagement_type == "Accuracy"').copy()
# Need to scale 'wave' and 'republican'
formula_s8 = 'response ~ veracity*scale(wave)*(both+order)*scale(republican)'
clusters_cols_s8 = ['ResponseId', 'item']

model_acc_s8 = fit_glm_clustered(formula_s8, acc_data_s8, clusters_cols_s8)
print_table(model_acc_s8)

# table s8b
print("\n--- Table S8b ---")
acc_data_s8b = combined.query('engagement_type == "Accuracy"').copy()
acc_data_s8b['republican_binary'] = (acc_data_s8b['republican'] > 3).astype(int)
# Need to scale 'wave' and 'republican_binary'
formula_s8b = 'response ~ veracity*scale(wave)*(both+order)*scale(republican_binary)'
clusters_cols_s8b = ['ResponseId', 'item']

model_acc_s8b = fit_glm_clustered(formula_s8b, acc_data_s8b, clusters_cols_s8b)
print_table(model_acc_s8b)

# table s9
print("\n--- Table S9 ---")
acc_data_wave1_demrep = combined.query('engagement_type == "Accuracy" and wave == 1').copy()
# Need to scale 'republican'
formula_s9 = 'response ~ veracity*(both+order)*scale(republican)'
acc_data_wave1_full = combined.query('wave == 1').copy() # Get the dataframe R used for clusters
# Filter cluster data to match the index of the model data
clusters_data_s9 = acc_data_wave1_full[['ResponseId', 'item']].loc[acc_data_wave1_demrep.index].copy()

# Use the helper function with the filtered cluster data
model_acc_w1_demrep = fit_glm_clustered_with_clusters_df(formula_s9, acc_data_wave1_demrep, clusters_data_s9)
print_table(model_acc_w1_demrep)

# table s10
print("\n--- Table S10 ---")
acc_data_wave2_demrep = combined.query('engagement_type == "Accuracy" and wave == 2').copy()
# Need to scale 'republican'
formula_s10 = 'response ~ veracity*(both+order)*scale(republican)'
clusters_cols_s10 = ['ResponseId', 'item']

# Use the original helper function that derives clusters from the model data
model_acc_w2_demrep = fit_glm_clustered(formula_s10, acc_data_wave2_demrep, clusters_cols_s10)
print_table(model_acc_w2_demrep)

# table s11
print("\n--- Table S11 ---")
acc_data_wave1_demrep_s11 = combined.query('engagement_type == "Accuracy" and wave == 1').copy()
acc_data_wave1_demrep_s11['republican_binary'] = (acc_data_wave1_demrep_s11['republican'] > 3).astype(int)
# Need to scale 'republican_binary'
formula_s11 = 'response ~ veracity*(both+order)*scale(republican_binary)'
# Filter cluster data to match the index of the model data.
acc_data_wave1_full = combined.query('wave == 1').copy() # Get the dataframe R used for clusters
clusters_data_s11 = acc_data_wave1_full[['ResponseId', 'item']].loc[acc_data_wave1_demrep_s11.index].copy()

# Use the helper function with the filtered cluster data
model_acc_w1_demrep_s11 = fit_glm_clustered_with_clusters_df(formula_s11, acc_data_wave1_demrep_s11, clusters_data_s11)
print_table(model_acc_w1_demrep_s11)

# table s12
print("\n--- Table S12 ---")
acc_data_wave2_demrep_s12 = combined.query('engagement_type == "Accuracy" and wave == 2').copy()
acc_data_wave2_demrep_s12['republican_binary'] = (acc_data_wave2_demrep_s12['republican'] > 3).astype(int)
# Need to scale 'republican_binary'
formula_s12 = 'response ~ veracity*(both+order)*scale(republican_binary)'
clusters_cols_s12 = ['ResponseId', 'item']

# Use the original helper function
model_acc_w2_demrep_s12 = fit_glm_clustered(formula_s12, acc_data_wave2_demrep_s12, clusters_cols_s12)
print_table(model_acc_w2_demrep_s12)

# table s13
print("\n--- Table S13 ---")
share_data_demrep_s13 = combined.query('engagement_type == "Sharing"').copy()
# Need to scale 'republican'
formula_s13 = 'response ~ veracity*(both+order)*scale(republican)*wave'
clusters_cols_s13 = ['ResponseId', 'item']

# Use the original helper function
model_share_data_demrep_s13 = fit_glm_clustered(formula_s13, share_data_demrep_s13, clusters_cols_s13)
print_table(model_share_data_demrep_s13)

# table s14
print("\n--- Table S14 ---")
share_data_demrep_s14 = combined.query('engagement_type == "Sharing"').copy()
# Need to create 'republican_binary' and scale it
share_data_demrep_s14['republican_binary'] = (share_data_demrep_s14['republican'] > 3).astype(int)
formula_s14 = 'response ~ veracity*(both+order)*scale(republican_binary)*wave'
clusters_cols_s14 = ['ResponseId', 'item']

# Use the original helper function
model_share_data_demrep_s14 = fit_glm_clustered(formula_s14, share_data_demrep_s14, clusters_cols_s14)
print_table(model_share_data_demrep_s14)

# --- Descriptive Statistics ---

w1a = combined.query('engagement_type == "Accuracy" and wave == 1').copy()
mean_w1a_response = w1a['response'].mean()
sd_w1a_response = w1a['response'].std()
print(f"\nWave 1 Accuracy Response Mean: {mean_w1a_response:.3f}, SD: {sd_w1a_response:.3f}")

w1s = combined.query('engagement_type == "Sharing" and wave == 1').copy()
mean_w1s_response = w1s['response'].mean()
sd_w1s_response = w1s['response'].std()
print(f"Wave 1 Sharing Response Mean: {mean_w1s_response:.3f}, SD: {sd_w1s_response:.3f}")

w2a = combined.query('engagement_type == "Accuracy" and wave == 2').copy()
mean_w2a_response = w2a['response'].mean()
sd_w2a_response = w2a['response'].std()
print(f"Wave 2 Accuracy Response Mean: {mean_w2a_response:.3f}, SD: {sd_w2a_response:.3f}")

w2s = combined.query('engagement_type == "Sharing" and wave == 2').copy()
mean_w2s_response = w2s['response'].mean()
sd_w2s_response = w2s['response'].std()
print(f"Wave 2 Sharing Response Mean: {mean_w2s_response:.3f}, SD: {sd_w2s_response:.3f}")

w1p = combined.query('wave == 1').groupby('ResponseId').agg(p=('republican', 'mean')).reset_index()
mean_w1p = w1p['p'].mean() # na.rm=T is default for pandas mean/std
sd_w1p = w1p['p'].std()
print(f"Wave 1 Republican Score (User Mean) Mean: {mean_w1p:.3f}, SD: {sd_w1p:.3f}")

w2p = combined.query('wave == 2').groupby('ResponseId').agg(p=('republican', 'mean')).reset_index() # Use w2p for clarity
mean_w2p = w2p['p'].mean()
sd_w2p = w2p['p'].std()
print(f"Wave 2 Republican Score (User Mean) Mean: {mean_w2p:.3f}, SD: {sd_w2p:.3f}")

# --- Plotting Functions ---

def fig2top():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) # Adjust figsize as needed

    # Wave 1 Plot
    plotter_s1 = s1_long.query("engagement_type == 'Accuracy'").groupby(['Condition', 'real']).agg(
        y=('response', 'mean'),
        response_std=('response', 'std'),
        response_n=('response', 'size')
    ).reset_index()
    plotter_s1['se'] = plotter_s1['response_std'] / np.sqrt(plotter_s1['response_n'].replace(0, np.nan))
    plotter_s1['se'] = plotter_s1['se'].fillna(0)

    # Ensure data is sorted by Condition then real for correct matrix formation
    plotter_s1 = plotter_s1.sort_values(['Condition', 'real']).reset_index(drop=True)

    # Arrange data into a matrix shape (real x Condition)
    y_matrix_s1 = plotter_s1.pivot(index='real', columns='Condition', values='y')
    se_matrix_s1 = plotter_s1.pivot(index='real', columns='Condition', values='se')

    # Ensure all expected conditions (1, 2, 4) and real (0, 1) are present, fill missing with NaN
    all_conditions = [1, 2, 4]
    all_real = [0, 1]
    y_matrix_s1 = y_matrix_s1.reindex(index=all_real, columns=all_conditions)
    se_matrix_s1 = se_matrix_s1.reindex(index=all_real, columns=all_conditions)

    # Convert to numpy arrays for plotting
    y_matrix_s1_vals = y_matrix_s1.values
    se_matrix_s1_vals = se_matrix_s1.values

    n_groups_s1 = y_matrix_s1_vals.shape[1] # Number of conditions (3)
    n_bars_per_group_s1 = y_matrix_s1_vals.shape[0] # Number of real values (2)
    bar_width_s1 = 0.35 # Choose a width
    group_spacing_s1 = 0.2 # Choose spacing between groups
    bar_positions_s1 = get_grouped_bar_positions(n_groups_s1, n_bars_per_group_s1, bar_width_s1, group_spacing_s1)

    # Plot bars for each 'real' value across conditions.
    x_pos_r0_s1 = bar_positions_s1[0::n_bars_per_group_s1]
    x_pos_r1_s1 = bar_positions_s1[1::n_bars_per_group_s1]

    bars_r0_s1 = axes[0].bar(x_pos_r0_s1, y_matrix_s1_vals[0, :], bar_width_s1, color='#FFBF00', label='Fake')
    bars_r1_s1 = axes[0].bar(x_pos_r1_s1, y_matrix_s1_vals[1, :], bar_width_s1, color='#2BFF67', label='Real')

    # Plot error bars
    # Error bars are centered at the top of the bar (y value).
    error_bar(axes[0], x_pos_r0_s1, y_matrix_s1_vals[0, :], 1.96 * se_matrix_s1_vals[0, :], capsize=5)
    error_bar(axes[0], x_pos_r1_s1, y_matrix_s1_vals[1, :], 1.96 * se_matrix_s1_vals[1, :], capsize=5)

    # Set plot properties
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Wave 1 (COVID)")
    axes[0].set_ylabel("Perceived Accuracy")
    axes[0].axhline(0, color='black', linewidth=0.8) # abline(h=0)

    # Set x-axis ticks and labels
    # The tick positions should be centered under each group of bars.
    group_centers_s1 = bar_positions_s1.reshape(n_groups_s1, n_bars_per_group_s1).mean(axis=1)
    axes[0].set_xticks(group_centers_s1)
    axes[0].set_xticklabels(["Accuracy", "Accuracy\nSharing","Sharing\nAccuracy"])

    # Need to find appropriate location. (6, 1) in R plot coordinates.
    # Let matplotlib find the best location or specify one.
    axes[0].legend(loc='upper right', fontsize=8, frameon=False) # bty="n" is frameon=False

    # Wave 2 Plot (Repetitive, translate exactly)
    plotter_s2 = s2_long.query("engagement_type == 'Accuracy'").groupby(['Condition', 'real']).agg(
        y=('response', 'mean'),
        response_std=('response', 'std'),
        response_n=('response', 'size')
    ).reset_index()
    plotter_s2['se'] = plotter_s2['response_std'] / np.sqrt(plotter_s2['response_n'].replace(0, np.nan))
    plotter_s2['se'] = plotter_s2['se'].fillna(0)

    plotter_s2 = plotter_s2.sort_values(['Condition', 'real']).reset_index(drop=True)

    # Arrange data into matrix shape (real x Condition)
    y_matrix_s2 = plotter_s2.pivot(index='real', columns='Condition', values='y')
    se_matrix_s2 = plotter_s2.pivot(index='real', columns='Condition', values='se')

    # Ensure all expected conditions (1, 2, 4) and real (0, 1) are present, fill missing with NaN
    y_matrix_s2 = y_matrix_s2.reindex(index=all_real, columns=all_conditions)
    se_matrix_s2 = se_matrix_s2.reindex(index=all_real, columns=all_conditions)

    y_matrix_s2_vals = y_matrix_s2.values
    se_matrix_s2_vals = se_matrix_s2.values

    n_groups_s2 = y_matrix_s2_vals.shape[1]
    n_bars_per_group_s2 = y_matrix_s2_vals.shape[0]
    bar_width_s2 = 0.35 # Use same width
    group_spacing_s2 = 0.2 # Use same spacing
    bar_positions_s2 = get_grouped_bar_positions(n_groups_s2, n_bars_per_group_s2, bar_width_s2, group_spacing_s2)

    x_pos_r0_s2 = bar_positions_s2[0::n_bars_per_group_s2]
    x_pos_r1_s2 = bar_positions_s2[1::n_bars_per_group_s2]

    bars_r0_s2 = axes[1].bar(x_pos_r0_s2, y_matrix_s2_vals[0, :], bar_width_s2, color='#FFBF00', label='Fake')
    bars_r1_s2 = axes[1].bar(x_pos_r1_s2, y_matrix_s2_vals[1, :], bar_width_s2, color='#2BFF67', label='Real')
    error_bar(axes[1], x_pos_r0_s2, y_matrix_s2_vals[0, :], 1.96 * se_matrix_s2_vals[0, :], capsize=5)
    error_bar(axes[1], x_pos_r1_s2, y_matrix_s2_vals[1, :], 1.96 * se_matrix_s2_vals[1, :], capsize=5)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Wave 2 (Politic)")
    axes[1].set_ylabel("Perceived Accuracy")
    axes[1].axhline(0, color='black', linewidth=0.8) # abline(h=0)
    group_centers_s2 = bar_positions_s2.reshape(n_groups_s2, n_bars_per_group_s2).mean(axis=1)
    axes[1].set_xticks(group_centers_s2)
    axes[1].set_xticklabels(["Accuracy", "Accuracy\nSharing","Sharing\nAccuracy"])
    axes[1].legend(loc='upper right', fontsize=8, frameon=False) # bty="n" is frameon=False

fig2top()