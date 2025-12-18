# Statistical Analysis Reference

Complete reference for descriptive statistics, correlation analysis, distributions, and hypothesis testing setup.

## Descriptive Statistics

### Basic Statistics

```python
import polars as pl
from polars import col

# Quick summary
df.describe()

# Specific statistics
stats = df.select([
    col("value").mean().alias("mean"),
    col("value").std().alias("std"),
    col("value").var().alias("variance"),
    col("value").min().alias("min"),
    col("value").max().alias("max"),
    col("value").median().alias("median"),
    col("value").sum().alias("sum"),
    col("value").count().alias("count"),
])
```

### Multiple Columns

```python
# Stats for all numeric columns
numeric_cols = [c for c in df.columns if df.schema[c] in [pl.Float64, pl.Int64]]

stats = df.select([
    col(c).mean().alias(f"{c}_mean") for c in numeric_cols
] + [
    col(c).std().alias(f"{c}_std") for c in numeric_cols
])
```

### Grouped Statistics

```python
df.group_by("category").agg([
    col("value").mean().alias("mean"),
    col("value").std().alias("std"),
    col("value").min().alias("min"),
    col("value").max().alias("max"),
    col("value").median().alias("median"),
    col("value").count().alias("n"),
])
```

---

## Percentiles and Quantiles

### Basic Percentiles

```python
percentiles = df.select([
    col("value").quantile(0.01).alias("p1"),
    col("value").quantile(0.05).alias("p5"),
    col("value").quantile(0.10).alias("p10"),
    col("value").quantile(0.25).alias("p25"),
    col("value").quantile(0.50).alias("p50"),
    col("value").quantile(0.75).alias("p75"),
    col("value").quantile(0.90).alias("p90"),
    col("value").quantile(0.95).alias("p95"),
    col("value").quantile(0.99).alias("p99"),
])
```

### Interquartile Range (IQR)

```python
# Calculate IQR
q1 = df.select(col("value").quantile(0.25))[0, 0]
q3 = df.select(col("value").quantile(0.75))[0, 0]
iqr = q3 - q1

print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")

# IQR as column
df = df.with_columns([
    (col("value").quantile(0.75) - col("value").quantile(0.25)).alias("iqr")
])
```

---

## Outlier Detection

### IQR Method

```python
# Calculate bounds
q1 = df.select(col("value").quantile(0.25))[0, 0]
q3 = df.select(col("value").quantile(0.75))[0, 0]
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Flag outliers
df = df.with_columns(
    ((col("value") < lower_bound) | (col("value") > upper_bound))
    .alias("is_outlier")
)

# Filter outliers
outliers = df.filter(col("is_outlier"))
non_outliers = df.filter(~col("is_outlier"))
```

### Z-Score Method

```python
# Calculate z-scores
df = df.with_columns(
    ((col("value") - col("value").mean()) / col("value").std())
    .alias("z_score")
)

# Flag outliers (|z| > 3)
df = df.with_columns(
    (col("z_score").abs() > 3).alias("is_outlier")
)
```

### Modified Z-Score (Robust)

```python
# Uses median and MAD (median absolute deviation)
median = df.select(col("value").median())[0, 0]
mad = df.select((col("value") - median).abs().median())[0, 0]

df = df.with_columns(
    (0.6745 * (col("value") - median) / mad).alias("modified_z")
)

# Outliers: |modified_z| > 3.5
df = df.with_columns(
    (col("modified_z").abs() > 3.5).alias("is_outlier")
)
```

---

## Standardization and Normalization

### Z-Score Standardization

```python
# Standard z-score
df = df.with_columns(
    ((col("value") - col("value").mean()) / col("value").std())
    .alias("z_standardized")
)

# Within groups
df = df.with_columns(
    ((col("value") - col("value").mean().over("group")) /
     col("value").std().over("group"))
    .alias("z_within_group")
)
```

### Min-Max Normalization

```python
# Scale to [0, 1]
df = df.with_columns(
    ((col("value") - col("value").min()) /
     (col("value").max() - col("value").min()))
    .alias("normalized")
)

# Scale to custom range [a, b]
a, b = -1, 1
df = df.with_columns(
    (a + (col("value") - col("value").min()) *
     (b - a) / (col("value").max() - col("value").min()))
    .alias("scaled")
)
```

### Robust Scaling (Median/IQR)

```python
df = df.with_columns(
    ((col("value") - col("value").median()) /
     (col("value").quantile(0.75) - col("value").quantile(0.25)))
    .alias("robust_scaled")
)
```

---

## Correlation Analysis

### Pearson Correlation

```python
# All numeric columns
numeric_cols = [c for c in df.columns if df.schema[c] in [pl.Float64, pl.Int64]]
corr_matrix = df.select(numeric_cols).pearson_corr()
print(corr_matrix)

# Specific pair
correlation = df.select(pl.corr("price", "quantity"))
print(f"Correlation: {correlation[0, 0]:.4f}")
```

### Finding Strong Correlations

```python
# Get correlation matrix as dict
numeric_cols = [c for c in df.columns if df.schema[c] in [pl.Float64, pl.Int64]]
corr_df = df.select(numeric_cols).pearson_corr()

# Convert to long format for analysis
import itertools
pairs = []
for i, col1 in enumerate(numeric_cols):
    for j, col2 in enumerate(numeric_cols):
        if i < j:  # Upper triangle only
            corr_val = corr_df[i, j]
            pairs.append({"col1": col1, "col2": col2, "correlation": corr_val})

corr_pairs = pl.DataFrame(pairs).sort("correlation", descending=True)
print("Strongest correlations:")
print(corr_pairs.head(10))
```

### Correlation Within Groups

```python
# Correlation by category
correlations = df.group_by("category").agg(
    pl.corr("x", "y").alias("correlation")
)
```

---

## Distribution Analysis

### Value Counts

```python
# Categorical distribution
value_counts = df.group_by("category").len().sort("len", descending=True)

# With percentages
total = len(df)
value_counts = (
    df.group_by("category")
    .len()
    .with_columns(
        (col("len") / total * 100).alias("percentage")
    )
    .sort("len", descending=True)
)
```

### Histogram Data

```python
# Create histogram bins
n_bins = 20
min_val = df.select(col("value").min())[0, 0]
max_val = df.select(col("value").max())[0, 0]
bin_width = (max_val - min_val) / n_bins

df = df.with_columns(
    ((col("value") - min_val) / bin_width).floor().alias("bin")
)

histogram = df.group_by("bin").len().sort("bin")
```

### Skewness and Kurtosis

```python
# Using scipy (convert to numpy)
from scipy import stats as scipy_stats
import numpy as np

values = df["value"].to_numpy()
skewness = scipy_stats.skew(values)
kurtosis = scipy_stats.kurtosis(values)

print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurtosis:.4f}")

# Interpretation:
# Skewness: 0 = symmetric, >0 = right-skewed, <0 = left-skewed
# Kurtosis: 0 = normal, >0 = heavy tails, <0 = light tails
```

---

## Hypothesis Testing Setup

### Group Comparison Statistics

```python
# Compare groups
group_stats = df.group_by("treatment").agg([
    col("outcome").mean().alias("mean"),
    col("outcome").std().alias("std"),
    col("outcome").count().alias("n"),
    col("outcome").median().alias("median"),
])
print(group_stats)
```

### T-Test Setup

```python
from scipy import stats

# Extract groups
control = df.filter(col("group") == "control")["value"].to_numpy()
treatment = df.filter(col("group") == "treatment")["value"].to_numpy()

# Independent samples t-test
t_stat, p_value = stats.ttest_ind(control, treatment)
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Welch's t-test (unequal variances)
t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=False)

# Paired t-test
before = df["before"].to_numpy()
after = df["after"].to_numpy()
t_stat, p_value = stats.ttest_rel(before, after)
```

### Chi-Square Test Setup

```python
from scipy import stats

# Create contingency table
contingency = df.pivot(
    on="category",
    index="group",
    values="count",
    aggregate_function="sum"
)

# Convert to numpy for scipy
table = contingency.drop("group").to_numpy()
chi2, p_value, dof, expected = stats.chi2_contingency(table)

print(f"Chi-square: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
```

### Proportion Test Setup

```python
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

# Data
successes = [100, 120]  # Successes in each group
totals = [1000, 1000]   # Total observations

# Two-proportion z-test
z_stat, p_value = proportions_ztest(successes, totals)
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Confidence interval for difference
ci_low, ci_high = confint_proportions_2indep(
    successes[1], totals[1],
    successes[0], totals[0]
)
print(f"95% CI for difference: [{ci_low:.4f}, {ci_high:.4f}]")
```

---

## Effect Size Calculations

### Cohen's d

```python
import numpy as np

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

# Usage
control = df.filter(col("group") == "control")["value"].to_numpy()
treatment = df.filter(col("group") == "treatment")["value"].to_numpy()
d = cohens_d(treatment, control)
print(f"Cohen's d: {d:.4f}")

# Interpretation:
# |d| < 0.2: small effect
# 0.2 <= |d| < 0.8: medium effect
# |d| >= 0.8: large effect
```

### Relative Lift

```python
# For conversion rates
control_rate = df.filter(col("group") == "control")["converted"].mean()
treatment_rate = df.filter(col("group") == "treatment")["converted"].mean()

lift = (treatment_rate - control_rate) / control_rate * 100
print(f"Relative lift: {lift:.2f}%")
```

---

## Confidence Intervals

### Mean CI (Normal Approximation)

```python
import numpy as np
from scipy import stats

values = df["value"].to_numpy()
mean = np.mean(values)
std_err = np.std(values, ddof=1) / np.sqrt(len(values))

# 95% CI
ci = stats.t.interval(0.95, len(values)-1, loc=mean, scale=std_err)
print(f"Mean: {mean:.4f}")
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
```

### Bootstrap CI

```python
import numpy as np

def bootstrap_ci(data, stat_func=np.mean, n_bootstrap=10000, ci=0.95):
    """Calculate bootstrap confidence interval."""
    boot_stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(stat_func(sample))

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_stats, alpha * 100)
    upper = np.percentile(boot_stats, (1 - alpha) * 100)
    return lower, upper

# Usage
values = df["value"].to_numpy()
ci_low, ci_high = bootstrap_ci(values, stat_func=np.median)
print(f"95% Bootstrap CI for median: [{ci_low:.4f}, {ci_high:.4f}]")
```

---

## Sample Size and Power

### Sample Size Calculator

```python
from statsmodels.stats.power import TTestIndPower

# Parameters
effect_size = 0.5  # Cohen's d
alpha = 0.05       # Significance level
power = 0.8        # Desired power

# Calculate required sample size (per group)
analysis = TTestIndPower()
n = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power)
print(f"Required sample size per group: {int(np.ceil(n))}")
```

### Power Analysis

```python
from statsmodels.stats.power import TTestIndPower

# Given sample size, what power do we have?
n = 100  # Sample size per group
effect_size = 0.3
alpha = 0.05

analysis = TTestIndPower()
power = analysis.power(effect_size=effect_size, nobs1=n, alpha=alpha)
print(f"Power: {power:.4f}")
```

---

## Summary Statistics Template

```python
def generate_summary(df: pl.DataFrame, value_col: str, group_col: str = None):
    """Generate comprehensive summary statistics."""

    if group_col:
        stats = df.group_by(group_col).agg([
            col(value_col).count().alias("n"),
            col(value_col).mean().alias("mean"),
            col(value_col).std().alias("std"),
            col(value_col).min().alias("min"),
            col(value_col).quantile(0.25).alias("q1"),
            col(value_col).median().alias("median"),
            col(value_col).quantile(0.75).alias("q3"),
            col(value_col).max().alias("max"),
            (col(value_col).quantile(0.75) - col(value_col).quantile(0.25)).alias("iqr"),
        ])
    else:
        stats = df.select([
            col(value_col).count().alias("n"),
            col(value_col).mean().alias("mean"),
            col(value_col).std().alias("std"),
            col(value_col).min().alias("min"),
            col(value_col).quantile(0.25).alias("q1"),
            col(value_col).median().alias("median"),
            col(value_col).quantile(0.75).alias("q3"),
            col(value_col).max().alias("max"),
            (col(value_col).quantile(0.75) - col(value_col).quantile(0.25)).alias("iqr"),
        ])

    return stats

# Usage
summary = generate_summary(df, "value", "category")
print(summary)
```
