# Data Visualization Reference

Complete reference for creating charts and visualizations using matplotlib, seaborn, and plotly.

## Setup and Basics

### Import Patterns

```python
import polars as pl
from polars import col
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Optional: Plotly for interactive charts
import plotly.express as px
import plotly.graph_objects as go

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
```

### Converting Polars to Visualization-Ready

```python
# To pandas (for seaborn/plotly)
pandas_df = df.to_pandas()

# To numpy arrays (for matplotlib)
x = df["x"].to_numpy()
y = df["y"].to_numpy()

# To lists
categories = df["category"].to_list()
values = df["value"].to_list()
```

---

## Line Charts

### Basic Line Plot

```python
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["date"].to_list(), df["value"].to_list())
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.set_title("Time Series")

plt.savefig("line_chart.png", dpi=150, bbox_inches="tight")
plt.close()
```

### Multiple Lines

```python
fig, ax = plt.subplots(figsize=(10, 6))

for category in df["category"].unique().to_list():
    subset = df.filter(col("category") == category)
    ax.plot(
        subset["date"].to_list(),
        subset["value"].to_list(),
        label=category
    )

ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.set_title("Trends by Category")
ax.legend()

plt.savefig("multi_line.png", dpi=150, bbox_inches="tight")
plt.close()
```

### With Confidence Intervals

```python
fig, ax = plt.subplots(figsize=(10, 6))

dates = df["date"].to_list()
values = df["value"].to_numpy()
std = df["std"].to_numpy()

ax.plot(dates, values, label="Mean")
ax.fill_between(dates, values - 2*std, values + 2*std, alpha=0.2, label="95% CI")

ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.legend()

plt.savefig("line_with_ci.png", dpi=150, bbox_inches="tight")
plt.close()
```

---

## Bar Charts

### Vertical Bar Chart

```python
fig, ax = plt.subplots(figsize=(10, 6))

summary = df.group_by("category").agg(col("value").sum()).sort("value", descending=True)
ax.bar(summary["category"].to_list(), summary["value"].to_list())

ax.set_xlabel("Category")
ax.set_ylabel("Total Value")
ax.set_title("Value by Category")
plt.xticks(rotation=45, ha="right")

plt.savefig("bar_chart.png", dpi=150, bbox_inches="tight")
plt.close()
```

### Horizontal Bar Chart

```python
fig, ax = plt.subplots(figsize=(10, 6))

summary = df.group_by("category").agg(col("value").sum()).sort("value")
ax.barh(summary["category"].to_list(), summary["value"].to_list())

ax.set_xlabel("Total Value")
ax.set_ylabel("Category")
ax.set_title("Value by Category")

plt.savefig("horizontal_bar.png", dpi=150, bbox_inches="tight")
plt.close()
```

### Grouped Bar Chart

```python
fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data
pivot_df = df.pivot(on="subcategory", index="category", values="value")
categories = pivot_df["category"].to_list()
subcategories = [c for c in pivot_df.columns if c != "category"]

x = np.arange(len(categories))
width = 0.8 / len(subcategories)

for i, subcat in enumerate(subcategories):
    offset = (i - len(subcategories)/2 + 0.5) * width
    ax.bar(x + offset, pivot_df[subcat].to_list(), width, label=subcat)

ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha="right")
ax.set_xlabel("Category")
ax.set_ylabel("Value")
ax.legend()

plt.savefig("grouped_bar.png", dpi=150, bbox_inches="tight")
plt.close()
```

### Stacked Bar Chart

```python
fig, ax = plt.subplots(figsize=(12, 6))

pivot_df = df.pivot(on="subcategory", index="category", values="value")
categories = pivot_df["category"].to_list()
subcategories = [c for c in pivot_df.columns if c != "category"]

bottom = np.zeros(len(categories))
for subcat in subcategories:
    values = pivot_df[subcat].to_numpy()
    ax.bar(categories, values, bottom=bottom, label=subcat)
    bottom += values

ax.set_xlabel("Category")
ax.set_ylabel("Value")
ax.legend()
plt.xticks(rotation=45, ha="right")

plt.savefig("stacked_bar.png", dpi=150, bbox_inches="tight")
plt.close()
```

---

## Scatter Plots

### Basic Scatter

```python
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(df["x"].to_numpy(), df["y"].to_numpy(), alpha=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Scatter Plot")

plt.savefig("scatter.png", dpi=150, bbox_inches="tight")
plt.close()
```

### Colored by Category

```python
fig, ax = plt.subplots(figsize=(10, 6))

for category in df["category"].unique().to_list():
    subset = df.filter(col("category") == category)
    ax.scatter(
        subset["x"].to_numpy(),
        subset["y"].to_numpy(),
        label=category,
        alpha=0.6
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()

plt.savefig("scatter_colored.png", dpi=150, bbox_inches="tight")
plt.close()
```

### With Regression Line

```python
from scipy import stats

fig, ax = plt.subplots(figsize=(10, 6))

x = df["x"].to_numpy()
y = df["y"].to_numpy()

# Scatter
ax.scatter(x, y, alpha=0.5)

# Regression line
slope, intercept, r_value, _, _ = stats.linregress(x, y)
line_x = np.linspace(x.min(), x.max(), 100)
line_y = slope * line_x + intercept
ax.plot(line_x, line_y, color="red", label=f"R² = {r_value**2:.3f}")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()

plt.savefig("scatter_regression.png", dpi=150, bbox_inches="tight")
plt.close()
```

---

## Histograms

### Basic Histogram

```python
fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(df["value"].to_numpy(), bins=50, edgecolor="black", alpha=0.7)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Values")

plt.savefig("histogram.png", dpi=150, bbox_inches="tight")
plt.close()
```

### Overlapping Histograms

```python
fig, ax = plt.subplots(figsize=(10, 6))

for category in df["category"].unique().to_list():
    subset = df.filter(col("category") == category)
    ax.hist(
        subset["value"].to_numpy(),
        bins=30,
        alpha=0.5,
        label=category
    )

ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.legend()

plt.savefig("histogram_overlay.png", dpi=150, bbox_inches="tight")
plt.close()
```

### With KDE (Density)

```python
fig, ax = plt.subplots(figsize=(10, 6))

values = df["value"].to_numpy()
ax.hist(values, bins=50, density=True, alpha=0.7, edgecolor="black")

# KDE
from scipy.stats import gaussian_kde
kde = gaussian_kde(values)
x_range = np.linspace(values.min(), values.max(), 200)
ax.plot(x_range, kde(x_range), color="red", linewidth=2)

ax.set_xlabel("Value")
ax.set_ylabel("Density")

plt.savefig("histogram_kde.png", dpi=150, bbox_inches="tight")
plt.close()
```

---

## Box Plots

### Basic Box Plot

```python
fig, ax = plt.subplots(figsize=(10, 6))

data = [
    df.filter(col("category") == cat)["value"].to_numpy()
    for cat in df["category"].unique().sort().to_list()
]
labels = df["category"].unique().sort().to_list()

ax.boxplot(data, labels=labels)
ax.set_xlabel("Category")
ax.set_ylabel("Value")
ax.set_title("Distribution by Category")

plt.savefig("boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
```

### Seaborn Box Plot

```python
fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(data=df.to_pandas(), x="category", y="value", ax=ax)
ax.set_xlabel("Category")
ax.set_ylabel("Value")
plt.xticks(rotation=45, ha="right")

plt.savefig("seaborn_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()
```

### Violin Plot

```python
fig, ax = plt.subplots(figsize=(10, 6))

sns.violinplot(data=df.to_pandas(), x="category", y="value", ax=ax)
ax.set_xlabel("Category")
ax.set_ylabel("Value")
plt.xticks(rotation=45, ha="right")

plt.savefig("violin_plot.png", dpi=150, bbox_inches="tight")
plt.close()
```

---

## Heatmaps

### Correlation Heatmap

```python
fig, ax = plt.subplots(figsize=(10, 8))

# Get numeric columns and calculate correlation
numeric_cols = [c for c in df.columns if df.schema[c] in [pl.Float64, pl.Int64]]
corr_matrix = df.select(numeric_cols).to_pandas().corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    ax=ax,
    fmt=".2f"
)
ax.set_title("Correlation Matrix")

plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
```

### Pivot Table Heatmap

```python
fig, ax = plt.subplots(figsize=(12, 8))

pivot_df = df.pivot(
    on="col_category",
    index="row_category",
    values="value",
    aggregate_function="mean"
)

# Convert to numpy for heatmap
matrix = pivot_df.drop("row_category").to_numpy()
row_labels = pivot_df["row_category"].to_list()
col_labels = [c for c in pivot_df.columns if c != "row_category"]

sns.heatmap(
    matrix,
    annot=True,
    cmap="YlOrRd",
    xticklabels=col_labels,
    yticklabels=row_labels,
    ax=ax,
    fmt=".1f"
)

plt.savefig("pivot_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
```

---

## Pie Charts

### Basic Pie Chart

```python
fig, ax = plt.subplots(figsize=(10, 8))

summary = df.group_by("category").agg(col("value").sum())
ax.pie(
    summary["value"].to_list(),
    labels=summary["category"].to_list(),
    autopct="%1.1f%%",
    startangle=90
)
ax.set_title("Distribution by Category")

plt.savefig("pie_chart.png", dpi=150, bbox_inches="tight")
plt.close()
```

### Donut Chart

```python
fig, ax = plt.subplots(figsize=(10, 8))

summary = df.group_by("category").agg(col("value").sum())
wedges, texts, autotexts = ax.pie(
    summary["value"].to_list(),
    labels=summary["category"].to_list(),
    autopct="%1.1f%%",
    startangle=90,
    pctdistance=0.85
)

# Create donut
centre_circle = plt.Circle((0, 0), 0.70, fc="white")
ax.add_artist(centre_circle)

plt.savefig("donut_chart.png", dpi=150, bbox_inches="tight")
plt.close()
```

---

## Subplots and Multi-Panel Figures

### Grid of Plots

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Line chart
axes[0, 0].plot(df["date"].to_list(), df["value"].to_list())
axes[0, 0].set_title("Time Series")

# Top-right: Bar chart
summary = df.group_by("category").agg(col("value").sum())
axes[0, 1].bar(summary["category"].to_list(), summary["value"].to_list())
axes[0, 1].set_title("By Category")
axes[0, 1].tick_params(axis="x", rotation=45)

# Bottom-left: Histogram
axes[1, 0].hist(df["value"].to_numpy(), bins=30, edgecolor="black")
axes[1, 0].set_title("Distribution")

# Bottom-right: Scatter
axes[1, 1].scatter(df["x"].to_numpy(), df["y"].to_numpy(), alpha=0.5)
axes[1, 1].set_title("Correlation")

plt.tight_layout()
plt.savefig("multi_panel.png", dpi=150, bbox_inches="tight")
plt.close()
```

### Shared Axis

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(df["date"].to_list(), df["metric1"].to_list())
axes[0].set_ylabel("Metric 1")

axes[1].plot(df["date"].to_list(), df["metric2"].to_list())
axes[1].set_ylabel("Metric 2")
axes[1].set_xlabel("Date")

plt.tight_layout()
plt.savefig("shared_axis.png", dpi=150, bbox_inches="tight")
plt.close()
```

---

## Interactive Charts (Plotly)

### Line Chart

```python
import plotly.express as px

fig = px.line(
    df.to_pandas(),
    x="date",
    y="value",
    color="category",
    title="Interactive Time Series"
)
fig.write_html("interactive_line.html")
```

### Scatter Plot

```python
fig = px.scatter(
    df.to_pandas(),
    x="x",
    y="y",
    color="category",
    size="value",
    hover_data=["name"],
    title="Interactive Scatter"
)
fig.write_html("interactive_scatter.html")
```

### Bar Chart

```python
fig = px.bar(
    df.to_pandas(),
    x="category",
    y="value",
    color="subcategory",
    barmode="group",
    title="Grouped Bar Chart"
)
fig.write_html("interactive_bar.html")
```

### Heatmap

```python
fig = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Correlation Heatmap"
)
fig.write_html("interactive_heatmap.html")
```

---

## Styling and Best Practices

### Color Palettes

```python
# Matplotlib
plt.style.use('seaborn-v0_8-whitegrid')

# Seaborn palettes
sns.set_palette("husl")        # Colorful
sns.set_palette("Set2")        # Qualitative
sns.set_palette("Blues")       # Sequential
sns.set_palette("coolwarm")    # Diverging

# Custom colors
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
```

### Figure Sizing

```python
# Standard sizes
# Single plot: (10, 6)
# Wide plot: (14, 6)
# Tall plot: (8, 10)
# Square: (8, 8)
# Grid 2x2: (12, 10)

fig, ax = plt.subplots(figsize=(10, 6))
```

### Saving Figures

```python
# PNG (raster)
plt.savefig("figure.png", dpi=150, bbox_inches="tight")

# SVG (vector)
plt.savefig("figure.svg", bbox_inches="tight")

# PDF (vector)
plt.savefig("figure.pdf", bbox_inches="tight")

# High resolution
plt.savefig("figure_hires.png", dpi=300, bbox_inches="tight")
```

### Adding Annotations

```python
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, y)

# Add text annotation
ax.annotate(
    "Peak",
    xy=(peak_x, peak_y),
    xytext=(peak_x + 5, peak_y + 10),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10
)

# Add horizontal line
ax.axhline(y=threshold, color="red", linestyle="--", label="Threshold")

# Add vertical line
ax.axvline(x=event_date, color="gray", linestyle=":", alpha=0.7)

# Add shaded region
ax.axvspan(start_x, end_x, alpha=0.2, color="yellow")

plt.savefig("annotated.png", dpi=150, bbox_inches="tight")
plt.close()
```
