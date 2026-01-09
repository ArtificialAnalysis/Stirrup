---
name: data_analysis
description: High-performance data analysis using Polars - load, transform, aggregate, visualize and export tabular data. Use for CSV/JSON/Parquet processing, statistical analysis, time series, and creating charts.
---

# Data Analysis Skill

Comprehensive data analysis toolkit using **Polars** - a blazingly fast DataFrame library. This skill provides instructions, reference documentation, and ready-to-use scripts for common data analysis tasks.

### Iteration Checkpoints

| Step | What to Present | User Input Type |
|------|-----------------|-----------------|
| Data Loading | Shape, columns, sample rows | "Is this the right data?" |
| Data Exploration | Summary stats, data quality issues | "Any columns to focus on?" |
| Transformation | Before/after comparison | "Does this transformation look correct?" |
| Analysis | Key findings, charts | "Should I dig deeper into anything?" |
| Export | Output preview | "Ready to save, or any changes?" |

## Quick Start

```python
import polars as pl
from polars import col

# Load data
df = pl.read_csv("data.csv")

# Explore
print(df.shape, df.schema)
df.describe()

# Transform and analyze
result = (
    df.filter(col("value") > 0)
    .group_by("category")
    .agg(col("value").sum().alias("total"))
    .sort("total", descending=True)
)

# Export
result.write_csv("output.csv")
```

## When to Use This Skill

- Loading datasets (CSV, JSON, Parquet, Excel, databases)
- Data cleaning, filtering, and transformation
- Aggregations, grouping, and pivot tables
- Statistical analysis and summary statistics
- Time series analysis and resampling
- Joining and merging multiple datasets
- Creating visualizations and charts
- Exporting results to various formats

## Skill Contents

### Reference Documentation

Detailed API reference and patterns for specific operations:

- `reference/loading.md` - Loading data from all supported formats
- `reference/transformations.md` - Column operations, filtering, sorting, type casting
- `reference/aggregations.md` - Group by, window functions, running totals
- `reference/time_series.md` - Date parsing, resampling, lag features
- `reference/statistics.md` - Correlations, distributions, hypothesis testing setup
- `reference/visualization.md` - Creating charts with matplotlib/plotly

### Ready-to-Use Scripts

Executable Python scripts for common tasks:

- `scripts/explore_data.py` - Quick dataset exploration and profiling
- `scripts/summary_stats.py` - Generate comprehensive statistics report

## Core Patterns

### Loading Data

```python
# CSV (most common)
df = pl.read_csv("data.csv")

# Lazy loading for large files
df = pl.scan_csv("large.csv").filter(col("x") > 0).collect()

# Parquet (recommended for large datasets)
df = pl.read_parquet("data.parquet")

# JSON
df = pl.read_json("data.json")
df = pl.read_ndjson("data.ndjson")  # Newline-delimited
```

### Filtering and Selection

```python
# Select columns
df.select("col1", "col2")
df.select(col("name"), col("value") * 2)

# Filter rows
df.filter(col("age") > 25)
df.filter((col("status") == "active") & (col("value") > 100))
df.filter(col("name").str.contains("Smith"))
```

### Transformations

```python
# Add/modify columns
df = df.with_columns(
    (col("price") * col("qty")).alias("total"),
    col("date_str").str.to_date("%Y-%m-%d").alias("date"),
)

# Conditional values
df = df.with_columns(
    pl.when(col("score") >= 90).then(pl.lit("A"))
    .when(col("score") >= 80).then(pl.lit("B"))
    .otherwise(pl.lit("C"))
    .alias("grade")
)
```

### Aggregations

```python
# Group by
df.group_by("category").agg(
    col("value").sum().alias("total"),
    col("value").mean().alias("avg"),
    pl.len().alias("count"),
)

# Window functions
df.with_columns(
    col("value").sum().over("group").alias("group_total"),
    col("value").rank().over("group").alias("rank_in_group"),
)
```

### Exporting

```python
df.write_csv("output.csv")
df.write_parquet("output.parquet")
df.write_json("output.json", row_oriented=True)
```

## Best Practices

1. **Use lazy evaluation** for large datasets: `pl.scan_csv()` + `.collect()`
2. **Filter early** to reduce data volume before expensive operations
3. **Select only needed columns** to minimize memory usage
4. **Prefer Parquet** for storage - faster I/O, better compression
5. **Use `.explain()`** to understand and optimize query plans
