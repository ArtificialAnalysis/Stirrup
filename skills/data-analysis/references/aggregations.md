# Aggregations Reference

Complete reference for group by operations, window functions, and running calculations.

## Global Aggregations

### Basic Statistics

```python
import polars as pl
from polars import col

# Single aggregation
total = df.select(col("value").sum())[0, 0]

# Multiple aggregations
stats = df.select([
    col("value").sum().alias("total"),
    col("value").mean().alias("average"),
    col("value").std().alias("std_dev"),
    col("value").var().alias("variance"),
    col("value").min().alias("minimum"),
    col("value").max().alias("maximum"),
    col("value").median().alias("median"),
    col("value").quantile(0.25).alias("q25"),
    col("value").quantile(0.75).alias("q75"),
    col("value").quantile(0.95).alias("p95"),
])
```

### Counting

```python
df.select([
    pl.len().alias("row_count"),
    col("id").count().alias("non_null_count"),
    col("id").n_unique().alias("unique_count"),
    col("category").value_counts(sort=True).alias("value_counts"),
])
```

---

## Group By Operations

### Basic Group By

```python
# Single grouping column
result = df.group_by("category").agg(
    col("value").sum().alias("total"),
    col("value").mean().alias("avg"),
    pl.len().alias("count"),
)

# Multiple grouping columns
result = df.group_by(["region", "category"]).agg(
    col("revenue").sum().alias("total_revenue"),
    col("quantity").sum().alias("total_quantity"),
)
```

### Common Aggregation Functions

```python
df.group_by("category").agg([
    # Numeric
    col("value").sum().alias("sum"),
    col("value").mean().alias("mean"),
    col("value").median().alias("median"),
    col("value").std().alias("std"),
    col("value").var().alias("var"),
    col("value").min().alias("min"),
    col("value").max().alias("max"),

    # Counting
    pl.len().alias("count"),
    col("id").n_unique().alias("unique_ids"),

    # First/Last
    col("timestamp").first().alias("first_time"),
    col("timestamp").last().alias("last_time"),
    col("timestamp").min().alias("earliest"),
    col("timestamp").max().alias("latest"),

    # Quantiles
    col("value").quantile(0.5).alias("median"),
    col("value").quantile(0.9).alias("p90"),
    col("value").quantile(0.99).alias("p99"),
])
```

### Computed Aggregations

```python
df.group_by("category").agg([
    # Ratio
    (col("successes").sum() / col("attempts").sum()).alias("success_rate"),

    # Weighted average
    (col("value") * col("weight")).sum() / col("weight").sum()
    ).alias("weighted_avg"),

    # Range
    (col("value").max() - col("value").min()).alias("range"),

    # Coefficient of variation
    (col("value").std() / col("value").mean()).alias("cv"),
])
```

### Collect Into Lists

```python
df.group_by("user_id").agg([
    col("product").alias("products"),  # List of all products
    col("product").unique().alias("unique_products"),
    col("product").n_unique().alias("product_count"),
])

# Concatenate strings
df.group_by("order_id").agg(
    col("item").str.concat(", ").alias("items_list")
)
```

### Sorting Grouped Results

```python
# Sort by aggregated value
df.group_by("category").agg(
    col("sales").sum().alias("total_sales")
).sort("total_sales", descending=True)

# Top N per category (requires window function - see below)
```

---

## Window Functions

Window functions compute values across related rows without collapsing them.

### Basic Window (over)

```python
# Sum per group (added to each row)
df = df.with_columns(
    col("value").sum().over("category").alias("category_total")
)

# Multiple partition columns
df = df.with_columns(
    col("value").mean().over(["region", "year"]).alias("regional_avg")
)
```

### Ranking

```python
# Row number within group
df = df.with_columns(
    col("value").rank(method="ordinal").over("category").alias("rank")
)

# Dense rank (no gaps)
df = df.with_columns(
    col("score").rank(method="dense", descending=True)
    .over("department")
    .alias("dense_rank")
)

# Percent rank
df = df.with_columns(
    (col("value").rank() / col("value").count())
    .over("group")
    .alias("percentile")
)

# Ranking methods:
# - "ordinal": 1, 2, 3, 4 (unique ranks)
# - "dense": 1, 2, 2, 3 (no gaps for ties)
# - "min": 1, 2, 2, 4 (min rank for ties)
# - "max": 1, 3, 3, 4 (max rank for ties)
# - "average": 1, 2.5, 2.5, 4 (average rank for ties)
```

### Top N Per Group

```python
# Top 3 per category by value
df = (
    df.with_columns(
        col("value").rank(descending=True).over("category").alias("rank")
    )
    .filter(col("rank") <= 3)
    .drop("rank")
)
```

### Lag and Lead

```python
# Previous value
df = df.with_columns(
    col("value").shift(1).over("user_id").alias("prev_value")
)

# Next value
df = df.with_columns(
    col("value").shift(-1).over("user_id").alias("next_value")
)

# N periods ago
df = df.with_columns([
    col("value").shift(1).over("user").alias("lag_1"),
    col("value").shift(7).over("user").alias("lag_7"),
    col("value").shift(30).over("user").alias("lag_30"),
])

# Change from previous
df = df.with_columns(
    (col("value") - col("value").shift(1))
    .over("user_id")
    .alias("change")
)

# Percent change
df = df.with_columns(
    ((col("value") / col("value").shift(1)) - 1)
    .over("user_id")
    .alias("pct_change")
)
```

### Cumulative Functions

```python
# Cumulative sum
df = df.with_columns(
    col("value").cum_sum().over("user_id").alias("running_total")
)

# Cumulative mean
df = df.with_columns(
    col("value").cum_mean().over("category").alias("running_avg")
)

# Cumulative count
df = df.with_columns(
    col("id").cum_count().over("user_id").alias("event_number")
)

# Cumulative min/max
df = df.with_columns([
    col("value").cum_min().over("id").alias("running_min"),
    col("value").cum_max().over("id").alias("running_max"),
])
```

---

## Rolling Windows

### Basic Rolling Aggregations

```python
# Rolling mean (3-period window)
df = df.with_columns(
    col("value").rolling_mean(window_size=3).alias("ma_3")
)

# Rolling with minimum periods
df = df.with_columns(
    col("value").rolling_mean(window_size=7, min_periods=1).alias("ma_7")
)
```

### Rolling Statistics

```python
df = df.with_columns([
    col("value").rolling_mean(window_size=7).alias("rolling_mean"),
    col("value").rolling_std(window_size=7).alias("rolling_std"),
    col("value").rolling_var(window_size=7).alias("rolling_var"),
    col("value").rolling_min(window_size=7).alias("rolling_min"),
    col("value").rolling_max(window_size=7).alias("rolling_max"),
    col("value").rolling_sum(window_size=7).alias("rolling_sum"),
    col("value").rolling_median(window_size=7).alias("rolling_median"),
    col("value").rolling_quantile(0.95, window_size=7).alias("rolling_p95"),
])
```

### Rolling Within Groups

```python
df = df.with_columns(
    col("value")
    .rolling_mean(window_size=3)
    .over("category")
    .alias("category_rolling_avg")
)
```

### Exponential Moving Average

```python
# EMA with span
df = df.with_columns(
    col("value").ewm_mean(span=10).alias("ema_10")
)

# EMA with alpha (smoothing factor)
df = df.with_columns(
    col("value").ewm_mean(alpha=0.1).alias("ema")
)
```

---

## Time-Based Grouping

### group_by_dynamic

For time-series data with datetime index:

```python
# Daily aggregation
daily = df.group_by_dynamic(
    "timestamp",
    every="1d",
).agg([
    col("value").sum().alias("daily_total"),
    col("value").mean().alias("daily_avg"),
    col("value").count().alias("daily_count"),
])

# Weekly (starting Monday)
weekly = df.group_by_dynamic(
    "timestamp",
    every="1w",
    start_by="monday",
).agg(col("value").sum())

# Monthly
monthly = df.group_by_dynamic(
    "timestamp",
    every="1mo",
).agg(col("revenue").sum())

# Hourly
hourly = df.group_by_dynamic(
    "timestamp",
    every="1h",
).agg(col("events").count())
```

### Rolling Time Windows

```python
# Rolling 7-day sum
rolling_weekly = df.group_by_dynamic(
    "timestamp",
    every="1d",
    period="7d",  # 7-day rolling window
).agg(col("value").sum().alias("rolling_7d_sum"))
```

### With Additional Group Columns

```python
# Daily by category
daily_by_cat = df.group_by_dynamic(
    "timestamp",
    every="1d",
    group_by="category",  # Additional grouping
).agg(col("value").sum())
```

---

## Advanced Patterns

### Conditional Aggregations

```python
df.group_by("category").agg([
    col("value").sum().alias("total"),
    col("value").filter(col("status") == "completed").sum().alias("completed_total"),
    col("value").filter(col("status") == "pending").sum().alias("pending_total"),
])
```

### Multiple Aggregations on Same Column

```python
df.group_by("category").agg([
    col("value").min().alias("min"),
    col("value").max().alias("max"),
    col("value").mean().alias("mean"),
    col("value").std().alias("std"),
    (col("value").max() - col("value").min()).alias("range"),
])
```

### Struct Aggregations

```python
# Aggregate into struct
df.group_by("category").agg(
    pl.struct(["min", "max", "mean"]).alias("stats")
)
```

---

## Performance Tips

1. **Filter before grouping** to reduce data volume:
   ```python
   df.filter(col("year") == 2024).group_by("category").agg(...)
   ```

2. **Use lazy evaluation** for complex aggregations:
   ```python
   result = (
       pl.scan_csv("data.csv")
       .group_by("key")
       .agg(col("value").sum())
       .collect()
   )
   ```

3. **Avoid unnecessary window computations** - compute once and reuse

4. **Use streaming** for large datasets:
   ```python
   result = (
       pl.scan_csv("huge.csv")
       .group_by("category")
       .agg(col("value").sum())
       .collect(streaming=True)
   )
   ```
