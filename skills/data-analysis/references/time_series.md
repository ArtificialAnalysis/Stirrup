# Time Series Reference

Complete reference for date/time parsing, time series operations, resampling, and temporal features.

## Date/Time Parsing

### String to Date

```python
import polars as pl
from polars import col

# Basic parsing
df = df.with_columns(
    col("date_str").str.to_date("%Y-%m-%d").alias("date")
)

# Common format strings
# %Y-%m-%d       -> 2024-01-15
# %d/%m/%Y       -> 15/01/2024
# %m/%d/%Y       -> 01/15/2024
# %Y%m%d         -> 20240115
# %b %d, %Y      -> Jan 15, 2024
# %B %d, %Y      -> January 15, 2024
```

### String to DateTime

```python
df = df.with_columns(
    col("datetime_str").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("datetime")
)

# Common datetime formats
# %Y-%m-%d %H:%M:%S        -> 2024-01-15 14:30:00
# %Y-%m-%dT%H:%M:%S        -> 2024-01-15T14:30:00 (ISO)
# %Y-%m-%dT%H:%M:%S%.f     -> 2024-01-15T14:30:00.123456 (with microseconds)
# %Y-%m-%d %H:%M:%S %Z     -> 2024-01-15 14:30:00 UTC (with timezone)
# %d/%m/%Y %I:%M %p        -> 15/01/2024 02:30 PM (12-hour)
```

### Unix Timestamps

```python
# From Unix timestamp (seconds)
df = df.with_columns(
    pl.from_epoch(col("timestamp"), time_unit="s").alias("datetime")
)

# From milliseconds
df = df.with_columns(
    pl.from_epoch(col("timestamp_ms"), time_unit="ms").alias("datetime")
)

# To Unix timestamp
df = df.with_columns(
    col("datetime").dt.epoch(time_unit="s").alias("unix_ts")
)
```

### Timezone Handling

```python
# Parse with timezone
df = df.with_columns(
    col("dt_str").str.to_datetime("%Y-%m-%d %H:%M:%S %Z").alias("dt_utc")
)

# Convert timezone
df = df.with_columns(
    col("datetime")
    .dt.replace_time_zone("UTC")
    .dt.convert_time_zone("America/New_York")
    .alias("dt_ny")
)

# Remove timezone
df = df.with_columns(
    col("datetime").dt.replace_time_zone(None).alias("dt_naive")
)
```

---

## Extracting Date Components

### Date Parts

```python
df = df.with_columns([
    col("datetime").dt.year().alias("year"),
    col("datetime").dt.month().alias("month"),
    col("datetime").dt.day().alias("day"),
    col("datetime").dt.ordinal_day().alias("day_of_year"),
    col("datetime").dt.weekday().alias("weekday"),  # 0=Monday, 6=Sunday
    col("datetime").dt.week().alias("week_of_year"),
    col("datetime").dt.quarter().alias("quarter"),
])
```

### Time Parts

```python
df = df.with_columns([
    col("datetime").dt.hour().alias("hour"),
    col("datetime").dt.minute().alias("minute"),
    col("datetime").dt.second().alias("second"),
    col("datetime").dt.millisecond().alias("millisecond"),
    col("datetime").dt.microsecond().alias("microsecond"),
    col("datetime").dt.nanosecond().alias("nanosecond"),
])
```

### Derived Features

```python
df = df.with_columns([
    # Is weekend
    (col("datetime").dt.weekday() >= 5).alias("is_weekend"),

    # Part of day
    pl.when(col("datetime").dt.hour() < 6).then(pl.lit("night"))
    .when(col("datetime").dt.hour() < 12).then(pl.lit("morning"))
    .when(col("datetime").dt.hour() < 18).then(pl.lit("afternoon"))
    .otherwise(pl.lit("evening"))
    .alias("part_of_day"),

    # Month name
    col("datetime").dt.strftime("%B").alias("month_name"),

    # Day name
    col("datetime").dt.strftime("%A").alias("day_name"),

    # ISO week
    col("datetime").dt.strftime("%G-W%V").alias("iso_week"),
])
```

---

## Date Arithmetic

### Adding/Subtracting Time

```python
from datetime import timedelta

# Add duration
df = df.with_columns([
    (col("date") + pl.duration(days=7)).alias("date_plus_week"),
    (col("date") + pl.duration(days=30)).alias("date_plus_month"),
    (col("datetime") + pl.duration(hours=24)).alias("dt_plus_day"),
    (col("datetime") + pl.duration(minutes=30)).alias("dt_plus_30min"),
])

# Subtract
df = df.with_columns(
    (col("date") - pl.duration(days=1)).alias("yesterday")
)
```

### Duration Between Dates

```python
# Difference in days
df = df.with_columns(
    (col("end_date") - col("start_date")).dt.total_days().alias("days_between")
)

# Difference in hours
df = df.with_columns(
    (col("end_datetime") - col("start_datetime")).dt.total_hours().alias("hours_between")
)

# Other duration extractions
df = df.with_columns([
    (col("end") - col("start")).dt.total_seconds().alias("seconds"),
    (col("end") - col("start")).dt.total_minutes().alias("minutes"),
    (col("end") - col("start")).dt.total_milliseconds().alias("ms"),
])
```

### Date Truncation

```python
df = df.with_columns([
    col("datetime").dt.truncate("1d").alias("date_only"),
    col("datetime").dt.truncate("1h").alias("hour_start"),
    col("datetime").dt.truncate("1mo").alias("month_start"),
    col("datetime").dt.truncate("1w").alias("week_start"),
])
```

---

## Filtering by Date

### Date Range

```python
from datetime import datetime, date

# Filter by date range
df.filter(
    col("date").is_between(date(2024, 1, 1), date(2024, 12, 31))
)

df.filter(
    col("datetime").is_between(
        datetime(2024, 1, 1, 0, 0, 0),
        datetime(2024, 12, 31, 23, 59, 59)
    )
)

# Last N days
df.filter(
    col("date") >= (pl.lit(datetime.now()) - pl.duration(days=30))
)
```

### Specific Periods

```python
# Current year
df.filter(col("date").dt.year() == 2024)

# Specific month
df.filter(
    (col("date").dt.year() == 2024) &
    (col("date").dt.month() == 6)
)

# Weekdays only
df.filter(col("date").dt.weekday() < 5)

# Specific day of week
df.filter(col("date").dt.weekday() == 0)  # Mondays
```

---

## Resampling (Time Grouping)

### group_by_dynamic

```python
# Daily aggregation
daily = df.sort("timestamp").group_by_dynamic(
    "timestamp",
    every="1d",
).agg([
    col("value").sum().alias("daily_sum"),
    col("value").mean().alias("daily_avg"),
    col("value").count().alias("daily_count"),
    col("value").min().alias("daily_min"),
    col("value").max().alias("daily_max"),
])
```

### Common Time Intervals

```python
# Hourly
hourly = df.sort("timestamp").group_by_dynamic("timestamp", every="1h").agg(...)

# Daily
daily = df.sort("timestamp").group_by_dynamic("timestamp", every="1d").agg(...)

# Weekly (starting Monday)
weekly = df.sort("timestamp").group_by_dynamic(
    "timestamp",
    every="1w",
    start_by="monday"
).agg(...)

# Monthly
monthly = df.sort("timestamp").group_by_dynamic("timestamp", every="1mo").agg(...)

# Quarterly
quarterly = df.sort("timestamp").group_by_dynamic("timestamp", every="3mo").agg(...)

# Yearly
yearly = df.sort("timestamp").group_by_dynamic("timestamp", every="1y").agg(...)
```

### With Additional Grouping

```python
# Daily totals by category
daily_by_cat = df.sort("timestamp").group_by_dynamic(
    "timestamp",
    every="1d",
    group_by="category",
).agg(col("value").sum().alias("daily_total"))
```

### Rolling Time Windows

```python
# 7-day rolling window
rolling_7d = df.sort("timestamp").group_by_dynamic(
    "timestamp",
    every="1d",
    period="7d",  # Look back 7 days
).agg([
    col("value").sum().alias("rolling_7d_sum"),
    col("value").mean().alias("rolling_7d_avg"),
])
```

---

## Lag and Lead Features

### Basic Lag/Lead

```python
# Sort by time first
df = df.sort("timestamp")

# Lag (previous values)
df = df.with_columns([
    col("value").shift(1).alias("lag_1"),
    col("value").shift(7).alias("lag_7"),
    col("value").shift(30).alias("lag_30"),
])

# Lead (future values)
df = df.with_columns(
    col("value").shift(-1).alias("next_value")
)
```

### Lag Within Groups

```python
df = df.sort("timestamp").with_columns([
    col("value").shift(1).over("user_id").alias("prev_value"),
    col("value").shift(7).over("user_id").alias("value_7_periods_ago"),
])
```

### Change Calculations

```python
df = df.sort("timestamp").with_columns([
    # Absolute change
    (col("value") - col("value").shift(1)).alias("change"),

    # Percent change
    ((col("value") / col("value").shift(1)) - 1).alias("pct_change"),

    # Log return
    (col("value").log() - col("value").shift(1).log()).alias("log_return"),
])
```

---

## Moving Averages and Rolling Statistics

### Simple Moving Average

```python
df = df.sort("timestamp").with_columns([
    col("value").rolling_mean(window_size=7).alias("ma_7"),
    col("value").rolling_mean(window_size=30).alias("ma_30"),
    col("value").rolling_mean(window_size=90).alias("ma_90"),
])
```

### Exponential Moving Average

```python
df = df.sort("timestamp").with_columns([
    col("value").ewm_mean(span=7).alias("ema_7"),
    col("value").ewm_mean(span=30).alias("ema_30"),
])
```

### Rolling Statistics

```python
df = df.sort("timestamp").with_columns([
    col("value").rolling_mean(window_size=7).alias("rolling_mean"),
    col("value").rolling_std(window_size=7).alias("rolling_std"),
    col("value").rolling_min(window_size=7).alias("rolling_min"),
    col("value").rolling_max(window_size=7).alias("rolling_max"),
    col("value").rolling_sum(window_size=7).alias("rolling_sum"),
    col("value").rolling_median(window_size=7).alias("rolling_median"),
])
```

### Bollinger Bands

```python
window = 20
df = df.sort("timestamp").with_columns([
    col("price").rolling_mean(window_size=window).alias("ma"),
    col("price").rolling_std(window_size=window).alias("std"),
]).with_columns([
    (col("ma") + 2 * col("std")).alias("upper_band"),
    (col("ma") - 2 * col("std")).alias("lower_band"),
])
```

---

## Seasonal Features

### Time-Based Features

```python
df = df.with_columns([
    # Cyclical encoding for hour (0-23)
    (col("datetime").dt.hour() * 2 * 3.14159 / 24).sin().alias("hour_sin"),
    (col("datetime").dt.hour() * 2 * 3.14159 / 24).cos().alias("hour_cos"),

    # Cyclical encoding for day of week (0-6)
    (col("datetime").dt.weekday() * 2 * 3.14159 / 7).sin().alias("dow_sin"),
    (col("datetime").dt.weekday() * 2 * 3.14159 / 7).cos().alias("dow_cos"),

    # Cyclical encoding for month (1-12)
    (col("datetime").dt.month() * 2 * 3.14159 / 12).sin().alias("month_sin"),
    (col("datetime").dt.month() * 2 * 3.14159 / 12).cos().alias("month_cos"),
])
```

### Year-over-Year Comparison

```python
# Get value from same period last year
df = df.sort("timestamp").with_columns(
    col("value").shift(365).alias("value_last_year")
).with_columns(
    ((col("value") / col("value_last_year")) - 1).alias("yoy_growth")
)
```

---

## Handling Gaps

### Fill Missing Dates

```python
import polars as pl
from datetime import date

# Create complete date range
date_range = pl.date_range(
    date(2024, 1, 1),
    date(2024, 12, 31),
    eager=True
).alias("date")

# Create DataFrame with all dates
all_dates = pl.DataFrame({"date": date_range})

# Join with data (fills gaps with null)
filled = all_dates.join(df, on="date", how="left")

# Forward fill missing values
filled = filled.with_columns(
    col("value").forward_fill()
)
```

### Interpolation

```python
df = df.with_columns(
    col("value").interpolate().alias("interpolated")
)
```

---

## Performance Tips

1. **Sort once, use many times**: Sort by timestamp at the start
   ```python
   df = df.sort("timestamp")
   ```

2. **Use lazy evaluation** for time series operations:
   ```python
   result = (
       pl.scan_csv("timeseries.csv")
       .sort("timestamp")
       .group_by_dynamic("timestamp", every="1d")
       .agg(col("value").sum())
       .collect()
   )
   ```

3. **Pre-compute frequently used features** like year, month, day

4. **Use appropriate time precision** (Date vs Datetime vs Time)

5. **Consider data partitioning** by date for very large datasets
