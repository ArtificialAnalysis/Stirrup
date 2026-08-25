# Data Transformations Reference

Complete reference for column operations, filtering, sorting, type casting, and data manipulation.

## Column Selection

### Basic Selection

```python
import polars as pl
from polars import col

# By name
df.select("name", "age")
df.select(["name", "age"])

# Using col()
df.select(col("name"), col("age"))
```

### Pattern-Based Selection

```python
# Regex pattern
df.select(pl.col("^value_.*$"))  # Columns starting with "value_"
df.select(pl.col("^.*_id$"))     # Columns ending with "_id"

# By data type
df.select(pl.col(pl.Float64))           # All float columns
df.select(pl.col(pl.Int64, pl.Int32))   # All integer columns
df.select(pl.col(pl.Utf8))              # All string columns

# All columns
df.select(pl.all())

# Exclude specific columns
df.select(pl.all().exclude("id", "created_at"))
df.select(pl.all().exclude(pl.Utf8))  # Exclude all strings
```

### Rename During Selection

```python
df.select(
    col("name").alias("full_name"),
    col("amt").alias("amount"),
)
```

---

## Row Filtering

### Comparison Operators

```python
# Equality
df.filter(col("status") == "active")
df.filter(col("status") != "inactive")

# Numeric comparisons
df.filter(col("age") > 25)
df.filter(col("age") >= 25)
df.filter(col("age") < 65)
df.filter(col("age") <= 65)

# Between
df.filter(col("score").is_between(80, 100))
df.filter(col("score").is_between(80, 100, closed="both"))  # inclusive
df.filter(col("score").is_between(80, 100, closed="left"))  # [80, 100)
```

### Logical Operators

```python
# AND
df.filter((col("age") > 25) & (col("salary") < 100000))

# OR
df.filter((col("dept") == "Engineering") | (col("dept") == "Data"))

# NOT
df.filter(~col("is_deleted"))
df.filter(col("status") != "cancelled")
```

### Null Checks

```python
# Is null
df.filter(col("email").is_null())

# Is not null
df.filter(col("email").is_not_null())

# Any column is null
df.filter(pl.any_horizontal(pl.all().is_null()))

# All columns are not null (complete rows)
df.filter(pl.all_horizontal(pl.all().is_not_null()))
```

### List Membership

```python
# Is in list
df.filter(col("category").is_in(["A", "B", "C"]))

# Not in list
df.filter(~col("category").is_in(["X", "Y", "Z"]))
```

### String Matching

```python
# Contains substring
df.filter(col("name").str.contains("Smith"))

# Starts/ends with
df.filter(col("email").str.starts_with("admin"))
df.filter(col("email").str.ends_with("@company.com"))

# Regex match
df.filter(col("phone").str.contains(r"^\d{3}-\d{3}-\d{4}$"))

# Exact match (case insensitive)
df.filter(col("name").str.to_lowercase() == "john")
```

---

## Adding and Modifying Columns

### with_columns()

```python
# Add single column
df = df.with_columns(
    (col("price") * col("quantity")).alias("total")
)

# Add multiple columns
df = df.with_columns([
    (col("price") * 1.1).alias("price_with_tax"),
    (col("first_name") + " " + col("last_name")).alias("full_name"),
    pl.lit("USD").alias("currency"),
    pl.lit(True).alias("is_processed"),
])

# Modify existing column (same name = replace)
df = df.with_columns(
    col("name").str.to_uppercase().alias("name")
)
```

### Conditional Columns (CASE WHEN)

```python
# Simple if-else
df = df.with_columns(
    pl.when(col("score") >= 60)
    .then(pl.lit("pass"))
    .otherwise(pl.lit("fail"))
    .alias("result")
)

# Multiple conditions (chained when)
df = df.with_columns(
    pl.when(col("score") >= 90).then(pl.lit("A"))
    .when(col("score") >= 80).then(pl.lit("B"))
    .when(col("score") >= 70).then(pl.lit("C"))
    .when(col("score") >= 60).then(pl.lit("D"))
    .otherwise(pl.lit("F"))
    .alias("grade")
)

# Conditional with column values
df = df.with_columns(
    pl.when(col("type") == "percentage")
    .then(col("value") / 100)
    .otherwise(col("value"))
    .alias("normalized_value")
)
```

### Replace Values

```python
# Map replacement
df = df.with_columns(
    col("status").replace({
        "active": "Active",
        "inactive": "Inactive",
        "pending": "Pending"
    })
)

# Replace with default for non-matches
df = df.with_columns(
    col("code").replace({"A": 1, "B": 2}, default=0)
)

# Replace null
df = df.with_columns(
    col("value").fill_null(0)
)
```

---

## Type Casting

### Basic Casting

```python
df = df.with_columns([
    col("id").cast(pl.Utf8).alias("id_str"),
    col("value").cast(pl.Int64).alias("value_int"),
    col("amount").cast(pl.Float64).alias("amount_float"),
    col("flag").cast(pl.Boolean).alias("flag_bool"),
])
```

### Safe Casting (Handle Errors)

```python
# Strict=False returns null on failure
df = df.with_columns(
    col("maybe_number").cast(pl.Int64, strict=False)
)
```

### String to Date/DateTime

```python
df = df.with_columns([
    col("date_str").str.to_date("%Y-%m-%d").alias("date"),
    col("datetime_str").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("datetime"),
    col("time_str").str.to_time("%H:%M:%S").alias("time"),
])

# Common date formats
# %Y-%m-%d       -> 2024-01-15
# %d/%m/%Y       -> 15/01/2024
# %m/%d/%Y       -> 01/15/2024
# %Y-%m-%d %H:%M:%S -> 2024-01-15 14:30:00
# %Y-%m-%dT%H:%M:%S -> 2024-01-15T14:30:00 (ISO)
```

### Categorical Type

```python
# Convert to categorical (memory efficient for low-cardinality)
df = df.with_columns(
    col("category").cast(pl.Categorical)
)

# Enable global string cache for joins
pl.enable_string_cache()
df1 = df1.with_columns(col("cat").cast(pl.Categorical))
df2 = df2.with_columns(col("cat").cast(pl.Categorical))
```

---

## Renaming Columns

```python
# Rename specific columns
df = df.rename({
    "old_name": "new_name",
    "col1": "column_1",
})

# Rename with function
df = df.rename(lambda c: c.lower().replace(" ", "_"))

# Prefix/suffix all columns
df = df.select([
    col(c).alias(f"prefix_{c}") for c in df.columns
])

# Remove prefix
df = df.rename(lambda c: c.removeprefix("tbl_"))
```

---

## Sorting

### Basic Sorting

```python
# Single column ascending
df = df.sort("date")

# Descending
df = df.sort("score", descending=True)

# Multiple columns
df = df.sort(["category", "date"])
df = df.sort(["category", "score"], descending=[False, True])
```

### Sort by Expression

```python
# By string length
df = df.sort(col("name").str.len_chars())

# By absolute value
df = df.sort(col("change").abs(), descending=True)

# Nulls first/last
df = df.sort("value", nulls_last=True)
df = df.sort("value", nulls_last=False)  # Nulls first
```

---

## Dropping Data

### Drop Columns

```python
df = df.drop("unnecessary_column")
df = df.drop(["col1", "col2", "col3"])
```

### Drop Duplicates

```python
# All columns
df = df.unique()

# Specific columns
df = df.unique(subset=["email"])

# Keep first/last occurrence
df = df.unique(subset=["id"], keep="first")
df = df.unique(subset=["id"], keep="last")
df = df.unique(subset=["id"], keep="none")  # Remove all duplicates
```

### Drop Nulls

```python
# Rows with any null
df = df.drop_nulls()

# Rows with null in specific columns
df = df.drop_nulls(subset=["email", "phone"])
```

---

## Slicing and Sampling

### Slicing

```python
# First N rows
df.head(10)
df.limit(10)

# Last N rows
df.tail(10)

# Slice by position
df.slice(0, 100)    # First 100 rows
df.slice(100, 50)   # 50 rows starting at index 100

# Get single row
row = df.row(0)  # Returns tuple
row_dict = df.row(0, named=True)  # Returns dict
```

### Sampling

```python
# Random sample (fixed size)
df.sample(n=100)

# Random sample (fraction)
df.sample(fraction=0.1)  # 10% of rows

# With seed for reproducibility
df.sample(n=100, seed=42)

# With replacement
df.sample(n=100, with_replacement=True)
```

---

## Row Operations

### Row Numbers

```python
# Add row number
df = df.with_row_index("row_num")

# Row number starting from 1
df = df.with_row_index("row_num", offset=1)
```

### Iterate Rows (Use Sparingly)

```python
# As tuples
for row in df.iter_rows():
    print(row)

# As dicts (named)
for row in df.iter_rows(named=True):
    print(row["name"], row["value"])

# Prefer vectorized operations instead!
```

---

## Performance Tips

1. **Chain operations** for better optimization:
   ```python
   result = (
       df
       .filter(col("x") > 0)
       .with_columns(col("x") * 2)
       .select("id", "x")
   )
   ```

2. **Use lazy evaluation** for complex transformations:
   ```python
   result = (
       pl.scan_csv("data.csv")
       .filter(...)
       .with_columns(...)
       .collect()
   )
   ```

3. **Avoid row-wise operations** - use vectorized expressions

4. **Select columns early** to reduce memory usage

5. **Cast to appropriate types** (e.g., Categorical for low-cardinality strings)
