# Data Loading Reference

Complete reference for loading data from various sources into Polars DataFrames.

## CSV Files

### Basic Loading

```python
import polars as pl

df = pl.read_csv("data.csv")
```

### Full Options

```python
df = pl.read_csv(
    "data.csv",
    separator=",",              # Field separator (default: ",")
    has_header=True,            # First row is header (default: True)
    skip_rows=0,                # Skip N rows at start
    skip_rows_after_header=0,   # Skip N rows after header
    n_rows=1000,                # Only read first N rows
    columns=["col1", "col2"],   # Select specific columns
    new_columns=["a", "b"],     # Rename columns on load
    dtypes={                    # Explicit column types
        "id": pl.Int64,
        "value": pl.Float64,
        "name": pl.Utf8,
        "active": pl.Boolean,
    },
    null_values=["NA", "N/A", "", "null", "NULL"],  # Treat as null
    ignore_errors=True,         # Skip malformed rows
    encoding="utf-8",           # File encoding
    low_memory=False,           # Trade memory for speed
    rechunk=True,               # Rechunk for contiguous memory
    try_parse_dates=True,       # Auto-detect date columns
    infer_schema_length=1000,   # Rows to scan for type inference
)
```

### Lazy Loading (Large Files)

```python
# Scan without loading into memory
lazy_df = pl.scan_csv("large_file.csv")

# Build query, then collect
result = (
    lazy_df
    .filter(pl.col("value") > 100)
    .select("id", "value")
    .collect()  # Execute query
)

# Streaming for very large files
result = (
    pl.scan_csv("huge_file.csv")
    .filter(pl.col("category") == "A")
    .group_by("region")
    .agg(pl.col("amount").sum())
    .collect(streaming=True)  # Process in chunks
)
```

### Multiple CSV Files

```python
# Glob pattern
df = pl.read_csv("data/*.csv")

# Concat multiple files
from pathlib import Path
dfs = [pl.read_csv(f) for f in Path("data").glob("*.csv")]
df = pl.concat(dfs)

# Lazy concat (memory efficient)
lazy_df = pl.concat([
    pl.scan_csv(f) for f in Path("data").glob("*.csv")
])
result = lazy_df.collect()
```

---

## JSON Files

### Standard JSON

```python
# Array of objects: [{"a": 1}, {"a": 2}]
df = pl.read_json("data.json")

# From string
json_str = '[{"name": "Alice", "age": 30}]'
df = pl.read_json(json_str.encode())
```

### Newline-Delimited JSON (NDJSON)

```python
# One JSON object per line (streaming friendly)
df = pl.read_ndjson("data.ndjson")

# Lazy loading
lazy_df = pl.scan_ndjson("large.ndjson")
```

### Nested JSON

```python
# Read nested structure
df = pl.read_json("nested.json")

# Unnest struct columns
df = df.unnest("nested_column")

# Access nested fields
df = df.with_columns(
    pl.col("data").struct.field("subfield").alias("extracted")
)
```

---

## Parquet Files

Parquet is the **recommended format** for large datasets due to:
- Columnar storage (fast column selection)
- Built-in compression
- Schema preservation
- Predicate pushdown support

### Basic Loading

```python
df = pl.read_parquet("data.parquet")
```

### Lazy Loading with Optimizations

```python
# Lazy scan enables predicate and projection pushdown
lazy_df = pl.scan_parquet("data.parquet")

# Only reads necessary columns and rows
result = (
    lazy_df
    .filter(pl.col("date") > "2024-01-01")
    .select("id", "value")
    .collect()
)
```

### Multiple Parquet Files

```python
# Glob pattern
df = pl.read_parquet("data/*.parquet")

# Partitioned data (Hive-style)
df = pl.read_parquet("data/**/*.parquet")
```

### Row Groups

```python
# Read specific row groups (for parallel processing)
df = pl.read_parquet("data.parquet", row_groups=[0, 1, 2])

# Get metadata
import pyarrow.parquet as pq
metadata = pq.read_metadata("data.parquet")
print(f"Row groups: {metadata.num_row_groups}")
```

---

## Excel Files

Requires: `pip install polars[xlsx2csv]` or `openpyxl`

```python
# Read specific sheet
df = pl.read_excel("data.xlsx", sheet_name="Sheet1")

# Read by sheet index (0-based)
df = pl.read_excel("data.xlsx", sheet_id=0)

# Read all sheets (returns dict)
sheets = pl.read_excel("data.xlsx", sheet_name=None)
for name, df in sheets.items():
    print(f"{name}: {df.shape}")

# With options
df = pl.read_excel(
    "data.xlsx",
    sheet_name="Data",
    read_options={"skip_rows": 2, "n_rows": 100},
)
```

---

## Database Connections

Requires: `pip install connectorx` (for best performance)

### SQLite

```python
df = pl.read_database(
    query="SELECT * FROM users WHERE active = 1",
    connection="sqlite:///database.db"
)
```

### PostgreSQL

```python
df = pl.read_database(
    query="SELECT * FROM orders WHERE date > '2024-01-01'",
    connection="postgresql://user:password@localhost:5432/dbname"
)

# Using connection URI
df = pl.read_database_uri(
    query="SELECT * FROM products",
    uri="postgresql://user:pass@host:5432/db"
)
```

### MySQL

```python
df = pl.read_database(
    query="SELECT * FROM customers",
    connection="mysql://user:password@localhost:3306/dbname"
)
```

### With SQLAlchemy

```python
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:pass@localhost/db")
with engine.connect() as conn:
    df = pl.read_database(
        query="SELECT * FROM table",
        connection=conn
    )
```

---

## From Python Objects

### From Dictionary

```python
df = pl.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "salary": [50000.0, 60000.0, 70000.0],
})
```

### From List of Dicts

```python
records = [
    {"id": 1, "name": "Alice", "score": 95},
    {"id": 2, "name": "Bob", "score": 87},
    {"id": 3, "name": "Charlie", "score": 92},
]
df = pl.DataFrame(records)
```

### From NumPy Arrays

```python
import numpy as np

df = pl.DataFrame({
    "x": np.random.randn(100),
    "y": np.random.randn(100),
})

# From 2D array
arr = np.array([[1, 2, 3], [4, 5, 6]])
df = pl.DataFrame(arr, schema=["a", "b", "c"])
```

### From Pandas DataFrame

```python
import pandas as pd

pandas_df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
polars_df = pl.from_pandas(pandas_df)
```

### From Arrow Table

```python
import pyarrow as pa

table = pa.table({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
df = pl.from_arrow(table)
```

---

## Data Type Inference

### Automatic Inference

```python
# Polars infers types from first N rows
df = pl.read_csv("data.csv", infer_schema_length=10000)

# Check inferred types
print(df.schema)
```

### Override Inference

```python
# Specify types explicitly
df = pl.read_csv(
    "data.csv",
    dtypes={
        "id": pl.Int64,
        "price": pl.Float64,
        "date": pl.Date,
        "category": pl.Categorical,
    }
)
```

### Common Type Issues

```python
# Numbers with commas (e.g., "1,234.56")
df = pl.read_csv("data.csv").with_columns(
    pl.col("amount").str.replace_all(",", "").cast(pl.Float64)
)

# Dates in non-standard format
df = pl.read_csv("data.csv").with_columns(
    pl.col("date").str.to_date("%d/%m/%Y")
)

# Boolean from strings
df = df.with_columns(
    pl.col("flag").str.to_lowercase().eq("true").alias("is_flag")
)
```

---

## Performance Tips

1. **Use lazy loading** for large files: `pl.scan_csv()` instead of `pl.read_csv()`

2. **Select columns early** to avoid loading unnecessary data:
   ```python
   df = pl.scan_csv("data.csv").select(["col1", "col2"]).collect()
   ```

3. **Use Parquet** for repeated access - much faster than CSV

4. **Limit rows for exploration**:
   ```python
   df = pl.read_csv("data.csv", n_rows=1000)
   ```

5. **Use streaming** for files larger than memory:
   ```python
   result = pl.scan_csv("huge.csv").group_by("key").agg(...).collect(streaming=True)
   ```

6. **Parallelize file loading**:
   ```python
   from concurrent.futures import ThreadPoolExecutor

   files = list(Path("data").glob("*.csv"))
   with ThreadPoolExecutor() as ex:
       dfs = list(ex.map(pl.read_csv, files))
   df = pl.concat(dfs)
   ```
