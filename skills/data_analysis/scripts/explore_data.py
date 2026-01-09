#!/usr/bin/env python3
"""Quick data exploration and profiling script.

Usage:
    python explore_data.py <file_path> [--sample N] [--output report.txt]

Examples:
    python explore_data.py data.csv
    python explore_data.py data.parquet --sample 10000
    python explore_data.py data.json --output exploration_report.txt
"""

import argparse
import sys
from pathlib import Path

import polars as pl  # type: ignore[import-unresolved]
from polars import col  # type: ignore[import-unresolved]


def load_data(file_path: str, sample_size: int | None = None) -> pl.DataFrame:
    """Load data from various file formats."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    loaders = {
        ".csv": pl.read_csv,
        ".parquet": pl.read_parquet,
        ".json": pl.read_json,
        ".ndjson": pl.read_ndjson,
    }

    if suffix not in loaders:
        raise ValueError(f"Unsupported file format: {suffix}")

    df = loaders[suffix](file_path)

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, seed=42)
        print(f"Sampled {sample_size:,} rows from {len(df):,} total rows")

    return df


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}\n")


def explore_schema(df: pl.DataFrame) -> None:
    """Explore DataFrame schema and structure."""
    print_section("SCHEMA & STRUCTURE")

    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    print("Columns and Types:")
    print("-" * 40)
    for col_name, dtype in df.schema.items():
        print(f"  {col_name:<30} {dtype}")


def explore_missing(df: pl.DataFrame) -> None:
    """Analyze missing data."""
    print_section("MISSING DATA ANALYSIS")

    null_counts = df.null_count()
    total_rows = len(df)

    print(f"{'Column':<30} {'Null Count':>12} {'Null %':>10}")
    print("-" * 52)

    for col_name in df.columns:
        null_count = null_counts[col_name][0]
        null_pct = null_count / total_rows * 100 if total_rows > 0 else 0
        if null_count > 0:
            print(f"{col_name:<30} {null_count:>12,} {null_pct:>9.2f}%")

    complete_rows = df.drop_nulls().shape[0]
    print(f"\nComplete rows (no nulls): {complete_rows:,} ({complete_rows / total_rows * 100:.1f}%)")


def explore_numeric(df: pl.DataFrame) -> None:
    """Explore numeric columns."""
    numeric_cols = [
        c for c in df.columns if df.schema[c] in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
    ]

    if not numeric_cols:
        return

    print_section("NUMERIC COLUMNS")

    for col_name in numeric_cols:
        print(f"\n{col_name}:")
        print("-" * 40)

        stats = df.select(
            [
                col(col_name).count().alias("count"),
                col(col_name).mean().alias("mean"),
                col(col_name).std().alias("std"),
                col(col_name).min().alias("min"),
                col(col_name).quantile(0.25).alias("q25"),
                col(col_name).median().alias("median"),
                col(col_name).quantile(0.75).alias("q75"),
                col(col_name).max().alias("max"),
            ]
        )

        row = stats.row(0, named=True)
        print(f"  Count:    {row['count']:>12,}")
        print(f"  Mean:     {row['mean']:>12.4f}" if row["mean"] is not None else "  Mean:     N/A")
        print(f"  Std Dev:  {row['std']:>12.4f}" if row["std"] is not None else "  Std Dev:  N/A")
        print(f"  Min:      {row['min']:>12.4f}" if row["min"] is not None else "  Min:      N/A")
        print(f"  25%:      {row['q25']:>12.4f}" if row["q25"] is not None else "  25%:      N/A")
        print(f"  Median:   {row['median']:>12.4f}" if row["median"] is not None else "  Median:   N/A")
        print(f"  75%:      {row['q75']:>12.4f}" if row["q75"] is not None else "  75%:      N/A")
        print(f"  Max:      {row['max']:>12.4f}" if row["max"] is not None else "  Max:      N/A")


def explore_categorical(df: pl.DataFrame, max_unique: int = 20) -> None:
    """Explore categorical/string columns."""
    cat_cols = [c for c in df.columns if df.schema[c] in [pl.Utf8, pl.Categorical]]

    if not cat_cols:
        return

    print_section("CATEGORICAL COLUMNS")

    for col_name in cat_cols:
        n_unique = df.select(col(col_name).n_unique())[0, 0]
        print(f"\n{col_name} ({n_unique:,} unique values):")
        print("-" * 40)

        if n_unique <= max_unique:
            value_counts = df.group_by(col_name).len().sort("len", descending=True)

            total = len(df)
            for row in value_counts.iter_rows(named=True):
                val = row[col_name] if row[col_name] is not None else "<null>"
                count = row["len"]
                pct = count / total * 100
                print(f"  {val!s:<25} {count:>8,} ({pct:>5.1f}%)")
        else:
            top_values = df.group_by(col_name).len().sort("len", descending=True).head(10)
            print(f"  (showing top 10 of {n_unique:,})")
            total = len(df)
            for row in top_values.iter_rows(named=True):
                val = row[col_name] if row[col_name] is not None else "<null>"
                count = row["len"]
                pct = count / total * 100
                print(f"  {str(val)[:25]:<25} {count:>8,} ({pct:>5.1f}%)")


def explore_datetime(df: pl.DataFrame) -> None:
    """Explore datetime columns."""
    dt_cols = [c for c in df.columns if df.schema[c] in [pl.Date, pl.Datetime, pl.Time]]

    if not dt_cols:
        return

    print_section("DATETIME COLUMNS")

    for col_name in dt_cols:
        print(f"\n{col_name}:")
        print("-" * 40)

        stats = df.select(
            [
                col(col_name).min().alias("min"),
                col(col_name).max().alias("max"),
                col(col_name).n_unique().alias("unique"),
            ]
        )

        row = stats.row(0, named=True)
        print(f"  Min:     {row['min']}")
        print(f"  Max:     {row['max']}")
        print(f"  Unique:  {row['unique']:,}")

        if row["min"] is not None and row["max"] is not None:
            try:
                span = row["max"] - row["min"]
                print(f"  Span:    {span}")
            except Exception:
                pass


def explore_sample(df: pl.DataFrame, n: int = 5) -> None:
    """Show sample rows."""
    print_section(f"SAMPLE DATA (first {n} rows)")
    print(df.head(n))


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore and profile a dataset")
    parser.add_argument("file_path", help="Path to the data file (CSV, Parquet, JSON)")
    parser.add_argument("--sample", type=int, help="Sample N rows for large files")
    parser.add_argument("--output", type=str, help="Output file for report")
    args = parser.parse_args()

    # Redirect output if specified
    output_file = None
    if args.output:
        output_file = open(args.output, "w")  # noqa: SIM115
        sys.stdout = output_file

    try:
        print(f"Exploring: {args.file_path}")
        print("=" * 60)

        df = load_data(args.file_path, args.sample)

        explore_schema(df)
        explore_missing(df)
        explore_numeric(df)
        explore_categorical(df)
        explore_datetime(df)
        explore_sample(df)

        print_section("EXPLORATION COMPLETE")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if output_file:
            output_file.close()


if __name__ == "__main__":
    main()
