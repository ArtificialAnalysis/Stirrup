#!/usr/bin/env python3
"""Generate comprehensive summary statistics report.

Usage:
    python summary_stats.py <file_path> [--group-by COLUMN] [--output stats.csv]

Examples:
    python summary_stats.py data.csv
    python summary_stats.py data.csv --group-by category
    python summary_stats.py data.csv --group-by category --output summary.csv
"""

import argparse
import sys
from pathlib import Path

import polars as pl  # type: ignore[import-unresolved]
from polars import col  # type: ignore[import-unresolved]


def load_data(file_path: str) -> pl.DataFrame:
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

    return loaders[suffix](file_path)


def compute_numeric_stats(df: pl.DataFrame, numeric_cols: list[str], group_by: str | None = None) -> pl.DataFrame:
    """Compute comprehensive statistics for numeric columns."""
    agg_exprs = []

    for col_name in numeric_cols:
        agg_exprs.extend(
            [
                col(col_name).count().alias(f"{col_name}_count"),
                col(col_name).null_count().alias(f"{col_name}_null_count"),
                col(col_name).mean().alias(f"{col_name}_mean"),
                col(col_name).std().alias(f"{col_name}_std"),
                col(col_name).min().alias(f"{col_name}_min"),
                col(col_name).quantile(0.25).alias(f"{col_name}_q25"),
                col(col_name).median().alias(f"{col_name}_median"),
                col(col_name).quantile(0.75).alias(f"{col_name}_q75"),
                col(col_name).max().alias(f"{col_name}_max"),
                col(col_name).sum().alias(f"{col_name}_sum"),
            ]
        )

    if group_by:
        return df.group_by(group_by).agg(agg_exprs).sort(group_by)
    else:
        return df.select(agg_exprs)


def compute_categorical_stats(
    df: pl.DataFrame, cat_cols: list[str], group_by: str | None = None
) -> dict[str, pl.DataFrame]:
    """Compute value counts for categorical columns."""
    results = {}

    for col_name in cat_cols:
        if group_by and col_name != group_by:
            counts = df.group_by([group_by, col_name]).len().sort([group_by, "len"], descending=[False, True])
        else:
            counts = df.group_by(col_name).len().sort("len", descending=True)

        results[col_name] = counts

    return results


def format_stats_table(stats: pl.DataFrame, numeric_cols: list[str]) -> str:
    """Format statistics into a readable table."""
    lines = []

    # Check if grouped
    is_grouped = any(
        c
        not in [
            f"{nc}_{s}"
            for nc in numeric_cols
            for s in ["count", "null_count", "mean", "std", "min", "q25", "median", "q75", "max", "sum"]
        ]
        for c in stats.columns
    )

    if is_grouped:
        group_col = next(
            c
            for c in stats.columns
            if not any(
                c.endswith(f"_{s}")
                for s in ["count", "null_count", "mean", "std", "min", "q25", "median", "q75", "max", "sum"]
            )
        )

        for col_name in numeric_cols:
            lines.append(f"\n{'=' * 80}")
            lines.append(f" Statistics for: {col_name}")
            lines.append(f"{'=' * 80}\n")

            header = f"{'Group':<20} {'Count':>10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Median':>12} {'Max':>12}"
            lines.append(header)
            lines.append("-" * len(header))

            for row in stats.iter_rows(named=True):
                group_val = str(row[group_col])[:20]
                count = row.get(f"{col_name}_count", 0)
                mean = row.get(f"{col_name}_mean")
                std = row.get(f"{col_name}_std")
                min_val = row.get(f"{col_name}_min")
                median = row.get(f"{col_name}_median")
                max_val = row.get(f"{col_name}_max")

                mean_str = f"{mean:>12.4f}" if mean is not None else f"{'N/A':>12}"
                std_str = f"{std:>12.4f}" if std is not None else f"{'N/A':>12}"
                min_str = f"{min_val:>12.4f}" if min_val is not None else f"{'N/A':>12}"
                median_str = f"{median:>12.4f}" if median is not None else f"{'N/A':>12}"
                max_str = f"{max_val:>12.4f}" if max_val is not None else f"{'N/A':>12}"

                lines.append(f"{group_val:<20} {count:>10,} {mean_str} {std_str} {min_str} {median_str} {max_str}")
    else:
        lines.append(f"\n{'=' * 80}")
        lines.append(" SUMMARY STATISTICS")
        lines.append(f"{'=' * 80}\n")

        header = f"{'Column':<25} {'Count':>10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Median':>12} {'Max':>12}"
        lines.append(header)
        lines.append("-" * len(header))

        row = stats.row(0, named=True)
        for col_name in numeric_cols:
            count = row.get(f"{col_name}_count", 0)
            mean = row.get(f"{col_name}_mean")
            std = row.get(f"{col_name}_std")
            min_val = row.get(f"{col_name}_min")
            median = row.get(f"{col_name}_median")
            max_val = row.get(f"{col_name}_max")

            mean_str = f"{mean:>12.4f}" if mean is not None else f"{'N/A':>12}"
            std_str = f"{std:>12.4f}" if std is not None else f"{'N/A':>12}"
            min_str = f"{min_val:>12.4f}" if min_val is not None else f"{'N/A':>12}"
            median_str = f"{median:>12.4f}" if median is not None else f"{'N/A':>12}"
            max_str = f"{max_val:>12.4f}" if max_val is not None else f"{'N/A':>12}"

            lines.append(f"{col_name:<25} {count:>10,} {mean_str} {std_str} {min_str} {median_str} {max_str}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summary statistics report")
    parser.add_argument("file_path", help="Path to the data file")
    parser.add_argument("--group-by", dest="group_by", help="Column to group statistics by")
    parser.add_argument("--output", help="Output CSV file for statistics")
    args = parser.parse_args()

    try:
        print(f"Loading: {args.file_path}")
        df = load_data(args.file_path)
        print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

        # Identify column types
        numeric_cols = [
            c for c in df.columns if df.schema[c] in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]
        ]
        cat_cols = [c for c in df.columns if df.schema[c] in [pl.Utf8, pl.Categorical]]

        # Compute numeric statistics
        if numeric_cols:
            print(f"Computing statistics for {len(numeric_cols)} numeric columns...")
            numeric_stats = compute_numeric_stats(df, numeric_cols, args.group_by)

            # Print formatted table
            print(format_stats_table(numeric_stats, numeric_cols))

            # Save to CSV if requested
            if args.output:
                numeric_stats.write_csv(args.output)
                print(f"\nStatistics saved to: {args.output}")

        # Summary of categorical columns
        if cat_cols:
            print(f"\n{'=' * 80}")
            print(" CATEGORICAL COLUMNS SUMMARY")
            print(f"{'=' * 80}\n")

            for col_name in cat_cols:
                n_unique = df.select(col(col_name).n_unique())[0, 0]
                null_count = df.select(col(col_name).null_count())[0, 0]
                print(f"{col_name}: {n_unique:,} unique values, {null_count:,} nulls")

        print(f"\n{'=' * 80}")
        print(" REPORT COMPLETE")
        print(f"{'=' * 80}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
