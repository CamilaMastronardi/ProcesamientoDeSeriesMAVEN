#!/usr/bin/env python3
"""
01b_MAG_gap_analysis.py

Detailed analysis of gaps inside uploaded MAVEN MAG source files.

A gap is computed ONLY between consecutive timestamps belonging to the same
source_file. Missing dates/files in BigQuery are never interpreted as gaps.

Default source:
    maven-vignes-crossings.maven.mag_fields

Outputs:
    results/MAG_gap_analysis/
        mag_gap_events.csv
        mag_gap_summary.csv
        mag_gap_duration_bins.csv
        mag_gaps_by_file.csv
        mag_interpolation_thresholds.csv
        figures/
            mag_gap_duration_histogram.png
            mag_gap_survival_curve.png
            mag_gap_counts_by_duration.png
            mag_missing_samples_by_duration.png
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import bigquery


DEFAULT_PROJECT = "maven-vignes-crossings"
DEFAULT_DATASET = "maven"
DEFAULT_TABLE = "mag_fields"

DEFAULT_OUTPUT_DIR = Path("results/MAG_gap_analysis")

DEFAULT_GAP_START_S = 1.5
DEFAULT_NOMINAL_CADENCE_S = 1.0

DEFAULT_INTERPOLATION_THRESHOLDS = [2.0, 3.0, 5.0, 10.0, 30.0]


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze time gaps strictly inside uploaded MAVEN MAG source files."
        )
    )

    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--table", default=DEFAULT_TABLE)

    parser.add_argument(
        "--start",
        default=None,
        help="Optional first UTC date, inclusive (YYYY-MM-DD).",
    )

    parser.add_argument(
        "--end",
        default=None,
        help="Optional last UTC date, inclusive (YYYY-MM-DD).",
    )

    parser.add_argument(
        "--gap-start",
        type=float,
        default=DEFAULT_GAP_START_S,
        help=(
            "Intervals larger than this are considered gaps. "
            f"Default: {DEFAULT_GAP_START_S} s"
        ),
    )

    parser.add_argument(
        "--nominal-cadence",
        type=float,
        default=DEFAULT_NOMINAL_CADENCE_S,
        help=(
            "Nominal MAG cadence used to estimate missing samples. "
            f"Default: {DEFAULT_NOMINAL_CADENCE_S} s"
        ),
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_INTERPOLATION_THRESHOLDS,
        help=(
            "Candidate maximum gap durations for interpolation. "
            "Default: 2 3 5 10 30"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser


def build_gap_query(project, dataset, table):
    full_table = f"{project}.{dataset}.{table}"

    return f"""
    WITH ordered AS (
      SELECT
        source_file,
        datetime_utc,
        LAG(datetime_utc) OVER (
          PARTITION BY source_file
          ORDER BY datetime_utc
        ) AS previous_time
      FROM `{full_table}`
      WHERE
        (@start_date IS NULL OR DATE(datetime_utc) >= @start_date)
        AND
        (@end_date IS NULL OR DATE(datetime_utc) <= @end_date)
    )

    SELECT
      source_file,
      DATE(datetime_utc) AS date_utc,
      previous_time AS gap_start_time,
      datetime_utc AS gap_end_time,
      TIMESTAMP_DIFF(
        datetime_utc,
        previous_time,
        MICROSECOND
      ) / 1000000.0 AS dt_s

    FROM ordered

    WHERE
      previous_time IS NOT NULL
      AND TIMESTAMP_DIFF(
        datetime_utc,
        previous_time,
        MICROSECOND
      ) / 1000000.0 > @gap_start

    ORDER BY gap_end_time
    """


def run_gap_query(client, args, start_date, end_date):
    query = build_gap_query(
        args.project,
        args.dataset,
        args.table,
    )

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "start_date",
                "DATE",
                start_date,
            ),
            bigquery.ScalarQueryParameter(
                "end_date",
                "DATE",
                end_date,
            ),
            bigquery.ScalarQueryParameter(
                "gap_start",
                "FLOAT64",
                args.gap_start,
            ),
        ]
    )

    print("Querying internal MAG gaps from BigQuery...")

    return (
        client.query(
            query,
            job_config=job_config,
        )
        .result()
        .to_dataframe()
    )


def estimate_missing_samples(dt_s, nominal_cadence):
    """
    For a nominal 1 s cadence:
        dt = 2 s -> ~1 missing sample
        dt = 3 s -> ~2 missing samples

    Rounding is used because actual MAVEN timestamps have small timing jitter.
    """
    intervals = np.rint(
        np.asarray(dt_s, dtype=float) / nominal_cadence
    ).astype(np.int64)

    return np.maximum(
        intervals - 1,
        0,
    )


def add_gap_columns(df, nominal_cadence):
    out = df.copy()

    out["gap_start_time"] = pd.to_datetime(
        out["gap_start_time"],
        utc=True,
    )

    out["gap_end_time"] = pd.to_datetime(
        out["gap_end_time"],
        utc=True,
    )

    out["estimated_missing_samples"] = estimate_missing_samples(
        out["dt_s"],
        nominal_cadence,
    )

    return out


def make_summary(gaps, nominal_cadence):
    dt = gaps["dt_s"].astype(float)
    missing = gaps["estimated_missing_samples"].astype(np.int64)

    summary = {
        "gap_count": int(len(gaps)),
        "files_with_gaps": int(gaps["source_file"].nunique()),
        "dates_with_gaps": int(gaps["date_utc"].nunique()),
        "nominal_cadence_s": float(nominal_cadence),

        "min_gap_s": float(dt.min()),
        "median_gap_s": float(dt.median()),
        "mean_gap_s": float(dt.mean()),

        "p75_gap_s": float(dt.quantile(0.75)),
        "p90_gap_s": float(dt.quantile(0.90)),
        "p95_gap_s": float(dt.quantile(0.95)),
        "p99_gap_s": float(dt.quantile(0.99)),

        "max_gap_s": float(dt.max()),

        "total_gap_duration_s": float(dt.sum()),
        "total_gap_duration_h": float(dt.sum() / 3600.0),

        "estimated_missing_samples_total": int(missing.sum()),
        "median_missing_samples_per_gap": float(missing.median()),
        "mean_missing_samples_per_gap": float(missing.mean()),
        "max_missing_samples_in_gap": int(missing.max()),
    }

    return pd.DataFrame([summary])


def duration_bins():
    """
    Bins chosen to preserve the important short-gap region and also describe
    long data outages.
    """
    edges = np.array(
        [
            1.5,
            2.5,
            3.5,
            5.0,
            10.0,
            30.0,
            60.0,
            300.0,
            1800.0,
            3600.0,
            21600.0,
            np.inf,
        ],
        dtype=float,
    )

    labels = [
        "1.5–2.5 s",
        "2.5–3.5 s",
        "3.5–5 s",
        "5–10 s",
        "10–30 s",
        "30–60 s",
        "1–5 min",
        "5–30 min",
        "30–60 min",
        "1–6 h",
        ">=6 h",
    ]

    return edges, labels


def make_duration_table(gaps):
    edges, labels = duration_bins()

    categories = pd.cut(
        gaps["dt_s"],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    temp = gaps.copy()
    temp["duration_bin"] = categories

    grouped = (
        temp.groupby(
            "duration_bin",
            observed=False,
        )
        .agg(
            gap_count=("dt_s", "size"),
            estimated_missing_samples=(
                "estimated_missing_samples",
                "sum",
            ),
        )
        .reset_index()
    )

    total_gaps = max(
        int(grouped["gap_count"].sum()),
        1,
    )

    total_missing = max(
        int(grouped["estimated_missing_samples"].sum()),
        1,
    )

    grouped["gap_fraction"] = (
        grouped["gap_count"] / total_gaps
    )

    grouped["missing_sample_fraction"] = (
        grouped["estimated_missing_samples"]
        / total_missing
    )

    return grouped


def make_by_file_table(gaps):
    grouped = (
        gaps.groupby(
            ["date_utc", "source_file"],
            as_index=False,
        )
        .agg(
            gap_count=("dt_s", "size"),
            min_gap_s=("dt_s", "min"),
            median_gap_s=("dt_s", "median"),
            mean_gap_s=("dt_s", "mean"),
            max_gap_s=("dt_s", "max"),
            estimated_missing_samples=(
                "estimated_missing_samples",
                "sum",
            ),
        )
    )

    return grouped.sort_values(
        ["estimated_missing_samples", "gap_count"],
        ascending=False,
    )


def make_threshold_table(gaps, thresholds):
    total_gaps = len(gaps)
    total_missing = int(
        gaps["estimated_missing_samples"].sum()
    )

    rows = []

    for threshold in sorted(set(thresholds)):
        eligible = gaps[
            gaps["dt_s"] <= threshold
        ]

        gap_count = len(eligible)

        missing_count = int(
            eligible["estimated_missing_samples"].sum()
        )

        rows.append(
            {
                "max_interpolated_gap_s": float(threshold),

                "eligible_gap_count": int(gap_count),

                "eligible_gap_fraction": (
                    gap_count / total_gaps
                    if total_gaps
                    else np.nan
                ),

                "estimated_samples_interpolated": (
                    missing_count
                ),

                "fraction_of_missing_samples_interpolated": (
                    missing_count / total_missing
                    if total_missing
                    else np.nan
                ),

                "files_affected": int(
                    eligible["source_file"].nunique()
                ),
            }
        )

    return pd.DataFrame(rows)


def make_log_histogram(gaps, output_path):
    values = gaps["dt_s"].to_numpy(dtype=float)

    xmin = values.min()
    xmax = values.max()

    bins = np.logspace(
        np.log10(xmin),
        np.log10(xmax),
        60,
    )

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Gap duration Δt (s)")
    plt.ylabel("Number of gaps")
    plt.title("Distribution of internal MAG gap durations")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=160,
    )
    plt.close()


def make_survival_curve(gaps, output_path):
    values = np.sort(
        gaps["dt_s"].to_numpy(dtype=float)
    )

    n = len(values)

    survival = (
        n - np.arange(n)
    ) / n

    plt.figure(figsize=(10, 6))
    plt.step(
        values,
        survival,
        where="post",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Gap duration threshold Δt (s)")
    plt.ylabel("Fraction of gaps with duration ≥ Δt")
    plt.title("MAG gap survival curve")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=160,
    )
    plt.close()


def make_duration_count_plot(duration_table, output_path):
    labels = duration_table["duration_bin"].astype(str)
    counts = duration_table["gap_count"]

    x = np.arange(len(labels))

    plt.figure(figsize=(11, 6))
    plt.bar(x, counts)
    plt.xticks(
        x,
        labels,
        rotation=45,
        ha="right",
    )
    plt.yscale("log")
    plt.xlabel("Gap duration")
    plt.ylabel("Number of gaps")
    plt.title("MAG gap count by duration range")
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=160,
    )
    plt.close()


def make_missing_sample_plot(duration_table, output_path):
    labels = duration_table["duration_bin"].astype(str)

    values = duration_table[
        "estimated_missing_samples"
    ]

    x = np.arange(len(labels))

    plt.figure(figsize=(11, 6))
    plt.bar(x, values)
    plt.xticks(
        x,
        labels,
        rotation=45,
        ha="right",
    )
    plt.yscale("log")
    plt.xlabel("Gap duration")
    plt.ylabel("Estimated missing MAG samples")
    plt.title(
        "Estimated missing samples contributed by each gap-duration range"
    )
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=160,
    )
    plt.close()


def pct(value):
    if pd.isna(value):
        return "NaN"
    return f"{100.0 * float(value):.2f}%"


def print_summary(summary, thresholds):
    row = summary.iloc[0]

    print()
    print("=" * 72)
    print("MAVEN MAG — INTERNAL GAP ANALYSIS")
    print("=" * 72)

    print(
        f"Total gaps:                         "
        f"{int(row['gap_count']):,}"
    )

    print(
        f"Files with gaps:                    "
        f"{int(row['files_with_gaps']):,}"
    )

    print(
        f"Median gap duration:                "
        f"{row['median_gap_s']:.3f} s"
    )

    print(
        f"Mean gap duration:                  "
        f"{row['mean_gap_s']:.3f} s"
    )

    print(
        f"90th percentile:                    "
        f"{row['p90_gap_s']:.3f} s"
    )

    print(
        f"95th percentile:                    "
        f"{row['p95_gap_s']:.3f} s"
    )

    print(
        f"99th percentile:                    "
        f"{row['p99_gap_s']:.3f} s"
    )

    print(
        f"Maximum gap:                        "
        f"{row['max_gap_s']:.3f} s"
    )

    print(
        f"Estimated missing samples:          "
        f"{int(row['estimated_missing_samples_total']):,}"
    )

    print(
        f"Total duration represented by gaps: "
        f"{row['total_gap_duration_h']:.2f} h"
    )

    print()
    print("Candidate interpolation thresholds:")

    for _, threshold_row in thresholds.iterrows():
        print(
            f"  <= {threshold_row['max_interpolated_gap_s']:>5.1f} s : "
            f"{int(threshold_row['eligible_gap_count']):>5,} gaps "
            f"({pct(threshold_row['eligible_gap_fraction'])}), "
            f"{int(threshold_row['estimated_samples_interpolated']):>8,} "
            f"estimated samples "
            f"({pct(threshold_row['fraction_of_missing_samples_interpolated'])})"
        )

    print("=" * 72)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.gap_start <= 0:
        parser.error("--gap-start must be > 0.")

    if args.nominal_cadence <= 0:
        parser.error("--nominal-cadence must be > 0.")

    if any(x <= args.gap_start for x in args.thresholds):
        parser.error(
            "Every interpolation threshold must be greater than --gap-start."
        )

    start_date = (
        pd.Timestamp(args.start).date()
        if args.start
        else None
    )

    end_date = (
        pd.Timestamp(args.end).date()
        if args.end
        else None
    )

    if (
        start_date is not None
        and end_date is not None
        and start_date > end_date
    ):
        parser.error("--start cannot be later than --end.")

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir = (
        args.output_dir
        / "figures"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = bigquery.Client(
        project=args.project
    )

    gaps = run_gap_query(
        client,
        args,
        start_date,
        end_date,
    )

    if gaps.empty:
        print("No internal MAG gaps were found.")
        return

    gaps = add_gap_columns(
        gaps,
        args.nominal_cadence,
    )

    summary = make_summary(
        gaps,
        args.nominal_cadence,
    )

    duration_table = make_duration_table(
        gaps
    )

    by_file = make_by_file_table(
        gaps
    )

    thresholds = make_threshold_table(
        gaps,
        args.thresholds,
    )

    gap_events_path = (
        args.output_dir
        / "mag_gap_events.csv"
    )

    summary_path = (
        args.output_dir
        / "mag_gap_summary.csv"
    )

    duration_path = (
        args.output_dir
        / "mag_gap_duration_bins.csv"
    )

    by_file_path = (
        args.output_dir
        / "mag_gaps_by_file.csv"
    )

    thresholds_path = (
        args.output_dir
        / "mag_interpolation_thresholds.csv"
    )

    gaps.to_csv(
        gap_events_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    duration_table.to_csv(
        duration_path,
        index=False,
    )

    by_file.to_csv(
        by_file_path,
        index=False,
    )

    thresholds.to_csv(
        thresholds_path,
        index=False,
    )

    make_log_histogram(
        gaps,
        figures_dir
        / "mag_gap_duration_histogram.png",
    )

    make_survival_curve(
        gaps,
        figures_dir
        / "mag_gap_survival_curve.png",
    )

    make_duration_count_plot(
        duration_table,
        figures_dir
        / "mag_gap_counts_by_duration.png",
    )

    make_missing_sample_plot(
        duration_table,
        figures_dir
        / "mag_missing_samples_by_duration.png",
    )

    print_summary(
        summary,
        thresholds,
    )

    print()
    print(f"Gap events:           {gap_events_path}")
    print(f"Summary:              {summary_path}")
    print(f"Duration bins:        {duration_path}")
    print(f"Gaps by file:         {by_file_path}")
    print(f"Interpolation table:  {thresholds_path}")
    print(f"Figures:              {figures_dir}")


if __name__ == "__main__":
    main()
