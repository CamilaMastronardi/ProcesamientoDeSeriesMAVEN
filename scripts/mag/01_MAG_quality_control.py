#!/usr/bin/env python3
"""
01_MAG_quality_control.py

Quality-control summary for all MAVEN MAG dates available in BigQuery.

Outputs
-------
1. mag_quality_by_day.csv
   One row per available UTC date.

2. mag_quality_summary.csv
   Global totals and averages across all available dates.

3. Global figures:
   - Histogram of delta t computed only within each uploaded source file
   - Scatter plot of candidate gaps computed only within each uploaded source file

Default BigQuery source
-----------------------
maven-vignes-crossings.maven.mag_fields
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

DEFAULT_OUTPUT_DIR = Path("results/MAG_quality")

DEFAULT_GAP_START_S = 1.5
DEFAULT_SMALL_GAP_END_S = 3.0
DEFAULT_MEDIUM_GAP_END_S = 30.0


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute MAVEN MAG quality-control statistics for every available "
            "date in BigQuery and generate global QC figures."
        )
    )

    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--start", default=None, help="Optional first UTC date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Optional last UTC date (YYYY-MM-DD).")
    parser.add_argument("--gap-start", type=float, default=DEFAULT_GAP_START_S)
    parser.add_argument("--small-gap-end", type=float, default=DEFAULT_SMALL_GAP_END_S)
    parser.add_argument("--medium-gap-end", type=float, default=DEFAULT_MEDIUM_GAP_END_S)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    return parser


def build_daily_query(project, dataset, table):
    """
    Daily QC query.

    IMPORTANT:
    LAG is partitioned by source_file, not by calendar continuity.
    Therefore a missing file/day in BigQuery is never interpreted as a gap.
    Only missing samples inside an uploaded MAG file can produce a gap.
    """
    full_table = f"{project}.{dataset}.{table}"

    return f"""
    WITH base AS (
      SELECT
        DATE(datetime_utc) AS date_utc,
        datetime_utc,
        bx_ss_nt,
        by_ss_nt,
        bz_ss_nt,
        b_total_nt,
        range_b,
        source_file,
        LAG(datetime_utc) OVER (
          PARTITION BY source_file
          ORDER BY datetime_utc
        ) AS previous_time
      FROM `{full_table}`
      WHERE
        (@start_date IS NULL OR DATE(datetime_utc) >= @start_date)
        AND (@end_date IS NULL OR DATE(datetime_utc) <= @end_date)
    ),
    with_dt AS (
      SELECT
        *,
        TIMESTAMP_DIFF(datetime_utc, previous_time, MICROSECOND) / 1000000.0 AS dt_s,
        (
          bx_ss_nt IS NOT NULL AND NOT IS_NAN(bx_ss_nt) AND NOT IS_INF(bx_ss_nt)
          AND by_ss_nt IS NOT NULL AND NOT IS_NAN(by_ss_nt) AND NOT IS_INF(by_ss_nt)
          AND bz_ss_nt IS NOT NULL AND NOT IS_NAN(bz_ss_nt) AND NOT IS_INF(bz_ss_nt)
          AND b_total_nt IS NOT NULL AND NOT IS_NAN(b_total_nt) AND NOT IS_INF(b_total_nt)
        ) AS valid_b
      FROM base
    ),
    daily_basic AS (
      SELECT
        date_utc,
        COUNT(*) AS sample_count,
        COUNTIF(valid_b) AS valid_sample_count,
        COUNTIF(NOT valid_b) AS invalid_sample_count,
        COUNTIF(bx_ss_nt IS NULL OR IS_NAN(bx_ss_nt) OR IS_INF(bx_ss_nt)) AS invalid_bx_count,
        COUNTIF(by_ss_nt IS NULL OR IS_NAN(by_ss_nt) OR IS_INF(by_ss_nt)) AS invalid_by_count,
        COUNTIF(bz_ss_nt IS NULL OR IS_NAN(bz_ss_nt) OR IS_INF(bz_ss_nt)) AS invalid_bz_count,
        COUNTIF(b_total_nt IS NULL OR IS_NAN(b_total_nt) OR IS_INF(b_total_nt)) AS invalid_b_total_count,
        COUNT(*) - COUNT(DISTINCT datetime_utc) AS duplicate_timestamp_count,
        MIN(datetime_utc) AS first_time,
        MAX(datetime_utc) AS last_time,
        COUNT(DISTINCT source_file) AS source_file_count,
        COUNTIF(range_b IS NULL) AS missing_range_count
      FROM with_dt
      GROUP BY date_utc
    ),
    daily_dt AS (
      SELECT
        date_utc,
        COUNTIF(dt_s > 0) AS positive_interval_count,
        COUNTIF(dt_s = 0) AS zero_interval_count,
        COUNTIF(dt_s < 0) AS negative_interval_count,
        AVG(IF(dt_s > 0, dt_s, NULL)) AS mean_dt_s,
        APPROX_QUANTILES(IF(dt_s > 0, dt_s, NULL), 100)[OFFSET(50)] AS median_dt_s,
        APPROX_QUANTILES(IF(dt_s > 0, dt_s, NULL), 100)[OFFSET(95)] AS p95_dt_s,
        MAX(IF(dt_s > 0, dt_s, NULL)) AS max_dt_s,
        COUNTIF(dt_s > @gap_start AND dt_s < @small_gap_end) AS small_gap_count,
        COUNTIF(dt_s >= @small_gap_end AND dt_s < @medium_gap_end) AS medium_gap_count,
        COUNTIF(dt_s >= @medium_gap_end) AS large_gap_count,
        COUNTIF(dt_s > @gap_start) AS total_gap_count,
        AVG(IF(dt_s > @gap_start, dt_s, NULL)) AS mean_gap_duration_s,
        MAX(IF(dt_s > @gap_start, dt_s, NULL)) AS largest_gap_s
      FROM with_dt
      GROUP BY date_utc
    )
    SELECT
      b.date_utc,
      b.sample_count,
      b.valid_sample_count,
      b.invalid_sample_count,
      SAFE_DIVIDE(b.valid_sample_count, b.sample_count) AS valid_fraction,
      SAFE_DIVIDE(b.invalid_sample_count, b.sample_count) AS invalid_fraction,
      b.invalid_bx_count,
      b.invalid_by_count,
      b.invalid_bz_count,
      b.invalid_b_total_count,
      b.duplicate_timestamp_count,
      b.first_time,
      b.last_time,
      TIMESTAMP_DIFF(b.last_time, b.first_time, MICROSECOND) / 1000000.0 AS observed_span_s,
      b.source_file_count,
      b.missing_range_count,
      d.positive_interval_count,
      d.zero_interval_count,
      d.negative_interval_count,
      d.mean_dt_s,
      d.median_dt_s,
      d.p95_dt_s,
      d.max_dt_s,
      d.small_gap_count,
      d.medium_gap_count,
      d.large_gap_count,
      d.total_gap_count,
      d.mean_gap_duration_s,
      d.largest_gap_s
    FROM daily_basic AS b
    LEFT JOIN daily_dt AS d USING (date_utc)
    ORDER BY date_utc
    """


def build_dt_histogram_query(project, dataset, table):
    full_table = f"{project}.{dataset}.{table}"

    return f"""
    WITH base AS (
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
        AND (@end_date IS NULL OR DATE(datetime_utc) <= @end_date)
    ),
    with_dt AS (
      SELECT
        TIMESTAMP_DIFF(datetime_utc, previous_time, MICROSECOND) / 1000000.0 AS dt_s
      FROM base
    )
    SELECT
      RANGE_BUCKET(dt_s, @bin_edges) AS bucket,
      COUNT(*) AS bucket_count
    FROM with_dt
    WHERE dt_s > 0
    GROUP BY bucket
    ORDER BY bucket
    """


def build_gap_events_query(project, dataset, table):
    full_table = f"{project}.{dataset}.{table}"

    return f"""
    WITH base AS (
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
        AND (@end_date IS NULL OR DATE(datetime_utc) <= @end_date)
    )
    SELECT
      source_file,
      datetime_utc,
      TIMESTAMP_DIFF(datetime_utc, previous_time, MICROSECOND) / 1000000.0 AS dt_s
    FROM base
    WHERE
      previous_time IS NOT NULL
      AND TIMESTAMP_DIFF(datetime_utc, previous_time, MICROSECOND) / 1000000.0 > @gap_start
    ORDER BY datetime_utc
    """


def run_daily_query(client, args, start_date, end_date):
    query = build_daily_query(args.project, args.dataset, args.table)
    cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
            bigquery.ScalarQueryParameter("gap_start", "FLOAT64", args.gap_start),
            bigquery.ScalarQueryParameter("small_gap_end", "FLOAT64", args.small_gap_end),
            bigquery.ScalarQueryParameter("medium_gap_end", "FLOAT64", args.medium_gap_end),
        ]
    )
    print("Running MAG daily quality-control query in BigQuery...")
    return client.query(query, job_config=cfg).result().to_dataframe()


def run_dt_histogram_query(client, args, start_date, end_date, bin_edges):
    query = build_dt_histogram_query(args.project, args.dataset, args.table)
    cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
            bigquery.ArrayQueryParameter("bin_edges", "FLOAT64", list(bin_edges)),
        ]
    )
    print("Running MAG global dt histogram query in BigQuery...")
    return client.query(query, job_config=cfg).result().to_dataframe()


def run_gap_events_query(client, args, start_date, end_date):
    query = build_gap_events_query(args.project, args.dataset, args.table)
    cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
            bigquery.ScalarQueryParameter("gap_start", "FLOAT64", args.gap_start),
        ]
    )
    print("Running MAG global gap-events query in BigQuery...")
    return client.query(query, job_config=cfg).result().to_dataframe()


def safe_mean(series):
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return np.nan
    return float(values.mean())


def safe_median(series):
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return np.nan
    return float(values.median())


def safe_max(series):
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return np.nan
    return float(values.max())


def make_summary(df):
    total_samples = int(df["sample_count"].sum())
    total_valid = int(df["valid_sample_count"].sum())
    total_invalid = int(df["invalid_sample_count"].sum())

    summary = {
        "available_days": int(len(df)),
        "available_source_files": int(df["source_file_count"].sum()),
        "first_date": df["date_utc"].min() if len(df) else None,
        "last_date": df["date_utc"].max() if len(df) else None,
        "total_samples": total_samples,
        "total_valid_samples": total_valid,
        "total_invalid_samples": total_invalid,
        "weighted_valid_fraction": total_valid / total_samples if total_samples else np.nan,
        "mean_samples_per_day": safe_mean(df["sample_count"]),
        "mean_valid_fraction_per_day": safe_mean(df["valid_fraction"]),
        "median_valid_fraction_per_day": safe_median(df["valid_fraction"]),
        "mean_duplicate_timestamps_per_day": safe_mean(df["duplicate_timestamp_count"]),
        "total_duplicate_timestamps": int(df["duplicate_timestamp_count"].sum()),
        "mean_daily_median_cadence_s": safe_mean(df["median_dt_s"]),
        "median_of_daily_median_cadence_s": safe_median(df["median_dt_s"]),
        "mean_daily_p95_cadence_s": safe_mean(df["p95_dt_s"]),
        "mean_small_gaps_per_day": safe_mean(df["small_gap_count"]),
        "mean_medium_gaps_per_day": safe_mean(df["medium_gap_count"]),
        "mean_large_gaps_per_day": safe_mean(df["large_gap_count"]),
        "mean_total_gaps_per_day": safe_mean(df["total_gap_count"]),
        "total_small_gaps": int(df["small_gap_count"].sum()),
        "total_medium_gaps": int(df["medium_gap_count"].sum()),
        "total_large_gaps": int(df["large_gap_count"].sum()),
        "total_gaps": int(df["total_gap_count"].sum()),
        "mean_daily_gap_duration_s": safe_mean(df["mean_gap_duration_s"]),
        "largest_gap_s": safe_max(df["largest_gap_s"]),
        "mean_source_files_per_day": safe_mean(df["source_file_count"]),
        "days_with_invalid_samples": int((df["invalid_sample_count"] > 0).sum()),
        "days_with_duplicate_timestamps": int((df["duplicate_timestamp_count"] > 0).sum()),
        "days_with_large_gaps": int((df["large_gap_count"] > 0).sum()),
        "days_with_negative_dt": int((df["negative_interval_count"] > 0).sum()),
    }

    return pd.DataFrame([summary])


def pct(value):
    if pd.isna(value):
        return "NaN"
    return f"{100.0 * float(value):.4f}%"


def fmt(value, decimals=3):
    if pd.isna(value):
        return "NaN"
    return f"{float(value):,.{decimals}f}"


def print_summary(summary, gap_start, small_end, medium_end):
    row = summary.iloc[0]

    print()
    print("=" * 68)
    print("MAVEN MAG QUALITY CONTROL — GLOBAL SUMMARY")
    print("=" * 68)
    print(f"Available dates in BigQuery:    {int(row['available_days']):,}")
    print(f"Available MAG source files:     {int(row['available_source_files']):,}")
    print(f"Date range:                     {row['first_date']} -> {row['last_date']}")
    print(f"Total samples:                  {int(row['total_samples']):,}")
    print(f"Mean samples/day:               {fmt(row['mean_samples_per_day'], 1)}")
    print(f"Weighted valid fraction:        {pct(row['weighted_valid_fraction'])}")
    print(f"Mean daily valid fraction:      {pct(row['mean_valid_fraction_per_day'])}")
    print(f"Days with invalid samples:      {int(row['days_with_invalid_samples']):,}")
    print(f"Duplicate timestamps total:     {int(row['total_duplicate_timestamps']):,}")
    print(f"Days with duplicates:           {int(row['days_with_duplicate_timestamps']):,}")
    print(f"Mean daily median cadence:      {fmt(row['mean_daily_median_cadence_s'])} s")
    print(f"Median daily median cadence:    {fmt(row['median_of_daily_median_cadence_s'])} s")
    print()
    print("Gap definition:")
    print(f"  candidate gap: dt > {gap_start:g} s")
    print(f"  small:  {gap_start:g} < dt < {small_end:g} s")
    print(f"  medium: {small_end:g} <= dt < {medium_end:g} s")
    print(f"  large:  dt >= {medium_end:g} s")
    print()
    print(f"Total candidate gaps:           {int(row['total_gaps']):,}")
    print(f"  small:                        {int(row['total_small_gaps']):,}")
    print(f"  medium:                       {int(row['total_medium_gaps']):,}")
    print(f"  large:                        {int(row['total_large_gaps']):,}")
    print(f"Mean gaps/day:                  {fmt(row['mean_total_gaps_per_day'])}")
    print(f"Days with large gaps:           {int(row['days_with_large_gaps']):,}")
    print(f"Largest intra-day gap:          {fmt(row['largest_gap_s'])} s")
    print(f"Days with negative dt:          {int(row['days_with_negative_dt']):,}")
    print("=" * 68)


def dt_bucket_labels(bin_edges):
    labels = []
    previous = 0.0
    for edge in bin_edges:
        labels.append(f"{previous:g}–{edge:g}")
        previous = edge
    labels.append(f">={bin_edges[-1]:g}")
    return labels


def make_global_dt_histogram(hist_df, bin_edges, output_path):
    labels = dt_bucket_labels(bin_edges)
    counts = np.zeros(len(labels), dtype=int)

    if not hist_df.empty:
        for _, row in hist_df.iterrows():
            bucket = int(row["bucket"])
            if 0 <= bucket < len(counts):
                counts[bucket] = int(row["bucket_count"])

    x = np.arange(len(labels))

    plt.figure(figsize=(14, 6))
    plt.bar(x, counts)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Δt bin (s)")
    plt.ylabel("Count")
    plt.title("Global histogram of MAG Δt within uploaded source files")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def make_global_gap_plot(gap_df, args, output_path):
    plt.figure(figsize=(14, 6))

    if not gap_df.empty:
        gap_df = gap_df.copy()
        gap_df["datetime_utc"] = pd.to_datetime(gap_df["datetime_utc"], utc=True)
        plt.scatter(gap_df["datetime_utc"], gap_df["dt_s"], s=8)
        plt.axhline(args.small_gap_end, linestyle="--")
        plt.axhline(args.medium_gap_end, linestyle="--")
        plt.yscale("log")
        plt.ylabel("Gap duration Δt (s)")
    else:
        plt.text(
            0.5,
            0.5,
            "No candidate gaps found",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.ylabel("Gap duration Δt (s)")

    plt.xlabel("Time (UTC)")
    plt.title("Candidate MAG gaps within uploaded source files")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.gap_start <= 0:
        parser.error("--gap-start must be > 0.")

    if args.small_gap_end <= args.gap_start:
        parser.error("--small-gap-end must be greater than --gap-start.")

    if args.medium_gap_end <= args.small_gap_end:
        parser.error("--medium-gap-end must be greater than --small-gap-end.")

    start_date = pd.Timestamp(args.start).date() if args.start else None
    end_date = pd.Timestamp(args.end).date() if args.end else None

    if start_date is not None and end_date is not None and start_date > end_date:
        parser.error("--start cannot be later than --end.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    client = bigquery.Client(project=args.project)

    daily = run_daily_query(client, args, start_date, end_date)

    if daily.empty:
        print("No MAG data found for the selected interval.")
        return

    daily["date_utc"] = pd.to_datetime(daily["date_utc"]).dt.date
    summary = make_summary(daily)

    daily_path = args.output_dir / "mag_quality_by_day.csv"
    summary_path = args.output_dir / "mag_quality_summary.csv"

    daily.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)

    # Global dt histogram bin edges in seconds
    # Chosen to resolve the near-1 s cadence while also keeping large gaps visible.
    bin_edges = np.array(
        [
            0.5,
            0.9,
            0.99,
            1.01,
            1.1,
            1.5,
            2.0,
            3.0,
            5.0,
            10.0,
            30.0,
            60.0,
            300.0,
            1800.0,
            3600.0,
            21600.0,
            86400.0,
        ],
        dtype=float,
    )

    hist_df = run_dt_histogram_query(
        client=client,
        args=args,
        start_date=start_date,
        end_date=end_date,
        bin_edges=bin_edges,
    )

    gap_df = run_gap_events_query(
        client=client,
        args=args,
        start_date=start_date,
        end_date=end_date,
    )

    dt_hist_path = figures_dir / "mag_dt_histogram_all_dates.png"
    gap_plot_path = figures_dir / "mag_gaps_all_dates.png"

    make_global_dt_histogram(
        hist_df=hist_df,
        bin_edges=bin_edges,
        output_path=dt_hist_path,
    )

    make_global_gap_plot(
        gap_df=gap_df,
        args=args,
        output_path=gap_plot_path,
    )

    print_summary(
        summary,
        gap_start=args.gap_start,
        small_end=args.small_gap_end,
        medium_end=args.medium_gap_end,
    )

    print()
    print(f"Daily results:      {daily_path}")
    print(f"Global summary:     {summary_path}")
    print(f"Δt histogram:       {dt_hist_path}")
    print(f"Gap figure:         {gap_plot_path}")


if __name__ == "__main__":
    main()
