#!/usr/bin/env python3
"""
MAVEN MAG — Boundary continuity analysis.

Uses:
    results/MAG_quality/mag_quality_by_day.csv

Creates:
    results/MAG_boundary_analysis/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_CSV = Path("results/MAG_quality/mag_quality_by_day.csv")
OUTPUT_DIR = Path("results/MAG_boundary_analysis")
FIGURE_DIR = OUTPUT_DIR / "figures"

EXPECTED_CADENCE_S = 1.0
GAP_THRESHOLD_S = 1.5
MEDIUM_THRESHOLD_S = 3.0
LARGE_THRESHOLD_S = 30.0


def find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(
        f"Could not identify the {label} column. "
        f"Available columns: {list(df.columns)}"
    )


def classify_boundary(dt_seconds: float) -> str:
    if pd.isna(dt_seconds):
        return "invalid"
    if dt_seconds < 0:
        return "overlap"
    if dt_seconds <= GAP_THRESHOLD_S:
        return "continuous"
    if dt_seconds < MEDIUM_THRESHOLD_S:
        return "small"
    if dt_seconds < LARGE_THRESHOLD_S:
        return "medium"
    return "large"


def survival_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return np.array([]), np.array([])
    x = np.sort(values)
    y = (len(x) - np.arange(len(x))) / len(x)
    return x, y


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_CSV}. "
            "Run 01_MAG_quality_control.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    date_col = find_column(
        df,
        ["date_utc", "date", "day", "file_date", "source_date"],
        "date",
    )
    first_col = find_column(
        df,
        ["first_time", "first_timestamp", "first_datetime", "min_time", "time_first"],
        "first timestamp",
    )
    last_col = find_column(
        df,
        ["last_time", "last_timestamp", "last_datetime", "max_time", "time_last"],
        "last timestamp",
    )

    data = df[[date_col, first_col, last_col]].copy()

    data["date_utc"] = pd.to_datetime(data[date_col], errors="coerce").dt.normalize()
    data["first_time"] = pd.to_datetime(data[first_col], utc=True, errors="coerce")
    data["last_time"] = pd.to_datetime(data[last_col], utc=True, errors="coerce")

    data = (
        data.dropna(subset=["date_utc"])
        .sort_values(["date_utc", "first_time"])
        .reset_index(drop=True)
    )

    if len(data) < 2:
        raise RuntimeError("At least two daily files are required.")

    boundaries = pd.DataFrame({
        "date_before": data["date_utc"].iloc[:-1].dt.date.astype(str).to_numpy(),
        "last_time_before": data["last_time"].iloc[:-1].to_numpy(),
        "date_after": data["date_utc"].iloc[1:].dt.date.astype(str).to_numpy(),
        "first_time_after": data["first_time"].iloc[1:].to_numpy(),
    })

    date_before = pd.to_datetime(boundaries["date_before"])
    date_after = pd.to_datetime(boundaries["date_after"])

    boundaries["calendar_day_step"] = (date_after - date_before).dt.days
    boundaries["missing_calendar_days"] = (
        boundaries["calendar_day_step"] - 1
    ).clip(lower=0)

    last_before = pd.to_datetime(
        boundaries["last_time_before"], utc=True, errors="coerce"
    )
    first_after = pd.to_datetime(
        boundaries["first_time_after"], utc=True, errors="coerce"
    )

    boundaries["boundary_dt_s"] = (
        first_after - last_before
    ).dt.total_seconds()

    boundaries["category"] = boundaries["boundary_dt_s"].apply(
        classify_boundary
    )

    boundaries["missing_time_s"] = np.maximum(
        boundaries["boundary_dt_s"] - EXPECTED_CADENCE_S, 0.0
    )
    boundaries["overlap_time_s"] = np.maximum(
        -boundaries["boundary_dt_s"], 0.0
    )

    boundaries["consecutive_calendar_days"] = (
        boundaries["calendar_day_step"] == 1
    )
    boundaries["has_missing_calendar_days"] = (
        boundaries["missing_calendar_days"] > 0
    )

    events_path = OUTPUT_DIR / "mag_boundary_events.csv"
    boundaries.to_csv(events_path, index=False)

    counts = boundaries["category"].value_counts()
    finite_dt = boundaries.loc[
        np.isfinite(boundaries["boundary_dt_s"]), "boundary_dt_s"
    ]
    positive_dt = finite_dt[finite_dt > 0]
    gap_rows = boundaries[
        boundaries["category"].isin(["small", "medium", "large"])
    ]

    summary = {
        "daily_files": len(data),
        "total_boundaries": len(boundaries),
        "consecutive_day_boundaries": int(
            boundaries["consecutive_calendar_days"].sum()
        ),
        "boundaries_spanning_missing_days": int(
            boundaries["has_missing_calendar_days"].sum()
        ),
        "total_missing_calendar_days": int(
            boundaries["missing_calendar_days"].sum()
        ),
        "continuous_boundaries": int(counts.get("continuous", 0)),
        "boundary_gaps": len(gap_rows),
        "small_boundary_gaps": int(counts.get("small", 0)),
        "medium_boundary_gaps": int(counts.get("medium", 0)),
        "large_boundary_gaps": int(counts.get("large", 0)),
        "overlapping_boundaries": int(counts.get("overlap", 0)),
        "invalid_boundaries": int(counts.get("invalid", 0)),
        "median_boundary_dt_s": float(positive_dt.median()) if len(positive_dt) else np.nan,
        "p95_boundary_dt_s": float(positive_dt.quantile(0.95)) if len(positive_dt) else np.nan,
        "p99_boundary_dt_s": float(positive_dt.quantile(0.99)) if len(positive_dt) else np.nan,
        "largest_boundary_dt_s": float(positive_dt.max()) if len(positive_dt) else np.nan,
        "total_boundary_missing_time_s": float(gap_rows["missing_time_s"].sum()),
        "largest_boundary_missing_time_s": (
            float(gap_rows["missing_time_s"].max()) if len(gap_rows) else 0.0
        ),
    }

    summary_path = OUTPUT_DIR / "mag_boundary_summary.csv"
    pd.DataFrame(
        {"metric": list(summary.keys()), "value": list(summary.values())}
    ).to_csv(summary_path, index=False)

    bin_edges = [0, 1.5, 3, 5, 10, 30, 60, 300, 3600, 86400, np.inf]
    bin_labels = [
        "<=1.5 s", "1.5-3 s", "3-5 s", "5-10 s", "10-30 s",
        "30-60 s", "1-5 min", "5-60 min", "1-24 h", ">24 h"
    ]

    valid_positive = boundaries.loc[
        boundaries["boundary_dt_s"] >= 0, ["boundary_dt_s"]
    ].copy()

    valid_positive["duration_bin"] = pd.cut(
        valid_positive["boundary_dt_s"],
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
    )

    bins_df = (
        valid_positive["duration_bin"]
        .value_counts(sort=False)
        .rename_axis("duration_bin")
        .reset_index(name="count")
    )
    bins_df["fraction"] = bins_df["count"] / bins_df["count"].sum()

    bins_path = OUTPUT_DIR / "mag_boundary_duration_bins.csv"
    bins_df.to_csv(bins_path, index=False)

    gap_dt = gap_rows["boundary_dt_s"].to_numpy(dtype=float)
    gap_dt = gap_dt[np.isfinite(gap_dt) & (gap_dt > 0)]

    if gap_dt.size:
        fig, ax = plt.subplots(figsize=(9, 5))
        bins = np.logspace(np.log10(gap_dt.min()), np.log10(gap_dt.max()), 50)
        ax.hist(gap_dt, bins=bins)
        ax.set_xscale("log")
        ax.set_xlabel("Boundary gap duration Δt (s)")
        ax.set_ylabel("Count")
        ax.set_title("MAG gaps across daily-file boundaries")
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "mag_boundary_dt_histogram.png", dpi=200)
        plt.close(fig)

        x_surv, y_surv = survival_curve(gap_dt)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.step(x_surv, y_surv, where="post")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Boundary gap duration threshold Δt (s)")
        ax.set_ylabel("Fraction of boundary gaps with duration ≥ Δt")
        ax.set_title("MAG boundary-gap survival curve")
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "mag_boundary_survival_curve.png", dpi=200)
        plt.close(fig)

    plot_data = boundaries[
        np.isfinite(boundaries["boundary_dt_s"])
        & (boundaries["boundary_dt_s"] > GAP_THRESHOLD_S)
    ].copy()

    if len(plot_data):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(
            pd.to_datetime(plot_data["date_after"]),
            plot_data["boundary_dt_s"],
            s=15,
        )
        ax.set_yscale("log")
        ax.set_xlabel("Date")
        ax.set_ylabel("Boundary gap duration Δt (s)")
        ax.set_title("MAG boundary gaps through the mission")
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "mag_boundary_timeline.png", dpi=200)
        plt.close(fig)

    print()
    print("=" * 72)
    print("MAVEN MAG — DAILY-FILE BOUNDARY CONTINUITY")
    print("=" * 72)
    print(f"Daily files:                         {summary['daily_files']:,}")
    print(f"Total boundaries:                    {summary['total_boundaries']:,}")
    print(f"Consecutive-day boundaries:          {summary['consecutive_day_boundaries']:,}")
    print(f"Boundaries spanning missing days:    {summary['boundaries_spanning_missing_days']:,}")
    print(f"Missing calendar days between files: {summary['total_missing_calendar_days']:,}")
    print()
    print(f"Continuous boundaries:               {summary['continuous_boundaries']:,}")
    print(f"Boundary gaps:                       {summary['boundary_gaps']:,}")
    print(f"  small:                             {summary['small_boundary_gaps']:,}")
    print(f"  medium:                            {summary['medium_boundary_gaps']:,}")
    print(f"  large:                             {summary['large_boundary_gaps']:,}")
    print(f"Overlapping boundaries:              {summary['overlapping_boundaries']:,}")
    print(f"Invalid boundaries:                  {summary['invalid_boundaries']:,}")
    print()
    print(f"Median positive boundary dt:         {summary['median_boundary_dt_s']:.3f} s")
    print(f"95th percentile:                     {summary['p95_boundary_dt_s']:.3f} s")
    print(f"99th percentile:                     {summary['p99_boundary_dt_s']:.3f} s")
    print(f"Largest boundary dt:                 {summary['largest_boundary_dt_s']:.3f} s")
    print(
        "Estimated missing time in boundary gaps: "
        f"{summary['total_boundary_missing_time_s'] / 3600:.2f} h"
    )
    print("=" * 72)
    print()
    print(f"Boundary events: {events_path}")
    print(f"Summary:         {summary_path}")
    print(f"Duration bins:   {bins_path}")
    print(f"Figures:         {FIGURE_DIR}")


if __name__ == "__main__":
    main()
