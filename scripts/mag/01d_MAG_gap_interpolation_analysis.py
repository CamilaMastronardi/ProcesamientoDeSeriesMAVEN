#!/usr/bin/env python3
"""
01d_MAG_gap_interpolation_analysis.py

Diagnostic study of MAVEN MAG gaps before choosing an interpolation policy.

This script does NOT modify or interpolate the MAG dataset. It:
1. Loads internal gaps found by 01b.
2. Loads boundary gaps found by 01c, if available.
3. Selects a stratified sample by gap duration.
4. Queries short MAG windows before and after each sampled gap.
5. Measures field jumps, field rotation and local variability.
6. Produces provisional interpolation-candidate flags and figures.

Run from the repository root:
    python scripts/mag/01d_MAG_gap_interpolation_analysis.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import bigquery

DEFAULT_TABLE_ID = "maven-vignes-crossings.maven.mag_fields"

INTERNAL_GAPS_CSV = Path("results/MAG_gap_analysis/mag_gap_events.csv")
BOUNDARY_GAPS_CSV = Path("results/MAG_boundary_analysis/mag_boundary_events.csv")

OUTPUT_DIR = Path("results/MAG_interpolation_analysis")
FIGURE_DIR = OUTPUT_DIR / "figures"
EXAMPLE_DIR = FIGURE_DIR / "examples"

GAP_THRESHOLD_S = 1.5

PROVISIONAL_MAX_GAP_S = 5.0
PROVISIONAL_MAX_RELATIVE_JUMP = 0.25
PROVISIONAL_MAX_ANGLE_DEG = 20.0

DEFAULT_SAMPLES_PER_BIN = 8
DEFAULT_WINDOW_SECONDS = 60.0
DEFAULT_RANDOM_SEED = 42

DURATION_BIN_ORDER = [
    "1.5-3 s",
    "3-5 s",
    "5-10 s",
    "10-30 s",
    "30-60 s",
    ">=60 s",
]


def find_column(df, candidates, label, required=True):
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if required:
        raise KeyError(
            f"Could not identify {label}. Available columns: {list(df.columns)}"
        )
    return None


def find_schema_column(names, candidates, label, required=True):
    lower = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in names:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if required:
        raise KeyError(
            f"Could not identify BigQuery field for {label}. "
            f"Available fields: {names}"
        )
    return None


def duration_bin(dt):
    if pd.isna(dt):
        return "invalid"
    if dt < 3:
        return "1.5-3 s"
    if dt < 5:
        return "3-5 s"
    if dt < 10:
        return "5-10 s"
    if dt < 30:
        return "10-30 s"
    if dt < 60:
        return "30-60 s"
    return ">=60 s"


def normalize_internal_gaps(path):
    """
    Normalize the output of 01b_MAG_gap_analysis.py.

    Expected columns:
      source_file, date_utc, gap_start_time, gap_end_time,
      dt_s, estimated_missing_samples
    """
    df = pd.read_csv(path)

    before_col = find_column(
        df,
        [
            "gap_start_time",
            "time_before",
            "timestamp_before",
            "last_time_before",
            "gap_start",
            "start_time",
        ],
        "internal gap start time",
    )
    after_col = find_column(
        df,
        [
            "gap_end_time",
            "time_after",
            "timestamp_after",
            "first_time_after",
            "gap_end",
            "end_time",
        ],
        "internal gap end time",
    )
    duration_col = find_column(
        df,
        [
            "dt_s",
            "dt_seconds",
            "gap_duration_s",
            "duration_s",
            "gap_seconds",
        ],
        "internal gap duration",
    )

    out = pd.DataFrame(
        {
            "source": "internal",
            "time_before": pd.to_datetime(
                df[before_col], utc=True, errors="coerce"
            ),
            "time_after": pd.to_datetime(
                df[after_col], utc=True, errors="coerce"
            ),
            "gap_duration_s": pd.to_numeric(
                df[duration_col], errors="coerce"
            ),
        }
    )

    source_file_col = find_column(
        df, ["source_file"], "source file", required=False
    )
    if source_file_col is not None:
        out["source_file"] = df[source_file_col].astype(str)

    return out


def normalize_boundary_gaps(path):
    df = pd.read_csv(path)

    before_col = find_column(
        df,
        ["last_time_before", "time_before", "timestamp_before"],
        "boundary gap start time",
    )
    after_col = find_column(
        df,
        ["first_time_after", "time_after", "timestamp_after"],
        "boundary gap end time",
    )
    duration_col = find_column(
        df,
        [
            "boundary_dt_s",
            "dt_s",
            "dt_seconds",
            "gap_duration_s",
            "duration_s",
        ],
        "boundary gap duration",
    )

    out = pd.DataFrame(
        {
            "source": "boundary",
            "time_before": pd.to_datetime(
                df[before_col], utc=True, errors="coerce"
            ),
            "time_after": pd.to_datetime(
                df[after_col], utc=True, errors="coerce"
            ),
            "gap_duration_s": pd.to_numeric(
                df[duration_col], errors="coerce"
            ),
        }
    )

    return out[out["gap_duration_s"] > GAP_THRESHOLD_S].copy()


def load_all_gap_events():
    frames = []

    if INTERNAL_GAPS_CSV.exists():
        frames.append(normalize_internal_gaps(INTERNAL_GAPS_CSV))
    else:
        print(f"Warning: file not found: {INTERNAL_GAPS_CSV}")

    if BOUNDARY_GAPS_CSV.exists():
        frames.append(normalize_boundary_gaps(BOUNDARY_GAPS_CSV))
    else:
        print(f"Warning: file not found: {BOUNDARY_GAPS_CSV}")

    if not frames:
        raise FileNotFoundError(
            "No gap-event CSV was found. Run 01b and 01c first."
        )

    events = pd.concat(frames, ignore_index=True, sort=False)
    events = events.dropna(
        subset=["time_before", "time_after", "gap_duration_s"]
    )
    events = events[events["gap_duration_s"] > GAP_THRESHOLD_S].copy()
    events["duration_bin"] = events["gap_duration_s"].apply(duration_bin)
    events["event_id"] = [
        f"{source}_{i:05d}"
        for i, source in enumerate(events["source"])
    ]

    return events.reset_index(drop=True)


def stratified_sample(events, samples_per_bin, seed):
    sampled = []

    for source in sorted(events["source"].unique()):
        source_df = events[events["source"] == source]

        for bin_name in DURATION_BIN_ORDER:
            group = source_df[source_df["duration_bin"] == bin_name]
            if group.empty:
                continue

            sampled.append(
                group.sample(
                    n=min(samples_per_bin, len(group)),
                    random_state=seed,
                )
            )

    if not sampled:
        raise RuntimeError("No gap events were available for sampling.")

    return (
        pd.concat(sampled, ignore_index=True)
        .sort_values(["source", "gap_duration_s"])
        .reset_index(drop=True)
    )


def discover_mag_schema(client, table_id):
    names = [field.name for field in client.get_table(table_id).schema]

    return {
        "time": find_schema_column(
            names, ["datetime_utc", "timestamp", "time_utc"], "timestamp"
        ),
        "bx": find_schema_column(
            names, ["bx_ss_nt", "bx_nt", "Bx", "bx"], "Bx"
        ),
        "by": find_schema_column(
            names, ["by_ss_nt", "by_nt", "By", "by"], "By"
        ),
        "bz": find_schema_column(
            names, ["bz_ss_nt", "bz_nt", "Bz", "bz"], "Bz"
        ),
        "btotal": find_schema_column(
            names,
            ["b_total_nt", "B_total", "Btot", "Bmag", "bmag_nt", "b_total"],
            "|B|",
            required=False,
        ),
    }


def query_sampled_windows(sampled, client, table_id, window_seconds):
    schema = discover_mag_schema(client, table_id)
    time_col = schema["time"]

    fields = [
        f"`{time_col}` AS time_utc",
        f"`{schema['bx']}` AS bx_nt",
        f"`{schema['by']}` AS by_nt",
        f"`{schema['bz']}` AS bz_nt",
    ]
    if schema["btotal"] is not None:
        fields.append(f"`{schema['btotal']}` AS btotal_nt")

    conditions = []
    params = []
    dates = set()
    width = pd.Timedelta(seconds=window_seconds)

    for i, row in sampled.iterrows():
        before = pd.Timestamp(row["time_before"])
        after = pd.Timestamp(row["time_after"])

        left_start = before - width
        left_end = before
        right_start = after
        right_end = after + width

        conditions.append(
            "("
            f"(`{time_col}` BETWEEN @ls{i} AND @le{i}) "
            "OR "
            f"(`{time_col}` BETWEEN @rs{i} AND @re{i})"
            ")"
        )

        params.extend(
            [
                bigquery.ScalarQueryParameter(
                    f"ls{i}", "TIMESTAMP", left_start.to_pydatetime()
                ),
                bigquery.ScalarQueryParameter(
                    f"le{i}", "TIMESTAMP", left_end.to_pydatetime()
                ),
                bigquery.ScalarQueryParameter(
                    f"rs{i}", "TIMESTAMP", right_start.to_pydatetime()
                ),
                bigquery.ScalarQueryParameter(
                    f"re{i}", "TIMESTAMP", right_end.to_pydatetime()
                ),
            ]
        )

        for t in [left_start, left_end, right_start, right_end]:
            dates.add(t.date())

    params.append(
        bigquery.ArrayQueryParameter("query_dates", "DATE", sorted(dates))
    )

    query = f"""
    SELECT {", ".join(fields)}
    FROM `{table_id}`
    WHERE DATE(`{time_col}`) IN UNNEST(@query_dates)
      AND (
        {" OR ".join(conditions)}
      )
    ORDER BY `{time_col}`
    """

    print()
    print(
        f"Querying MAG around {len(sampled)} sampled gaps "
        f"({len(dates)} UTC dates)..."
    )

    job_config = bigquery.QueryJobConfig(query_parameters=params)
    mag = client.query(query, job_config=job_config).to_dataframe()

    mag["time_utc"] = pd.to_datetime(
        mag["time_utc"], utc=True, errors="coerce"
    )

    for column in ["bx_nt", "by_nt", "bz_nt"]:
        mag[column] = pd.to_numeric(mag[column], errors="coerce")

    if "btotal_nt" not in mag.columns:
        mag["btotal_nt"] = np.sqrt(
            mag["bx_nt"] ** 2
            + mag["by_nt"] ** 2
            + mag["bz_nt"] ** 2
        )
    else:
        mag["btotal_nt"] = pd.to_numeric(
            mag["btotal_nt"], errors="coerce"
        )

    mag = (
        mag.dropna(subset=["time_utc"])
        .sort_values("time_utc")
        .drop_duplicates(subset=["time_utc"])
        .reset_index(drop=True)
    )

    return mag, schema


def vector_angle_deg(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if not np.isfinite(na) or not np.isfinite(nb) or na == 0 or nb == 0:
        return np.nan

    cosine = np.dot(a, b) / (na * nb)
    return float(
        np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    )


def analyze_one_gap(row, mag, window_seconds):
    before = pd.Timestamp(row["time_before"])
    after = pd.Timestamp(row["time_after"])
    width = pd.Timedelta(seconds=window_seconds)

    left = mag[
        (mag["time_utc"] >= before - width)
        & (mag["time_utc"] <= before)
    ]
    right = mag[
        (mag["time_utc"] >= after)
        & (mag["time_utc"] <= after + width)
    ]

    result = {
        "event_id": row["event_id"],
        "source": row["source"],
        "duration_bin": row["duration_bin"],
        "gap_duration_s": row["gap_duration_s"],
        "time_before": before,
        "time_after": after,
        "n_left": len(left),
        "n_right": len(right),
    }

    if left.empty or right.empty:
        result["status"] = "missing_edge_data"
        return result

    lrow = left.iloc[-1]
    rrow = right.iloc[0]

    left_time = pd.Timestamp(lrow["time_utc"])
    right_time = pd.Timestamp(rrow["time_utc"])

    before_offset = abs((before - left_time).total_seconds())
    after_offset = abs((right_time - after).total_seconds())

    b0 = np.array(
        [lrow["bx_nt"], lrow["by_nt"], lrow["bz_nt"]], dtype=float
    )
    b1 = np.array(
        [rrow["bx_nt"], rrow["by_nt"], rrow["bz_nt"]], dtype=float
    )

    if not (np.isfinite(b0).all() and np.isfinite(b1).all()):
        result["status"] = "invalid_edge_field"
        return result

    b0_norm = float(np.linalg.norm(b0))
    b1_norm = float(np.linalg.norm(b1))
    delta_vector = float(np.linalg.norm(b1 - b0))
    mean_b = np.mean([b0_norm, b1_norm])

    relative_jump = (
        delta_vector / mean_b if mean_b > 0 else np.nan
    )
    angle = vector_angle_deg(b0, b1)

    left_std = (
        float(left["btotal_nt"].std(ddof=1))
        if left["btotal_nt"].notna().sum() >= 2
        else np.nan
    )
    right_std = (
        float(right["btotal_nt"].std(ddof=1))
        if right["btotal_nt"].notna().sum() >= 2
        else np.nan
    )
    local_std = np.nanmedian([left_std, right_std])

    jump_over_local_std = (
        delta_vector / local_std
        if np.isfinite(local_std) and local_std > 0
        else np.nan
    )

    candidate = bool(
        row["gap_duration_s"] <= PROVISIONAL_MAX_GAP_S
        and np.isfinite(relative_jump)
        and relative_jump <= PROVISIONAL_MAX_RELATIVE_JUMP
        and np.isfinite(angle)
        and angle <= PROVISIONAL_MAX_ANGLE_DEG
        and before_offset <= GAP_THRESHOLD_S
        and after_offset <= GAP_THRESHOLD_S
    )

    result.update(
        {
            "status": "ok",
            "edge_before_offset_s": before_offset,
            "edge_after_offset_s": after_offset,
            "bx_before_nt": b0[0],
            "by_before_nt": b0[1],
            "bz_before_nt": b0[2],
            "bx_after_nt": b1[0],
            "by_after_nt": b1[1],
            "bz_after_nt": b1[2],
            "b_before_nt": b0_norm,
            "b_after_nt": b1_norm,
            "delta_b_vector_nt": delta_vector,
            "delta_b_total_nt": abs(
                float(rrow["btotal_nt"]) - float(lrow["btotal_nt"])
            ),
            "relative_vector_jump": relative_jump,
            "field_rotation_deg": angle,
            "local_std_before_nt": left_std,
            "local_std_after_nt": right_std,
            "local_std_nt": local_std,
            "jump_over_local_std": jump_over_local_std,
            "provisional_candidate": candidate,
        }
    )

    return result


def analyze_sampled_gaps(sampled, mag, window_seconds):
    return pd.DataFrame(
        [
            analyze_one_gap(row, mag, window_seconds)
            for _, row in sampled.iterrows()
        ]
    )


def build_summary(metrics):
    rows = []

    for source in sorted(metrics["source"].dropna().unique()):
        for bin_name in DURATION_BIN_ORDER:
            all_group = metrics[
                (metrics["source"] == source)
                & (metrics["duration_bin"] == bin_name)
            ]

            if all_group.empty:
                continue

            group = all_group[all_group["status"] == "ok"]

            row = {
                "source": source,
                "duration_bin": bin_name,
                "n_selected": len(all_group),
                "n_valid": len(group),
            }

            if len(group):
                row.update(
                    {
                        "median_gap_duration_s": group[
                            "gap_duration_s"
                        ].median(),
                        "median_relative_vector_jump": group[
                            "relative_vector_jump"
                        ].median(),
                        "median_field_rotation_deg": group[
                            "field_rotation_deg"
                        ].median(),
                        "median_jump_over_local_std": group[
                            "jump_over_local_std"
                        ].median(),
                        "n_provisional_candidates": int(
                            group["provisional_candidate"].sum()
                        ),
                        "candidate_fraction": float(
                            group["provisional_candidate"].mean()
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "median_gap_duration_s": np.nan,
                        "median_relative_vector_jump": np.nan,
                        "median_field_rotation_deg": np.nan,
                        "median_jump_over_local_std": np.nan,
                        "n_provisional_candidates": 0,
                        "candidate_fraction": np.nan,
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows)


def make_global_figures(metrics):
    valid = metrics[metrics["status"] == "ok"].copy()

    if valid.empty:
        return

    finite = valid[
        np.isfinite(valid["gap_duration_s"])
        & np.isfinite(valid["relative_vector_jump"])
        & (valid["gap_duration_s"] > 0)
        & (valid["relative_vector_jump"] > 0)
    ]

    if len(finite):
        fig, ax = plt.subplots(figsize=(9, 5))

        for source in sorted(finite["source"].unique()):
            g = finite[finite["source"] == source]
            ax.scatter(
                g["gap_duration_s"],
                g["relative_vector_jump"],
                s=24,
                alpha=0.7,
                label=source,
            )

        ax.axvline(PROVISIONAL_MAX_GAP_S, linestyle="--", linewidth=1)
        ax.axhline(
            PROVISIONAL_MAX_RELATIVE_JUMP, linestyle="--", linewidth=1
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Gap duration (s)")
        ax.set_ylabel("|ΔB| / mean(|B_before|, |B_after|)")
        ax.set_title("MAG field jump across sampled gaps")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "gap_duration_vs_relative_jump.png", dpi=200
        )
        plt.close(fig)

    finite = valid[
        np.isfinite(valid["gap_duration_s"])
        & np.isfinite(valid["field_rotation_deg"])
        & (valid["gap_duration_s"] > 0)
    ]

    if len(finite):
        fig, ax = plt.subplots(figsize=(9, 5))

        for source in sorted(finite["source"].unique()):
            g = finite[finite["source"] == source]
            ax.scatter(
                g["gap_duration_s"],
                g["field_rotation_deg"],
                s=24,
                alpha=0.7,
                label=source,
            )

        ax.axvline(PROVISIONAL_MAX_GAP_S, linestyle="--", linewidth=1)
        ax.axhline(PROVISIONAL_MAX_ANGLE_DEG, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Gap duration (s)")
        ax.set_ylabel("Field rotation across gap (deg)")
        ax.set_title("MAG directional change across sampled gaps")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "gap_duration_vs_angle.png", dpi=200)
        plt.close(fig)

    groups = []
    labels = []

    for bin_name in DURATION_BIN_ORDER:
        values = valid.loc[
            valid["duration_bin"] == bin_name,
            "relative_vector_jump",
        ].dropna()

        if len(values):
            groups.append(values.to_numpy())
            labels.append(bin_name)

    if groups:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(groups, labels=labels, showfliers=True)
        ax.set_yscale("log")
        ax.set_xlabel("Gap duration bin")
        ax.set_ylabel("Relative vector jump")
        ax.set_title("MAG field jump by gap-duration interval")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / "relative_jump_by_duration_bin.png", dpi=200
        )
        plt.close(fig)


def plot_example(event, mag, window_seconds, suffix):
    before = pd.Timestamp(event["time_before"])
    after = pd.Timestamp(event["time_after"])
    width = pd.Timedelta(seconds=window_seconds)

    left = mag[
        (mag["time_utc"] >= before - width)
        & (mag["time_utc"] <= before)
    ].copy()

    right = mag[
        (mag["time_utc"] >= after)
        & (mag["time_utc"] <= after + width)
    ].copy()

    if left.empty or right.empty:
        return

    left["relative_s"] = (
        left["time_utc"] - before
    ).dt.total_seconds()

    right["relative_s"] = (
        right["time_utc"] - before
    ).dt.total_seconds()

    fig, ax = plt.subplots(figsize=(11, 6))

    for column, label in [
        ("bx_nt", "Bx"),
        ("by_nt", "By"),
        ("bz_nt", "Bz"),
        ("btotal_nt", "|B|"),
    ]:
        ax.plot(left["relative_s"], left[column], label=label)
        ax.plot(right["relative_s"], right[column])

    ax.axvspan(
        0,
        float(event["gap_duration_s"]),
        alpha=0.15,
    )
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.axvline(
        float(event["gap_duration_s"]),
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("Time relative to last sample before gap (s)")
    ax.set_ylabel("Magnetic field (nT)")
    ax.set_title(
        f"{event['source']} gap — {event['gap_duration_s']:.3f} s"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=4)

    fig.tight_layout()

    filename = (
        f"{suffix}_{event['event_id']}_"
        f"{event['gap_duration_s']:.2f}s.png"
    )
    fig.savefig(EXAMPLE_DIR / filename, dpi=200)
    plt.close(fig)


def make_example_figures(sampled, metrics, mag, window_seconds):
    valid = metrics[
        (metrics["status"] == "ok")
        & (metrics["gap_duration_s"] <= 30)
        & np.isfinite(metrics["relative_vector_jump"])
    ].copy()

    if valid.empty:
        return

    smooth_ids = (
        valid.sort_values("relative_vector_jump")
        .head(5)["event_id"]
        .tolist()
    )

    abrupt_ids = (
        valid.sort_values("relative_vector_jump", ascending=False)
        .head(5)["event_id"]
        .tolist()
    )

    # Keep event_id as both index and column.
    by_id = sampled.set_index("event_id", drop=False)

    for event_id in smooth_ids:
        if event_id in by_id.index:
            plot_example(
                by_id.loc[event_id],
                mag,
                window_seconds,
                "smooth",
            )

    for event_id in abrupt_ids:
        if event_id in by_id.index:
            plot_example(
                by_id.loc[event_id],
                mag,
                window_seconds,
                "abrupt",
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate physical suitability of MAVEN MAG gaps "
            "for interpolation."
        )
    )

    parser.add_argument("--table", default=DEFAULT_TABLE_ID)
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=DEFAULT_SAMPLES_PER_BIN,
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )

    args = parser.parse_args()

    if args.samples_per_bin <= 0:
        raise ValueError("--samples-per-bin must be positive.")

    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be positive.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("Loading previously detected MAG gap events...")

    events = load_all_gap_events()

    print(f"Available gap events: {len(events):,}")
    print(
        events.groupby(
            ["source", "duration_bin"],
            observed=True,
        ).size()
    )

    sampled = stratified_sample(
        events,
        args.samples_per_bin,
        args.seed,
    )

    print()
    print(
        f"Selected gaps for physical analysis: {len(sampled):,}"
    )

    client = bigquery.Client()

    mag, schema = query_sampled_windows(
        sampled,
        client,
        args.table,
        args.window_seconds,
    )

    print(f"MAG samples retrieved: {len(mag):,}")
    print()
    print("Detected BigQuery MAG schema:")

    for key, value in schema.items():
        print(f"  {key:8s}: {value}")

    metrics = analyze_sampled_gaps(
        sampled,
        mag,
        args.window_seconds,
    )

    metrics_path = (
        OUTPUT_DIR / "mag_gap_interpolation_metrics.csv"
    )
    metrics.to_csv(metrics_path, index=False)

    summary = build_summary(metrics)

    summary_path = (
        OUTPUT_DIR / "mag_gap_interpolation_summary_by_bin.csv"
    )
    summary.to_csv(summary_path, index=False)

    valid = metrics[metrics["status"] == "ok"].copy()

    candidate_count = (
        int(valid["provisional_candidate"].sum())
        if len(valid)
        else 0
    )

    candidate_summary = pd.DataFrame(
        {
            "metric": [
                "selected_gaps",
                "valid_gaps",
                "provisional_candidates",
                "candidate_fraction_valid",
                "provisional_max_gap_s",
                "provisional_max_relative_jump",
                "provisional_max_angle_deg",
            ],
            "value": [
                len(metrics),
                len(valid),
                candidate_count,
                (
                    float(valid["provisional_candidate"].mean())
                    if len(valid)
                    else np.nan
                ),
                PROVISIONAL_MAX_GAP_S,
                PROVISIONAL_MAX_RELATIVE_JUMP,
                PROVISIONAL_MAX_ANGLE_DEG,
            ],
        }
    )

    candidate_summary_path = (
        OUTPUT_DIR / "mag_gap_interpolation_candidate_summary.csv"
    )
    candidate_summary.to_csv(candidate_summary_path, index=False)

    make_global_figures(metrics)
    make_example_figures(
        sampled,
        metrics,
        mag,
        args.window_seconds,
    )

    print()
    print("=" * 76)
    print("MAVEN MAG — GAP INTERPOLATION SUITABILITY ANALYSIS")
    print("=" * 76)
    print(f"All available gap events:        {len(events):,}")
    print(f"Sampled gap events:              {len(metrics):,}")
    print(f"Successfully analyzed:           {len(valid):,}")

    if len(valid):
        print(
            f"Provisional candidates:          {candidate_count:,} "
            f"({100 * candidate_count / len(valid):.1f}%)"
        )

        print()
        print("Median relative |ΔB| by duration bin:")

        medians = (
            valid.groupby(
                "duration_bin",
                observed=True,
            )["relative_vector_jump"]
            .median()
            .reindex(DURATION_BIN_ORDER)
            .dropna()
        )

        for bin_name, value in medians.items():
            print(f"  {bin_name:10s}: {value:.4f}")

        print()
        print("Median field rotation by duration bin:")

        angles = (
            valid.groupby(
                "duration_bin",
                observed=True,
            )["field_rotation_deg"]
            .median()
            .reindex(DURATION_BIN_ORDER)
            .dropna()
        )

        for bin_name, value in angles.items():
            print(f"  {bin_name:10s}: {value:.2f} deg")

    print()
    print("IMPORTANT:")
    print("  provisional_candidate is only a screening flag.")
    print("  It does not prove that interpolation is physically correct.")
    print(
        "  Final validation should include pseudo-gap experiments "
        "on continuous MAG intervals."
    )
    print("=" * 76)

    print()
    print(f"Metrics:           {metrics_path}")
    print(f"Summary:           {summary_path}")
    print(f"Candidate summary: {candidate_summary_path}")
    print(f"Figures:           {FIGURE_DIR}")


if __name__ == "__main__":
    main()
