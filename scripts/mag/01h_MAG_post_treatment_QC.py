#!/usr/bin/env python3
"""
01h_MAG_post_treatment_QC.py

Post-treatment quality control for MAVEN MAG.

Validates the outputs of 01f and 01g without querying BigQuery.
The script reconstructs the original time sequence from rows with
is_original=True and compares it with the treated sequence.

Checks
------
1) Temporal integrity:
   - no duplicate timestamps
   - no non-positive dt
   - raw gaps reconstructed from original rows match the 01f policy
   - residual gaps match exactly the gaps rejected by 01f

2) Interpolation integrity:
   - every synthetic sample belongs to an approved gap
   - every approved gap has the expected number of synthetic samples
   - Bx, By, Bz follow the exact linear interpolation formula
   - |B| = sqrt(Bx^2 + By^2 + Bz^2)
   - no rejected gap contains synthetic samples

3) Flag/accounting integrity:
   - original and interpolated flags are mutually exclusive
   - original + interpolated = processed
   - daily counts are compared with the 01g manifest when available
   - number of daily files is compared with the original MAG QC date list

Outputs
-------
results/MAG_post_treatment_QC/
    mag_post_treatment_qc_summary.csv
    mag_post_treatment_daily_qc.csv
    mag_raw_gaps_from_processed_originals.csv
    mag_residual_gaps.csv
    mag_interpolation_validation.csv
    figures/
        gap_count_before_after.png
        gap_duration_before_after.png
        interpolated_samples_by_day.png

Run
---
python3 scripts/mag/01h_MAG_post_treatment_QC.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed/MAG_gap_treated")
POLICY_CSV = Path(
    "results/MAG_gap_treatment_policy/mag_gap_treatment_policy.csv"
)
MANIFEST_CSV = Path(
    "results/MAG_gap_treatment_applied/mag_gap_treatment_manifest.csv"
)
QUALITY_CSV = Path(
    "results/MAG_quality/mag_quality_by_day.csv"
)
OUTPUT_DIR = Path("results/MAG_post_treatment_QC")
FIGURE_DIR = OUTPUT_DIR / "figures"

GAP_THRESHOLD_S = 1.5
NOMINAL_CADENCE_S = 1.0
DEFAULT_ATOL = 1e-8


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


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


def ts_key(ts):
    return int(pd.Timestamp(ts).value)


def gap_key(t1, t2):
    return (ts_key(t1), ts_key(t2))


def expected_missing_count(t1, t2, cadence_s=NOMINAL_CADENCE_S):
    dt_s = (pd.Timestamp(t2) - pd.Timestamp(t1)).total_seconds()
    intervals = int(np.rint(dt_s / cadence_s))
    return max(intervals - 1, 0)


def load_policy():
    if not POLICY_CSV.exists():
        raise FileNotFoundError(f"Missing {POLICY_CSV}. Run 01f first.")

    p = pd.read_csv(POLICY_CSV)
    p["time_before"] = pd.to_datetime(p["time_before"], utc=True, errors="coerce")
    p["time_after"] = pd.to_datetime(p["time_after"], utc=True, errors="coerce")
    p["interpolate"] = parse_bool(p["interpolate"])
    p["gap_duration_s"] = pd.to_numeric(p["gap_duration_s"], errors="coerce")
    p = p.dropna(
        subset=["gap_id", "time_before", "time_after", "gap_duration_s"]
    ).copy()

    if p["gap_id"].duplicated().any():
        raise ValueError("Duplicated gap_id values found in treatment policy.")

    return p


def load_expected_dates():
    if not QUALITY_CSV.exists():
        return None
    q = pd.read_csv(QUALITY_CSV)
    col = find_column(
        q,
        ["date_utc", "date", "day", "file_date", "source_date"],
        "MAG QC date",
    )
    return set(
        pd.to_datetime(q[col], errors="coerce")
        .dropna()
        .dt.date
        .drop_duplicates()
        .tolist()
    )


def load_manifest():
    if not MANIFEST_CSV.exists():
        return None
    m = pd.read_csv(MANIFEST_CSV)
    col = find_column(m, ["date_utc", "date"], "manifest date")
    m["date_utc"] = pd.to_datetime(m[col], errors="coerce").dt.date
    if "status" in m.columns:
        m = m[m["status"].astype(str) == "written"].copy()
    return m.drop_duplicates(subset=["date_utc"], keep="last")


def load_day(path):
    df = pd.read_parquet(path)
    required = [
        "datetime_utc",
        "bx_ss_nt",
        "by_ss_nt",
        "bz_ss_nt",
        "b_total_nt",
        "is_original",
        "is_interpolated",
        "gap_id",
        "interpolation_method",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing required columns {missing}")

    df["datetime_utc"] = pd.to_datetime(
        df["datetime_utc"], utc=True, errors="coerce"
    )
    for c in ["bx_ss_nt", "by_ss_nt", "bz_ss_nt", "b_total_nt"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_original"] = parse_bool(df["is_original"])
    df["is_interpolated"] = parse_bool(df["is_interpolated"])
    return df.sort_values("datetime_utc").reset_index(drop=True)


def collect_gap_transitions(times, prev_time=None):
    gaps = []
    duplicates = 0
    nonpositive = 0

    times = pd.DatetimeIndex(times)
    if len(times) == 0:
        return gaps, duplicates, nonpositive

    values = times.asi8

    def inspect_pair(left_ns, right_ns, source):
        nonlocal duplicates, nonpositive
        dt_s = (right_ns - left_ns) / 1e9
        if dt_s == 0:
            duplicates += 1
        if dt_s <= 0:
            nonpositive += 1
        if dt_s > GAP_THRESHOLD_S:
            gaps.append(
                {
                    "time_before": pd.Timestamp(left_ns, unit="ns", tz="UTC"),
                    "time_after": pd.Timestamp(right_ns, unit="ns", tz="UTC"),
                    "dt_s": float(dt_s),
                    "source": source,
                }
            )

    if prev_time is not None:
        inspect_pair(ts_key(prev_time), int(values[0]), "boundary")

    if len(values) >= 2:
        delta = np.diff(values) / 1e9
        duplicates += int(np.sum(delta == 0))
        nonpositive += int(np.sum(delta <= 0))
        idx = np.flatnonzero(delta > GAP_THRESHOLD_S)
        for i in idx:
            gaps.append(
                {
                    "time_before": pd.Timestamp(
                        int(values[i]), unit="ns", tz="UTC"
                    ),
                    "time_after": pd.Timestamp(
                        int(values[i + 1]), unit="ns", tz="UTC"
                    ),
                    "dt_s": float(delta[i]),
                    "source": "internal",
                }
            )

    return gaps, duplicates, nonpositive


def scan_files(files):
    daily_rows = []
    raw_gaps = []
    processed_gaps = []
    synthetic_parts = []

    prev_processed = None
    prev_raw = None

    totals = {
        "processed_rows": 0,
        "original_rows": 0,
        "interpolated_rows": 0,
        "processed_duplicate_timestamps": 0,
        "processed_nonpositive_dt": 0,
        "raw_duplicate_timestamps": 0,
        "raw_nonpositive_dt": 0,
        "flag_overlap_rows": 0,
        "flag_neither_rows": 0,
        "original_rows_with_gap_id": 0,
        "interpolated_rows_without_gap_id": 0,
        "interpolated_rows_bad_method": 0,
        "nonfinite_B_rows": 0,
        "invalid_time_rows": 0,
    }

    for n, path in enumerate(files, start=1):
        df = load_day(path)
        if df.empty:
            continue

        valid_times = df["datetime_utc"].dropna()
        date_utc = valid_times.iloc[0].date() if len(valid_times) else None

        totals["invalid_time_rows"] += int(df["datetime_utc"].isna().sum())

        finite = np.isfinite(
            df[["bx_ss_nt", "by_ss_nt", "bz_ss_nt", "b_total_nt"]].to_numpy(
                dtype=float
            )
        ).all(axis=1)
        nonfinite_count = int((~finite).sum())
        totals["nonfinite_B_rows"] += nonfinite_count

        overlap = df["is_original"] & df["is_interpolated"]
        neither = ~df["is_original"] & ~df["is_interpolated"]
        totals["flag_overlap_rows"] += int(overlap.sum())
        totals["flag_neither_rows"] += int(neither.sum())

        original = df[df["is_original"]].copy()
        interpolated = df[df["is_interpolated"]].copy()

        totals["original_rows_with_gap_id"] += int(original["gap_id"].notna().sum())
        totals["interpolated_rows_without_gap_id"] += int(
            interpolated["gap_id"].isna().sum()
        )

        if len(interpolated):
            bad_method = (
                interpolated["interpolation_method"].astype(str).str.lower()
                != "linear"
            )
            totals["interpolated_rows_bad_method"] += int(bad_method.sum())
            keep = [
                c
                for c in [
                    "datetime_utc",
                    "bx_ss_nt",
                    "by_ss_nt",
                    "bz_ss_nt",
                    "b_total_nt",
                    "gap_id",
                    "interpolation_method",
                ]
                if c in interpolated.columns
            ]
            synthetic_parts.append(interpolated[keep].copy())

        pg, pdup, pnonpos = collect_gap_transitions(
            valid_times, prev_time=prev_processed
        )
        rg, rdup, rnonpos = collect_gap_transitions(
            original["datetime_utc"].dropna(), prev_time=prev_raw
        )

        processed_gaps.extend(pg)
        raw_gaps.extend(rg)
        totals["processed_duplicate_timestamps"] += pdup
        totals["processed_nonpositive_dt"] += pnonpos
        totals["raw_duplicate_timestamps"] += rdup
        totals["raw_nonpositive_dt"] += rnonpos

        if len(valid_times):
            prev_processed = valid_times.iloc[-1]
        if len(original):
            prev_raw = original["datetime_utc"].dropna().iloc[-1]

        totals["processed_rows"] += len(df)
        totals["original_rows"] += len(original)
        totals["interpolated_rows"] += len(interpolated)

        daily_rows.append(
            {
                "date_utc": date_utc,
                "file": str(path),
                "rows": len(df),
                "original_samples": len(original),
                "interpolated_samples": len(interpolated),
                "raw_gaps": len(rg),
                "processed_gaps": len(pg),
                "duplicate_timestamps": pdup,
                "nonpositive_dt": pnonpos,
                "nonfinite_B_rows": nonfinite_count,
            }
        )

        if n == 1 or n % 250 == 0 or n == len(files):
            print(f"  scanned {n:,}/{len(files):,} files")

    synthetic = (
        pd.concat(synthetic_parts, ignore_index=True, sort=False)
        if synthetic_parts
        else pd.DataFrame()
    )

    return (
        pd.DataFrame(daily_rows),
        pd.DataFrame(raw_gaps),
        pd.DataFrame(processed_gaps),
        synthetic,
        totals,
    )


def compare_gap_sets(policy, raw_gaps, residual_gaps):
    all_policy = {
        gap_key(r["time_before"], r["time_after"])
        for _, r in policy.iterrows()
    }
    approved = policy[policy["interpolate"]]
    rejected = policy[~policy["interpolate"]]
    approved_keys = {
        gap_key(r["time_before"], r["time_after"])
        for _, r in approved.iterrows()
    }
    rejected_keys = {
        gap_key(r["time_before"], r["time_after"])
        for _, r in rejected.iterrows()
    }
    raw_keys = {
        gap_key(r["time_before"], r["time_after"])
        for _, r in raw_gaps.iterrows()
    }
    residual_keys = {
        gap_key(r["time_before"], r["time_after"])
        for _, r in residual_gaps.iterrows()
    }

    return {
        "policy_gap_count": len(all_policy),
        "raw_gap_count": len(raw_keys),
        "raw_policy_missing": len(all_policy - raw_keys),
        "raw_extra_gaps": len(raw_keys - all_policy),
        "rejected_gap_count": len(rejected_keys),
        "residual_gap_count": len(residual_keys),
        "rejected_missing_from_residual": len(rejected_keys - residual_keys),
        "unexpected_residual_gaps": len(residual_keys - rejected_keys),
        "approved_gaps_still_present": len(approved_keys & residual_keys),
    }


def edge_value(row, component, side):
    col = f"{component}_{side}"
    if col not in row.index:
        return np.nan
    value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else np.nan


def validate_interpolation(policy, synthetic, atol):
    approved = policy[policy["interpolate"]].set_index("gap_id", drop=False)
    policy_ids = set(policy["gap_id"].astype(str))
    approved_ids = set(approved.index.astype(str))
    rows = []

    synthetic = synthetic.copy()
    if not synthetic.empty:
        synthetic["gap_id"] = synthetic["gap_id"].astype(str)
        synthetic["datetime_utc"] = pd.to_datetime(
            synthetic["datetime_utc"], utc=True, errors="coerce"
        )
        synthetic_ids = set(synthetic["gap_id"].dropna().unique())
    else:
        synthetic_ids = set()

    for gap_id in sorted(synthetic_ids - policy_ids):
        rows.append(
            {
                "gap_id": gap_id,
                "expected_samples": 0,
                "actual_samples": int((synthetic["gap_id"] == gap_id).sum()),
                "max_component_abs_error_nt": np.nan,
                "max_btotal_abs_error_nt": np.nan,
                "status": "unknown_gap_id",
            }
        )

    for gap_id in sorted((synthetic_ids & policy_ids) - approved_ids):
        rows.append(
            {
                "gap_id": gap_id,
                "expected_samples": 0,
                "actual_samples": int((synthetic["gap_id"] == gap_id).sum()),
                "max_component_abs_error_nt": np.nan,
                "max_btotal_abs_error_nt": np.nan,
                "status": "synthetic_samples_in_rejected_gap",
            }
        )

    for gap_id, gap in approved.iterrows():
        gap_id = str(gap_id)
        group = synthetic[synthetic["gap_id"] == gap_id].copy()
        expected_count = expected_missing_count(
            gap["time_before"], gap["time_after"]
        )

        if group.empty:
            rows.append(
                {
                    "gap_id": gap_id,
                    "expected_samples": expected_count,
                    "actual_samples": 0,
                    "max_component_abs_error_nt": np.nan,
                    "max_btotal_abs_error_nt": np.nan,
                    "status": "missing_synthetic_samples",
                }
            )
            continue

        t1 = pd.Timestamp(gap["time_before"])
        t2 = pd.Timestamp(gap["time_after"])
        dt_s = (t2 - t1).total_seconds()

        b1 = np.array(
            [
                edge_value(gap, "bx_nt", "before"),
                edge_value(gap, "by_nt", "before"),
                edge_value(gap, "bz_nt", "before"),
            ]
        )
        b2 = np.array(
            [
                edge_value(gap, "bx_nt", "after"),
                edge_value(gap, "by_nt", "after"),
                edge_value(gap, "bz_nt", "after"),
            ]
        )

        if not (np.isfinite(b1).all() and np.isfinite(b2).all() and dt_s > 0):
            rows.append(
                {
                    "gap_id": gap_id,
                    "expected_samples": expected_count,
                    "actual_samples": len(group),
                    "max_component_abs_error_nt": np.nan,
                    "max_btotal_abs_error_nt": np.nan,
                    "status": "missing_policy_edge_values",
                }
            )
            continue

        alpha = (
            (group["datetime_utc"] - t1).dt.total_seconds() / dt_s
        ).to_numpy(dtype=float)
        expected_b = b1[None, :] + alpha[:, None] * (b2 - b1)[None, :]
        observed_b = group[["bx_ss_nt", "by_ss_nt", "bz_ss_nt"]].to_numpy(
            dtype=float
        )
        max_component_error = float(np.nanmax(np.abs(observed_b - expected_b)))
        expected_btotal = np.linalg.norm(expected_b, axis=1)
        observed_btotal = group["b_total_nt"].to_numpy(dtype=float)
        max_btotal_error = float(
            np.nanmax(np.abs(observed_btotal - expected_btotal))
        )

        issues = []
        if len(group) != expected_count:
            issues.append("wrong_sample_count")
        if max_component_error > atol:
            issues.append("interpolation_mismatch")
        if max_btotal_error > atol:
            issues.append("btotal_mismatch")
        if (alpha <= 0).any() or (alpha >= 1).any():
            issues.append("synthetic_time_outside_gap")

        rows.append(
            {
                "gap_id": gap_id,
                "expected_samples": expected_count,
                "actual_samples": len(group),
                "max_component_abs_error_nt": max_component_error,
                "max_btotal_abs_error_nt": max_btotal_error,
                "status": ";".join(issues) if issues else "ok",
            }
        )

    return pd.DataFrame(rows)


def compare_manifest(daily, manifest):
    daily = daily.copy()
    daily["manifest_match"] = np.nan
    if manifest is None:
        return daily

    cols = [
        c
        for c in [
            "date_utc",
            "original_samples",
            "interpolated_samples",
            "output_samples",
        ]
        if c in manifest.columns
    ]
    m = manifest[cols].copy()
    merged = daily.merge(m, on="date_utc", how="left", suffixes=("", "_manifest"))

    matches = []
    for _, row in merged.iterrows():
        checks = []
        if "original_samples_manifest" in merged.columns and pd.notna(
            row["original_samples_manifest"]
        ):
            checks.append(
                int(row["original_samples"])
                == int(row["original_samples_manifest"])
            )
        if "interpolated_samples_manifest" in merged.columns and pd.notna(
            row["interpolated_samples_manifest"]
        ):
            checks.append(
                int(row["interpolated_samples"])
                == int(row["interpolated_samples_manifest"])
            )
        if "output_samples" in merged.columns and pd.notna(row["output_samples"]):
            checks.append(int(row["rows"]) == int(row["output_samples"]))
        matches.append(all(checks) if checks else np.nan)

    merged["manifest_match"] = matches
    return merged


def make_figures(raw_gaps, residual_gaps, daily):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        ["Raw", "After treatment"],
        [len(raw_gaps), len(residual_gaps)],
    )
    ax.set_ylabel("Number of MAG gaps")
    ax.set_title("MAG gaps before and after treatment")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "gap_count_before_after.png", dpi=200)
    plt.close(fig)

    if not raw_gaps.empty:
        raw_dt = raw_gaps["dt_s"].to_numpy(dtype=float)
        res_dt = (
            residual_gaps["dt_s"].to_numpy(dtype=float)
            if not residual_gaps.empty
            else np.array([])
        )
        positive = np.concatenate([raw_dt[raw_dt > 0], res_dt[res_dt > 0]])
        if len(positive):
            lo = max(GAP_THRESHOLD_S, float(np.nanmin(positive)))
            hi = float(np.nanmax(positive))
            if hi <= lo:
                hi = lo * 1.01
            bins = np.logspace(np.log10(lo), np.log10(hi), 50)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(raw_dt, bins=bins, histtype="step", linewidth=1.5, label="Raw gaps")
            ax.hist(res_dt, bins=bins, histtype="step", linewidth=1.5, label="Residual gaps")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Gap duration [s]")
            ax.set_ylabel("Count")
            ax.set_title("MAG gap-duration distribution")
            ax.legend()
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(FIGURE_DIR / "gap_duration_before_after.png", dpi=200)
            plt.close(fig)

    plot_daily = daily.copy()
    plot_daily["date_utc"] = pd.to_datetime(plot_daily["date_utc"], errors="coerce")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        plot_daily["date_utc"],
        plot_daily["interpolated_samples"],
        linewidth=0.8,
    )
    ax.set_xlabel("UTC date")
    ax.set_ylabel("Interpolated samples")
    ax.set_title("Interpolated MAG samples by day")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "interpolated_samples_by_day.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Post-treatment QC for MAVEN MAG gap interpolation."
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help="Absolute numerical tolerance for interpolation validation.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    policy = load_policy()
    expected_dates = load_expected_dates()
    manifest = load_manifest()

    files = sorted(DATA_DIR.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {DATA_DIR}")

    print()
    print(f"Policy gaps:     {len(policy):,}")
    print(f"Approved gaps:   {int(policy['interpolate'].sum()):,}")
    print(f"Rejected gaps:   {int((~policy['interpolate']).sum()):,}")
    print()
    print(f"Scanning {len(files):,} processed daily files...")

    daily, raw_gaps, residual_gaps, synthetic, totals = scan_files(files)
    daily = compare_manifest(daily, manifest)

    gap_checks = compare_gap_sets(policy, raw_gaps, residual_gaps)

    print()
    print("Validating synthetic samples...")
    validation = validate_interpolation(policy, synthetic, args.atol)

    approved = policy[policy["interpolate"]]
    rejected = policy[~policy["interpolate"]]
    expected_synthetic = int(
        sum(
            expected_missing_count(r["time_before"], r["time_after"])
            for _, r in approved.iterrows()
        )
    )
    interpolation_failures = int((validation["status"] != "ok").sum())

    expected_file_count = len(expected_dates) if expected_dates is not None else None
    processed_dates = set(daily["date_utc"].dropna().tolist())
    missing_dates = (
        len(expected_dates - processed_dates) if expected_dates is not None else 0
    )
    extra_dates = (
        len(processed_dates - expected_dates) if expected_dates is not None else 0
    )

    manifest_known = daily["manifest_match"].notna()
    manifest_mismatch_days = int(
        (~daily.loc[manifest_known, "manifest_match"].astype(bool)).sum()
    ) if manifest_known.any() else 0

    checks = {
        "file_count_matches_expected": (
            True
            if expected_file_count is None
            else len(files) == expected_file_count and missing_dates == 0 and extra_dates == 0
        ),
        "processed_accounting": (
            totals["processed_rows"]
            == totals["original_rows"] + totals["interpolated_rows"]
        ),
        "no_processed_duplicate_timestamps": totals["processed_duplicate_timestamps"] == 0,
        "no_processed_nonpositive_dt": totals["processed_nonpositive_dt"] == 0,
        "no_nonfinite_B_rows": totals["nonfinite_B_rows"] == 0,
        "no_invalid_times": totals["invalid_time_rows"] == 0,
        "flags_are_consistent": (
            totals["flag_overlap_rows"] == 0
            and totals["flag_neither_rows"] == 0
            and totals["original_rows_with_gap_id"] == 0
            and totals["interpolated_rows_without_gap_id"] == 0
            and totals["interpolated_rows_bad_method"] == 0
        ),
        "raw_gap_set_matches_policy": (
            gap_checks["raw_policy_missing"] == 0
            and gap_checks["raw_extra_gaps"] == 0
            and gap_checks["raw_gap_count"] == gap_checks["policy_gap_count"]
        ),
        "residual_gap_set_matches_rejected_policy": (
            gap_checks["rejected_missing_from_residual"] == 0
            and gap_checks["unexpected_residual_gaps"] == 0
            and gap_checks["approved_gaps_still_present"] == 0
            and gap_checks["residual_gap_count"] == gap_checks["rejected_gap_count"]
        ),
        "synthetic_sample_count_matches": totals["interpolated_rows"] == expected_synthetic,
        "all_interpolations_validate": interpolation_failures == 0,
        "manifest_daily_counts_match": manifest_mismatch_days == 0,
    }

    overall = all(checks.values())

    summary_rows = [
        ("overall_qc_status", "PASS" if overall else "FAIL"),
        ("processed_files", len(files)),
        ("expected_files", expected_file_count),
        ("missing_expected_dates", missing_dates),
        ("extra_processed_dates", extra_dates),
        ("original_samples", totals["original_rows"]),
        ("interpolated_samples", totals["interpolated_rows"]),
        ("processed_samples", totals["processed_rows"]),
        (
            "interpolated_fraction",
            totals["interpolated_rows"] / totals["processed_rows"]
            if totals["processed_rows"]
            else np.nan,
        ),
        ("policy_total_gaps", len(policy)),
        ("policy_approved_gaps", len(approved)),
        ("policy_rejected_gaps", len(rejected)),
        ("raw_gaps_reconstructed", len(raw_gaps)),
        ("residual_gaps_after_treatment", len(residual_gaps)),
        ("expected_synthetic_samples", expected_synthetic),
        ("interpolation_validation_failures", interpolation_failures),
        ("processed_duplicate_timestamps", totals["processed_duplicate_timestamps"]),
        ("processed_nonpositive_dt", totals["processed_nonpositive_dt"]),
        ("nonfinite_B_rows", totals["nonfinite_B_rows"]),
        ("manifest_mismatch_days", manifest_mismatch_days),
        ("raw_policy_missing", gap_checks["raw_policy_missing"]),
        ("raw_extra_gaps", gap_checks["raw_extra_gaps"]),
        (
            "rejected_missing_from_residual",
            gap_checks["rejected_missing_from_residual"],
        ),
        ("unexpected_residual_gaps", gap_checks["unexpected_residual_gaps"]),
        ("approved_gaps_still_present", gap_checks["approved_gaps_still_present"]),
    ]
    summary_rows.extend(
        (f"check_{name}", "PASS" if passed else "FAIL")
        for name, passed in checks.items()
    )
    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

    summary.to_csv(OUTPUT_DIR / "mag_post_treatment_qc_summary.csv", index=False)
    daily.to_csv(OUTPUT_DIR / "mag_post_treatment_daily_qc.csv", index=False)
    raw_gaps.to_csv(
        OUTPUT_DIR / "mag_raw_gaps_from_processed_originals.csv", index=False
    )
    residual_gaps.to_csv(OUTPUT_DIR / "mag_residual_gaps.csv", index=False)
    validation.to_csv(
        OUTPUT_DIR / "mag_interpolation_validation.csv", index=False
    )
    make_figures(raw_gaps, residual_gaps, daily)

    print()
    print("=" * 82)
    print("MAVEN MAG — POST-TREATMENT QUALITY CONTROL")
    print("=" * 82)
    print(f"Processed daily files:             {len(files):,}")
    if expected_file_count is not None:
        print(f"Expected MAG dates:                {expected_file_count:,}")
    print()
    print(f"Original samples:                  {totals['original_rows']:,}")
    print(f"Interpolated samples:              {totals['interpolated_rows']:,}")
    print(f"Processed samples:                 {totals['processed_rows']:,}")
    if totals["processed_rows"]:
        frac = totals["interpolated_rows"] / totals["processed_rows"]
        print(f"Interpolated fraction:             {100 * frac:.6f}%")
    print()
    print(f"Raw gaps reconstructed:            {len(raw_gaps):,}")
    print(f"Policy gaps:                       {len(policy):,}")
    print(f"Approved gaps:                     {len(approved):,}")
    print(f"Rejected gaps:                     {len(rejected):,}")
    print(f"Residual gaps after treatment:     {len(residual_gaps):,}")
    print()
    print(f"Expected synthetic samples:        {expected_synthetic:,}")
    print(f"Actual synthetic samples:          {totals['interpolated_rows']:,}")
    print(f"Interpolation validation failures: {interpolation_failures:,}")
    print()
    print(f"Duplicate processed timestamps:    {totals['processed_duplicate_timestamps']:,}")
    print(f"Non-positive processed dt:         {totals['processed_nonpositive_dt']:,}")
    print(f"Non-finite B rows:                 {totals['nonfinite_B_rows']:,}")
    print(f"Manifest mismatch days:            {manifest_mismatch_days:,}")
    print()
    print("Gap-set consistency:")
    print(f"  Raw policy gaps not recovered:   {gap_checks['raw_policy_missing']:,}")
    print(f"  Unexpected raw gaps:             {gap_checks['raw_extra_gaps']:,}")
    print(f"  Rejected gaps missing after QC:  {gap_checks['rejected_missing_from_residual']:,}")
    print(f"  Unexpected residual gaps:        {gap_checks['unexpected_residual_gaps']:,}")
    print(f"  Approved gaps still present:     {gap_checks['approved_gaps_still_present']:,}")
    print()
    print("Critical checks:")
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL':4s}  {name}")
    print()
    print("-" * 82)
    print(f"FINAL QC STATUS: {'PASS' if overall else 'FAIL'}")
    print("-" * 82)
    print()
    print(f"Tables:  {OUTPUT_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print("=" * 82)


if __name__ == "__main__":
    main()
