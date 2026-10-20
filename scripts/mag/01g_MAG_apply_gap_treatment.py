#!/usr/bin/env python3
"""
01g_MAG_apply_gap_treatment.py

Apply the validated MAVEN MAG real-gap treatment policy.

Important design choice
-----------------------
This script preserves every original MAG timestamp and inserts ONLY the
synthetic samples authorized by 01f. It does not force the whole mission onto
an exact integer-second grid.

Why?
----
The exact common time grid should be defined later, during synchronization of
MAG, SWEA and SWIA. Regularizing MAG independently at this stage would mix two
different preprocessing decisions: gap treatment and instrument resampling.

Approved gaps
-------------
Only rows with interpolate=True in

    results/MAG_gap_treatment_policy/mag_gap_treatment_policy.csv

are reconstructed.

Interpolation
-------------
Bx, By and Bz are linearly interpolated between the observed gap edges.
|B| is recomputed from the interpolated vector.

If x/y/z edge positions were available in the policy table, spacecraft
position is also linearly interpolated.

Unfilled gaps
-------------
Rejected gaps are NOT expanded into millions of NaN rows. Instead, their
intervals are saved as an explicit gap mask:

    results/MAG_gap_treatment_applied/mag_unfilled_gap_intervals.csv

This is more scalable and preserves the distinction between missing data and
real measurements. A sample-level is_gap mask can be created later when the
multi-instrument common time grid is defined.

Output data
-----------
Processed MAG is written as daily Parquet files under:

    data/processed/MAG_gap_treated/YYYY/MM/mag_YYYY-MM-DD.parquet

Each row includes:
    datetime_utc
    bx_ss_nt
    by_ss_nt
    bz_ss_nt
    b_total_nt
    [x_ss_km, y_ss_km, z_ss_km if available]
    is_original
    is_interpolated
    gap_id
    interpolation_method

Original samples are never replaced by synthetic ones.

Policy timestamps are parsed with ``format="mixed"`` so timestamps with and
without fractional seconds are handled consistently. Approved gaps with
invalid timestamps or missing magnetic-field edge values raise an error
instead of being silently skipped.

Results
-------
results/MAG_gap_treatment_applied/
    mag_gap_treatment_manifest.csv
    mag_gap_treatment_applied_summary.csv
    mag_unfilled_gap_intervals.csv

Run
---
First test a short interval:

python3 scripts/mag/01g_MAG_apply_gap_treatment.py \
    --start 2014-10-11 \
    --end 2014-10-15

Then full mission:

python3 scripts/mag/01g_MAG_apply_gap_treatment.py

Use --overwrite to replace existing daily Parquet files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery


DEFAULT_TABLE_ID = "maven-vignes-crossings.maven.mag_fields"

POLICY_CSV = Path(
    "results/MAG_gap_treatment_policy/mag_gap_treatment_policy.csv"
)

QUALITY_CSV = Path(
    "results/MAG_quality/mag_quality_by_day.csv"
)

DATA_OUTPUT_DIR = Path(
    "data/processed/MAG_gap_treated"
)

RESULTS_DIR = Path(
    "results/MAG_gap_treatment_applied"
)

DEFAULT_NOMINAL_CADENCE_S = 1.0
DEFAULT_CHUNK_DAYS = 30


# =====================================================================
# Helpers
# =====================================================================

def find_column(
    df,
    candidates,
    label,
    required=True,
):
    lower = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

        if candidate.lower() in lower:
            return lower[
                candidate.lower()
            ]

    if required:
        raise KeyError(
            f"Could not identify {label}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


def find_schema_column(
    names,
    candidates,
    label,
    required=True,
):
    lower = {
        name.lower(): name
        for name in names
    }

    for candidate in candidates:
        if candidate in names:
            return candidate

        if candidate.lower() in lower:
            return lower[
                candidate.lower()
            ]

    if required:
        raise KeyError(
            f"Could not identify BigQuery field for {label}. "
            f"Available fields: {names}"
        )

    return None


# =====================================================================
# Available dates
# =====================================================================

def load_available_dates(
    path,
    start_date,
    end_date,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. "
            "Run 01_MAG_quality_control.py first."
        )

    df = pd.read_csv(path)

    date_col = find_column(
        df,
        [
            "date_utc",
            "date",
            "day",
            "file_date",
            "source_date",
        ],
        "date",
    )

    dates = (
        pd.to_datetime(
            df[date_col],
            errors="coerce",
        )
        .dropna()
        .dt.date
        .drop_duplicates()
        .sort_values()
    )

    if start_date is not None:
        dates = dates[
            dates
            >= start_date
        ]

    if end_date is not None:
        dates = dates[
            dates
            <= end_date
        ]

    return dates.tolist()


# =====================================================================
# Policy and synthetic samples
# =====================================================================

def load_policy(
    path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. "
            "Run 01f_MAG_gap_treatment_policy.py first."
        )

    policy = pd.read_csv(path)

    policy["time_before"] = pd.to_datetime(
        policy["time_before"],
        utc=True,
        format="mixed",
        errors="coerce",
    )

    policy["time_after"] = pd.to_datetime(
        policy["time_after"],
        utc=True,
        format="mixed",
        errors="coerce",
    )

    if policy["interpolate"].dtype != bool:
        policy["interpolate"] = (
            policy["interpolate"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
            .fillna(False)
            .astype(bool)
        )

    approved = policy[
        policy["interpolate"]
    ]

    bad_approved_time = (
        approved["time_before"].isna()
        | approved["time_after"].isna()
    )

    if bad_approved_time.any():
        cols = [
            c
            for c in [
                "gap_id",
                "source",
                "time_before",
                "time_after",
                "gap_duration_s",
            ]
            if c in approved.columns
        ]

        raise ValueError(
            "Approved gaps contain invalid timestamps after parsing. "
            f"Count: {int(bad_approved_time.sum())}. "
            "First rows:\n"
            + approved.loc[
                bad_approved_time,
                cols,
            ].head(10).to_string(index=False)
        )

    return policy


def edge_value(
    row,
    canonical,
    side,
):
    candidates = [
        f"{canonical}_{side}",
    ]

    for candidate in candidates:
        if candidate in row.index:
            value = pd.to_numeric(
                pd.Series(
                    [
                        row[
                            candidate
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[
                0
            ]

            return (
                float(value)
                if np.isfinite(value)
                else np.nan
            )

    return np.nan


def build_synthetic_samples(
    policy,
    nominal_cadence_s,
):
    approved = policy[
        policy[
            "interpolate"
        ]
    ].copy()

    rows = []

    cadence = pd.Timedelta(
        seconds=nominal_cadence_s
    )

    for _, gap in approved.iterrows():
        before = pd.Timestamp(
            gap[
                "time_before"
            ]
        )

        after = pd.Timestamp(
            gap[
                "time_after"
            ]
        )

        if (
            pd.isna(before)
            or pd.isna(after)
            or after <= before
        ):
            raise ValueError(
                "Approved gap has invalid temporal endpoints: "
                f"gap_id={gap.get('gap_id', 'unknown')}, "
                f"time_before={before}, time_after={after}"
            )

        dt_s = (
            after
            - before
        ).total_seconds()

        estimated_intervals = int(
            np.rint(
                dt_s
                / nominal_cadence_s
            )
        )

        n_missing = max(
            estimated_intervals
            - 1,
            0,
        )

        if n_missing == 0:
            continue

        b_before = np.array(
            [
                edge_value(
                    gap,
                    "bx_nt",
                    "before",
                ),
                edge_value(
                    gap,
                    "by_nt",
                    "before",
                ),
                edge_value(
                    gap,
                    "bz_nt",
                    "before",
                ),
            ],
            dtype=float,
        )

        b_after = np.array(
            [
                edge_value(
                    gap,
                    "bx_nt",
                    "after",
                ),
                edge_value(
                    gap,
                    "by_nt",
                    "after",
                ),
                edge_value(
                    gap,
                    "bz_nt",
                    "after",
                ),
            ],
            dtype=float,
        )

        if not (
            np.isfinite(
                b_before
            ).all()
            and np.isfinite(
                b_after
            ).all()
        ):
            raise ValueError(
                "Approved gap is missing finite magnetic-field edge "
                "values and cannot be interpolated: "
                f"gap_id={gap.get('gap_id', 'unknown')}, "
                f"time_before={before}, time_after={after}"
            )

        pos_before = np.array(
            [
                edge_value(
                    gap,
                    "x_ss_km",
                    "before",
                ),
                edge_value(
                    gap,
                    "y_ss_km",
                    "before",
                ),
                edge_value(
                    gap,
                    "z_ss_km",
                    "before",
                ),
            ],
            dtype=float,
        )

        pos_after = np.array(
            [
                edge_value(
                    gap,
                    "x_ss_km",
                    "after",
                ),
                edge_value(
                    gap,
                    "y_ss_km",
                    "after",
                ),
                edge_value(
                    gap,
                    "z_ss_km",
                    "after",
                ),
            ],
            dtype=float,
        )

        positions_available = (
            np.isfinite(
                pos_before
            ).all()
            and np.isfinite(
                pos_after
            ).all()
        )

        for missing_index in range(
            1,
            n_missing + 1,
        ):
            synthetic_time = (
                before
                + missing_index
                * cadence
            )

            if synthetic_time >= after:
                break

            alpha = (
                (
                    synthetic_time
                    - before
                ).total_seconds()
                / dt_s
            )

            b_interp = (
                b_before
                + alpha
                * (
                    b_after
                    - b_before
                )
            )

            row = {
                "datetime_utc":
                    synthetic_time,

                "bx_ss_nt":
                    float(
                        b_interp[
                            0
                        ]
                    ),

                "by_ss_nt":
                    float(
                        b_interp[
                            1
                        ]
                    ),

                "bz_ss_nt":
                    float(
                        b_interp[
                            2
                        ]
                    ),

                "b_total_nt":
                    float(
                        np.linalg.norm(
                            b_interp
                        )
                    ),

                "is_original":
                    False,

                "is_interpolated":
                    True,

                "gap_id":
                    gap[
                        "gap_id"
                    ],

                "interpolation_method":
                    "linear",
            }

            if positions_available:
                pos_interp = (
                    pos_before
                    + alpha
                    * (
                        pos_after
                        - pos_before
                    )
                )

                row[
                    "x_ss_km"
                ] = float(
                    pos_interp[
                        0
                    ]
                )

                row[
                    "y_ss_km"
                ] = float(
                    pos_interp[
                        1
                    ]
                )

                row[
                    "z_ss_km"
                ] = float(
                    pos_interp[
                        2
                    ]
                )

            rows.append(
                row
            )

    if not rows:
        return pd.DataFrame(
            columns=[
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
        )

    synthetic = pd.DataFrame(
        rows
    )

    synthetic[
        "datetime_utc"
    ] = pd.to_datetime(
        synthetic[
            "datetime_utc"
        ],
        utc=True,
    )

    synthetic[
        "date_utc"
    ] = synthetic[
        "datetime_utc"
    ].dt.date

    return synthetic


# =====================================================================
# BigQuery raw data
# =====================================================================

def discover_mag_schema(
    client,
    table_id,
):
    names = [
        field.name
        for field
        in client.get_table(
            table_id
        ).schema
    ]

    return {
        "time": find_schema_column(
            names,
            [
                "datetime_utc",
                "timestamp",
                "time_utc",
            ],
            "timestamp",
        ),
        "bx": find_schema_column(
            names,
            [
                "bx_ss_nt",
                "bx_nt",
                "Bx",
                "bx",
            ],
            "Bx",
        ),
        "by": find_schema_column(
            names,
            [
                "by_ss_nt",
                "by_nt",
                "By",
                "by",
            ],
            "By",
        ),
        "bz": find_schema_column(
            names,
            [
                "bz_ss_nt",
                "bz_nt",
                "Bz",
                "bz",
            ],
            "Bz",
        ),
        "btotal": find_schema_column(
            names,
            [
                "b_total_nt",
                "bmag_nt",
                "Bmag",
                "B_total",
                "b_total",
            ],
            "|B|",
            required=False,
        ),
        "x": find_schema_column(
            names,
            [
                "x_ss_km",
                "posx_ss_km",
                "x_km",
            ],
            "x position",
            required=False,
        ),
        "y": find_schema_column(
            names,
            [
                "y_ss_km",
                "posy_ss_km",
                "y_km",
            ],
            "y position",
            required=False,
        ),
        "z": find_schema_column(
            names,
            [
                "z_ss_km",
                "posz_ss_km",
                "z_km",
            ],
            "z position",
            required=False,
        ),
    }


def query_raw_dates(
    client,
    table_id,
    schema,
    dates,
):
    time_col = schema[
        "time"
    ]

    fields = [
        f"`{time_col}` AS datetime_utc",
        f"`{schema['bx']}` AS bx_ss_nt",
        f"`{schema['by']}` AS by_ss_nt",
        f"`{schema['bz']}` AS bz_ss_nt",
    ]

    if schema[
        "btotal"
    ] is not None:
        fields.append(
            f"`{schema['btotal']}` "
            "AS b_total_nt"
        )

    for canonical in [
        "x",
        "y",
        "z",
    ]:
        if schema[
            canonical
        ] is not None:
            fields.append(
                f"`{schema[canonical]}` "
                f"AS {canonical}_ss_km"
            )

    query = f"""
    SELECT
        {", ".join(fields)}
    FROM `{table_id}`
    WHERE
        DATE(`{time_col}`)
        IN UNNEST(@query_dates)
    ORDER BY `{time_col}`
    """

    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter(
                "query_dates",
                "DATE",
                dates,
            )
        ]
    )

    raw = (
        client.query(
            query,
            job_config=config,
        )
        .to_dataframe()
    )

    raw[
        "datetime_utc"
    ] = pd.to_datetime(
        raw[
            "datetime_utc"
        ],
        utc=True,
        errors="coerce",
    )

    for column in [
        "bx_ss_nt",
        "by_ss_nt",
        "bz_ss_nt",
        "b_total_nt",
        "x_ss_km",
        "y_ss_km",
        "z_ss_km",
    ]:
        if column in raw.columns:
            raw[
                column
            ] = pd.to_numeric(
                raw[
                    column
                ],
                errors="coerce",
            )

    if "b_total_nt" not in raw.columns:
        raw[
            "b_total_nt"
        ] = np.sqrt(
            raw[
                "bx_ss_nt"
            ] ** 2
            + raw[
                "by_ss_nt"
            ] ** 2
            + raw[
                "bz_ss_nt"
            ] ** 2
        )

    raw = (
        raw.dropna(
            subset=[
                "datetime_utc",
            ]
        )
        .sort_values(
            "datetime_utc"
        )
        .drop_duplicates(
            subset=[
                "datetime_utc"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    raw[
        "date_utc"
    ] = raw[
        "datetime_utc"
    ].dt.date

    raw[
        "is_original"
    ] = True

    raw[
        "is_interpolated"
    ] = False

    raw[
        "gap_id"
    ] = None

    raw[
        "interpolation_method"
    ] = None

    return raw


# =====================================================================
# Daily output
# =====================================================================

def daily_output_path(
    date_value,
):
    return (
        DATA_OUTPUT_DIR
        / f"{date_value.year:04d}"
        / f"{date_value.month:02d}"
        / (
            "mag_"
            f"{date_value.isoformat()}"
            ".parquet"
        )
    )


def combine_day(
    raw_day,
    synthetic_day,
):
    frames = [
        raw_day.copy()
    ]

    if not synthetic_day.empty:
        frames.append(
            synthetic_day.copy()
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    # Original observations always win if an exact timestamp collision occurs.
    combined[
        "_priority"
    ] = np.where(
        combined[
            "is_original"
        ],
        0,
        1,
    )

    combined = (
        combined.sort_values(
            [
                "datetime_utc",
                "_priority",
            ]
        )
        .drop_duplicates(
            subset=[
                "datetime_utc"
            ],
            keep="first",
        )
        .drop(
            columns=[
                "_priority",
                "date_utc",
            ],
            errors="ignore",
        )
        .sort_values(
            "datetime_utc"
        )
        .reset_index(
            drop=True
        )
    )

    return combined


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply the validated MAVEN MAG "
            "gap-interpolation policy."
        )
    )

    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE_ID,
    )

    parser.add_argument(
        "--start",
        default=None,
        help=(
            "Optional first UTC date "
            "(YYYY-MM-DD)."
        ),
    )

    parser.add_argument(
        "--end",
        default=None,
        help=(
            "Optional last UTC date "
            "(YYYY-MM-DD)."
        ),
    )

    parser.add_argument(
        "--nominal-cadence",
        type=float,
        default=DEFAULT_NOMINAL_CADENCE_S,
    )

    parser.add_argument(
        "--chunk-days",
        type=int,
        default=DEFAULT_CHUNK_DAYS,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite daily Parquet files "
            "that already exist."
        ),
    )

    args = parser.parse_args()

    if args.nominal_cadence <= 0:
        parser.error(
            "--nominal-cadence must be positive."
        )

    if args.chunk_days <= 0:
        parser.error(
            "--chunk-days must be positive."
        )

    start_date = (
        pd.Timestamp(
            args.start
        ).date()
        if args.start
        else None
    )

    end_date = (
        pd.Timestamp(
            args.end
        ).date()
        if args.end
        else None
    )

    if (
        start_date is not None
        and end_date is not None
        and start_date > end_date
    ):
        parser.error(
            "--start cannot be later than --end."
        )

    DATA_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy = load_policy(
        POLICY_CSV
    )

    synthetic = build_synthetic_samples(
        policy=policy,
        nominal_cadence_s=args.nominal_cadence,
    )

    unfilled = policy[
        ~policy[
            "interpolate"
        ]
    ].copy()

    unfilled_path = (
        RESULTS_DIR
        / "mag_unfilled_gap_intervals.csv"
    )

    unfilled.to_csv(
        unfilled_path,
        index=False,
    )

    available_dates = load_available_dates(
        path=QUALITY_CSV,
        start_date=start_date,
        end_date=end_date,
    )

    if not available_dates:
        raise RuntimeError(
            "No MAG dates found in the requested range."
        )

    if not synthetic.empty:
        synthetic = synthetic[
            synthetic[
                "date_utc"
            ].isin(
                available_dates
            )
        ].copy()

    client = bigquery.Client()

    schema = discover_mag_schema(
        client,
        args.table,
    )

    print()
    print(
        "Applying real-gap treatment..."
    )

    print(
        f"Dates to process:            "
        f"{len(available_dates):,}"
    )

    print(
        f"Approved real gaps:          "
        f"{int(policy['interpolate'].sum()):,}"
    )

    print(
        f"Synthetic samples prepared:  "
        f"{len(synthetic):,}"
    )

    print(
        f"Unfilled gap intervals:      "
        f"{len(unfilled):,}"
    )

    manifest_rows = []

    total_chunks = int(
        np.ceil(
            len(
                available_dates
            )
            / args.chunk_days
        )
    )

    for chunk_number, start in enumerate(
        range(
            0,
            len(
                available_dates
            ),
            args.chunk_days,
        ),
        start=1,
    ):
        chunk_dates = available_dates[
            start:
            start
            + args.chunk_days
        ]

        print()
        print(
            f"Chunk {chunk_number}/{total_chunks}: "
            f"{chunk_dates[0]} -> "
            f"{chunk_dates[-1]}"
        )

        raw = query_raw_dates(
            client=client,
            table_id=args.table,
            schema=schema,
            dates=chunk_dates,
        )

        for date_value in chunk_dates:
            output_path = daily_output_path(
                date_value
            )

            if (
                output_path.exists()
                and not args.overwrite
            ):
                print(
                    f"  {date_value}: "
                    "exists, skipped"
                )

                manifest_rows.append(
                    {
                        "date_utc":
                            date_value,

                        "status":
                            "skipped_existing",

                        "original_samples":
                            np.nan,

                        "interpolated_samples":
                            np.nan,

                        "output_samples":
                            np.nan,

                        "output_file":
                            str(
                                output_path
                            ),
                    }
                )

                continue

            raw_day = raw[
                raw[
                    "date_utc"
                ]
                == date_value
            ].copy()

            if synthetic.empty:
                synthetic_day = synthetic.copy()
            else:
                synthetic_day = synthetic[
                    synthetic[
                        "date_utc"
                    ]
                    == date_value
                ].copy()

            combined = combine_day(
                raw_day,
                synthetic_day,
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            combined.to_parquet(
                output_path,
                index=False,
            )

            interpolated_count = int(
                combined[
                    "is_interpolated"
                ].sum()
            )

            print(
                f"  {date_value}: "
                f"{len(raw_day):,} original + "
                f"{interpolated_count:,} interpolated "
                f"-> {len(combined):,}"
            )

            manifest_rows.append(
                {
                    "date_utc":
                        date_value,

                    "status":
                        "written",

                    "original_samples":
                        len(
                            raw_day
                        ),

                    "interpolated_samples":
                        interpolated_count,

                    "output_samples":
                        len(
                            combined
                        ),

                    "output_file":
                        str(
                            output_path
                        ),
                }
            )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest_path = (
        RESULTS_DIR
        / "mag_gap_treatment_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
    )

    written = manifest[
        manifest[
            "status"
        ]
        == "written"
    ]

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "dates_requested",
                "value":
                    len(
                        available_dates
                    ),
            },
            {
                "metric":
                    "dates_written",
                "value":
                    len(
                        written
                    ),
            },
            {
                "metric":
                    "approved_real_gaps",
                "value":
                    int(
                        policy[
                            "interpolate"
                        ].sum()
                    ),
            },
            {
                "metric":
                    "unfilled_real_gaps",
                "value":
                    len(
                        unfilled
                    ),
            },
            {
                "metric":
                    "synthetic_samples_prepared",
                "value":
                    len(
                        synthetic
                    ),
            },
            {
                "metric":
                    "original_samples_written",
                "value":
                    float(
                        written[
                            "original_samples"
                        ].sum()
                    ),
            },
            {
                "metric":
                    "interpolated_samples_written",
                "value":
                    float(
                        written[
                            "interpolated_samples"
                        ].sum()
                    ),
            },
        ]
    )

    summary_path = (
        RESULTS_DIR
        / "mag_gap_treatment_applied_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        "="
        * 78
    )

    print(
        "MAVEN MAG — GAP TREATMENT APPLIED"
    )

    print(
        "="
        * 78
    )

    print(
        f"Dates requested:              "
        f"{len(available_dates):,}"
    )

    print(
        f"Dates written:                "
        f"{len(written):,}"
    )

    print(
        f"Approved real gaps:           "
        f"{int(policy['interpolate'].sum()):,}"
    )

    print(
        f"Unfilled real gaps retained:  "
        f"{len(unfilled):,}"
    )

    print(
        f"Synthetic samples prepared:   "
        f"{len(synthetic):,}"
    )

    if len(
        written
    ):
        print(
            f"Interpolated samples written: "
            f"{int(written['interpolated_samples'].sum()):,}"
        )

    print(
        "="
        * 78
    )

    print()
    print(
        f"Manifest:       "
        f"{manifest_path}"
    )

    print(
        f"Summary:        "
        f"{summary_path}"
    )

    print(
        f"Unfilled gaps:  "
        f"{unfilled_path}"
    )

    print(
        f"Processed data: "
        f"{DATA_OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
