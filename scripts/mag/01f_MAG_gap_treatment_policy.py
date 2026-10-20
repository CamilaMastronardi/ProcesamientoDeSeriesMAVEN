#!/usr/bin/env python3
"""
01f_MAG_gap_treatment_policy.py

Classify every real MAVEN MAG gap according to the interpolation policy
derived from the pseudo-gap experiment.

This script DOES NOT modify the MAG time series.

Timestamp parsing
-----------------
Gap timestamps are parsed with ``format="mixed"`` so ISO-8601 strings with
and without fractional seconds are both preserved. For boundary events, if
one endpoint is missing but the other endpoint and boundary duration are
available, the missing endpoint is reconstructed from those two quantities.
If both one endpoint and the stored duration were lost upstream, the nearest
raw MAG timestamp is recovered from BigQuery and the duration is rebuilt from
the final endpoint pair. Any remaining recovery failure is treated as an error.

Policy
------
Local variability is

    V = median(|B[i+1] - B[i]|) / median(|B|)

computed from observed MAG samples in windows immediately before and after
each real gap. Increments are calculated separately on each side so the gap
itself is never crossed.

Regimes
-------
quiet:
    V <= 0.0471873

intermediate:
    0.0471873 < V <= 0.174766

variable:
    V > 0.174766

Treatment rule
--------------
quiet:
    linear interpolation only if gap duration <= 5 s

intermediate:
    linear interpolation only if gap duration <= 3 s

variable:
    never interpolate automatically

The script also requires the observed gap-edge samples to be recovered within
1.5 s of the timestamps stored by the previous gap-analysis scripts.

Inputs
------
results/MAG_gap_analysis/mag_gap_events.csv
results/MAG_boundary_analysis/mag_boundary_events.csv

BigQuery default
----------------
maven-vignes-crossings.maven.mag_fields

Outputs
-------
results/MAG_gap_treatment_policy/
    mag_gap_treatment_policy.csv
    mag_gap_treatment_summary.csv
    mag_gap_treatment_summary_by_regime.csv
    figures/
        gap_treatment_by_regime.png

Run from repository root
------------------------
python3 scripts/mag/01f_MAG_gap_treatment_policy.py

Useful options
--------------
python3 scripts/mag/01f_MAG_gap_treatment_policy.py --batch-size 200
python3 scripts/mag/01f_MAG_gap_treatment_policy.py --context-seconds 90
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import bigquery


DEFAULT_TABLE_ID = "maven-vignes-crossings.maven.mag_fields"

INTERNAL_GAPS_CSV = Path(
    "results/MAG_gap_analysis/mag_gap_events.csv"
)

BOUNDARY_GAPS_CSV = Path(
    "results/MAG_boundary_analysis/mag_boundary_events.csv"
)

OUTPUT_DIR = Path(
    "results/MAG_gap_treatment_policy"
)

FIGURE_DIR = OUTPUT_DIR / "figures"

GAP_THRESHOLD_S = 1.5
EDGE_TOLERANCE_S = 1.5

QUIET_MAX = 0.0471873
INTERMEDIATE_MAX = 0.174766

QUIET_MAX_GAP_S = 5.0
INTERMEDIATE_MAX_GAP_S = 3.0

DEFAULT_CONTEXT_SECONDS = 60.0
DEFAULT_BATCH_SIZE = 250


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


def variability_regime(
    value,
):
    if not np.isfinite(value):
        return "unclassified"

    if value <= QUIET_MAX:
        return "quiet"

    if value <= INTERMEDIATE_MAX:
        return "intermediate"

    return "variable"


# =====================================================================
# Load real gaps
# =====================================================================

def normalize_internal_gaps(
    path,
):
    df = pd.read_csv(path)

    before_col = find_column(
        df,
        [
            "gap_start_time",
            "time_before",
            "timestamp_before",
        ],
        "internal gap start time",
    )

    after_col = find_column(
        df,
        [
            "gap_end_time",
            "time_after",
            "timestamp_after",
        ],
        "internal gap end time",
    )

    duration_col = find_column(
        df,
        [
            "dt_s",
            "gap_duration_s",
            "duration_s",
        ],
        "internal gap duration",
    )

    source_file_col = find_column(
        df,
        ["source_file"],
        "source file",
        required=False,
    )

    out = pd.DataFrame(
        {
            "source": "internal",
            "time_before": pd.to_datetime(
                df[before_col],
                utc=True,
                format="mixed",
                errors="coerce",
            ),
            "time_after": pd.to_datetime(
                df[after_col],
                utc=True,
                format="mixed",
                errors="coerce",
            ),
            "gap_duration_s": pd.to_numeric(
                df[duration_col],
                errors="coerce",
            ),
        }
    )

    bad_time = (
        out["time_before"].isna()
        | out["time_after"].isna()
    )

    if bad_time.any():
        bad_rows = df.loc[
            bad_time,
            [before_col, after_col],
        ]

        raise ValueError(
            "Internal gap timestamp parsing failed for "
            f"{int(bad_time.sum())} row(s). "
            "First problematic rows:\n"
            f"{bad_rows.head(10).to_string(index=False)}"
        )

    out["original_category"] = None

    if source_file_col is not None:
        out["source_file"] = (
            df[source_file_col]
            .astype(str)
        )
    else:
        out["source_file"] = None

    return out



def query_nearest_boundary_timestamp(
    client,
    table_id,
    time_col,
    anchor_time,
    direction,
):
    """
    Recover the nearest raw MAG timestamp around a boundary from BigQuery.

    Used only as a fallback when an upstream boundary CSV lost one endpoint
    and also lost the stored boundary duration.
    """
    anchor_time = pd.Timestamp(anchor_time)

    if anchor_time.tzinfo is None:
        anchor_time = anchor_time.tz_localize("UTC")
    else:
        anchor_time = anchor_time.tz_convert("UTC")

    window = pd.Timedelta(days=2)

    if direction == "previous":
        start_time = anchor_time - window
        end_time = anchor_time
        comparison = "<"
        ordering = "DESC"
    elif direction == "next":
        start_time = anchor_time
        end_time = anchor_time + window
        comparison = ">"
        ordering = "ASC"
    else:
        raise ValueError(
            "direction must be 'previous' or 'next'"
        )

    query = f"""
    SELECT
        `{time_col}` AS time_utc
    FROM `{table_id}`
    WHERE
        DATE(`{time_col}`) BETWEEN @start_date AND @end_date
        AND `{time_col}` >= @start_time
        AND `{time_col}` <= @end_time
        AND `{time_col}` {comparison} @anchor_time
    ORDER BY `{time_col}` {ordering}
    LIMIT 1
    """

    config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "start_date",
                "DATE",
                start_time.date(),
            ),
            bigquery.ScalarQueryParameter(
                "end_date",
                "DATE",
                end_time.date(),
            ),
            bigquery.ScalarQueryParameter(
                "start_time",
                "TIMESTAMP",
                start_time.to_pydatetime(),
            ),
            bigquery.ScalarQueryParameter(
                "end_time",
                "TIMESTAMP",
                end_time.to_pydatetime(),
            ),
            bigquery.ScalarQueryParameter(
                "anchor_time",
                "TIMESTAMP",
                anchor_time.to_pydatetime(),
            ),
        ]
    )

    result = (
        client.query(
            query,
            job_config=config,
        )
        .to_dataframe()
    )

    if result.empty:
        return pd.NaT

    return pd.to_datetime(
        result.iloc[0]["time_utc"],
        utc=True,
        format="mixed",
        errors="coerce",
    )


def normalize_boundary_gaps(
    path,
    client,
    table_id,
    schema,
):
    df = pd.read_csv(path)

    before_col = find_column(
        df,
        [
            "last_time_before",
            "time_before",
            "timestamp_before",
        ],
        "boundary gap start time",
    )

    after_col = find_column(
        df,
        [
            "first_time_after",
            "time_after",
            "timestamp_after",
        ],
        "boundary gap end time",
    )

    duration_col = find_column(
        df,
        [
            "boundary_dt_s",
            "dt_s",
            "gap_duration_s",
            "duration_s",
        ],
        "boundary gap duration",
    )

    category_col = find_column(
        df,
        [
            "category",
            "gap_category",
        ],
        "boundary category",
        required=False,
    )

    out = pd.DataFrame(
        {
            "source": "boundary",
            "time_before": pd.to_datetime(
                df[before_col],
                utc=True,
                format="mixed",
                errors="coerce",
            ),
            "time_after": pd.to_datetime(
                df[after_col],
                utc=True,
                format="mixed",
                errors="coerce",
            ),
            "gap_duration_s": pd.to_numeric(
                df[duration_col],
                errors="coerce",
            ),
        }
    )

    # -------------------------------------------------------------
    # First recover from a valid stored duration, when possible.
    # -------------------------------------------------------------
    valid_duration = (
        np.isfinite(
            out["gap_duration_s"]
        )
        & (
            out["gap_duration_s"]
            > 0
        )
    )

    recover_before_from_duration = (
        out["time_before"].isna()
        & out["time_after"].notna()
        & valid_duration
    )

    if recover_before_from_duration.any():
        out.loc[
            recover_before_from_duration,
            "time_before",
        ] = (
            out.loc[
                recover_before_from_duration,
                "time_after",
            ]
            - pd.to_timedelta(
                out.loc[
                    recover_before_from_duration,
                    "gap_duration_s",
                ],
                unit="s",
            )
        )

    recover_after_from_duration = (
        out["time_after"].isna()
        & out["time_before"].notna()
        & valid_duration
    )

    if recover_after_from_duration.any():
        out.loc[
            recover_after_from_duration,
            "time_after",
        ] = (
            out.loc[
                recover_after_from_duration,
                "time_before",
            ]
            + pd.to_timedelta(
                out.loc[
                    recover_after_from_duration,
                    "gap_duration_s",
                ],
                unit="s",
            )
        )

    recovered_from_duration = int(
        recover_before_from_duration.sum()
        + recover_after_from_duration.sum()
    )

    if recovered_from_duration:
        print(
            "Recovered boundary timestamp endpoint(s) "
            f"from stored duration: {recovered_from_duration:,}"
        )

    # -------------------------------------------------------------
    # If both one endpoint and the stored duration were lost upstream,
    # recover the missing endpoint directly from raw MAG in BigQuery.
    # -------------------------------------------------------------
    recovered_from_bigquery = 0

    incomplete_indices = out.index[
        out["time_before"].isna()
        | out["time_after"].isna()
    ]

    for index in incomplete_indices:
        before = out.at[
            index,
            "time_before",
        ]
        after = out.at[
            index,
            "time_after",
        ]

        if pd.isna(before) and pd.notna(after):
            recovered = query_nearest_boundary_timestamp(
                client=client,
                table_id=table_id,
                time_col=schema["time"],
                anchor_time=after,
                direction="previous",
            )

            if pd.notna(recovered):
                out.at[
                    index,
                    "time_before",
                ] = recovered
                recovered_from_bigquery += 1

        elif pd.isna(after) and pd.notna(before):
            recovered = query_nearest_boundary_timestamp(
                client=client,
                table_id=table_id,
                time_col=schema["time"],
                anchor_time=before,
                direction="next",
            )

            if pd.notna(recovered):
                out.at[
                    index,
                    "time_after",
                ] = recovered
                recovered_from_bigquery += 1

    if recovered_from_bigquery:
        print(
            "Recovered boundary timestamp endpoint(s) "
            f"from raw MAG BigQuery data: {recovered_from_bigquery:,}"
        )

    bad_time = (
        out["time_before"].isna()
        | out["time_after"].isna()
    )

    if bad_time.any():
        bad_rows = df.loc[
            bad_time,
            [before_col, after_col, duration_col],
        ]

        raise ValueError(
            "Boundary gap timestamp recovery failed for "
            f"{int(bad_time.sum())} row(s). "
            "First problematic rows:\n"
            f"{bad_rows.head(10).to_string(index=False)}"
        )

    reconstructed_duration_s = (
        out["time_after"]
        - out["time_before"]
    ).dt.total_seconds()

    original_duration = (
        out["gap_duration_s"].copy()
    )

    comparable = (
        np.isfinite(
            original_duration
        )
        & np.isfinite(
            reconstructed_duration_s
        )
    )

    duration_difference_s = np.abs(
        reconstructed_duration_s
        - original_duration
    )

    inconsistent_duration = (
        comparable
        & (
            duration_difference_s
            > 1e-6
        )
    )

    if inconsistent_duration.any():
        raise ValueError(
            "Boundary timestamps and stored duration are inconsistent "
            f"for {int(inconsistent_duration.sum())} row(s)."
        )

    # Reconstructed endpoints are authoritative if the stored duration
    # was missing upstream.
    out["gap_duration_s"] = (
        reconstructed_duration_s
    )

    if category_col is not None:
        out["original_category"] = (
            df[category_col]
            .astype(str)
        )
    else:
        out["original_category"] = None

    out["source_file"] = None

    return out[
        out["gap_duration_s"]
        > GAP_THRESHOLD_S
    ].copy()


def load_gap_events(
    client,
    table_id,
    schema,
):
    frames = []

    if INTERNAL_GAPS_CSV.exists():
        frames.append(
            normalize_internal_gaps(
                INTERNAL_GAPS_CSV
            )
        )
    else:
        raise FileNotFoundError(
            f"Missing {INTERNAL_GAPS_CSV}"
        )

    if BOUNDARY_GAPS_CSV.exists():
        frames.append(
            normalize_boundary_gaps(
                BOUNDARY_GAPS_CSV,
                client=client,
                table_id=table_id,
                schema=schema,
            )
        )
    else:
        print(
            f"Warning: {BOUNDARY_GAPS_CSV} not found. "
            "Only internal gaps will be classified."
        )

    events = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    events = events.dropna(
        subset=[
            "time_before",
            "time_after",
            "gap_duration_s",
        ]
    )

    events = events[
        events["gap_duration_s"]
        > GAP_THRESHOLD_S
    ].copy()

    events = events.sort_values(
        [
            "time_before",
            "time_after",
        ]
    ).reset_index(
        drop=True
    )

    print(
        "Gap catalog after timestamp normalization:"
    )

    source_counts = (
        events["source"]
        .value_counts()
        .sort_index()
    )

    for source, count in source_counts.items():
        print(
            f"  {source:8s}: "
            f"{count:,}"
        )

    print(
        f"  total   : "
        f"{len(events):,}"
    )

    events["gap_id"] = [
        f"gap_{index:06d}"
        for index
        in range(
            1,
            len(events) + 1,
        )
    ]

    return events


# =====================================================================
# BigQuery schema and context query
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

    schema = {
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

    return schema


def query_gap_batch(
    client,
    table_id,
    schema,
    batch,
    context_seconds,
):
    time_col = schema["time"]

    fields = [
        f"`{time_col}` AS time_utc",
        f"`{schema['bx']}` AS bx_nt",
        f"`{schema['by']}` AS by_nt",
        f"`{schema['bz']}` AS bz_nt",
    ]

    if schema["btotal"] is not None:
        fields.append(
            f"`{schema['btotal']}` "
            "AS btotal_nt"
        )

    for canonical in [
        "x",
        "y",
        "z",
    ]:
        if schema[canonical] is not None:
            fields.append(
                f"`{schema[canonical]}` "
                f"AS {canonical}_ss_km"
            )

    conditions = []
    parameters = []
    dates = set()

    context = pd.Timedelta(
        seconds=context_seconds
    )

    for local_index, (
        _,
        row,
    ) in enumerate(
        batch.iterrows()
    ):
        before = pd.Timestamp(
            row["time_before"]
        )

        after = pd.Timestamp(
            row["time_after"]
        )

        left_start = (
            before
            - context
        )

        left_end = before

        right_start = after

        right_end = (
            after
            + context
        )

        conditions.append(
            "("
            f"(`{time_col}` BETWEEN "
            f"@ls_{local_index} "
            f"AND @le_{local_index}) "
            "OR "
            f"(`{time_col}` BETWEEN "
            f"@rs_{local_index} "
            f"AND @re_{local_index})"
            ")"
        )

        parameters.extend(
            [
                bigquery.ScalarQueryParameter(
                    f"ls_{local_index}",
                    "TIMESTAMP",
                    left_start.to_pydatetime(),
                ),
                bigquery.ScalarQueryParameter(
                    f"le_{local_index}",
                    "TIMESTAMP",
                    left_end.to_pydatetime(),
                ),
                bigquery.ScalarQueryParameter(
                    f"rs_{local_index}",
                    "TIMESTAMP",
                    right_start.to_pydatetime(),
                ),
                bigquery.ScalarQueryParameter(
                    f"re_{local_index}",
                    "TIMESTAMP",
                    right_end.to_pydatetime(),
                ),
            ]
        )

        for timestamp in [
            left_start,
            left_end,
            right_start,
            right_end,
        ]:
            dates.add(
                timestamp.date()
            )

    parameters.append(
        bigquery.ArrayQueryParameter(
            "query_dates",
            "DATE",
            sorted(dates),
        )
    )

    query = f"""
    SELECT
        {", ".join(fields)}
    FROM `{table_id}`
    WHERE
        DATE(`{time_col}`)
        IN UNNEST(@query_dates)
        AND (
            {" OR ".join(conditions)}
        )
    ORDER BY `{time_col}`
    """

    config = bigquery.QueryJobConfig(
        query_parameters=parameters
    )

    data = (
        client.query(
            query,
            job_config=config,
        )
        .to_dataframe()
    )

    data["time_utc"] = pd.to_datetime(
        data["time_utc"],
        utc=True,
        format="mixed",
        errors="coerce",
    )

    for column in [
        "bx_nt",
        "by_nt",
        "bz_nt",
        "btotal_nt",
        "x_ss_km",
        "y_ss_km",
        "z_ss_km",
    ]:
        if column in data.columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

    if "btotal_nt" not in data.columns:
        data["btotal_nt"] = np.sqrt(
            data["bx_nt"] ** 2
            + data["by_nt"] ** 2
            + data["bz_nt"] ** 2
        )

    return (
        data.dropna(
            subset=[
                "time_utc",
                "bx_nt",
                "by_nt",
                "bz_nt",
            ]
        )
        .sort_values(
            "time_utc"
        )
        .drop_duplicates(
            subset=[
                "time_utc"
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =====================================================================
# Gap metrics and policy
# =====================================================================

def local_variability(
    left,
    right,
):
    increment_chunks = []

    for context in [
        left,
        right,
    ]:
        vectors = context[
            [
                "bx_nt",
                "by_nt",
                "bz_nt",
            ]
        ].to_numpy(
            dtype=float
        )

        if len(vectors) >= 2:
            increment_chunks.append(
                np.linalg.norm(
                    np.diff(
                        vectors,
                        axis=0,
                    ),
                    axis=1,
                )
            )

    if not increment_chunks:
        return np.nan

    increments = np.concatenate(
        increment_chunks
    )

    context_vectors = pd.concat(
        [
            left[
                [
                    "bx_nt",
                    "by_nt",
                    "bz_nt",
                ]
            ],
            right[
                [
                    "bx_nt",
                    "by_nt",
                    "bz_nt",
                ]
            ],
        ],
        ignore_index=True,
    ).to_numpy(
        dtype=float
    )

    field_scale = np.nanmedian(
        np.linalg.norm(
            context_vectors,
            axis=1,
        )
    )

    if (
        not np.isfinite(
            field_scale
        )
        or field_scale <= 0
    ):
        return np.nan

    return float(
        np.nanmedian(
            increments
        )
        / field_scale
    )


def apply_policy(
    regime,
    gap_duration_s,
):
    if regime == "quiet":
        if (
            gap_duration_s
            <= QUIET_MAX_GAP_S
        ):
            return (
                True,
                "linear",
                "quiet_and_short_enough",
            )

        return (
            False,
            None,
            "quiet_but_gap_too_long",
        )

    if regime == "intermediate":
        if (
            gap_duration_s
            <= INTERMEDIATE_MAX_GAP_S
        ):
            return (
                True,
                "linear",
                "intermediate_and_short_enough",
            )

        return (
            False,
            None,
            "intermediate_gap_too_long",
        )

    if regime == "variable":
        return (
            False,
            None,
            "high_local_variability",
        )

    return (
        False,
        None,
        "variability_unclassified",
    )


def analyze_gap(
    event,
    context_data,
    context_seconds,
):
    before = pd.Timestamp(
        event["time_before"]
    )

    after = pd.Timestamp(
        event["time_after"]
    )

    context = pd.Timedelta(
        seconds=context_seconds
    )

    left = context_data[
        (
            context_data[
                "time_utc"
            ]
            >= before - context
        )
        & (
            context_data[
                "time_utc"
            ]
            <= before
        )
    ].copy()

    right = context_data[
        (
            context_data[
                "time_utc"
            ]
            >= after
        )
        & (
            context_data[
                "time_utc"
            ]
            <= after + context
        )
    ].copy()

    result = {
        "gap_id":
            event["gap_id"],

        "source":
            event["source"],

        "source_file":
            event.get(
                "source_file",
                None,
            ),

        "original_category":
            event.get(
                "original_category",
                None,
            ),

        "time_before":
            before,

        "time_after":
            after,

        "gap_duration_s":
            float(
                event[
                    "gap_duration_s"
                ]
            ),

        "n_context_left":
            len(left),

        "n_context_right":
            len(right),
    }

    if (
        left.empty
        or right.empty
    ):
        result.update(
            {
                "status":
                    "missing_context",

                "local_variability_score":
                    np.nan,

                "variability_regime":
                    "unclassified",

                "interpolate":
                    False,

                "interpolation_method":
                    None,

                "decision_reason":
                    "missing_context",
            }
        )

        return result

    left_edge = left.iloc[
        -1
    ]

    right_edge = right.iloc[
        0
    ]

    left_edge_time = pd.Timestamp(
        left_edge[
            "time_utc"
        ]
    )

    right_edge_time = pd.Timestamp(
        right_edge[
            "time_utc"
        ]
    )

    left_offset = abs(
        (
            before
            - left_edge_time
        ).total_seconds()
    )

    right_offset = abs(
        (
            right_edge_time
            - after
        ).total_seconds()
    )

    result[
        "edge_before_offset_s"
    ] = left_offset

    result[
        "edge_after_offset_s"
    ] = right_offset

    for column in [
        "bx_nt",
        "by_nt",
        "bz_nt",
        "btotal_nt",
        "x_ss_km",
        "y_ss_km",
        "z_ss_km",
    ]:
        if column in context_data.columns:
            result[
                f"{column}_before"
            ] = left_edge.get(
                column,
                np.nan,
            )

            result[
                f"{column}_after"
            ] = right_edge.get(
                column,
                np.nan,
            )

    if (
        left_offset
        > EDGE_TOLERANCE_S
        or right_offset
        > EDGE_TOLERANCE_S
    ):
        result.update(
            {
                "status":
                    "edge_mismatch",

                "local_variability_score":
                    np.nan,

                "variability_regime":
                    "unclassified",

                "interpolate":
                    False,

                "interpolation_method":
                    None,

                "decision_reason":
                    "edge_timestamp_mismatch",
            }
        )

        return result

    variability = local_variability(
        left,
        right,
    )

    regime = variability_regime(
        variability
    )

    (
        interpolate,
        method,
        reason,
    ) = apply_policy(
        regime,
        float(
            event[
                "gap_duration_s"
            ]
        ),
    )

    result.update(
        {
            "status":
                "ok",

            "local_variability_score":
                variability,

            "variability_regime":
                regime,

            "interpolate":
                bool(
                    interpolate
                ),

            "interpolation_method":
                method,

            "decision_reason":
                reason,
        }
    )

    return result


# =====================================================================
# Summaries and figure
# =====================================================================

def make_summary(
    policy,
):
    rows = [
        {
            "metric":
                "total_gaps",
            "value":
                len(policy),
        },
        {
            "metric":
                "classified_ok",
            "value":
                int(
                    (
                        policy[
                            "status"
                        ]
                        == "ok"
                    ).sum()
                ),
        },
        {
            "metric":
                "approved_for_interpolation",
            "value":
                int(
                    policy[
                        "interpolate"
                    ].sum()
                ),
        },
        {
            "metric":
                "not_interpolated",
            "value":
                int(
                    (
                        ~policy[
                            "interpolate"
                        ]
                    ).sum()
                ),
        },
        {
            "metric":
                "approved_fraction",
            "value":
                float(
                    policy[
                        "interpolate"
                    ].mean()
                ),
        },
        {
            "metric":
                "quiet_threshold",
            "value":
                QUIET_MAX,
        },
        {
            "metric":
                "intermediate_threshold",
            "value":
                INTERMEDIATE_MAX,
        },
        {
            "metric":
                "quiet_max_gap_s",
            "value":
                QUIET_MAX_GAP_S,
        },
        {
            "metric":
                "intermediate_max_gap_s",
            "value":
                INTERMEDIATE_MAX_GAP_S,
        },
    ]

    return pd.DataFrame(
        rows
    )


def make_regime_summary(
    policy,
):
    return (
        policy.groupby(
            [
                "source",
                "variability_regime",
                "interpolate",
                "decision_reason",
            ],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "gap_count"
            }
        )
    )


def make_regime_figure(
    policy,
    output_path,
):
    regimes = [
        "quiet",
        "intermediate",
        "variable",
        "unclassified",
    ]

    approved = []
    rejected = []

    for regime in regimes:
        group = policy[
            policy[
                "variability_regime"
            ]
            == regime
        ]

        approved.append(
            int(
                group[
                    "interpolate"
                ].sum()
            )
        )

        rejected.append(
            int(
                (
                    ~group[
                        "interpolate"
                    ]
                ).sum()
            )
        )

    x = np.arange(
        len(
            regimes
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    ax.bar(
        x,
        approved,
        label="interpolate",
    )

    ax.bar(
        x,
        rejected,
        bottom=approved,
        label="keep gap",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        regimes
    )

    ax.set_ylabel(
        "Number of real MAG gaps"
    )

    ax.set_xlabel(
        "Local variability regime"
    )

    ax.set_title(
        "MAG real-gap treatment policy"
    )

    ax.legend()

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        fig
    )


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classify all real MAVEN MAG gaps "
            "according to the validated interpolation policy."
        )
    )

    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE_ID,
    )

    parser.add_argument(
        "--context-seconds",
        type=float,
        default=DEFAULT_CONTEXT_SECONDS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    args = parser.parse_args()

    if args.context_seconds <= 0:
        parser.error(
            "--context-seconds must be positive."
        )

    if args.batch_size <= 0:
        parser.error(
            "--batch-size must be positive."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = bigquery.Client()

    schema = discover_mag_schema(
        client,
        args.table,
    )

    print()
    print(
        "Detected BigQuery MAG schema:"
    )

    for key, value in schema.items():
        print(
            f"  {key:8s}: "
            f"{value}"
        )

    print()
    print(
        "Loading real MAG gaps..."
    )

    events = load_gap_events(
        client=client,
        table_id=args.table,
        schema=schema,
    )

    print(
        f"Real gaps to classify: "
        f"{len(events):,}"
    )

    results = []

    total_batches = int(
        np.ceil(
            len(events)
            / args.batch_size
        )
    )

    for batch_number, start in enumerate(
        range(
            0,
            len(events),
            args.batch_size,
        ),
        start=1,
    ):
        stop = min(
            start
            + args.batch_size,
            len(events),
        )

        batch = events.iloc[
            start:stop
        ]

        print(
            f"Batch "
            f"{batch_number}/{total_batches}: "
            f"gaps {start + 1:,}-{stop:,}"
        )

        context_data = query_gap_batch(
            client=client,
            table_id=args.table,
            schema=schema,
            batch=batch,
            context_seconds=args.context_seconds,
        )

        for _, event in batch.iterrows():
            results.append(
                analyze_gap(
                    event=event,
                    context_data=context_data,
                    context_seconds=args.context_seconds,
                )
            )

    policy = pd.DataFrame(
        results
    )

    policy = policy.sort_values(
        [
            "time_before",
            "time_after",
        ]
    ).reset_index(
        drop=True
    )

    summary = make_summary(
        policy
    )

    regime_summary = make_regime_summary(
        policy
    )

    policy_path = (
        OUTPUT_DIR
        / "mag_gap_treatment_policy.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "mag_gap_treatment_summary.csv"
    )

    regime_summary_path = (
        OUTPUT_DIR
        / "mag_gap_treatment_summary_by_regime.csv"
    )

    policy.to_csv(
        policy_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    regime_summary.to_csv(
        regime_summary_path,
        index=False,
    )

    make_regime_figure(
        policy,
        FIGURE_DIR
        / "gap_treatment_by_regime.png",
    )

    approved = int(
        policy[
            "interpolate"
        ].sum()
    )

    print()
    print(
        "="
        * 78
    )

    print(
        "MAVEN MAG — REAL GAP TREATMENT POLICY"
    )

    print(
        "="
        * 78
    )

    print(
        f"Real gaps classified:          "
        f"{len(policy):,}"
    )

    print(
        f"Approved for interpolation:    "
        f"{approved:,} "
        f"({100 * approved / len(policy):.2f}%)"
    )

    print(
        f"Kept as real gaps:             "
        f"{len(policy) - approved:,}"
    )

    print()
    print(
        "By variability regime:"
    )

    regime_counts = (
        policy.groupby(
            "variability_regime"
        )
        .agg(
            total=(
                "gap_id",
                "size",
            ),
            interpolate=(
                "interpolate",
                "sum",
            ),
        )
    )

    print(
        regime_counts
    )

    print()
    print(
        "Policy:"
    )

    print(
        f"  quiet        V <= {QUIET_MAX:.7f}: "
        f"linear if gap <= {QUIET_MAX_GAP_S:g} s"
    )

    print(
        f"  intermediate {QUIET_MAX:.7f} < V <= "
        f"{INTERMEDIATE_MAX:.6f}: "
        f"linear if gap <= {INTERMEDIATE_MAX_GAP_S:g} s"
    )

    print(
        f"  variable     V > {INTERMEDIATE_MAX:.6f}: "
        "do not interpolate"
    )

    print(
        "="
        * 78
    )

    print()
    print(
        f"Policy table:    "
        f"{policy_path}"
    )

    print(
        f"Summary:         "
        f"{summary_path}"
    )

    print(
        f"Regime summary:  "
        f"{regime_summary_path}"
    )

    print(
        f"Figures:         "
        f"{FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()
