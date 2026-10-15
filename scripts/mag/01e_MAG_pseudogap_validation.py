#!/usr/bin/env python3
"""
01e_MAG_pseudogap_validation.py

Pseudo-gap validation for MAVEN MAG interpolation.

Purpose
-------
Quantitatively compare several reconstruction methods before defining a real
gap-filling policy for MAVEN MAG.

For each pseudo-gap the script:
1. Selects a truly continuous MAG interval.
2. Hides the real samples inside an artificial gap.
3. Reconstructs the SAME hidden samples with:
       - linear interpolation
       - cubic spline interpolation
       - bidirectional local autoregression (AR)
4. Compares every reconstruction with the hidden true data.
5. Repeats the experiment for several gap durations.
6. Separates cases by local MAG variability.
7. Reports which method performs best for each gap-duration regime.

The experiment is deliberately paired: all methods see the same pseudo-gaps.

Gap-duration convention
-----------------------
The gap duration is the separation between the last visible sample before the
gap and the first visible sample after it.

At nominal 1 s cadence:
    2 s gap  -> 1 hidden sample
    3 s gap  -> 2 hidden samples
    5 s gap  -> 4 hidden samples
    10 s gap -> 9 hidden samples
    30 s gap -> 29 hidden samples

Interpolation methods
---------------------
LINEAR
    B(t) is reconstructed component-by-component between the two gap edges.

SPLINE
    A cubic spline is fitted independently to Bx, By and Bz using observable
    samples from both sides of the gap. The hidden samples are never used.

AR
    A local AutoReg model is fitted separately to Bx, By and Bz.
    One model forecasts forward from the left context and another forecasts
    backward from the right context. Both estimates are blended across the gap.

Inputs
------
results/MAG_quality/mag_quality_by_day.csv

BigQuery:
    maven-vignes-crossings.maven.mag_fields

Outputs
-------
results/MAG_pseudogap_validation/
    mag_pseudogap_trials.csv
    mag_pseudogap_summary_by_gap_method.csv
    mag_pseudogap_summary_by_gap_method_regime.csv
    mag_pseudogap_winner_summary.csv
    mag_pseudogap_variability_thresholds.csv

    figures/
        normalized_vector_rmse_by_gap_method.png
        angle_error_by_gap_method.png
        winner_fraction_by_gap.png
        error_vs_variability.png

Default experiment
------------------
- 12 dates distributed over the mission
- 100 pseudo-gaps for each duration
- durations: 2, 3, 5, 10, 30 s
- 60 s continuous context on each side
- AR order: 10

Run from repository root
------------------------
python3 scripts/mag/01e_MAG_pseudogap_validation.py

Examples
--------
python3 scripts/mag/01e_MAG_pseudogap_validation.py --gap-durations 2 3 5
python3 scripts/mag/01e_MAG_pseudogap_validation.py --trials-per-gap 200
python3 scripts/mag/01e_MAG_pseudogap_validation.py --ar-lags 15
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import bigquery
from scipy.interpolate import CubicSpline


# =====================================================================
# Configuration
# =====================================================================

DEFAULT_TABLE_ID = "maven-vignes-crossings.maven.mag_fields"

QUALITY_CSV = Path(
    "results/MAG_quality/mag_quality_by_day.csv"
)

OUTPUT_DIR = Path(
    "results/MAG_pseudogap_validation"
)

FIGURE_DIR = OUTPUT_DIR / "figures"

DEFAULT_GAP_DURATIONS_S = [
    2.0,
    3.0,
    5.0,
    10.0,
    30.0,
]

DEFAULT_SAMPLE_DAYS = 12
DEFAULT_TRIALS_PER_GAP = 100
DEFAULT_CONTEXT_SECONDS = 60.0
DEFAULT_NOMINAL_CADENCE_S = 1.0
DEFAULT_GAP_THRESHOLD_S = 1.5
DEFAULT_AR_LAGS = 10
DEFAULT_RANDOM_SEED = 42

METHOD_ORDER = [
    "linear",
    "spline",
    "ar",
]

REGIME_ORDER = [
    "quiet",
    "intermediate",
    "variable",
]


# =====================================================================
# Generic helpers
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


def safe_quantile(
    series,
    q,
):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return np.nan

    return float(
        values.quantile(q)
    )


# =====================================================================
# Mission-spanning date selection
# =====================================================================

def load_available_dates(
    path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input: {path}. "
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
        .tolist()
    )

    if not dates:
        raise RuntimeError(
            "No valid MAG dates were found."
        )

    return dates


def select_mission_spanning_dates(
    available_dates,
    n_days,
):
    """
    Select dates approximately uniformly across the mission.
    """
    if n_days >= len(
        available_dates
    ):
        return available_dates

    positions = np.linspace(
        0,
        len(available_dates) - 1,
        n_days,
    )

    indices = np.unique(
        np.rint(
            positions
        ).astype(int)
    )

    return [
        available_dates[index]
        for index in indices
    ]


# =====================================================================
# BigQuery
# =====================================================================

def discover_mag_schema(
    client,
    table_id,
):
    table = client.get_table(
        table_id
    )

    names = [
        field.name
        for field in table.schema
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
                "B_total",
                "Btot",
                "Bmag",
                "bmag_nt",
                "b_total",
                "magnetic_field_magnitude",
            ],
            "|B|",
            required=False,
        ),
    }


def query_selected_dates(
    client,
    table_id,
    dates,
):
    schema = discover_mag_schema(
        client,
        table_id,
    )

    time_col = schema["time"]

    fields = [
        f"`{time_col}` AS time_utc",
        f"`{schema['bx']}` AS bx_nt",
        f"`{schema['by']}` AS by_nt",
        f"`{schema['bz']}` AS bz_nt",
    ]

    if schema[
        "btotal"
    ] is not None:
        fields.append(
            f"`{schema['btotal']}` "
            "AS btotal_nt"
        )

    query = f"""
    SELECT
        {", ".join(fields)}
    FROM `{table_id}`
    WHERE DATE(`{time_col}`)
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

    print()
    print(
        f"Querying {len(dates)} "
        "mission-spanning MAG dates..."
    )

    mag = (
        client.query(
            query,
            job_config=config,
        )
        .to_dataframe()
    )

    mag["time_utc"] = pd.to_datetime(
        mag["time_utc"],
        utc=True,
        errors="coerce",
    )

    for column in [
        "bx_nt",
        "by_nt",
        "bz_nt",
    ]:
        mag[column] = pd.to_numeric(
            mag[column],
            errors="coerce",
        )

    if "btotal_nt" not in mag.columns:
        mag["btotal_nt"] = np.sqrt(
            mag["bx_nt"] ** 2
            + mag["by_nt"] ** 2
            + mag["bz_nt"] ** 2
        )
    else:
        mag["btotal_nt"] = pd.to_numeric(
            mag["btotal_nt"],
            errors="coerce",
        )

    mag = (
        mag.dropna(
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

    mag["date_utc"] = (
        mag[
            "time_utc"
        ].dt.date
    )

    return (
        mag,
        schema,
    )


# =====================================================================
# Pseudo-gap candidate selection
# =====================================================================

def valid_left_indices(
    day,
    gap_steps,
    context_samples,
    gap_threshold_s,
):
    """
    Return indices that provide a fully continuous context around the
    artificial gap.

    If left=i and right=i+gap_steps, then i+1 ... right-1 are hidden.
    """
    n = len(day)

    minimum_length = (
        gap_steps
        + 2 * context_samples
        + 1
    )

    if n < minimum_length:
        return np.array(
            [],
            dtype=int,
        )

    times_ns = (
        day[
            "time_utc"
        ]
        .astype("int64")
        .to_numpy()
    )

    dt_s = (
        np.diff(
            times_ns
        )
        / 1e9
    )

    bad_interval = (
        ~np.isfinite(dt_s)
        | (dt_s <= 0)
        | (
            dt_s
            > gap_threshold_s
        )
    )

    prefix = np.concatenate(
        [
            np.array(
                [0],
                dtype=np.int64,
            ),
            np.cumsum(
                bad_interval.astype(
                    np.int64
                )
            ),
        ]
    )

    starts = np.arange(
        context_samples,
        n
        - gap_steps
        - context_samples,
        dtype=int,
    )

    if starts.size == 0:
        return starts

    right = (
        starts
        + gap_steps
    )

    context_start = (
        starts
        - context_samples
    )

    context_end = (
        right
        + context_samples
    )

    bad_counts = (
        prefix[
            context_end
        ]
        - prefix[
            context_start
        ]
    )

    return starts[
        bad_counts == 0
    ]


def build_candidate_pool(
    mag,
    gap_duration_s,
    nominal_cadence_s,
    context_seconds,
    gap_threshold_s,
    rng,
    max_candidates_per_day,
):
    gap_steps = int(
        round(
            gap_duration_s
            / nominal_cadence_s
        )
    )

    if gap_steps < 2:
        raise ValueError(
            "Pseudo-gap must hide at least one sample."
        )

    context_samples = max(
        4,
        int(
            round(
                context_seconds
                / nominal_cadence_s
            )
        ),
    )

    candidates = []

    for (
        date_value,
        day,
    ) in mag.groupby(
        "date_utc",
        sort=True,
    ):
        day = (
            day.sort_values(
                "time_utc"
            )
            .reset_index(
                drop=True
            )
        )

        valid = valid_left_indices(
            day=day,
            gap_steps=gap_steps,
            context_samples=context_samples,
            gap_threshold_s=gap_threshold_s,
        )

        if valid.size == 0:
            continue

        n_take = min(
            max_candidates_per_day,
            len(valid),
        )

        chosen = rng.choice(
            valid,
            size=n_take,
            replace=False,
        )

        for left_idx in chosen:
            candidates.append(
                {
                    "date_utc":
                        date_value,

                    "left_idx":
                        int(
                            left_idx
                        ),

                    "gap_steps":
                        int(
                            gap_steps
                        ),

                    "context_samples":
                        int(
                            context_samples
                        ),

                    "target_gap_duration_s":
                        float(
                            gap_duration_s
                        ),
                }
            )

    return candidates


# =====================================================================
# Reconstruction methods
# =====================================================================

def interpolate_linear(
    left_vector,
    right_vector,
    n_hidden,
):
    """
    Linear interpolation between the two visible edge vectors.
    """
    alpha = (
        np.arange(
            1,
            n_hidden + 1,
            dtype=float,
        )
        / (
            n_hidden + 1
        )
    )

    return (
        left_vector[
            None,
            :
        ]
        + alpha[
            :,
            None,
        ]
        * (
            right_vector
            - left_vector
        )[
            None,
            :
        ]
    )


def interpolate_spline(
    day,
    left_idx,
    right_idx,
    context_samples,
):
    """
    Natural cubic spline using visible samples on both sides of the gap.

    The hidden samples are excluded from the fit.
    """
    left_context = day.iloc[
        left_idx
        - context_samples
        : left_idx
        + 1
    ]

    right_context = day.iloc[
        right_idx
        : right_idx
        + context_samples
        + 1
    ]

    visible = pd.concat(
        [
            left_context,
            right_context,
        ],
        ignore_index=True,
    )

    hidden = day.iloc[
        left_idx + 1
        : right_idx
    ]

    if len(hidden) == 0:
        return np.empty(
            (
                0,
                3,
            )
        )

    reference_time = pd.Timestamp(
        day.iloc[
            left_idx
        ][
            "time_utc"
        ]
    )

    x_visible = (
        (
            visible[
                "time_utc"
            ]
            - reference_time
        )
        .dt.total_seconds()
        .to_numpy(
            dtype=float
        )
    )

    x_hidden = (
        (
            hidden[
                "time_utc"
            ]
            - reference_time
        )
        .dt.total_seconds()
        .to_numpy(
            dtype=float
        )
    )

    prediction = np.empty(
        (
            len(hidden),
            3,
        ),
        dtype=float,
    )

    columns = [
        "bx_nt",
        "by_nt",
        "bz_nt",
    ]

    for j, column in enumerate(
        columns
    ):
        y_visible = (
            visible[
                column
            ]
            .to_numpy(
                dtype=float
            )
        )

        spline = CubicSpline(
            x_visible,
            y_visible,
            bc_type="natural",
            extrapolate=False,
        )

        prediction[
            :,
            j,
        ] = spline(
            x_hidden
        )

    return prediction


def _fit_ar_forecast(
    values,
    n_forecast,
    ar_lags,
):
    """
    Fit a local AR(p) model by ordinary least squares and forecast forward.

    The implementation is explicit instead of relying on statsmodels so the
    pseudo-gap experiment is reproducible across Python/statsmodels versions.

    Model:
        z_t = c + phi_1 z_(t-1) + ... + phi_p z_(t-p) + eps_t

    The input series is standardized before fitting to improve numerical
    conditioning. The requested AR order is reduced automatically if the
    local context is too short.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 6:
        raise ValueError(
            "Insufficient data for AR fit."
        )

    max_safe_lag = max(
        1,
        min(
            int(ar_lags),
            (
                len(values)
                - 2
            )
            // 3,
        ),
    )

    mean = float(
        np.mean(values)
    )

    scale = float(
        np.std(values, ddof=0)
    )

    if (
        not np.isfinite(scale)
        or scale < 1e-12
    ):
        return np.full(
            int(n_forecast),
            mean,
            dtype=float,
        )

    z = (
        values
        - mean
    ) / scale

    p = max_safe_lag

    X = []
    y = []

    for t in range(
        p,
        len(z),
    ):
        X.append(
            [1.0]
            + [
                z[t - lag]
                for lag in range(
                    1,
                    p + 1,
                )
            ]
        )

        y.append(
            z[t]
        )

    X = np.asarray(
        X,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if (
        len(y) <= p
        or X.ndim != 2
    ):
        raise ValueError(
            "Insufficient effective observations for AR fit."
        )

    coefficients, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    history = list(
        z.astype(float)
    )

    forecasts_z = []

    for _ in range(
        int(n_forecast)
    ):
        predictors = np.asarray(
            [1.0]
            + [
                history[-lag]
                for lag in range(
                    1,
                    p + 1,
                )
            ],
            dtype=float,
        )

        prediction = float(
            predictors
            @ coefficients
        )

        if not np.isfinite(
            prediction
        ):
            raise FloatingPointError(
                "Non-finite AR forecast."
            )

        forecasts_z.append(
            prediction
        )

        history.append(
            prediction
        )

    return (
        np.asarray(
            forecasts_z,
            dtype=float,
        )
        * scale
        + mean
    )


def interpolate_ar_bidirectional(
    day,
    left_idx,
    right_idx,
    context_samples,
    ar_lags,
):
    """
    Bidirectional local AR reconstruction.

    Forward AR:
        trained on samples before the gap.

    Backward AR:
        trained on future samples in reverse time, so it predicts backward
        from the right edge.

    The two predictions are blended according to position inside the gap.
    """
    n_hidden = (
        right_idx
        - left_idx
        - 1
    )

    if n_hidden <= 0:
        return np.empty(
            (
                0,
                3,
            )
        )

    left_context = day.iloc[
        left_idx
        - context_samples
        : left_idx
        + 1
    ]

    right_context = day.iloc[
        right_idx
        : right_idx
        + context_samples
        + 1
    ]

    prediction = np.empty(
        (
            n_hidden,
            3,
        ),
        dtype=float,
    )

    alpha = (
        np.arange(
            1,
            n_hidden + 1,
            dtype=float,
        )
        / (
            n_hidden + 1
        )
    )

    columns = [
        "bx_nt",
        "by_nt",
        "bz_nt",
    ]

    for j, column in enumerate(
        columns
    ):
        left_values = (
            left_context[
                column
            ]
            .to_numpy(
                dtype=float
            )
        )

        right_values = (
            right_context[
                column
            ]
            .to_numpy(
                dtype=float
            )
        )

        forward = _fit_ar_forecast(
            left_values,
            n_hidden,
            ar_lags,
        )

        backward_from_right = (
            _fit_ar_forecast(
                right_values[
                    ::-1
                ],
                n_hidden,
                ar_lags,
            )
        )

        # Forecast from reversed right context is ordered from the right edge
        # toward earlier times. Reverse it to match chronological hidden times.
        backward = (
            backward_from_right[
                ::-1
            ]
        )

        prediction[
            :,
            j,
        ] = (
            (
                1.0
                - alpha
            )
            * forward
            + alpha
            * backward
        )

    return prediction


# =====================================================================
# Error metrics
# =====================================================================

def angle_errors_deg(
    true_vectors,
    predicted_vectors,
):
    true_norm = np.linalg.norm(
        true_vectors,
        axis=1,
    )

    pred_norm = np.linalg.norm(
        predicted_vectors,
        axis=1,
    )

    denominator = (
        true_norm
        * pred_norm
    )

    result = np.full(
        len(
            true_vectors
        ),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(
            denominator
        )
        & (
            denominator
            > 0
        )
    )

    if valid.any():
        cosine = (
            np.sum(
                true_vectors[
                    valid
                ]
                * predicted_vectors[
                    valid
                ],
                axis=1,
            )
            / denominator[
                valid
            ]
        )

        result[
            valid
        ] = np.degrees(
            np.arccos(
                np.clip(
                    cosine,
                    -1.0,
                    1.0,
                )
            )
        )

    return result


def local_variability_score(
    day,
    left_idx,
    right_idx,
    context_samples,
):
    """
    Dimensionless local variability using only observed context:

        median(|B[i+1]-B[i]|) / median(|B|)
    """
    left_context = day.iloc[
        left_idx
        - context_samples
        : left_idx
        + 1
    ]

    right_context = day.iloc[
        right_idx
        : right_idx
        + context_samples
        + 1
    ]

    increment_chunks = []

    for context in [
        left_context,
        right_context,
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

    observed_vectors = pd.concat(
        [
            left_context[
                [
                    "bx_nt",
                    "by_nt",
                    "bz_nt",
                ]
            ],
            right_context[
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

    scale = np.nanmedian(
        np.linalg.norm(
            observed_vectors,
            axis=1,
        )
    )

    if (
        not np.isfinite(
            scale
        )
        or scale <= 0
    ):
        return np.nan

    return float(
        np.nanmedian(
            increments
        )
        / scale
    )


def reconstruction_metrics(
    true_vectors,
    predicted_vectors,
):
    if (
        predicted_vectors.shape
        != true_vectors.shape
    ):
        raise ValueError(
            "Prediction and truth have different shapes."
        )

    errors = (
        predicted_vectors
        - true_vectors
    )

    vector_error = np.linalg.norm(
        errors,
        axis=1,
    )

    true_bmag = np.linalg.norm(
        true_vectors,
        axis=1,
    )

    pred_bmag = np.linalg.norm(
        predicted_vectors,
        axis=1,
    )

    bmag_error = (
        pred_bmag
        - true_bmag
    )

    field_scale = np.nanmedian(
        true_bmag
    )

    vector_rmse = float(
        np.sqrt(
            np.nanmean(
                vector_error ** 2
            )
        )
    )

    bmag_rmse = float(
        np.sqrt(
            np.nanmean(
                bmag_error ** 2
            )
        )
    )

    angles = angle_errors_deg(
        true_vectors,
        predicted_vectors,
    )

    return {
        "rmse_bx_nt":
            float(
                np.sqrt(
                    np.nanmean(
                        errors[
                            :,
                            0,
                        ]
                        ** 2
                    )
                )
            ),

        "rmse_by_nt":
            float(
                np.sqrt(
                    np.nanmean(
                        errors[
                            :,
                            1,
                        ]
                        ** 2
                    )
                )
            ),

        "rmse_bz_nt":
            float(
                np.sqrt(
                    np.nanmean(
                        errors[
                            :,
                            2,
                        ]
                        ** 2
                    )
                )
            ),

        "vector_rmse_nt":
            vector_rmse,

        "normalized_vector_rmse":
            (
                vector_rmse
                / field_scale
                if (
                    np.isfinite(
                        field_scale
                    )
                    and field_scale > 0
                )
                else np.nan
            ),

        "max_vector_error_nt":
            float(
                np.nanmax(
                    vector_error
                )
            ),

        "bmag_rmse_nt":
            bmag_rmse,

        "normalized_bmag_rmse":
            (
                bmag_rmse
                / field_scale
                if (
                    np.isfinite(
                        field_scale
                    )
                    and field_scale > 0
                )
                else np.nan
            ),

        "bmag_mae_nt":
            float(
                np.nanmean(
                    np.abs(
                        bmag_error
                    )
                )
            ),

        "max_abs_bmag_error_nt":
            float(
                np.nanmax(
                    np.abs(
                        bmag_error
                    )
                )
            ),

        "median_angle_error_deg":
            float(
                np.nanmedian(
                    angles
                )
            ),

        "max_angle_error_deg":
            float(
                np.nanmax(
                    angles
                )
            ),
    }


# =====================================================================
# One paired pseudo-gap experiment
# =====================================================================

def evaluate_candidate(
    day,
    candidate,
    trial_id,
    ar_lags,
):
    left_idx = candidate[
        "left_idx"
    ]

    gap_steps = candidate[
        "gap_steps"
    ]

    context_samples = candidate[
        "context_samples"
    ]

    right_idx = (
        left_idx
        + gap_steps
    )

    hidden = day.iloc[
        left_idx + 1
        : right_idx
    ]

    n_hidden = len(
        hidden
    )

    if n_hidden == 0:
        return []

    true_vectors = hidden[
        [
            "bx_nt",
            "by_nt",
            "bz_nt",
        ]
    ].to_numpy(
        dtype=float
    )

    left_row = day.iloc[
        left_idx
    ]

    right_row = day.iloc[
        right_idx
    ]

    left_vector = np.array(
        [
            left_row[
                "bx_nt"
            ],
            left_row[
                "by_nt"
            ],
            left_row[
                "bz_nt"
            ],
        ],
        dtype=float,
    )

    right_vector = np.array(
        [
            right_row[
                "bx_nt"
            ],
            right_row[
                "by_nt"
            ],
            right_row[
                "bz_nt"
            ],
        ],
        dtype=float,
    )

    time_before = pd.Timestamp(
        left_row[
            "time_utc"
        ]
    )

    time_after = pd.Timestamp(
        right_row[
            "time_utc"
        ]
    )

    actual_edge_dt_s = (
        time_after
        - time_before
    ).total_seconds()

    edge_scale = np.nanmean(
        [
            np.linalg.norm(
                left_vector
            ),
            np.linalg.norm(
                right_vector
            ),
        ]
    )

    edge_jump = np.linalg.norm(
        right_vector
        - left_vector
    )

    edge_relative_jump = (
        float(
            edge_jump
            / edge_scale
        )
        if (
            np.isfinite(
                edge_scale
            )
            and edge_scale > 0
        )
        else np.nan
    )

    variability = local_variability_score(
        day=day,
        left_idx=left_idx,
        right_idx=right_idx,
        context_samples=context_samples,
    )

    common = {
        "trial_id":
            trial_id,

        "date_utc":
            candidate[
                "date_utc"
            ],

        "target_gap_duration_s":
            candidate[
                "target_gap_duration_s"
            ],

        "actual_edge_dt_s":
            float(
                actual_edge_dt_s
            ),

        "hidden_sample_count":
            int(
                n_hidden
            ),

        "time_before":
            time_before,

        "time_after":
            time_after,

        "edge_relative_jump":
            edge_relative_jump,

        "local_variability_score":
            variability,
    }

    predictions = {}
    failures = {}

    predictions[
        "linear"
    ] = interpolate_linear(
        left_vector=left_vector,
        right_vector=right_vector,
        n_hidden=n_hidden,
    )

    try:
        predictions[
            "spline"
        ] = interpolate_spline(
            day=day,
            left_idx=left_idx,
            right_idx=right_idx,
            context_samples=context_samples,
        )
    except Exception as exc:
        predictions[
            "spline"
        ] = None
        failures[
            "spline"
        ] = f"{type(exc).__name__}: {exc}"

    try:
        predictions[
            "ar"
        ] = interpolate_ar_bidirectional(
            day=day,
            left_idx=left_idx,
            right_idx=right_idx,
            context_samples=context_samples,
            ar_lags=ar_lags,
        )
    except Exception as exc:
        predictions[
            "ar"
        ] = None
        failures[
            "ar"
        ] = f"{type(exc).__name__}: {exc}"

    rows = []

    for method in METHOD_ORDER:
        prediction = predictions[
            method
        ]

        row = dict(
            common
        )

        row[
            "method"
        ] = method

        row[
            "failure_reason"
        ] = failures.get(
            method,
            "",
        )

        if (
            prediction is None
            or prediction.shape
            != true_vectors.shape
            or not np.isfinite(
                prediction
            ).all()
        ):
            row[
                "status"
            ] = "failed"

            rows.append(
                row
            )

            continue

        row[
            "status"
        ] = "ok"

        row.update(
            reconstruction_metrics(
                true_vectors,
                prediction,
            )
        )

        rows.append(
            row
        )

    return rows


# =====================================================================
# Full experiment
# =====================================================================

def run_pseudogap_experiment(
    mag,
    gap_durations,
    trials_per_gap,
    context_seconds,
    nominal_cadence_s,
    gap_threshold_s,
    ar_lags,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    grouped_days = {
        date_value:
            day.sort_values(
                "time_utc"
            )
            .reset_index(
                drop=True
            )

        for (
            date_value,
            day,
        ) in mag.groupby(
            "date_utc",
            sort=True,
        )
    }

    rows = []

    trial_counter = 0

    n_days = max(
        1,
        len(
            grouped_days
        ),
    )

    max_candidates_per_day = max(
        10,
        int(
            np.ceil(
                3
                * trials_per_gap
                / n_days
            )
        ),
    )

    for gap_duration in gap_durations:
        print()
        print(
            f"Building paired pseudo-gaps "
            f"for {gap_duration:g} s..."
        )

        candidates = build_candidate_pool(
            mag=mag,
            gap_duration_s=gap_duration,
            nominal_cadence_s=nominal_cadence_s,
            context_seconds=context_seconds,
            gap_threshold_s=gap_threshold_s,
            rng=rng,
            max_candidates_per_day=max_candidates_per_day,
        )

        if not candidates:
            print(
                "  No valid continuous intervals."
            )
            continue

        if len(
            candidates
        ) > trials_per_gap:
            chosen = rng.choice(
                len(
                    candidates
                ),
                size=trials_per_gap,
                replace=False,
            )

            candidates = [
                candidates[
                    index
                ]
                for index in chosen
            ]

        valid_trial_count = 0

        for candidate in candidates:
            trial_counter += 1

            trial_id = (
                f"pg_{trial_counter:06d}"
            )

            day = grouped_days[
                candidate[
                    "date_utc"
                ]
            ]

            trial_rows = evaluate_candidate(
                day=day,
                candidate=candidate,
                trial_id=trial_id,
                ar_lags=ar_lags,
            )

            if trial_rows:
                valid_trial_count += 1
                rows.extend(
                    trial_rows
                )

        print(
            f"  Pseudo-gap locations: "
            f"{valid_trial_count:,}"
        )

        if valid_trial_count:
            method_counts = (
                pd.DataFrame(
                    [
                        row
                        for row in rows
                        if row[
                            "target_gap_duration_s"
                        ]
                        == gap_duration
                    ]
                )
                .groupby(
                    [
                        "method",
                        "status",
                    ]
                )
                .size()
            )

            for (
                method,
                status,
            ), count in method_counts.items():
                print(
                    f"    {method:7s} "
                    f"{status:6s}: "
                    f"{count:,}"
                )

    if not rows:
        raise RuntimeError(
            "No pseudo-gap trials were evaluated."
        )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# Variability regimes
# =====================================================================

def add_variability_regimes(
    trials,
):
    out = trials.copy()

    # Variability is identical for all methods in a paired trial.
    unique_trials = (
        out[
            [
                "trial_id",
                "local_variability_score",
            ]
        ]
        .drop_duplicates(
            subset=[
                "trial_id"
            ]
        )
    )

    finite = pd.to_numeric(
        unique_trials[
            "local_variability_score"
        ],
        errors="coerce",
    )

    finite = finite[
        np.isfinite(
            finite
        )
    ]

    if len(
        finite
    ) < 3:
        out[
            "variability_regime"
        ] = "unclassified"

        thresholds = pd.DataFrame(
            {
                "metric": [
                    "q33_local_variability",
                    "q67_local_variability",
                ],
                "value": [
                    np.nan,
                    np.nan,
                ],
            }
        )

        return (
            out,
            thresholds,
        )

    q33 = float(
        finite.quantile(
            1 / 3
        )
    )

    q67 = float(
        finite.quantile(
            2 / 3
        )
    )

    def classify(
        value,
    ):
        if not np.isfinite(
            value
        ):
            return "unclassified"

        if value <= q33:
            return "quiet"

        if value <= q67:
            return "intermediate"

        return "variable"

    out[
        "variability_regime"
    ] = pd.to_numeric(
        out[
            "local_variability_score"
        ],
        errors="coerce",
    ).apply(
        classify
    )

    thresholds = pd.DataFrame(
        {
            "metric": [
                "q33_local_variability",
                "q67_local_variability",
            ],
            "value": [
                q33,
                q67,
            ],
        }
    )

    return (
        out,
        thresholds,
    )


# =====================================================================
# Summaries
# =====================================================================

def summarize_group(
    group,
):
    valid = group[
        group[
            "status"
        ]
        == "ok"
    ]

    return pd.Series(
        {
            "n_selected":
                len(
                    group
                ),

            "n_valid":
                len(
                    valid
                ),

            "median_normalized_vector_rmse":
                valid[
                    "normalized_vector_rmse"
                ].median(),

            "p90_normalized_vector_rmse":
                safe_quantile(
                    valid[
                        "normalized_vector_rmse"
                    ],
                    0.90,
                ),

            "p95_normalized_vector_rmse":
                safe_quantile(
                    valid[
                        "normalized_vector_rmse"
                    ],
                    0.95,
                ),

            "median_vector_rmse_nt":
                valid[
                    "vector_rmse_nt"
                ].median(),

            "p90_vector_rmse_nt":
                safe_quantile(
                    valid[
                        "vector_rmse_nt"
                    ],
                    0.90,
                ),

            "median_normalized_bmag_rmse":
                valid[
                    "normalized_bmag_rmse"
                ].median(),

            "p90_normalized_bmag_rmse":
                safe_quantile(
                    valid[
                        "normalized_bmag_rmse"
                    ],
                    0.90,
                ),

            "median_angle_error_deg":
                valid[
                    "median_angle_error_deg"
                ].median(),

            "p90_angle_error_deg":
                safe_quantile(
                    valid[
                        "median_angle_error_deg"
                    ],
                    0.90,
                ),

            "median_max_angle_error_deg":
                valid[
                    "max_angle_error_deg"
                ].median(),

            "median_edge_relative_jump":
                valid[
                    "edge_relative_jump"
                ].median(),

            "median_local_variability":
                valid[
                    "local_variability_score"
                ].median(),
        }
    )


def build_summaries(
    trials,
):
    by_gap_method = (
        trials.groupby(
            [
                "target_gap_duration_s",
                "method",
            ],
            as_index=False,
        )
        .apply(
            summarize_group,
            include_groups=False,
        )
        .reset_index(
            drop=True
        )
    )

    by_gap_method_regime = (
        trials.groupby(
            [
                "target_gap_duration_s",
                "method",
                "variability_regime",
            ],
            as_index=False,
        )
        .apply(
            summarize_group,
            include_groups=False,
        )
        .reset_index(
            drop=True
        )
    )

    return (
        by_gap_method,
        by_gap_method_regime,
    )


def build_winner_summary(
    trials,
):
    columns = [
        "target_gap_duration_s",
        "winner_method",
        "wins",
        "n_complete_trials",
        "win_fraction",
    ]

    valid = trials[
        (
            trials[
                "status"
            ]
            == "ok"
        )
        & np.isfinite(
            trials[
                "normalized_vector_rmse"
            ]
        )
    ].copy()

    if valid.empty:
        return pd.DataFrame(
            columns=columns
        )

    counts_per_trial = (
        valid.groupby(
            "trial_id"
        )[
            "method"
        ]
        .nunique()
    )

    complete_ids = counts_per_trial[
        counts_per_trial
        == len(
            METHOD_ORDER
        )
    ].index

    valid = valid[
        valid[
            "trial_id"
        ].isin(
            complete_ids
        )
    ].copy()

    if valid.empty:
        return pd.DataFrame(
            columns=columns
        )

    winner_index = (
        valid.groupby(
            "trial_id"
        )[
            "normalized_vector_rmse"
        ]
        .idxmin()
    )

    winners = valid.loc[
        winner_index,
        [
            "trial_id",
            "target_gap_duration_s",
            "variability_regime",
            "method",
            "normalized_vector_rmse",
        ],
    ].copy()

    winners = winners.rename(
        columns={
            "method":
                "winner_method",

            "normalized_vector_rmse":
                "winner_normalized_vector_rmse",
        }
    )

    total_by_gap = (
        winners.groupby(
            "target_gap_duration_s"
        )
        .size()
        .rename(
            "n_complete_trials"
        )
    )

    summary = (
        winners.groupby(
            [
                "target_gap_duration_s",
                "winner_method",
            ]
        )
        .size()
        .rename(
            "wins"
        )
        .reset_index()
    )

    summary = summary.merge(
        total_by_gap,
        on="target_gap_duration_s",
        how="left",
    )

    summary[
        "win_fraction"
    ] = (
        summary[
            "wins"
        ]
        / summary[
            "n_complete_trials"
        ]
    )

    return summary


# =====================================================================
# Figures
# =====================================================================

def make_metric_line_plot(
    summary,
    metric,
    ylabel,
    title,
    output_path,
):
    if summary.empty:
        return

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    for method in METHOD_ORDER:
        group = summary[
            summary[
                "method"
            ]
            == method
        ].sort_values(
            "target_gap_duration_s"
        )

        if group.empty:
            continue

        ax.plot(
            group[
                "target_gap_duration_s"
            ],
            group[
                metric
            ],
            marker="o",
            label=method,
        )

    ax.set_xscale(
        "log"
    )

    values = pd.to_numeric(
        summary[
            metric
        ],
        errors="coerce",
    )

    finite_positive = values[
        np.isfinite(
            values
        )
        & (
            values
            > 0
        )
    ]

    if (
        len(
            finite_positive
        )
        == values[
            np.isfinite(
                values
            )
        ].shape[0]
        and len(
            finite_positive
        )
        > 0
    ):
        ax.set_yscale(
            "log"
        )

    ax.set_xlabel(
        "Artificial gap duration (s)"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        fig
    )


def make_winner_plot(
    winner_summary,
    output_path,
):
    if winner_summary.empty:
        return

    gaps = sorted(
        winner_summary[
            "target_gap_duration_s"
        ].unique()
    )

    x = np.arange(
        len(
            gaps
        ),
        dtype=float,
    )

    width = (
        0.8
        / len(
            METHOD_ORDER
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    for j, method in enumerate(
        METHOD_ORDER
    ):
        fractions = []

        for gap in gaps:
            row = winner_summary[
                (
                    winner_summary[
                        "target_gap_duration_s"
                    ]
                    == gap
                )
                & (
                    winner_summary[
                        "winner_method"
                    ]
                    == method
                )
            ]

            if row.empty:
                fractions.append(
                    0.0
                )
            else:
                fractions.append(
                    float(
                        row.iloc[
                            0
                        ][
                            "win_fraction"
                        ]
                    )
                )

        offset = (
            j
            - (
                len(
                    METHOD_ORDER
                )
                - 1
            )
            / 2
        ) * width

        ax.bar(
            x
            + offset,
            fractions,
            width=width,
            label=method,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            f"{gap:g}"
            for gap in gaps
        ]
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_xlabel(
        "Artificial gap duration (s)"
    )

    ax.set_ylabel(
        "Fraction of paired trials won"
    )

    ax.set_title(
        "Best reconstruction method by gap duration"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        fig
    )


def make_error_vs_variability(
    trials,
    output_path,
):
    valid = trials[
        (
            trials[
                "status"
            ]
            == "ok"
        )
        & np.isfinite(
            trials[
                "local_variability_score"
            ]
        )
        & np.isfinite(
            trials[
                "normalized_vector_rmse"
            ]
        )
        & (
            trials[
                "local_variability_score"
            ]
            > 0
        )
        & (
            trials[
                "normalized_vector_rmse"
            ]
            > 0
        )
    ]

    if valid.empty:
        return

    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    for method in METHOD_ORDER:
        group = valid[
            valid[
                "method"
            ]
            == method
        ]

        if group.empty:
            continue

        ax.scatter(
            group[
                "local_variability_score"
            ],
            group[
                "normalized_vector_rmse"
            ],
            s=18,
            alpha=0.45,
            label=method,
        )

    ax.set_xscale(
        "log"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Local variability score"
    )

    ax.set_ylabel(
        "Normalized vector RMSE"
    )

    ax.set_title(
        "Interpolation error versus local MAG variability"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        fig
    )


def make_figures(
    trials,
    summary_by_gap_method,
    winner_summary,
):
    make_metric_line_plot(
        summary=summary_by_gap_method,
        metric=(
            "median_normalized_vector_rmse"
        ),
        ylabel=(
            "Median normalized vector RMSE"
        ),
        title=(
            "MAG pseudo-gap reconstruction error"
        ),
        output_path=(
            FIGURE_DIR
            / "normalized_vector_rmse_by_gap_method.png"
        ),
    )

    make_metric_line_plot(
        summary=summary_by_gap_method,
        metric=(
            "median_angle_error_deg"
        ),
        ylabel=(
            "Median angular error (deg)"
        ),
        title=(
            "MAG pseudo-gap directional reconstruction error"
        ),
        output_path=(
            FIGURE_DIR
            / "angle_error_by_gap_method.png"
        ),
    )

    make_winner_plot(
        winner_summary=winner_summary,
        output_path=(
            FIGURE_DIR
            / "winner_fraction_by_gap.png"
        ),
    )

    make_error_vs_variability(
        trials=trials,
        output_path=(
            FIGURE_DIR
            / "error_vs_variability.png"
        ),
    )


# =====================================================================
# Console report
# =====================================================================

def print_report(
    trials,
    summary_by_gap_method,
    winner_summary,
    thresholds,
    selected_dates,
):
    print()
    print(
        "="
        * 82
    )

    print(
        "MAVEN MAG — PSEUDO-GAP INTERPOLATION METHOD COMPARISON"
    )

    print(
        "="
        * 82
    )

    print(
        f"Mission-spanning dates queried: "
        f"{len(selected_dates):,}"
    )

    print(
        f"Paired pseudo-gap locations:     "
        f"{trials['trial_id'].nunique():,}"
    )

    print(
        f"Method-evaluation rows:          "
        f"{len(trials):,}"
    )

    print()
    print(
        "Local variability thresholds:"
    )

    for _, row in thresholds.iterrows():
        print(
            f"  {row['metric']}: "
            f"{row['value']:.6g}"
        )

    gaps = sorted(
        summary_by_gap_method[
            "target_gap_duration_s"
        ].unique()
    )

    for gap in gaps:
        print()
        print(
            f"Gap duration: {gap:g} s"
        )

        subset = summary_by_gap_method[
            summary_by_gap_method[
                "target_gap_duration_s"
            ]
            == gap
        ]

        for method in METHOD_ORDER:
            row = subset[
                subset[
                    "method"
                ]
                == method
            ]

            if row.empty:
                continue

            row = row.iloc[
                0
            ]

            print(
                f"  {method:7s} | "
                f"valid={int(row['n_valid']):3d} | "
                f"median nRMSE="
                f"{row['median_normalized_vector_rmse']:.4f} | "
                f"p90="
                f"{row['p90_normalized_vector_rmse']:.4f} | "
                f"median angle="
                f"{row['median_angle_error_deg']:.2f} deg"
            )

        if winner_summary.empty:
            winners = pd.DataFrame()
        else:
            winners = winner_summary[
                winner_summary[
                    "target_gap_duration_s"
                ]
                == gap
            ]

        if not winners.empty:
            winner = winners.sort_values(
                "win_fraction",
                ascending=False,
            ).iloc[
                0
            ]

            print(
                f"  Most frequent winner: "
                f"{winner['winner_method']} "
                f"({100 * winner['win_fraction']:.1f}% "
                "of complete paired trials)"
            )

    print()
    print(
        "Interpretation:"
    )

    print(
        "  Each method is evaluated on the SAME hidden samples."
    )

    print(
        "  Lower normalized vector RMSE and lower angular error are better."
    )

    print(
        "  Winner fractions show which method most often gives the lowest "
        "vector reconstruction error."
    )

    print(
        "  The final interpolation policy should depend on both gap duration "
        "and local signal variability, not only on a global average."
    )

    print(
        "="
        * 82
    )


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare linear, cubic-spline and bidirectional-AR "
            "reconstruction on artificial MAVEN MAG gaps."
        )
    )

    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE_ID,
    )

    parser.add_argument(
        "--gap-durations",
        nargs="+",
        type=float,
        default=DEFAULT_GAP_DURATIONS_S,
        help=(
            "Artificial edge-to-edge gap durations in seconds. "
            "Default: 2 3 5 10 30"
        ),
    )

    parser.add_argument(
        "--sample-days",
        type=int,
        default=DEFAULT_SAMPLE_DAYS,
    )

    parser.add_argument(
        "--trials-per-gap",
        type=int,
        default=DEFAULT_TRIALS_PER_GAP,
    )

    parser.add_argument(
        "--context-seconds",
        type=float,
        default=DEFAULT_CONTEXT_SECONDS,
    )

    parser.add_argument(
        "--nominal-cadence",
        type=float,
        default=DEFAULT_NOMINAL_CADENCE_S,
    )

    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=DEFAULT_GAP_THRESHOLD_S,
    )

    parser.add_argument(
        "--ar-lags",
        type=int,
        default=DEFAULT_AR_LAGS,
        help=(
            "Requested local AR order. It is automatically reduced "
            "when the available context is too short."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )

    args = parser.parse_args()

    if args.sample_days <= 0:
        parser.error(
            "--sample-days must be positive."
        )

    if args.trials_per_gap <= 0:
        parser.error(
            "--trials-per-gap must be positive."
        )

    if args.context_seconds <= 0:
        parser.error(
            "--context-seconds must be positive."
        )

    if args.nominal_cadence <= 0:
        parser.error(
            "--nominal-cadence must be positive."
        )

    if args.gap_threshold <= 0:
        parser.error(
            "--gap-threshold must be positive."
        )

    if args.ar_lags <= 0:
        parser.error(
            "--ar-lags must be positive."
        )

    if any(
        gap
        <= args.nominal_cadence
        for gap in args.gap_durations
    ):
        parser.error(
            "Every pseudo-gap duration must exceed nominal cadence."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    available_dates = load_available_dates(
        QUALITY_CSV
    )

    selected_dates = select_mission_spanning_dates(
        available_dates,
        args.sample_days,
    )

    print()
    print(
        "Selected mission-spanning dates:"
    )

    for date_value in selected_dates:
        print(
            f"  {date_value}"
        )

    client = bigquery.Client()

    mag, schema = query_selected_dates(
        client=client,
        table_id=args.table,
        dates=selected_dates,
    )

    print(
        f"MAG samples retrieved: "
        f"{len(mag):,}"
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

    trials = run_pseudogap_experiment(
        mag=mag,
        gap_durations=sorted(
            set(
                args.gap_durations
            )
        ),
        trials_per_gap=args.trials_per_gap,
        context_seconds=args.context_seconds,
        nominal_cadence_s=args.nominal_cadence,
        gap_threshold_s=args.gap_threshold,
        ar_lags=args.ar_lags,
        seed=args.seed,
    )

    trials, thresholds = add_variability_regimes(
        trials
    )

    (
        summary_by_gap_method,
        summary_by_gap_method_regime,
    ) = build_summaries(
        trials
    )

    winner_summary = build_winner_summary(
        trials
    )

    trials_path = (
        OUTPUT_DIR
        / "mag_pseudogap_trials.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "mag_pseudogap_summary_by_gap_method.csv"
    )

    regime_path = (
        OUTPUT_DIR
        / "mag_pseudogap_summary_by_gap_method_regime.csv"
    )

    winner_path = (
        OUTPUT_DIR
        / "mag_pseudogap_winner_summary.csv"
    )

    thresholds_path = (
        OUTPUT_DIR
        / "mag_pseudogap_variability_thresholds.csv"
    )

    trials.to_csv(
        trials_path,
        index=False,
    )

    summary_by_gap_method.to_csv(
        summary_path,
        index=False,
    )

    summary_by_gap_method_regime.to_csv(
        regime_path,
        index=False,
    )

    winner_summary.to_csv(
        winner_path,
        index=False,
    )

    thresholds.to_csv(
        thresholds_path,
        index=False,
    )

    make_figures(
        trials=trials,
        summary_by_gap_method=summary_by_gap_method,
        winner_summary=winner_summary,
    )

    print_report(
        trials=trials,
        summary_by_gap_method=summary_by_gap_method,
        winner_summary=winner_summary,
        thresholds=thresholds,
        selected_dates=selected_dates,
    )

    print()
    print(
        f"Trials:             "
        f"{trials_path}"
    )

    print(
        f"Summary:            "
        f"{summary_path}"
    )

    print(
        f"Regime summary:     "
        f"{regime_path}"
    )

    print(
        f"Winner summary:     "
        f"{winner_path}"
    )

    print(
        f"Variability limits: "
        f"{thresholds_path}"
    )

    print(
        f"Figures:            "
        f"{FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()
