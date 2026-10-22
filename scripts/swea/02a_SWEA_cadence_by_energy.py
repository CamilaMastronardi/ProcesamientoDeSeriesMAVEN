#!/usr/bin/env python3
"""
MAVEN SWEA L2 svy3d — cadence analysis by energy.

For each SWEA energy channel, this script builds the sequence of timestamps
for which that energy has at least one valid angular flux value and computes:

    - mean delta-t
    - median delta-t
    - percentage of delta-t values different from the nominal cadence

The nominal cadence is inferred from the global mode of positive timestamp
differences; it is not hard-coded.

Validity criterion
------------------
An energy channel is considered present at a timestamp when at least one
angular bin for that energy has a finite, non-negative differential energy
flux.  No filtering by SWEA quality flag is applied in this first temporal
analysis.

Data layout
-----------
The BigQuery SWEA table stores one row per epoch:

    datetime_utc
    diff_en_fluxes ARRAY<FLOAT64>
    quality
    source_file

The original CDF flux cube has shape:

    (n_elev, n_azim, n_energy)

and was flattened in C order, so energy is the fastest-varying dimension.

Before cadence statistics are computed, the script verifies that the energy
grid is stable across source files.  If the grid changes materially, the
script stops rather than mixing different physical energies by array index.

Outputs
-------
results/SWEA_cadence/
    swea_cadence_by_energy.csv
    swea_energy_grid_summary.csv
    swea_cadence_global_summary.csv
    figures/
        swea_cadence_by_energy.png

Notes
-----
The flux-array scan is much more expensive than a timestamp-only query.
The script performs a BigQuery dry-run first and prints the estimated bytes
processed before executing the analysis.  Use --estimate-only to stop after
that estimate.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import bigquery


# =====================================================================
# Configuration
# =====================================================================

PROJECT_ID = "maven-vignes-crossings"
DATA_TABLE = f"{PROJECT_ID}.maven.swea_svy3d"
AXES_TABLE = f"{PROJECT_ID}.maven.swea_svy3d_axes"

OUTPUT_DIR = Path("results/SWEA_cadence")
FIGURE_DIR = OUTPUT_DIR / "figures"

DEFAULT_CHUNK_DAYS = 30
MIN_VALID_FLUX = 0.0

ENERGY_RTOL = 1e-6
ENERGY_ATOL_EV = 1e-6


# =====================================================================
# Utilities
# =====================================================================

def parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def format_bytes(n_bytes: int) -> str:
    value = float(n_bytes)

    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if value < 1024.0 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{n_bytes:,} B"


def weighted_median_from_counter(counter: Counter) -> float:
    """
    Exact median of integer-valued observations stored as counts.

    Returns the median in the same integer units as the Counter keys.
    """
    total = int(sum(counter.values()))

    if total == 0:
        return np.nan

    left_rank = (total - 1) // 2
    right_rank = total // 2

    cumulative = 0
    left_value = None
    right_value = None

    for value in sorted(counter):
        cumulative += counter[value]

        if left_value is None and cumulative > left_rank:
            left_value = value

        if cumulative > right_rank:
            right_value = value
            break

    return (float(left_value) + float(right_value)) / 2.0


def update_counter(counter: Counter, values: np.ndarray) -> None:
    if len(values) == 0:
        return

    unique, counts = np.unique(values, return_counts=True)

    counter.update(
        {
            int(value): int(count)
            for value, count in zip(unique, counts)
        }
    )


# =====================================================================
# BigQuery metadata
# =====================================================================

def validate_tables(
    client: bigquery.Client,
    data_table: str,
    axes_table: str,
) -> None:
    data = client.get_table(data_table)
    axes = client.get_table(axes_table)

    data_required = {
        "datetime_utc",
        "diff_en_fluxes",
        "quality",
        "source_file",
    }

    axes_required = {
        "source_file",
        "energy_ev",
        "n_elev",
        "n_azim",
        "n_energy",
    }

    data_columns = {field.name for field in data.schema}
    axes_columns = {field.name for field in axes.schema}

    missing_data = sorted(data_required - data_columns)
    missing_axes = sorted(axes_required - axes_columns)

    if missing_data:
        raise RuntimeError(
            f"{data_table} is missing columns: {missing_data}"
        )

    if missing_axes:
        raise RuntimeError(
            f"{axes_table} is missing columns: {missing_axes}"
        )

    if data.time_partitioning is None:
        raise RuntimeError(
            f"{data_table} is expected to be partitioned by datetime_utc."
        )


def get_date_range(
    client: bigquery.Client,
    data_table: str,
) -> tuple[date, date]:
    query = f"""
    SELECT
        MIN(DATE(datetime_utc)) AS first_date,
        MAX(DATE(datetime_utc)) AS last_date
    FROM `{data_table}`
    """

    row = next(iter(client.query(query).result()))

    if row.first_date is None or row.last_date is None:
        raise RuntimeError(
            f"No SWEA rows found in {data_table}."
        )

    return row.first_date, row.last_date


def load_axes(
    client: bigquery.Client,
    axes_table: str,
) -> pd.DataFrame:
    query = f"""
    SELECT
        source_file,
        energy_ev,
        n_elev,
        n_azim,
        n_energy
    FROM `{axes_table}`
    ORDER BY source_file
    """

    return client.query(query).to_dataframe(
        create_bqstorage_client=False
    )


# =====================================================================
# Energy-grid validation
# =====================================================================

def analyze_energy_grid(
    axes: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    if axes.empty:
        raise RuntimeError("The SWEA axes table is empty.")

    if axes["source_file"].duplicated().any():
        duplicated = (
            axes.loc[
                axes["source_file"].duplicated(keep=False),
                "source_file",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise RuntimeError(
            "Duplicate source_file entries found in SWEA axes table. "
            f"Examples: {duplicated[:10]}"
        )

    reference = np.asarray(
        axes.iloc[0]["energy_ev"],
        dtype=float,
    )

    reference_n_energy = len(reference)

    rows = []

    for row in axes.itertuples(index=False):
        energy = np.asarray(
            row.energy_ev,
            dtype=float,
        )

        same_length = len(energy) == reference_n_energy

        if same_length:
            diff = energy - reference

            max_abs_diff = float(
                np.nanmax(
                    np.abs(diff)
                )
            )

            denom = np.maximum(
                np.abs(reference),
                ENERGY_ATOL_EV,
            )

            max_rel_diff = float(
                np.nanmax(
                    np.abs(diff) / denom
                )
            )

            stable = bool(
                np.allclose(
                    energy,
                    reference,
                    rtol=ENERGY_RTOL,
                    atol=ENERGY_ATOL_EV,
                    equal_nan=False,
                )
            )
        else:
            max_abs_diff = np.nan
            max_rel_diff = np.nan
            stable = False

        dimensions_consistent = (
            int(row.n_energy) == reference_n_energy
            and int(row.n_elev) > 0
            and int(row.n_azim) > 0
        )

        rows.append(
            {
                "source_file": row.source_file,
                "n_elev": int(row.n_elev),
                "n_azim": int(row.n_azim),
                "n_energy": int(row.n_energy),
                "energy_array_length": len(energy),
                "max_abs_energy_difference_ev": max_abs_diff,
                "max_relative_energy_difference": max_rel_diff,
                "energy_grid_matches_reference": stable,
                "dimensions_consistent": dimensions_consistent,
            }
        )

    summary = pd.DataFrame(rows)

    grid_ok = bool(
        summary["energy_grid_matches_reference"].all()
        and summary["dimensions_consistent"].all()
    )

    if not grid_ok:
        bad = summary[
            ~(
                summary["energy_grid_matches_reference"]
                & summary["dimensions_consistent"]
            )
        ]

        raise RuntimeError(
            "SWEA energy grid is not stable across all source files. "
            f"{len(bad):,} file(s) differ from the reference grid. "
            "The grid summary CSV was written; inspect it before "
            "combining cadence statistics by energy index."
        )

    return reference, summary


# =====================================================================
# BigQuery flux-validity query
# =====================================================================

def build_validity_query(
    data_table: str,
    axes_table: str,
) -> str:
    """
    Return one row per SWEA epoch with a 64-element Boolean validity array.

    Each flux cube was flattened in C order from
    (elevation, azimuth, energy), so MOD(offset, n_energy) is the
    energy index.
    """
    return f"""
    WITH axes AS (
        SELECT
            source_file,
            ANY_VALUE(n_energy) AS n_energy
        FROM `{axes_table}`
        GROUP BY source_file
    )

    SELECT
        d.datetime_utc,
        d.source_file,
        ARRAY(
            SELECT AS VALUE valid
            FROM (
                SELECT
                    MOD(offset_index, a.n_energy) AS energy_index,
                    LOGICAL_OR(
                        IF(
                            flux IS NULL,
                            FALSE,
                            IF(
                                IS_NAN(flux) OR IS_INF(flux),
                                FALSE,
                                flux >= @min_valid_flux
                            )
                        )
                    ) AS valid
                FROM UNNEST(d.diff_en_fluxes) AS flux
                WITH OFFSET AS offset_index
                GROUP BY energy_index
            )
            ORDER BY energy_index
        ) AS energy_valid
    FROM `{data_table}` AS d
    JOIN axes AS a
        USING (source_file)
    WHERE
        DATE(d.datetime_utc)
        BETWEEN @start_date AND @end_date
    ORDER BY d.datetime_utc
    """


def query_config(
    start_date: date,
    end_date: date,
    min_valid_flux: float,
    dry_run: bool = False,
) -> bigquery.QueryJobConfig:
    return bigquery.QueryJobConfig(
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
                "min_valid_flux",
                "FLOAT64",
                min_valid_flux,
            ),
        ],
        dry_run=dry_run,
        use_query_cache=False if dry_run else True,
    )


def estimate_bytes(
    client: bigquery.Client,
    query: str,
    start_date: date,
    end_date: date,
    min_valid_flux: float,
) -> int:
    job = client.query(
        query,
        job_config=query_config(
            start_date=start_date,
            end_date=end_date,
            min_valid_flux=min_valid_flux,
            dry_run=True,
        ),
    )

    return int(job.total_bytes_processed or 0)


def query_validity_chunk(
    client: bigquery.Client,
    query: str,
    start_date: date,
    end_date: date,
    min_valid_flux: float,
) -> pd.DataFrame:
    job = client.query(
        query,
        job_config=query_config(
            start_date=start_date,
            end_date=end_date,
            min_valid_flux=min_valid_flux,
            dry_run=False,
        ),
    )

    return job.to_dataframe(
        create_bqstorage_client=False
    )


# =====================================================================
# Streaming cadence statistics
# =====================================================================

class CadenceAccumulator:
    def __init__(self, n_energy: int):
        self.n_energy = n_energy

        self.valid_sample_count = np.zeros(
            n_energy,
            dtype=np.int64,
        )

        self.dt_count = np.zeros(
            n_energy,
            dtype=np.int64,
        )

        self.dt_sum_us = np.zeros(
            n_energy,
            dtype=np.float64,
        )

        self.last_valid_time_us = np.full(
            n_energy,
            -1,
            dtype=np.int64,
        )

        self.energy_dt_counter = [
            Counter()
            for _ in range(n_energy)
        ]

        self.global_dt_counter = Counter()
        self.global_dt_count = 0
        self.global_dt_sum_us = 0.0
        self.last_global_time_us = -1

        self.total_records = 0
        self.records_with_invalid_energy = 0

    def _update_global_times(
        self,
        times_us: np.ndarray,
    ) -> None:
        if len(times_us) == 0:
            return

        if self.last_global_time_us >= 0:
            values = np.concatenate(
                [
                    np.array(
                        [self.last_global_time_us],
                        dtype=np.int64,
                    ),
                    times_us,
                ]
            )
        else:
            values = times_us

        dt = np.diff(values)
        dt = dt[dt > 0]

        self.global_dt_count += len(dt)
        self.global_dt_sum_us += float(dt.sum())

        update_counter(
            self.global_dt_counter,
            dt,
        )

        self.last_global_time_us = int(
            times_us[-1]
        )

    def update(
        self,
        frame: pd.DataFrame,
    ) -> None:
        if frame.empty:
            return

        frame = frame.copy()

        frame["datetime_utc"] = pd.to_datetime(
            frame["datetime_utc"],
            utc=True,
            format="mixed",
            errors="raise",
        )

        frame = frame.sort_values(
            "datetime_utc"
        ).reset_index(drop=True)

        times_us = (
            frame["datetime_utc"]
            .astype("int64")
            .to_numpy(dtype=np.int64)
            // 1000
        )

        masks = []

        for value in frame["energy_valid"]:
            arr = np.asarray(
                value,
                dtype=bool,
            )

            if len(arr) != self.n_energy:
                raise ValueError(
                    "Unexpected energy-validity array length: "
                    f"{len(arr)}; expected {self.n_energy}."
                )

            masks.append(arr)

        valid = np.vstack(masks)

        self.total_records += len(frame)

        self.records_with_invalid_energy += int(
            np.sum(
                ~np.all(
                    valid,
                    axis=1,
                )
            )
        )

        self._update_global_times(
            times_us
        )

        for energy_index in range(
            self.n_energy
        ):
            energy_times = times_us[
                valid[:, energy_index]
            ]

            n_valid = len(
                energy_times
            )

            self.valid_sample_count[
                energy_index
            ] += n_valid

            if n_valid == 0:
                continue

            previous = self.last_valid_time_us[
                energy_index
            ]

            if previous >= 0:
                values = np.concatenate(
                    [
                        np.array(
                            [previous],
                            dtype=np.int64,
                        ),
                        energy_times,
                    ]
                )
            else:
                values = energy_times

            dt = np.diff(
                values
            )

            dt = dt[
                dt > 0
            ]

            self.dt_count[
                energy_index
            ] += len(dt)

            self.dt_sum_us[
                energy_index
            ] += float(
                dt.sum()
            )

            update_counter(
                self.energy_dt_counter[
                    energy_index
                ],
                dt,
            )

            self.last_valid_time_us[
                energy_index
            ] = int(
                energy_times[-1]
            )


# =====================================================================
# Results
# =====================================================================

def build_energy_results(
    accumulator: CadenceAccumulator,
    energy_ev: np.ndarray,
) -> tuple[pd.DataFrame, int]:
    if not accumulator.global_dt_counter:
        raise RuntimeError(
            "No positive SWEA timestamp differences were found."
        )

    nominal_dt_us = int(
        accumulator.global_dt_counter.most_common(1)[0][0]
    )

    rows = []

    for index, energy in enumerate(
        energy_ev
    ):
        n_dt = int(
            accumulator.dt_count[index]
        )

        counter = accumulator.energy_dt_counter[
            index
        ]

        if n_dt:
            mean_dt_s = (
                accumulator.dt_sum_us[index]
                / n_dt
                / 1e6
            )

            median_dt_s = (
                weighted_median_from_counter(
                    counter
                )
                / 1e6
            )

            nominal_count = int(
                counter.get(
                    nominal_dt_us,
                    0,
                )
            )

            non_nominal_count = (
                n_dt
                - nominal_count
            )

            pct_non_nominal = (
                100.0
                * non_nominal_count
                / n_dt
            )
        else:
            mean_dt_s = np.nan
            median_dt_s = np.nan
            nominal_count = 0
            non_nominal_count = 0
            pct_non_nominal = np.nan

        valid_fraction_pct = (
            100.0
            * accumulator.valid_sample_count[index]
            / accumulator.total_records
            if accumulator.total_records
            else np.nan
        )

        rows.append(
            {
                "energy_index": index,
                "energy_ev": float(energy),
                "n_valid_samples": int(
                    accumulator.valid_sample_count[index]
                ),
                "valid_fraction_pct": valid_fraction_pct,
                "n_dt": n_dt,
                "dt_mean_s": mean_dt_s,
                "dt_median_s": median_dt_s,
                "dt_nominal_s": nominal_dt_us / 1e6,
                "n_dt_nominal": nominal_count,
                "n_dt_non_nominal": non_nominal_count,
                "pct_dt_non_nominal": pct_non_nominal,
            }
        )

    return pd.DataFrame(rows), nominal_dt_us


def build_global_summary(
    accumulator: CadenceAccumulator,
    nominal_dt_us: int,
    start_date: date,
    end_date: date,
    n_source_files: int,
) -> pd.DataFrame:
    global_median_us = weighted_median_from_counter(
        accumulator.global_dt_counter
    )

    global_nominal_count = int(
        accumulator.global_dt_counter.get(
            nominal_dt_us,
            0,
        )
    )

    global_non_nominal = (
        accumulator.global_dt_count
        - global_nominal_count
    )

    if accumulator.global_dt_count:
        global_mean_s = (
            accumulator.global_dt_sum_us
            / accumulator.global_dt_count
            / 1e6
        )

        global_pct_non_nominal = (
            100.0
            * global_non_nominal
            / accumulator.global_dt_count
        )
    else:
        global_mean_s = np.nan
        global_pct_non_nominal = np.nan

    return pd.DataFrame(
        [
            {
                "start_date": start_date,
                "end_date": end_date,
                "source_files_in_axes_table": n_source_files,
                "total_swea_records": accumulator.total_records,
                "records_with_at_least_one_invalid_energy": (
                    accumulator.records_with_invalid_energy
                ),
                "nominal_dt_s": nominal_dt_us / 1e6,
                "global_dt_mean_s": global_mean_s,
                "global_dt_median_s": global_median_us / 1e6,
                "global_n_dt": accumulator.global_dt_count,
                "global_n_dt_nominal": global_nominal_count,
                "global_n_dt_non_nominal": global_non_nominal,
                "global_pct_dt_non_nominal": global_pct_non_nominal,
            }
        ]
    )


def make_figure(
    results: pd.DataFrame,
    output_path: Path,
) -> None:
    energy = results["energy_ev"].to_numpy()

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 10),
        sharex=True,
    )

    axes[0].plot(
        energy,
        results["dt_mean_s"],
        marker="o",
        markersize=3,
    )

    axes[0].set_ylabel(
        r"Mean $\Delta t$ [s]"
    )

    axes[0].grid(
        True,
        alpha=0.25,
    )

    axes[1].plot(
        energy,
        results["dt_median_s"],
        marker="o",
        markersize=3,
    )

    axes[1].set_ylabel(
        r"Median $\Delta t$ [s]"
    )

    axes[1].grid(
        True,
        alpha=0.25,
    )

    axes[2].plot(
        energy,
        results["pct_dt_non_nominal"],
        marker="o",
        markersize=3,
    )

    axes[2].set_ylabel(
        r"$\Delta t \neq \Delta t_{\rm nominal}$ [%]"
    )

    axes[2].set_xlabel(
        "SWEA energy [eV]"
    )

    axes[2].grid(
        True,
        alpha=0.25,
    )

    for ax in axes:
        ax.set_xscale(
            "log"
        )

    nominal = results[
        "dt_nominal_s"
    ].iloc[0]

    fig.suptitle(
        "MAVEN SWEA svy3d — temporal cadence by energy\n"
        f"Global nominal cadence = {nominal:g} s"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze MAVEN SWEA svy3d temporal cadence for each "
            "energy channel."
        )
    )

    parser.add_argument(
        "--data-table",
        default=DATA_TABLE,
    )

    parser.add_argument(
        "--axes-table",
        default=AXES_TABLE,
    )

    parser.add_argument(
        "--start",
        type=parse_date,
        default=None,
        help="First UTC date, YYYY-MM-DD.",
    )

    parser.add_argument(
        "--end",
        type=parse_date,
        default=None,
        help="Last UTC date, YYYY-MM-DD.",
    )

    parser.add_argument(
        "--chunk-days",
        type=int,
        default=DEFAULT_CHUNK_DAYS,
    )

    parser.add_argument(
        "--min-valid-flux",
        type=float,
        default=MIN_VALID_FLUX,
        help=(
            "Minimum flux accepted as physically valid. "
            "Default: 0."
        ),
    )

    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help=(
            "Estimate BigQuery bytes processed and exit "
            "without running the flux scan."
        ),
    )

    args = parser.parse_args()

    if args.chunk_days <= 0:
        parser.error(
            "--chunk-days must be positive."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = bigquery.Client(
        project=PROJECT_ID
    )

    validate_tables(
        client,
        args.data_table,
        args.axes_table,
    )

    available_start, available_end = get_date_range(
        client,
        args.data_table,
    )

    start_date = (
        args.start
        if args.start is not None
        else available_start
    )

    end_date = (
        args.end
        if args.end is not None
        else available_end
    )

    if start_date < available_start:
        start_date = available_start

    if end_date > available_end:
        end_date = available_end

    if end_date < start_date:
        parser.error(
            "Selected end date is earlier than start date."
        )

    print()
    print("=" * 78)
    print("MAVEN SWEA — CADENCE BY ENERGY")
    print("=" * 78)
    print(
        f"Available data range: {available_start} -> {available_end}"
    )
    print(
        f"Selected range:       {start_date} -> {end_date}"
    )
    print(
        f"Chunk size:           {args.chunk_days} days"
    )
    print(
        "Validity criterion:   at least one finite, "
        f"non-negative angular flux (>= {args.min_valid_flux:g})"
    )

    print()
    print("Loading and validating SWEA energy grids...")

    axes = load_axes(
        client,
        args.axes_table,
    )

    # Write the grid summary even if the stability check fails.
    try:
        energy_ev, grid_summary = analyze_energy_grid(
            axes
        )
    except RuntimeError:
        # Re-run the non-raising part in a compact way so there is still
        # a diagnostic file if a grid mismatch is encountered.
        reference = np.asarray(
            axes.iloc[0]["energy_ev"],
            dtype=float,
        )

        rows = []

        for row in axes.itertuples(index=False):
            energy = np.asarray(
                row.energy_ev,
                dtype=float,
            )

            same_length = len(energy) == len(reference)

            if same_length:
                max_abs = float(
                    np.nanmax(
                        np.abs(
                            energy - reference
                        )
                    )
                )

                matches = bool(
                    np.allclose(
                        energy,
                        reference,
                        rtol=ENERGY_RTOL,
                        atol=ENERGY_ATOL_EV,
                        equal_nan=False,
                    )
                )
            else:
                max_abs = np.nan
                matches = False

            rows.append(
                {
                    "source_file": row.source_file,
                    "n_elev": int(row.n_elev),
                    "n_azim": int(row.n_azim),
                    "n_energy": int(row.n_energy),
                    "energy_array_length": len(energy),
                    "max_abs_energy_difference_ev": max_abs,
                    "energy_grid_matches_reference": matches,
                }
            )

        diagnostic = pd.DataFrame(
            rows
        )

        diagnostic.to_csv(
            OUTPUT_DIR
            / "swea_energy_grid_summary.csv",
            index=False,
        )

        raise

    grid_summary.to_csv(
        OUTPUT_DIR
        / "swea_energy_grid_summary.csv",
        index=False,
    )

    print(
        f"Source files in axes table: {len(axes):,}"
    )
    print(
        f"Energy channels:            {len(energy_ev):,}"
    )
    print(
        "Energy grid stable:         YES"
    )
    print(
        f"Energy range:               "
        f"{np.nanmin(energy_ev):.6g} -> "
        f"{np.nanmax(energy_ev):.6g} eV"
    )

    validity_query = build_validity_query(
        args.data_table,
        args.axes_table,
    )

    print()
    print("Estimating BigQuery scan size...")

    estimated_bytes = estimate_bytes(
        client=client,
        query=validity_query,
        start_date=start_date,
        end_date=end_date,
        min_valid_flux=args.min_valid_flux,
    )

    print(
        f"Estimated bytes processed:  "
        f"{format_bytes(estimated_bytes)}"
    )

    if args.estimate_only:
        print()
        print(
            "Estimate-only mode: no flux data were processed."
        )
        return

    accumulator = CadenceAccumulator(
        n_energy=len(
            energy_ev
        )
    )

    current = start_date
    chunk_number = 0

    total_days = (
        end_date
        - start_date
    ).days + 1

    total_chunks = int(
        np.ceil(
            total_days
            / args.chunk_days
        )
    )

    print()
    print("Processing SWEA flux validity by date chunk...")

    while current <= end_date:
        chunk_number += 1

        chunk_end = min(
            current
            + timedelta(
                days=args.chunk_days - 1
            ),
            end_date,
        )

        print(
            f"  Chunk {chunk_number}/{total_chunks}: "
            f"{current} -> {chunk_end}"
        )

        frame = query_validity_chunk(
            client=client,
            query=validity_query,
            start_date=current,
            end_date=chunk_end,
            min_valid_flux=args.min_valid_flux,
        )

        print(
            f"    SWEA records: {len(frame):,}"
        )

        accumulator.update(
            frame
        )

        current = (
            chunk_end
            + timedelta(days=1)
        )

    results, nominal_dt_us = build_energy_results(
        accumulator=accumulator,
        energy_ev=energy_ev,
    )

    global_summary = build_global_summary(
        accumulator=accumulator,
        nominal_dt_us=nominal_dt_us,
        start_date=start_date,
        end_date=end_date,
        n_source_files=len(axes),
    )

    results_path = (
        OUTPUT_DIR
        / "swea_cadence_by_energy.csv"
    )

    global_path = (
        OUTPUT_DIR
        / "swea_cadence_global_summary.csv"
    )

    figure_path = (
        FIGURE_DIR
        / "swea_cadence_by_energy.png"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    global_summary.to_csv(
        global_path,
        index=False,
    )

    make_figure(
        results,
        figure_path,
    )

    nominal_s = (
        nominal_dt_us
        / 1e6
    )

    print()
    print("=" * 78)
    print("MAVEN SWEA — CADENCE BY ENERGY SUMMARY")
    print("=" * 78)
    print(
        f"Records processed:          {accumulator.total_records:,}"
    )
    print(
        f"Energy channels:            {len(energy_ev):,}"
    )
    print(
        f"Nominal global delta-t:     {nominal_s:g} s"
    )
    print(
        "Mean delta-t range:        "
        f"{results['dt_mean_s'].min():.6g} -> "
        f"{results['dt_mean_s'].max():.6g} s"
    )
    print(
        "Median delta-t range:      "
        f"{results['dt_median_s'].min():.6g} -> "
        f"{results['dt_median_s'].max():.6g} s"
    )
    print(
        "% non-nominal range:       "
        f"{results['pct_dt_non_nominal'].min():.6g} -> "
        f"{results['pct_dt_non_nominal'].max():.6g} %"
    )
    print("=" * 78)
    print()
    print(f"Energy results: {results_path}")
    print(f"Grid summary:   {OUTPUT_DIR / 'swea_energy_grid_summary.csv'}")
    print(f"Global summary: {global_path}")
    print(f"Figure:         {figure_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
