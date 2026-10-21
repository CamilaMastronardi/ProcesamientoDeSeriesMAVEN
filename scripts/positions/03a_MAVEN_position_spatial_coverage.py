#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03a — MAVEN position spatial coverage from the existing BigQuery table.

Updated version:
    - the main spatial plot uses signed rho = sign(z)*sqrt(y^2+z^2),
      so negative z is represented explicitly;
    - an additional figure separates the spatial coverage by year.


Source (independent of treated MAG Parquet):
    maven-vignes-crossings.maven.positions
    datetime_utc, x_ss_km, y_ss_km, z_ss_km

The MAG magnetic-field table does NOT contain spacecraft coordinates.

To avoid downloading tens of millions of rows, BigQuery computes:
  1) per-day position/altitude summaries,
  2) x-rho spatial bin counts,
  3) altitude bin counts,
  4) frontal Y-Z projection bin counts, including per-year splits.
Only the small aggregated results are transferred to Python.

Outputs:
    results/MAVEN_positions/position_summary.csv
    results/MAVEN_positions/position_by_day.csv
    results/MAVEN_positions/position_by_year.csv
    results/MAVEN_positions/position_x_signed_rho_counts.csv
    results/MAVEN_positions/position_x_signed_rho_counts_by_year.csv
    results/MAVEN_positions/position_altitude_histogram.csv
    results/MAVEN_positions/position_y_z_counts_by_year.csv
    results/MAVEN_positions/figures/*.png

Usage:
    python3 scripts/positions/03a_MAVEN_position_spatial_coverage.py \
        --limit-files 10 --estimate-only

    python3 scripts/positions/03a_MAVEN_position_spatial_coverage.py \
        --limit-files 10

    python3 scripts/positions/03a_MAVEN_position_spatial_coverage.py \
        --estimate-only

    python3 scripts/positions/03a_MAVEN_position_spatial_coverage.py

Note: --limit-files is retained as an alias for --limit-days from the
previous (local-Parquet) version. This script reads the position table,
NOT files, and never modifies it or the MAG processed dataset.

Coordinate labels are kept in the stored SS reference frame. Before
comparing with boundary surfaces, verify the needed frame/aberration.

Spatial count maps are SAMPLE COUNTS, not volume-normalized densities.

The frontal Y-Z maps show the actual y_ss and z_ss coordinates,
viewed along the X axis, with no sign folding. They are projections
of all X positions, not slices through X = 0. Run --front-only to obtain
these maps without rerunning the previous position analyses.

The main spatial map uses the signed cylindrical coordinate

    rho_signed = sign(z) * sqrt(y^2 + z^2)

so the upper half corresponds to z > 0 and the lower half to z < 0.


Vignes model overlays
---------------------
On the x-rho coverage plot, this script overlays the classic conic-section
models from Vignes et al. (2000) for the Martian bow shock and magnetic
pile-up boundary (MPB), using the parameters also summarized by
Edberg et al. (2008):

    Bow shock:  x0 = 0.64 R_M, epsilon = 1.03, L = 2.04 R_M
    MPB:        x0 = 0.78 R_M, epsilon = 0.90, L = 0.96 R_M

with conic polar form

    r = L / (1 + epsilon cos(theta))

about the focus located at (x0, 0), and cylindrical plotting coordinates

    x   = x0 + r cos(theta)
    rho = r sin(theta)

Only the rho >= 0 branch is drawn.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import bigquery


DEFAULT_TABLE = "maven-vignes-crossings.maven.positions"
DEFAULT_OUTPUT_DIR = Path("results/MAVEN_positions")
DEFAULT_MARS_RADIUS_KM = 3389.5

X_EDGES_RM = np.linspace(-6.0, 6.0, 241)
RHO_SIGNED_EDGES_RM = np.linspace(-6.0, 6.0, 241)
ALT_EDGES_KM = np.linspace(-250.0, 20000.0, 406)

# Frontal projection as seen from +X: horizontal Y, vertical Z.
# Plot both axes with the same limits and bin widths.
Y_EDGES_RM = np.linspace(-5.0, 5.0, 201)
Z_EDGES_RM = np.linspace(-5.0, 5.0, 201)

# Vignes et al. (2000) conic fits in R_M units.
VIGNES_BOW_SHOCK = {
    "label": "Vignes BS",
    "x0_rm": 0.64,
    "epsilon": 1.03,
    "L_rm": 2.04,
}

VIGNES_MPB = {
    "label": "Vignes MPB",
    "x0_rm": 0.78,
    "epsilon": 0.90,
    "L_rm": 0.96,
}

REQUIRED_COLUMNS = {
    "datetime_utc", "x_ss_km", "y_ss_km", "z_ss_km"
}


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected date YYYY-MM-DD") from exc


def human_bytes(value: int) -> str:
    n = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if n < 1024 or unit == "PiB":
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{value} B"


def partitions_and_validate(client, table_id: str) -> list[date]:
    table = client.get_table(table_id)
    names = {field.name for field in table.schema}
    missing = REQUIRED_COLUMNS - names
    if missing:
        raise RuntimeError(
            f"BigQuery table {table_id} is missing {sorted(missing)}"
        )

    partitioning = table.time_partitioning
    if partitioning is None or partitioning.field != "datetime_utc":
        raise RuntimeError(
            "Expected positions to be partitioned on datetime_utc. "
            "Partition pruning is required to control query costs."
        )

    components = table_id.split(".")
    if len(components) != 3:
        raise ValueError("Use PROJECT.DATASET.TABLE for --table")
    project, dataset, tablename = components

    sql = f"""
        SELECT
            partition_id,
            SUM(total_rows) AS total_rows
        FROM `{project}.{dataset}.INFORMATION_SCHEMA.PARTITIONS`
        WHERE table_name = @table_name
          AND REGEXP_CONTAINS(partition_id, r'^[0-9]{{8}}$')
        GROUP BY partition_id
        ORDER BY partition_id
    """
    params = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("table_name", "STRING", tablename)
    ])

    result = client.query(sql, job_config=params).result()
    dates = [
        date(int(r.partition_id[:4]), int(r.partition_id[4:6]),
             int(r.partition_id[6:8]))
        for r in result if int(r.total_rows or 0) > 0
    ]
    if not dates:
        raise RuntimeError(
            "No non-empty daily partitions in the positions table."
        )
    return sorted(set(dates))


def valid_xyz_sql() -> str:
    parts = []
    for col in ("x_ss_km", "y_ss_km", "z_ss_km"):
        parts.extend([
            f"{col} IS NOT NULL",
            f"NOT IS_NAN({col})",
            f"NOT IS_INF({col})",
        ])
    return " AND ".join(parts)


def sql_daily(table_id: str) -> str:
    valid = valid_xyz_sql()
    return f"""
    WITH coords AS (
        SELECT
            DATE(datetime_utc) AS date_utc,
            IF(
                {valid},
                SQRT(
                    POW(x_ss_km, 2) +
                    POW(y_ss_km, 2) +
                    POW(z_ss_km, 2)
                ),
                NULL
            ) AS r_km
        FROM `{table_id}`
        WHERE DATE(datetime_utc) BETWEEN @start_date AND @end_date
    )
    SELECT
        date_utc,
        COUNT(*) AS total_rows,
        COUNT(r_km) AS valid_positions,
        COUNT(*) - COUNT(r_km) AS invalid_positions,
        MIN(r_km) AS min_r_km,
        MAX(r_km) AS max_r_km,
        MIN(r_km) - @radius_km AS min_altitude_km,
        MAX(r_km) - @radius_km AS max_altitude_km,
        APPROX_QUANTILES(r_km, 100)[OFFSET(50)]
            - @radius_km AS approx_median_altitude_km,
        COUNTIF(r_km - @radius_km < 0) AS negative_altitude_count,
        COUNTIF(r_km - @radius_km < 500)
            AS samples_altitude_below_500_km,
        COUNTIF(r_km - @radius_km < 1000)
            AS samples_altitude_below_1000_km
    FROM coords
    GROUP BY date_utc
    ORDER BY date_utc
    """


def sql_spatial(table_id: str) -> str:
    valid = valid_xyz_sql()
    return f"""
    WITH points AS (
        SELECT
            x_ss_km / @radius_km AS x_rm,
            SIGN(z_ss_km) * SQRT(
                POW(y_ss_km, 2) + POW(z_ss_km, 2)
            ) / @radius_km AS rho_signed_rm
        FROM `{table_id}`
        WHERE DATE(datetime_utc) BETWEEN @start_date AND @end_date
          AND {valid}
    ),
    binned AS (
        SELECT
            LEAST(
                @nx - 1,
                CAST(FLOOR((x_rm - @xmin) / @dx) AS INT64)
            ) AS ix,
            LEAST(
                @nrho - 1,
                CAST(FLOOR((rho_signed_rm - @rhomin) / @drho) AS INT64)
            ) AS irho
        FROM points
        WHERE x_rm BETWEEN @xmin AND @xmax
          AND rho_signed_rm BETWEEN @rhomin AND @rhomax
    )
    SELECT ix, irho, COUNT(*) AS sample_count
    FROM binned
    GROUP BY ix, irho
    ORDER BY ix, irho
    """

def sql_spatial_by_year(table_id: str) -> str:
    valid = valid_xyz_sql()
    return f"""
    WITH points AS (
        SELECT
            EXTRACT(YEAR FROM DATE(datetime_utc)) AS year,
            x_ss_km / @radius_km AS x_rm,
            SIGN(z_ss_km) * SQRT(
                POW(y_ss_km, 2) + POW(z_ss_km, 2)
            ) / @radius_km AS rho_signed_rm
        FROM `{table_id}`
        WHERE DATE(datetime_utc) BETWEEN @start_date AND @end_date
          AND {valid}
    ),
    binned AS (
        SELECT
            year,
            LEAST(
                @nx - 1,
                CAST(FLOOR((x_rm - @xmin) / @dx) AS INT64)
            ) AS ix,
            LEAST(
                @nrho - 1,
                CAST(FLOOR((rho_signed_rm - @rhomin) / @drho) AS INT64)
            ) AS irho
        FROM points
        WHERE x_rm BETWEEN @xmin AND @xmax
          AND rho_signed_rm BETWEEN @rhomin AND @rhomax
    )
    SELECT year, ix, irho, COUNT(*) AS sample_count
    FROM binned
    GROUP BY year, ix, irho
    ORDER BY year, ix, irho
    """

def sql_frontal_by_year(table_id: str) -> str:
    """Binned Y-Z projected sampling by UTC year (aggregated in BigQuery)."""
    valid = valid_xyz_sql()
    return f"""
    WITH points AS (
        SELECT
            EXTRACT(YEAR FROM datetime_utc) AS year,
            y_ss_km / @radius_km AS y_rm,
            z_ss_km / @radius_km AS z_rm
        FROM `{table_id}`
        WHERE DATE(datetime_utc) BETWEEN @start_date AND @end_date
          AND {valid}
    ),
    binned AS (
        SELECT
            year,
            LEAST(@ny - 1,
                  CAST(FLOOR((y_rm - @ymin) / @dy) AS INT64)) AS iy,
            LEAST(@nz - 1,
                  CAST(FLOOR((z_rm - @zmin) / @dz) AS INT64)) AS iz
        FROM points
        WHERE y_rm BETWEEN @ymin AND @ymax
          AND z_rm BETWEEN @zmin AND @zmax
    )
    SELECT year, iy, iz, COUNT(*) AS sample_count
    FROM binned
    GROUP BY year, iy, iz
    ORDER BY year, iy, iz
    """


def sql_altitude(table_id: str) -> str:
    valid = valid_xyz_sql()
    return f"""
    WITH positions AS (
        SELECT
            SQRT(
                POW(x_ss_km, 2) +
                POW(y_ss_km, 2) +
                POW(z_ss_km, 2)
            ) - @radius_km AS altitude_km
        FROM `{table_id}`
        WHERE DATE(datetime_utc) BETWEEN @start_date AND @end_date
          AND {valid}
    ),
    binned AS (
        SELECT
            LEAST(
                @nalt - 1,
                CAST(FLOOR(
                    (altitude_km - @altmin) / @dalt
                ) AS INT64)
            ) AS ialt
        FROM positions
        WHERE altitude_km BETWEEN @altmin AND @altmax
    )
    SELECT ialt, COUNT(*) AS sample_count
    FROM binned
    GROUP BY ialt
    ORDER BY ialt
    """


def query_config(
    start_date: date,
    end_date: date,
    radius_km: float,
    query_kind: str,
    dry_run: bool,
) -> bigquery.QueryJobConfig:
    params = [
        bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
        bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
        bigquery.ScalarQueryParameter("radius_km", "FLOAT64", radius_km),
    ]
    if query_kind in ("spatial", "spatial_by_year"):
        params += [
            bigquery.ScalarQueryParameter("xmin", "FLOAT64", float(X_EDGES_RM[0])),
            bigquery.ScalarQueryParameter("xmax", "FLOAT64", float(X_EDGES_RM[-1])),
            bigquery.ScalarQueryParameter("dx", "FLOAT64", float(np.diff(X_EDGES_RM)[0])),
            bigquery.ScalarQueryParameter("nx", "INT64", len(X_EDGES_RM) - 1),
            bigquery.ScalarQueryParameter("rhomin", "FLOAT64", float(RHO_SIGNED_EDGES_RM[0])),
            bigquery.ScalarQueryParameter("rhomax", "FLOAT64", float(RHO_SIGNED_EDGES_RM[-1])),
            bigquery.ScalarQueryParameter("drho", "FLOAT64", float(np.diff(RHO_SIGNED_EDGES_RM)[0])),
            bigquery.ScalarQueryParameter("nrho", "INT64", len(RHO_SIGNED_EDGES_RM) - 1),
        ]
    if query_kind == "frontal_by_year":
        params += [
            bigquery.ScalarQueryParameter("ymin", "FLOAT64", float(Y_EDGES_RM[0])),
            bigquery.ScalarQueryParameter("ymax", "FLOAT64", float(Y_EDGES_RM[-1])),
            bigquery.ScalarQueryParameter("dy", "FLOAT64", float(np.diff(Y_EDGES_RM)[0])),
            bigquery.ScalarQueryParameter("ny", "INT64", len(Y_EDGES_RM) - 1),
            bigquery.ScalarQueryParameter("zmin", "FLOAT64", float(Z_EDGES_RM[0])),
            bigquery.ScalarQueryParameter("zmax", "FLOAT64", float(Z_EDGES_RM[-1])),
            bigquery.ScalarQueryParameter("dz", "FLOAT64", float(np.diff(Z_EDGES_RM)[0])),
            bigquery.ScalarQueryParameter("nz", "INT64", len(Z_EDGES_RM) - 1),
        ]
    if query_kind == "altitude":
        params += [
            bigquery.ScalarQueryParameter("altmin", "FLOAT64", float(ALT_EDGES_KM[0])),
            bigquery.ScalarQueryParameter("altmax", "FLOAT64", float(ALT_EDGES_KM[-1])),
            bigquery.ScalarQueryParameter("dalt", "FLOAT64", float(np.diff(ALT_EDGES_KM)[0])),
            bigquery.ScalarQueryParameter("nalt", "INT64", len(ALT_EDGES_KM) - 1),
        ]
    return bigquery.QueryJobConfig(
        query_parameters=params,
        dry_run=dry_run,
        use_query_cache=not dry_run,
    )


def run_query(client, sql, kind, start, end, radius, estimate_only):
    cfg = query_config(start, end, radius, kind, dry_run=estimate_only)
    job = client.query(sql, job_config=cfg)
    if estimate_only:
        return int(job.total_bytes_processed or 0)
    return job.to_dataframe(create_bqstorage_client=False)


def vignes_curve_xy(model: dict, x_min_rm: float, x_max_rm: float,
                    n_theta: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the rho >= 0 branch of a Vignes conic section in x-rho coordinates.

    The boundary is parameterized by polar angle theta around the focus at
    (x0, 0), with conic equation r = L / (1 + epsilon cos(theta)).
    """
    epsilon = float(model["epsilon"])
    x0 = float(model["x0_rm"])
    L = float(model["L_rm"])

    # Keep only the angular sector where the conic radius is positive.
    if epsilon < 1:
        theta_max = np.pi
    else:
        theta_max = np.arccos(-1.0 / epsilon) - 1e-6

    theta = np.linspace(0.0, theta_max, n_theta)
    denom = 1.0 + epsilon * np.cos(theta)
    mask = denom > 0

    theta = theta[mask]
    denom = denom[mask]

    r = L / denom
    x = x0 + r * np.cos(theta)
    rho = r * np.sin(theta)

    finite = np.isfinite(x) & np.isfinite(rho) & (rho >= 0)
    x = x[finite]
    rho = rho[finite]

    inside = (x >= x_min_rm) & (x <= x_max_rm)
    return x[inside], rho[inside]




def make_figures(output_dir, day_df, year_df, spatial_counts,
                 spatial_year_mats, altitude_counts, radius_km):
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Total spatial coverage in x vs signed-rho
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7.2))
    mesh = ax.pcolormesh(
        X_EDGES_RM, RHO_SIGNED_EDGES_RM,
        np.log10(spatial_counts.T + 1),
        shading="auto",
    )
    fig.colorbar(mesh, ax=ax, label=r"$\log_{10}(N_\mathrm{samples}+1)$")

    xx = np.linspace(-1, 1, 300)
    mars_upper = np.sqrt(np.maximum(0, 1 - xx**2))
    ax.plot(xx, mars_upper, "--", linewidth=1, label="Mars surface")
    ax.plot(xx, -mars_upper, "--", linewidth=1)

    for model, style in [
        (VIGNES_BOW_SHOCK, {"linestyle": "-", "linewidth": 1.8}),
        (VIGNES_MPB, {"linestyle": "--", "linewidth": 1.8}),
    ]:
        x_model, rho_model = vignes_curve_xy(
            model=model,
            x_min_rm=float(X_EDGES_RM[0]),
            x_max_rm=float(X_EDGES_RM[-1]),
        )
        ax.plot(
            x_model,
            rho_model,
            label=(
                f"{model['label']} "
                f"(x0={model['x0_rm']}, "
                f"ε={model['epsilon']}, "
                f"L={model['L_rm']})"
            ),
            **style,
        )
        ax.plot(x_model, -rho_model, **style)

    ax.axhline(0.0, linewidth=0.8, alpha=0.4)
    ax.set_xlabel(r"$x_\mathrm{SS}/R_M$")
    ax.set_ylabel(r"$\mathrm{sign}(z_\mathrm{SS})\,\rho_\mathrm{SS}/R_M$")
    ax.set_title(
        "MAVEN — spatial sampling in x vs signed-rho with Vignes BS/MPB models"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "position_x_signed_rho_coverage.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Spatial coverage by year: dedicated colorbar column (no overlap)
    # -----------------------------------------------------------------
    years = sorted(spatial_year_mats)
    if years:
        ncols = 3
        nrows = int(np.ceil(len(years) / ncols))

        # Explicit GridSpec layout instead of tight_layout + fig.colorbar(ax=...)
        # because the old combination placed the colorbar ON TOP of the
        # subplots and squeezed year titles into neighbouring axes.
        fig = plt.figure(figsize=(14.0, 4.15 * nrows))
        gs = fig.add_gridspec(
            nrows=nrows,
            ncols=ncols + 1,
            width_ratios=[1.0, 1.0, 1.0, 0.065],
            left=0.075,
            right=0.91,
            bottom=0.055,
            top=0.945,
            wspace=0.26,
            hspace=0.35,
        )

        max_count = max(int(mat.max()) for mat in spatial_year_mats.values())
        vmax = max(1.0, float(np.log10(max_count + 1)))
        first_ax = None
        mesh = None

        for i, year in enumerate(years):
            irow, icol = divmod(i, ncols)
            if first_ax is None:
                ax = fig.add_subplot(gs[irow, icol])
                first_ax = ax
            else:
                ax = fig.add_subplot(gs[irow, icol],
                                     sharex=first_ax, sharey=first_ax)

            counts = spatial_year_mats[year]
            mesh = ax.pcolormesh(
                X_EDGES_RM,
                RHO_SIGNED_EDGES_RM,
                np.log10(counts.T + 1),
                shading="auto",
                vmin=0.0,
                vmax=vmax,
            )

            ax.plot(xx, mars_upper, "--", linewidth=0.8)
            ax.plot(xx, -mars_upper, "--", linewidth=0.8)

            for model, style in [
                (VIGNES_BOW_SHOCK, {"linestyle": "-", "linewidth": 1.0}),
                (VIGNES_MPB, {"linestyle": "--", "linewidth": 1.0}),
            ]:
                x_model, rho_model = vignes_curve_xy(
                    model=model,
                    x_min_rm=float(X_EDGES_RM[0]),
                    x_max_rm=float(X_EDGES_RM[-1]),
                )
                ax.plot(x_model, rho_model, **style)
                ax.plot(x_model, -rho_model, **style)

            ax.axhline(0.0, linewidth=0.6, alpha=0.35)
            ax.set_title(str(year), pad=9)
            ax.set_xlim(float(X_EDGES_RM[0]), float(X_EDGES_RM[-1]))
            ax.set_ylim(float(RHO_SIGNED_EDGES_RM[0]),
                        float(RHO_SIGNED_EDGES_RM[-1]))
            ax.set_aspect("equal", adjustable="box")

            # Keep labels only where they are useful to avoid collisions.
            if irow == nrows - 1 or (i + ncols >= len(years)):
                ax.set_xlabel(r"$x_\mathrm{SS}/R_M$")
            elif irow < nrows - 1:
                ax.tick_params(axis="x", labelbottom=False)
            if icol == 0:
                ax.set_ylabel(
                    r"$\mathrm{sign}(z_\mathrm{SS})\,\rho_\mathrm{SS}/R_M$"
                )
            else:
                ax.tick_params(axis="y", labelleft=False)

        # The fourth GridSpec column is reserved solely for the colorbar.
        cax = fig.add_subplot(gs[:, -1])
        fig.colorbar(
            mesh,
            cax=cax,
            label=r"$\log_{10}(N_\mathrm{samples}+1)$",
        )
        fig.suptitle(
            "MAVEN — spatial coverage by year in x vs signed-rho",
            y=0.985,
        )
        # Do not call tight_layout(): GridSpec already reserves spacing.
        fig.savefig(
            fig_dir / "position_x_signed_rho_by_year.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    # -----------------------------------------------------------------
    # Altitude histogram
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        ALT_EDGES_KM[:-1],
        altitude_counts,
        width=np.diff(ALT_EDGES_KM),
        align="edge",
        linewidth=0,
    )
    ax.set_xlabel("Altitude above mean Mars radius [km]")
    ax.set_ylabel("Position samples")
    ax.set_yscale("log")
    ax.set_title(f"MAVEN — altitude distribution (R_M = {radius_km:g} km)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "position_altitude_histogram.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Daily min/max altitude
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = pd.to_datetime(day_df["date_utc"])
    ax.plot(x, day_df["min_altitude_km"], lw=0.8, label="Daily minimum")
    ax.plot(x, day_df["max_altitude_km"], lw=0.8, label="Daily maximum")
    ax.set_xlabel("UTC date")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("MAVEN — daily observed altitude range")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "position_altitude_range_by_day.png", dpi=200)
    plt.close(fig)

    # -----------------------------------------------------------------
    # Sampling by year
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(year_df["year"].astype(str), year_df["valid_positions"])
    ax.set_xlabel("UTC year")
    ax.set_ylabel("Positions with finite coordinates")
    ax.set_title("MAVEN — spatial sampling by year")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "position_samples_by_year.png", dpi=200)
    plt.close(fig)


def frontal_matrices_from_result(grouped_df: pd.DataFrame):
    """Build year-wise sample-count matrices; no averaging across years."""
    shape = (len(Y_EDGES_RM) - 1, len(Z_EDGES_RM) - 1)
    matrices = {}
    for row in grouped_df.itertuples(index=False):
        year, iy, iz = int(row.year), int(row.iy), int(row.iz)
        if not (0 <= iy < shape[0] and 0 <= iz < shape[1]):
            raise ValueError(f"Invalid frontal bin: {year}, {iy}, {iz}")
        if year not in matrices:
            matrices[year] = np.zeros(shape, dtype=np.int64)
        matrices[year][iy, iz] += int(row.sample_count)
    if not matrices:
        raise RuntimeError("The frontal query returned no position bins.")
    return matrices


def save_frontal_csv(grouped_df: pd.DataFrame, output_dir: Path) -> Path:
    """Store small binned counts instead of downloading positions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = grouped_df.copy()
    result["year"] = result["year"].astype(int)
    result["iy"] = result["iy"].astype(int)
    result["iz"] = result["iz"].astype(int)
    result["sample_count"] = result["sample_count"].astype(np.int64)
    result["y_min_rm"] = Y_EDGES_RM[result["iy"].to_numpy()]
    result["y_max_rm"] = Y_EDGES_RM[result["iy"].to_numpy() + 1]
    result["z_min_rm"] = Z_EDGES_RM[result["iz"].to_numpy()]
    result["z_max_rm"] = Z_EDGES_RM[result["iz"].to_numpy() + 1]
    path = output_dir / "position_y_z_counts_by_year.csv"
    result[["year", "iy", "iz", "y_min_rm", "y_max_rm",
            "z_min_rm", "z_max_rm", "sample_count"]].to_csv(
        path, index=False
    )
    return path


def frontal_matrices_from_csv(csv_path: Path):
    counts = pd.read_csv(csv_path)
    required = {"year", "iy", "iz", "sample_count"}
    if not required.issubset(counts.columns):
        raise ValueError(f"Invalid frontal CSV schema at {csv_path}")
    return frontal_matrices_from_result(counts)


def make_frontal_figures(output_dir: Path, year_matrices) -> None:
    """Render Y-Z coverage for the whole period and year-by-year."""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    years = sorted(year_matrices)
    shape = (len(Y_EDGES_RM) - 1, len(Z_EDGES_RM) - 1)
    total = np.zeros(shape, dtype=np.int64)
    for year in years:
        total += year_matrices[year]

    theta = np.linspace(0, 2*np.pi, 400)
    mars_y, mars_z = np.cos(theta), np.sin(theta)

    # Figure 1 — all years. No boundary curves: a Vignes bow-shock/MPB
    # surface does not project to a single Y-Z curve without selecting X.
    fig, ax = plt.subplots(figsize=(9, 7))
    mesh = ax.pcolormesh(
        Y_EDGES_RM, Z_EDGES_RM, np.log10(total.T + 1),
        shading="auto", vmin=0,
    )
    fig.colorbar(mesh, ax=ax, label=r"$\log_{10}(N_\mathrm{samples}+1)$")
    ax.plot(mars_y, mars_z, color="white", linewidth=1.3,
            linestyle="--", label="Mars limb")
    ax.axhline(0, color="white", linewidth=0.5, alpha=0.35)
    ax.axvline(0, color="white", linewidth=0.5, alpha=0.35)
    ax.set(xlabel=r"$y_\mathrm{SS}/R_M$",
           ylabel=r"$z_\mathrm{SS}/R_M$",
           title="MAVEN — frontal spatial coverage (Y–Z projection)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "position_y_z_coverage.png", dpi=200)
    plt.close(fig)

    # Figure 2 — per-year facets with a dedicated GridSpec colorbar column.
    ncols = 3
    nrows = int(np.ceil(len(years) / ncols))
    fig = plt.figure(figsize=(14.0, 4.15*nrows))
    gs = fig.add_gridspec(
        nrows, ncols+1, width_ratios=[1.0, 1.0, 1.0, 0.065],
        left=0.075, right=0.91, bottom=0.055, top=0.945,
        wspace=0.26, hspace=0.35,
    )
    vmax = max(1.0, float(np.log10(
        max(int(a.max()) for a in year_matrices.values()) + 1
    )))
    mesh = None
    first_ax = None
    for i, year in enumerate(years):
        row, col = divmod(i, ncols)
        if first_ax is None:
            ax = fig.add_subplot(gs[row, col])
            first_ax = ax
        else:
            ax = fig.add_subplot(gs[row, col],
                                 sharex=first_ax, sharey=first_ax)
        mesh = ax.pcolormesh(
            Y_EDGES_RM, Z_EDGES_RM,
            np.log10(year_matrices[year].T + 1),
            shading="auto", vmin=0, vmax=vmax,
        )
        ax.plot(mars_y, mars_z, color="white", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
        ax.axvline(0, color="white", linewidth=0.5, alpha=0.3)
        ax.set_title(str(year), pad=9)
        ax.set_xlim(float(Y_EDGES_RM[0]), float(Y_EDGES_RM[-1]))
        ax.set_ylim(float(Z_EDGES_RM[0]), float(Z_EDGES_RM[-1]))
        ax.set_aspect("equal", adjustable="box")
        if row == nrows-1 or i+ncols >= len(years):
            ax.set_xlabel(r"$y_\mathrm{SS}/R_M$")
        else:
            ax.tick_params(axis="x", labelbottom=False)
        if col == 0:
            ax.set_ylabel(r"$z_\mathrm{SS}/R_M$")
        else:
            ax.tick_params(axis="y", labelleft=False)
    fig.colorbar(
        mesh, cax=fig.add_subplot(gs[:, -1]),
        label=r"$\log_{10}(N_\mathrm{samples}+1)$",
    )
    fig.suptitle("MAVEN — frontal (Y–Z) spatial coverage by year", y=0.985)
    fig.savefig(figure_dir / "position_y_z_by_year.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Frontal Y-Z figures saved:")
    print(f"  {figure_dir / 'position_y_z_coverage.png'}")
    print(f"  {figure_dir / 'position_y_z_by_year.png'}")


def rerender_from_saved_csvs(output_dir: Path, radius_km: float) -> None:
    """Rebuild all figures from the CSV summaries saved by a prior run."""
    required = {
        "day": output_dir / "position_by_day.csv",
        "year": output_dir / "position_by_year.csv",
        "spatial": output_dir / "position_x_signed_rho_counts.csv",
        "spatial_by_year": (
            output_dir / "position_x_signed_rho_counts_by_year.csv"
        ),
        "altitude": output_dir / "position_altitude_histogram.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing saved results needed for --plots-only:\n  "
            + "\n  ".join(missing)
            + "\nRun the normal position analysis once to create them."
        )

    day_df = pd.read_csv(required["day"])
    year_df = pd.read_csv(required["year"])
    spatial_df = pd.read_csv(required["spatial"])
    by_year_df = pd.read_csv(required["spatial_by_year"])
    altitude_df = pd.read_csv(required["altitude"])

    shape = (len(X_EDGES_RM) - 1, len(RHO_SIGNED_EDGES_RM) - 1)
    spatial_counts = np.zeros(shape, dtype=np.int64)
    for row in spatial_df.itertuples(index=False):
        ix = int(round((float(row.x_min_rm) - X_EDGES_RM[0]) /
                       (X_EDGES_RM[1] - X_EDGES_RM[0])))
        irho = int(round((float(row.rho_signed_min_rm)
                          - RHO_SIGNED_EDGES_RM[0]) /
                         (RHO_SIGNED_EDGES_RM[1] - RHO_SIGNED_EDGES_RM[0])))
        if not (0 <= ix < shape[0] and 0 <= irho < shape[1]):
            raise ValueError("Invalid spatial bin in stored CSV")
        spatial_counts[ix, irho] += int(row.sample_count)

    spatial_year_mats = {}
    for row in by_year_df.itertuples(index=False):
        year = int(row.year)
        if year not in spatial_year_mats:
            spatial_year_mats[year] = np.zeros(shape, dtype=np.int64)
        ix = int(round((float(row.x_min_rm) - X_EDGES_RM[0]) /
                       (X_EDGES_RM[1] - X_EDGES_RM[0])))
        irho = int(round((float(row.rho_signed_min_rm)
                          - RHO_SIGNED_EDGES_RM[0]) /
                         (RHO_SIGNED_EDGES_RM[1] - RHO_SIGNED_EDGES_RM[0])))
        if not (0 <= ix < shape[0] and 0 <= irho < shape[1]):
            raise ValueError("Invalid annual spatial bin in stored CSV")
        spatial_year_mats[year][ix, irho] += int(row.sample_count)

    altitude_counts = altitude_df["sample_count"].to_numpy(dtype=np.int64)
    if len(altitude_counts) != len(ALT_EDGES_KM) - 1:
        raise ValueError("Altitude CSV bins do not match current grid")

    make_figures(
        output_dir, day_df, year_df,
        spatial_counts, spatial_year_mats,
        altitude_counts, radius_km,
    )
    frontal_path = output_dir / "position_y_z_counts_by_year.csv"
    if frontal_path.is_file():
        frontal_matrices = frontal_matrices_from_csv(frontal_path)
        make_frontal_figures(output_dir, frontal_matrices)
    else:
        print("Frontal Y-Z counts are not available yet; run --front-only once.")
    print("Figures regenerated from CSVs (no BigQuery queries):")
    print(f"  {output_dir / 'figures'}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MAVEN spatial coverage from BigQuery positions."
    )
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", type=parse_date)
    parser.add_argument("--end", type=parse_date)
    parser.add_argument("--limit-days", "--limit-files",
                        dest="limit_days", type=int)
    parser.add_argument("--radius-km", type=float,
                        default=DEFAULT_MARS_RADIUS_KM)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--front-only", action="store_true",
        help=("Compute only the new Y-Z (frontal) coverage from BigQuery, "
              "preserving the previous x-rho / altitude outputs."),
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help=(
            "Regenerate figures from existing results/MAVEN_positions CSVs "
            "without querying BigQuery."
        ),
    )
    args = parser.parse_args()

    if args.radius_km <= 0:
        parser.error("--radius-km must be positive")
    if args.limit_days is not None and args.limit_days <= 0:
        parser.error("--limit-days must be positive")
    if args.start and args.end and args.start > args.end:
        parser.error("--start must not be after --end")

    if args.plots_only and args.front_only:
        parser.error("--plots-only and --front-only are mutually exclusive")
    if args.plots_only:
        rerender_from_saved_csvs(args.output_dir, args.radius_km)
        return

    client = bigquery.Client(project=args.table.split(".")[0])
    dates = partitions_and_validate(client, args.table)

    start = args.start if args.start is not None else dates[0]
    end = args.end if args.end is not None else dates[-1]
    selected_dates = [d for d in dates if start <= d <= end]
    if not selected_dates:
        raise RuntimeError(
            "No position table partitions within the requested range."
        )

    if args.limit_days is not None:
        selected_dates = selected_dates[:args.limit_days]

    first, last = selected_dates[0], selected_dates[-1]

    print("=" * 76)
    print("MAVEN — POSITION SPATIAL COVERAGE (BIGQUERY)")
    print("=" * 76)
    print(f"Table:                  {args.table}")
    print(f"Available partitions:   {len(dates):,}")
    print(f"Selected partitions:    {len(selected_dates):,}")
    print(f"UTC range:              {first} -> {last}")
    print(f"Mars radius:            {args.radius_km:g} km")
    if args.limit_days is not None:
        print("MODE:                   TEST SUBSET (not full coverage)")

    if args.front_only:
        sqls = {
            "frontal_by_year": sql_frontal_by_year(args.table),
        }
    else:
        sqls = {
            "daily": sql_daily(args.table),
            "spatial": sql_spatial(args.table),
            "spatial_by_year": sql_spatial_by_year(args.table),
            "altitude": sql_altitude(args.table),
            "frontal_by_year": sql_frontal_by_year(args.table),
        }

    print("\nEstimating BigQuery scan sizes...")
    sizes = {}
    for kind, sql in sqls.items():
        n = run_query(client, sql, kind, first, last, args.radius_km,
                      estimate_only=True)
        sizes[kind] = n
        print(f"  {kind:8s}: {human_bytes(n)}")
    total_estimated = sum(sizes.values())
    print(f"  total   : {human_bytes(total_estimated)}")

    if args.estimate_only:
        print("\nEstimate-only: no position data were scanned.")
        return

    print("\nQuerying aggregated spatial statistics...")
    outputs = {}
    for kind, sql in sqls.items():
        print(f"  {kind}...")
        outputs[kind] = run_query(client, sql, kind, first, last,
                                  args.radius_km, estimate_only=False)
        print(f"    grouped rows: {len(outputs[kind]):,}")

    frontal_df = outputs["frontal_by_year"].copy()
    if frontal_df.empty:
        raise RuntimeError("No Y-Z bins were returned from BigQuery.")
    frontal_csv = save_frontal_csv(frontal_df, args.output_dir)
    frontal_by_year = frontal_matrices_from_result(frontal_df)
    make_frontal_figures(args.output_dir, frontal_by_year)
    print(f"Frontal Y-Z counts saved: {frontal_csv}")

    if args.front_only:
        print("Front-only: previous x-rho and altitude outputs unchanged.")
        return

    day_df = outputs["daily"].copy()
    if day_df.empty:
        raise RuntimeError("No position samples returned.")

    day_df["date_utc"] = pd.to_datetime(
        day_df["date_utc"]
    ).dt.strftime("%Y-%m-%d")
    day_df["year"] = pd.to_datetime(day_df["date_utc"]).dt.year
    for c in day_df.columns:
        if c not in {"date_utc", "year"}:
            day_df[c] = pd.to_numeric(day_df[c], errors="coerce")

    year_df = day_df.groupby("year", as_index=False).agg(
        days_with_data=("date_utc", "count"),
        total_rows=("total_rows", "sum"),
        valid_positions=("valid_positions", "sum"),
        invalid_positions=("invalid_positions", "sum"),
        min_altitude_km=("min_altitude_km", "min"),
        max_altitude_km=("max_altitude_km", "max"),
        samples_altitude_below_500_km=(
            "samples_altitude_below_500_km", "sum"
        ),
        samples_altitude_below_1000_km=(
            "samples_altitude_below_1000_km", "sum"
        ),
    )
    year_df["pct_altitude_below_1000_km"] = (
        100 * year_df["samples_altitude_below_1000_km"] /
        year_df["valid_positions"].replace(0, np.nan)
    )

    spatial_counts = np.zeros(
        (len(X_EDGES_RM) - 1, len(RHO_SIGNED_EDGES_RM) - 1),
        dtype=np.int64,
    )
    for row in outputs["spatial"].itertuples(index=False):
        ix, irho = int(row.ix), int(row.irho)
        if not (0 <= ix < spatial_counts.shape[0] and
                0 <= irho < spatial_counts.shape[1]):
            raise RuntimeError(f"Unexpected spatial histogram bin: {ix}, {irho}")
        spatial_counts[ix, irho] += int(row.sample_count)

    spatial_year_mats = {}
    for row in outputs["spatial_by_year"].itertuples(index=False):
        year = int(row.year)
        if year not in spatial_year_mats:
            spatial_year_mats[year] = np.zeros(
                (len(X_EDGES_RM) - 1, len(RHO_SIGNED_EDGES_RM) - 1),
                dtype=np.int64,
            )
        ix, irho = int(row.ix), int(row.irho)
        if not (0 <= ix < spatial_counts.shape[0] and
                0 <= irho < spatial_counts.shape[1]):
            raise RuntimeError(
                f"Unexpected spatial-by-year histogram bin: {year}, {ix}, {irho}"
            )
        spatial_year_mats[year][ix, irho] += int(row.sample_count)

    altitude_counts = np.zeros(len(ALT_EDGES_KM) - 1, dtype=np.int64)
    for row in outputs["altitude"].itertuples(index=False):
        ialt = int(row.ialt)
        if not (0 <= ialt < len(altitude_counts)):
            raise RuntimeError(f"Unexpected altitude histogram bin: {ialt}")
        altitude_counts[ialt] += int(row.sample_count)

    total_rows = int(day_df["total_rows"].sum())
    total_valid = int(day_df["valid_positions"].sum())
    total_invalid = int(day_df["invalid_positions"].sum())

    if not total_valid:
        raise RuntimeError("No finite spacecraft positions found.")

    summary = pd.DataFrame([{
        "source_table": args.table,
        "first_date": first.isoformat(),
        "last_date": last.isoformat(),
        "partitions_selected": len(selected_dates),
        "is_test_subset": args.limit_days is not None,
        "coordinate_frame": "SS (as stored)",
        "mars_radius_km": args.radius_km,
        "total_rows": total_rows,
        "valid_positions": total_valid,
        "invalid_positions": total_invalid,
        "pct_valid_positions": 100 * total_valid / total_rows,
        "r_min_km": day_df["min_r_km"].min(),
        "r_max_km": day_df["max_r_km"].max(),
        "altitude_min_km": day_df["min_altitude_km"].min(),
        "altitude_max_km": day_df["max_altitude_km"].max(),
        "negative_altitude_count": int(
            day_df["negative_altitude_count"].sum()
        ),
        "altitude_below_500_km_count": int(
            day_df["samples_altitude_below_500_km"].sum()
        ),
        "altitude_below_1000_km_count": int(
            day_df["samples_altitude_below_1000_km"].sum()
        ),
        "pct_altitude_below_1000_km": (
            100 * day_df["samples_altitude_below_1000_km"].sum() / total_valid
        ),
        "out_of_x_signed_rho_plot_range": (
            total_valid - int(spatial_counts.sum())
        ),
        "out_of_altitude_plot_range": (
            total_valid - int(altitude_counts.sum())
        ),
        "vignes_bs_x0_rm": VIGNES_BOW_SHOCK["x0_rm"],
        "vignes_bs_epsilon": VIGNES_BOW_SHOCK["epsilon"],
        "vignes_bs_L_rm": VIGNES_BOW_SHOCK["L_rm"],
        "vignes_mpb_x0_rm": VIGNES_MPB["x0_rm"],
        "vignes_mpb_epsilon": VIGNES_MPB["epsilon"],
        "vignes_mpb_L_rm": VIGNES_MPB["L_rm"],
    }])

    spatial_csv_rows = []
    for ix, irho in np.argwhere(spatial_counts > 0):
        spatial_csv_rows.append({
            "x_min_rm": X_EDGES_RM[ix],
            "x_max_rm": X_EDGES_RM[ix + 1],
            "rho_signed_min_rm": RHO_SIGNED_EDGES_RM[irho],
            "rho_signed_max_rm": RHO_SIGNED_EDGES_RM[irho + 1],
            "sample_count": int(spatial_counts[ix, irho]),
        })
    spatial_csv = pd.DataFrame(spatial_csv_rows, columns=[
        "x_min_rm", "x_max_rm",
        "rho_signed_min_rm", "rho_signed_max_rm", "sample_count"
    ])

    spatial_year_csv_rows = []
    for year, mat in sorted(spatial_year_mats.items()):
        for ix, irho in np.argwhere(mat > 0):
            spatial_year_csv_rows.append({
                "year": year,
                "x_min_rm": X_EDGES_RM[ix],
                "x_max_rm": X_EDGES_RM[ix + 1],
                "rho_signed_min_rm": RHO_SIGNED_EDGES_RM[irho],
                "rho_signed_max_rm": RHO_SIGNED_EDGES_RM[irho + 1],
                "sample_count": int(mat[ix, irho]),
            })
    spatial_year_csv = pd.DataFrame(spatial_year_csv_rows, columns=[
        "year", "x_min_rm", "x_max_rm",
        "rho_signed_min_rm", "rho_signed_max_rm", "sample_count"
    ])
    altitude_csv = pd.DataFrame({
        "altitude_min_km": ALT_EDGES_KM[:-1],
        "altitude_max_km": ALT_EDGES_KM[1:],
        "sample_count": altitude_counts,
    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "position_summary.csv", index=False)
    day_df.to_csv(args.output_dir / "position_by_day.csv", index=False)
    year_df.to_csv(args.output_dir / "position_by_year.csv", index=False)
    spatial_csv.to_csv(
        args.output_dir / "position_x_signed_rho_counts.csv", index=False
    )
    spatial_year_csv.to_csv(
        args.output_dir / "position_x_signed_rho_counts_by_year.csv",
        index=False,
    )
    altitude_csv.to_csv(
        args.output_dir / "position_altitude_histogram.csv", index=False
    )
    make_figures(
        args.output_dir, day_df, year_df,
        spatial_counts, spatial_year_mats,
        altitude_counts, args.radius_km
    )

    print("\n" + "=" * 76)
    print("MAVEN — POSITION SPATIAL COVERAGE SUMMARY")
    print("=" * 76)
    print(f"Position records:        {total_rows:,}")
    print(f"Finite positions:        {total_valid:,}")
    print(f"Invalid positions:       {total_invalid:,}")
    print(f"Altitude range:          "
          f"{float(day_df['min_altitude_km'].min()):.3f} - "
          f"{float(day_df['max_altitude_km'].max()):.3f} km")
    print(f"Altitude < 1000 km:      "
          f"{float(summary['pct_altitude_below_1000_km'].iloc[0]):.3f}%")
    print("Vignes model overlays:   "
          f"BS(x0={VIGNES_BOW_SHOCK['x0_rm']}, ε={VIGNES_BOW_SHOCK['epsilon']}, L={VIGNES_BOW_SHOCK['L_rm']}); "
          f"MPB(x0={VIGNES_MPB['x0_rm']}, ε={VIGNES_MPB['epsilon']}, L={VIGNES_MPB['L_rm']})")
    print(f"Outside x-signed-rho map:"
          f"       {int(summary['out_of_x_signed_rho_plot_range'].iloc[0]):,}")
    print(f"Outside altitude plot:   "
          f"{int(summary['out_of_altitude_plot_range'].iloc[0]):,}")
    print("Extra outputs:           "
          "position_x_signed_rho_coverage.png, "
          "position_x_signed_rho_by_year.png")
    print(f"Outputs:                 {args.output_dir}")
    print("=" * 76)


if __name__ == "__main__":
    main()
