#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03b — MAVEN frontal orbit coverage by month within each year.

This script queries the existing BigQuery positions table and builds
one figure per UTC year. Each figure contains a 4x3 grid of monthly
Y-Z coverage panels (frontal view, looking from +X toward Mars).

The plot is based on 2D binned sample counts (heatmaps), which show
the projection of the observed MAVEN orbital paths onto the Y-Z plane.
This is not a slice at X = 0: all X values are projected onto the
frontal plane.

Table expected:
    maven-vignes-crossings.maven.positions
with columns:
    datetime_utc, x_ss_km, y_ss_km, z_ss_km

Outputs:
    results/MAVEN_positions_monthly_front/
    ├── frontal_y_z_counts_by_year_month.csv
    ├── frontal_y_z_year_summary.csv
    └── figures/
        ├── frontal_y_z_by_month_2014.png
        ├── frontal_y_z_by_month_2015.png
        └── ...

Usage:
    python3 scripts/positions/03b_MAVEN_frontal_by_month.py --estimate-only
    python3 scripts/positions/03b_MAVEN_frontal_by_month.py
    python3 scripts/positions/03b_MAVEN_frontal_by_month.py --plots-only
    python3 scripts/positions/03b_MAVEN_frontal_by_month.py --start 2018-01-01 --end 2020-12-31
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
DEFAULT_OUTPUT_DIR = Path("results/MAVEN_positions_monthly_front")
DEFAULT_MARS_RADIUS_KM = 3389.5
Y_EDGES_RM = np.linspace(-5.0, 5.0, 201)
Z_EDGES_RM = np.linspace(-5.0, 5.0, 201)
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

REQUIRED_COLUMNS = {"datetime_utc", "x_ss_km", "y_ss_km", "z_ss_km"}


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


def valid_xyz_sql() -> str:
    parts = []
    for col in ("x_ss_km", "y_ss_km", "z_ss_km"):
        parts.extend([f"{col} IS NOT NULL", f"NOT IS_NAN({col})", f"NOT IS_INF({col})"])
    return " AND ".join(parts)


def partitions_and_validate(client, table_id: str) -> list[date]:
    table = client.get_table(table_id)
    names = {field.name for field in table.schema}
    missing = REQUIRED_COLUMNS - names
    if missing:
        raise RuntimeError(
            f"BigQuery table {table_id} is missing required columns: {sorted(missing)}"
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
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("table_name", "STRING", tablename),
    ])
    result = client.query(sql, job_config=job_config).result()
    dates = [
        date(int(r.partition_id[:4]), int(r.partition_id[4:6]), int(r.partition_id[6:8]))
        for r in result if int(r.total_rows or 0) > 0
    ]
    if not dates:
        raise RuntimeError("No non-empty daily partitions found in the positions table.")
    return sorted(set(dates))


def sql_frontal_by_year_month(table_id: str) -> str:
    valid = valid_xyz_sql()
    return f"""
    WITH points AS (
        SELECT
            EXTRACT(YEAR FROM datetime_utc) AS year,
            EXTRACT(MONTH FROM datetime_utc) AS month,
            y_ss_km / @radius_km AS y_rm,
            z_ss_km / @radius_km AS z_rm
        FROM `{table_id}`
        WHERE DATE(datetime_utc) BETWEEN @start_date AND @end_date
          AND {valid}
    ),
    binned AS (
        SELECT
            year,
            month,
            LEAST(
                @ny - 1,
                CAST(FLOOR((y_rm - @ymin) / @dy) AS INT64)
            ) AS iy,
            LEAST(
                @nz - 1,
                CAST(FLOOR((z_rm - @zmin) / @dz) AS INT64)
            ) AS iz
        FROM points
        WHERE y_rm BETWEEN @ymin AND @ymax
          AND z_rm BETWEEN @zmin AND @zmax
    )
    SELECT
        year,
        month,
        iy,
        iz,
        COUNT(*) AS sample_count
    FROM binned
    GROUP BY year, month, iy, iz
    ORDER BY year, month, iy, iz
    """


def frontal_query_config(start_date: date, end_date: date, radius_km: float,
                         dry_run: bool) -> bigquery.QueryJobConfig:
    return bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
            bigquery.ScalarQueryParameter("radius_km", "FLOAT64", radius_km),
            bigquery.ScalarQueryParameter("ymin", "FLOAT64", float(Y_EDGES_RM[0])),
            bigquery.ScalarQueryParameter("ymax", "FLOAT64", float(Y_EDGES_RM[-1])),
            bigquery.ScalarQueryParameter("dy", "FLOAT64", float(np.diff(Y_EDGES_RM)[0])),
            bigquery.ScalarQueryParameter("ny", "INT64", len(Y_EDGES_RM) - 1),
            bigquery.ScalarQueryParameter("zmin", "FLOAT64", float(Z_EDGES_RM[0])),
            bigquery.ScalarQueryParameter("zmax", "FLOAT64", float(Z_EDGES_RM[-1])),
            bigquery.ScalarQueryParameter("dz", "FLOAT64", float(np.diff(Z_EDGES_RM)[0])),
            bigquery.ScalarQueryParameter("nz", "INT64", len(Z_EDGES_RM) - 1),
        ],
        dry_run=dry_run,
        use_query_cache=not dry_run,
    )


def run_query(client, sql: str, start_date: date, end_date: date,
              radius_km: float, estimate_only: bool):
    cfg = frontal_query_config(start_date, end_date, radius_km, dry_run=estimate_only)
    job = client.query(sql, job_config=cfg)
    if estimate_only:
        return int(job.total_bytes_processed or 0)
    return job.to_dataframe(create_bqstorage_client=False)


def matrices_from_grouped(grouped_df: pd.DataFrame):
    matrices = {}
    for row in grouped_df.itertuples(index=False):
        year = int(row.year)
        month = int(row.month)
        iy = int(row.iy)
        iz = int(row.iz)
        if year not in matrices:
            matrices[year] = {}
        if month not in matrices[year]:
            matrices[year][month] = np.zeros(
                (len(Y_EDGES_RM) - 1, len(Z_EDGES_RM) - 1),
                dtype=np.int64,
            )
        if not (0 <= iy < len(Y_EDGES_RM) - 1 and 0 <= iz < len(Z_EDGES_RM) - 1):
            raise ValueError(f"Invalid frontal bin: {year}, {month}, {iy}, {iz}")
        matrices[year][month][iy, iz] += int(row.sample_count)
    if not matrices:
        raise RuntimeError("The frontal monthly query returned no position bins.")
    return matrices


def save_grouped_csv(grouped_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "frontal_y_z_counts_by_year_month.csv"
    grouped_df.to_csv(csv_path, index=False)
    return csv_path


def save_year_summary(year_month_mats, output_dir: Path) -> Path:
    rows = []
    for year, month_dict in sorted(year_month_mats.items()):
        months_present = sorted(month_dict.keys())
        total_samples = int(sum(mat.sum() for mat in month_dict.values()))
        for month in range(1, 13):
            rows.append({
                "year": year,
                "month": month,
                "month_name": MONTH_NAMES[month - 1],
                "has_data": month in month_dict,
                "sample_count": int(month_dict.get(month, np.zeros((1, 1), dtype=int)).sum()),
                "months_present_in_year": len(months_present),
                "total_samples_in_year": total_samples,
            })
    df = pd.DataFrame(rows)
    path = output_dir / "frontal_y_z_year_summary.csv"
    df.to_csv(path, index=False)
    return path


def load_grouped_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    required = {"year", "month", "iy", "iz", "sample_count"}
    if not required.issubset(df.columns):
        raise ValueError(f"Invalid grouped frontal CSV schema at {csv_path}")
    return df


def make_monthly_figures(output_dir: Path, year_month_mats) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    vmax = 0.0
    for month_dict in year_month_mats.values():
        for mat in month_dict.values():
            vmax = max(vmax, float(np.log10(int(mat.max()) + 1)))
    if vmax <= 0:
        vmax = None

    yy = np.linspace(-1, 1, 300)
    mars = np.sqrt(np.maximum(0, 1 - yy**2))

    for year, month_dict in sorted(year_month_mats.items()):
        fig, axes = plt.subplots(
            4, 3,
            figsize=(11.5, 14.0),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        last_mesh = None
        for month in range(1, 13):
            r = (month - 1) // 3
            c = (month - 1) % 3
            ax = axes[r, c]
            counts = month_dict.get(
                month,
                np.zeros((len(Y_EDGES_RM) - 1, len(Z_EDGES_RM) - 1), dtype=np.int64),
            )

            last_mesh = ax.pcolormesh(
                Y_EDGES_RM,
                Z_EDGES_RM,
                np.log10(counts.T + 1),
                shading="auto",
                vmin=0.0,
                vmax=vmax,
            )
            ax.plot(yy, mars, "--", linewidth=0.8)
            ax.plot(yy, -mars, "--", linewidth=0.8)
            ax.axhline(0.0, linewidth=0.5, alpha=0.35)
            ax.axvline(0.0, linewidth=0.5, alpha=0.35)
            ax.set_title(MONTH_NAMES[month - 1], fontsize=10)
            ax.set_aspect("equal", adjustable="box")

        for ax in axes[-1, :]:
            ax.set_xlabel(r"$Y_{\mathrm{SS}}/R_M$")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$Z_{\mathrm{SS}}/R_M$")

        fig.subplots_adjust(
            left=0.08, right=0.88, bottom=0.06, top=0.95,
            wspace=0.08, hspace=0.16,
        )
        cax = fig.add_axes([0.90, 0.18, 0.03, 0.64])
        fig.colorbar(
            last_mesh,
            cax=cax,
            label=r"$\log_{10}(N_{\mathrm{samples}}+1)$",
        )
        fig.suptitle(
            f"MAVEN — frontal Y-Z coverage by month for {year}",
            y=0.985,
        )
        fig.savefig(fig_dir / f"frontal_y_z_by_month_{year}.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Make one frontal Y-Z figure per year, with 12 monthly panels."
        )
    )
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", type=parse_date)
    parser.add_argument("--end", type=parse_date)
    parser.add_argument("--radius-km", type=float, default=DEFAULT_MARS_RADIUS_KM)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate the figures from the saved CSV without querying BigQuery.",
    )
    args = parser.parse_args()

    if args.radius_km <= 0:
        parser.error("--radius-km must be positive")
    if args.start and args.end and args.start > args.end:
        parser.error("--start must not be after --end")
    if args.plots_only and args.estimate_only:
        parser.error("--plots-only and --estimate-only are mutually exclusive")

    csv_path = args.output_dir / "frontal_y_z_counts_by_year_month.csv"
    if args.plots_only:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing CSV for --plots-only: {csv_path}. Run the full query first."
            )
        grouped_df = load_grouped_csv(csv_path)
        year_month_mats = matrices_from_grouped(grouped_df)
        make_monthly_figures(args.output_dir, year_month_mats)
        print("=" * 78)
        print("MAVEN — FRONTAL MONTHLY COVERAGE FIGURES REGENERATED")
        print("=" * 78)
        print(f"Loaded grouped CSV:      {csv_path}")
        print(f"Years present:           {len(year_month_mats):,}")
        print(f"Figures directory:       {args.output_dir / 'figures'}")
        print("=" * 78)
        return

    client = bigquery.Client(project=args.table.split(".")[0])
    dates = partitions_and_validate(client, args.table)

    start = args.start if args.start is not None else dates[0]
    end = args.end if args.end is not None else dates[-1]
    selected_dates = [d for d in dates if start <= d <= end]
    if not selected_dates:
        raise RuntimeError("No table partitions within the requested date range.")
    first, last = selected_dates[0], selected_dates[-1]

    print("=" * 78)
    print("MAVEN — FRONTAL MONTHLY ORBIT COVERAGE")
    print("=" * 78)
    print(f"Table:                  {args.table}")
    print(f"Available partitions:   {len(dates):,}")
    print(f"Selected partitions:    {len(selected_dates):,}")
    print(f"UTC range:              {first} -> {last}")
    print(f"Mars radius:            {args.radius_km:g} km")

    sql = sql_frontal_by_year_month(args.table)
    estimated = run_query(
        client, sql, first, last, args.radius_km, estimate_only=True
    )
    print(f"Estimated bytes:        {human_bytes(estimated)}")

    if args.estimate_only:
        print("\nEstimate-only: no position data were scanned.")
        return

    print("\nQuerying grouped frontal Y-Z counts by year and month...")
    grouped_df = run_query(
        client, sql, first, last, args.radius_km, estimate_only=False
    )
    if grouped_df.empty:
        raise RuntimeError("The grouped query returned no rows.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_grouped_csv(grouped_df, args.output_dir)
    year_month_mats = matrices_from_grouped(grouped_df)
    summary_path = save_year_summary(year_month_mats, args.output_dir)
    make_monthly_figures(args.output_dir, year_month_mats)

    print("\n" + "=" * 78)
    print("MAVEN — FRONTAL MONTHLY ORBIT COVERAGE SUMMARY")
    print("=" * 78)
    print(f"Years with data:         {len(year_month_mats):,}")
    print(f"Grouped rows returned:   {len(grouped_df):,}")
    print(f"Counts CSV:              {csv_path}")
    print(f"Year summary CSV:        {summary_path}")
    print(f"Figures directory:       {args.output_dir / 'figures'}")
    print("Figure pattern:          frontal_y_z_by_month_<year>.png")
    print("=" * 78)


if __name__ == "__main__":
    main()
