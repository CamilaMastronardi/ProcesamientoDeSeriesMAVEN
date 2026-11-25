#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CWT Morlet sobre MAG en ventanas TOTALES de 10 min centradas en MPB.

Se usan exclusivamente tiempos MPB clasificados por la usuaria, no Vignes.
Los Parquet MAG tratados se leen localmente; no se consulta BigQuery.

Cada segmento MAG continuo se interpola SOLO a una grilla de 1 s y se
transforma por separado. Nunca se atraviesa un gap >1.5 s ni un dato inválido.
Se enmascara una aproximación conservadora del cono de influencia (COI):
para cmorB-C, se excluyen distancias a bordes < sqrt(B)*escala*dt.

Produce resúmenes por evento y período, algunos escalogramas individuales,
y el escalograma PROMEDIO de potencia normalizada entre cruces MPB.
La potencia wavelet no es un test de significancia ni demuestra oscilaciones:
un salto de campo en la propia MPB puede producir potencia de banda ancha.

Ejemplo:
  python3 scripts/wavelets/05a_MPB_MAG_wavelet_10min.py \
    --events data/classified_crossings/ --max-plots 12
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt


MAG_DIR = Path("data/processed/MAG_gap_treated")
OUT_ROOT = Path("results/MAG_MPB_wavelets_10min")
FIELDS = ["b_total_nt", "bx_ss_nt", "by_ss_nt", "bz_ss_nt"]
WAVELET = "cmor1.5-1.0"
BANDWIDTH = 1.5


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")[:85] or "evento"


def load_mpb_catalog(location):
    """Lee los cuatro CSV originales o un catálogo long/wide equivalente."""
    path = Path(location)
    if path.is_dir():
        files = sorted(path.glob("MAVEN_MPB_Data_Group*.csv"))
        if not files:
            files = sorted(path.glob("*.csv"))
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError("No se encontró el catálogo: {}".format(path))
    if not files:
        raise FileNotFoundError("No hay CSV de cruces MPB en {}".format(path))

    records = []
    for file in files:
        raw = pd.read_csv(file, dtype=str, keep_default_na=False,
                          encoding="utf-8-sig")
        raw.columns = [str(c).strip() for c in raw.columns]
        group_match = re.search(r"Group\s*(\d+)", file.stem, re.I)
        group = "G" + group_match.group(1) if group_match else file.stem
        original = {"yyyy-mm-dd", "MPB time"}.issubset(raw.columns)
        long = {"event_type", "event_time_utc"}.issubset(raw.columns)
        if not original and not long and "mpb_datetime_utc" not in raw.columns:
            raise ValueError("Catálogo no reconocido: {}".format(file))
        for i, row in raw.iterrows():
            row_group = row.get("group", "").strip() or group
            case_id = row.get("case_id", "").strip() or "{}_row{:04d}".format(row_group, i + 2)
            bs_value = ""
            if original:
                day = row["yyyy-mm-dd"].strip()
                mpb = row["MPB time"].strip()
                if not day and not mpb:
                    continue
                if not day or not mpb:
                    raise ValueError("MPB incompleta: {} línea {}".format(file, i + 2))
                value = day + " " + mpb
                if row.get("BS time", "").strip():
                    bs_value = day + " " + row["BS time"].strip()
            elif long:
                if row["event_type"].strip().lower() != "mpb":
                    continue
                value = row["event_time_utc"].strip()
            else:
                value = row["mpb_datetime_utc"].strip()
                bs_value = row.get("bow_shock_datetime_utc", "").strip()
            if not value:
                continue
            t = pd.to_datetime(value, errors="coerce", utc=True)
            if pd.isna(t):
                raise ValueError("Hora MPB inválida: {} línea {}".format(file, i + 2))
            bs = pd.to_datetime(bs_value, errors="coerce", utc=True) if bs_value else pd.NaT
            source_file = row.get("source_file", "").strip() or file.name
            source_row_raw = row.get("source_row", "").strip()
            source_row = int(source_row_raw) if source_row_raw.isdigit() else i + 2
            records.append(dict(case_id=case_id, group=row_group, mpb_utc=t,
                                bs_utc=bs, source_file=source_file, source_row=source_row))
    if not records:
        raise ValueError("El catálogo no contiene eventos MPB")
    events = pd.DataFrame(records).sort_values(["mpb_utc", "case_id"])
    duplicated = int(events.duplicated("mpb_utc").sum())
    events = events.drop_duplicates("mpb_utc").reset_index(drop=True)
    events["event_id"] = [
        "{:04d}_{}_{}".format(i + 1, safe_name(row.case_id),
                               row.mpb_utc.strftime("%Y%m%dT%H%M%S"))
        for i, row in enumerate(events.itertuples(index=False))
    ]
    print("MPB clasificadas únicas: {} ({} duplicados descartados)".format(
        len(events), duplicated))
    return events


def read_day(day, mag_root, field):
    p = mag_root / day.strftime("%Y") / day.strftime("%m") / (
        "mag_{}.parquet".format(day.strftime("%Y-%m-%d")))
    if not p.is_file():
        raise FileNotFoundError(str(p))
    columns = ["datetime_utc", field, "is_interpolated"]
    frame = pd.read_parquet(p, columns=columns)
    frame["datetime_utc"] = pd.to_datetime(frame["datetime_utc"], utc=True)
    return frame


def extract_mag_window(event_time, mag_root, field, half_window_s, day_cache):
    start = event_time - pd.Timedelta(seconds=half_window_s)
    end = event_time + pd.Timedelta(seconds=half_window_s)
    pieces = []
    day = start.floor("D")
    while day <= end.floor("D"):
        daily = day_cache(day.strftime("%Y-%m-%d"))
        pieces.append(daily.loc[(daily.datetime_utc >= start) &
                                (daily.datetime_utc <= end)])
        day += pd.Timedelta(days=1)
    if not pieces:
        return pd.DataFrame(columns=["datetime_utc", field, "is_interpolated"])
    mag = pd.concat(pieces, ignore_index=True)
    return mag.sort_values("datetime_utc").drop_duplicates("datetime_utc")


def cwt_event(mag, event_time, field, relative_seconds, periods,
              cadence_s=1.0, max_gap_s=1.5, min_segment_s=60.0,
              exclude_interpolated=False, detrend="linear"):
    """Scalograma sin interpolar discontinuidades; COI excluido por segmento."""
    n = len(relative_seconds)
    values = np.full(n, np.nan)
    raw_power = np.full((len(periods), n), np.nan)
    field_values = pd.to_numeric(mag[field], errors="coerce").to_numpy(dtype=float)
    relative_raw = ((mag["datetime_utc"] - event_time).dt.total_seconds()
                    .to_numpy(dtype=float))
    reconstructed = mag["is_interpolated"].fillna(False).to_numpy(dtype=bool)
    good = np.isfinite(relative_raw) & np.isfinite(field_values)
    if exclude_interpolated:
        good &= ~reconstructed
    traw = relative_raw[good]
    braw = field_values[good]

    segments = []
    if len(traw):
        edges = np.r_[0, np.where(np.diff(traw) > max_gap_s)[0] + 1, len(traw)]
        for left, right in zip(edges[:-1], edges[1:]):
            t, b = traw[left:right], braw[left:right]
            if len(t) < 2:
                continue
            indices = np.flatnonzero((relative_seconds >= t[0]) &
                                     (relative_seconds <= t[-1]))
            if len(indices) < 2:
                continue
            # Solo entre puntos MAG consecutivos de la misma región continua.
            values[indices] = np.interp(relative_seconds[indices], t, b)
            if len(indices) * cadence_s >= min_segment_s:
                segments.append(indices)

    wavelet = pywt.ContinuousWavelet(WAVELET)
    scales = pywt.central_frequency(wavelet) * periods / cadence_s
    detrended_segments = []
    n_transform = 0
    for indices in segments:
        signal = values[indices].astype(float)
        if detrend == "linear":
            x = np.arange(len(signal), dtype=float)
            coef = np.polyfit(x, signal, deg=1)
            signal = signal - np.polyval(coef, x)
        elif detrend == "constant":
            signal = signal - np.mean(signal)
        # Con "none" conservamos el nivel medio, opción para comparar.
        detrended_segments.append(signal)
        coeff, _ = pywt.cwt(signal, scales, wavelet,
                            sampling_period=cadence_s, method="fft")
        power = np.abs(coeff) ** 2
        # Para cmorB-C, la potencia de la envolvente gaussiana cae por e^-2
        # a sqrt(B)*escala*dt del borde (aproximación COI por segmento).
        dist = np.minimum(np.arange(len(signal)),
                          np.arange(len(signal))[::-1]) * cadence_s
        trustworthy = dist[None, :] >= (
            np.sqrt(BANDWIDTH) * scales[:, None] * cadence_s)
        raw_power[:, indices] = np.where(trustworthy, power, np.nan)
        n_transform += 1

    if detrended_segments:
        all_residuals = np.concatenate(detrended_segments)
        variance = float(np.var(all_residuals))
    else:
        variance = np.nan
    if not np.isfinite(variance) or variance <= 0:
        normalized = np.full_like(raw_power, np.nan)
    else:
        normalized = raw_power / variance
    count = np.isfinite(normalized).sum(axis=1)
    integrated = np.nansum(normalized, axis=1)
    mean_by_period = np.divide(integrated, count,
                               out=np.full(len(periods), np.nan),
                               where=count > 0)
    if np.any(np.isfinite(mean_by_period)):
        best = int(np.nanargmax(mean_by_period))
        dominant_period = float(periods[best])
    else:
        dominant_period = np.nan
    stats = dict(n_mag_samples=int(len(mag)),
                 n_original_or_filled_valid=int(good.sum()),
                 n_interpolated_mag=int(reconstructed.sum()),
                 fraction_window_covered=float(np.isfinite(values).mean()),
                 n_contiguous_segments=int(n_transform),
                 n_valid_cwt_pixels=int(np.isfinite(normalized).sum()),
                 dominant_period_s=dominant_period,
                 residual_variance=float(variance),
                 status="ok" if np.any(np.isfinite(normalized)) else "insufficient_data")
    return values, normalized, mean_by_period, count, stats


def paired_power_by_period(power, relative_seconds, min_side_points=20):
    """Potencia normalizada media por período y por lado, para el mismo evento.

    Evita el punto central t=0 y descarta un período si cualquiera de los dos
    lados no tiene al menos min_side_points tiempos fuera del COI y gaps.
    La diferencia logarítmica es log10(P_despues/P_antes), no un test.
    """
    before = np.asarray(relative_seconds, dtype=float) < 0
    after = np.asarray(relative_seconds, dtype=float) > 0
    if power.ndim != 2 or power.shape[1] != len(relative_seconds):
        raise ValueError("Las dimensiones de potencia y tiempo no coinciden")
    vals = []
    for mask in (before, after):
        region = power[:, mask]
        n = np.isfinite(region).sum(axis=1)
        sums = np.nansum(region, axis=1)
        avg = np.divide(sums, n, out=np.full(len(n), np.nan),
                        where=n >= min_side_points)
        vals.append((avg, n))
    before_avg, before_n = vals[0]
    after_avg, after_n = vals[1]
    valid = (np.isfinite(before_avg) & (before_avg > 0) &
             np.isfinite(after_avg) & (after_avg > 0))
    delta = np.full(len(before_avg), np.nan)
    delta[valid] = np.log10(after_avg[valid] / before_avg[valid])
    return before_avg, after_avg, before_n, after_n, delta


def plot_map(times_min, periods, values, title, filename, signal=None,
             label="log10 potencia normalizada"):
    from matplotlib.colors import Normalize
    import warnings
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 6.3))
    if signal is not None:
        ax1 = fig.add_axes([0.10, 0.77, 0.76, 0.17])
        ax1.plot(times_min, signal, lw=0.8, color="black")
        ax1.set_ylabel("|B| [nT]" if "b_total" in title else "B [nT]")
        ax1.set_xlim(times_min[0], times_min[-1])
        ax1.axvline(0, color="crimson", ls="--", lw=1.1)
        ax1.tick_params(labelbottom=False)
        plot_rect = [0.10, 0.11, 0.76, 0.58]
    else:
        plot_rect = [0.10, 0.12, 0.76, 0.78]
    ax = fig.add_axes(plot_rect)
    finite = values[np.isfinite(values) & (values > 0)]
    if len(finite):
        log_power = np.log10(np.maximum(values, 1e-14))
        lo, hi = np.nanpercentile(np.log10(finite), [2, 98])
        if lo >= hi:
            hi = lo + 0.1
        boundaries_t = np.r_[times_min[0] - (times_min[1]-times_min[0])/2,
                             (times_min[1:] + times_min[:-1])/2,
                             times_min[-1] + (times_min[-1]-times_min[-2])/2]
        boundaries_p = np.r_[periods[0] / np.sqrt(periods[1]/periods[0]),
                             np.sqrt(periods[1:] * periods[:-1]),
                             periods[-1] * np.sqrt(periods[-1]/periods[-2])]
        mesh = ax.pcolormesh(boundaries_t, boundaries_p,
                             np.ma.masked_invalid(log_power), shading="flat",
                             cmap="viridis", norm=Normalize(vmin=lo, vmax=hi))
        fig.colorbar(mesh, ax=ax, pad=0.02, label=label)
    else:
        ax.text(0.5, 0.5, "Sin potencia válida fuera del COI",
                transform=ax.transAxes, ha="center", va="center")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.axvline(0, color="crimson", ls="--", lw=1.1)
    ax.set_xlim(times_min[0], times_min[-1])
    ax.set_ylim(periods[-1], periods[0])
    ax.set_ylabel("Período [s]")
    ax.set_xlabel("Tiempo relativo a MPB [min]")
    ax.set_title(title, pad=9)
    fig.savefig(str(filename), dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Wavelets Morlet de MAG: 10 min TOTALES centrados en MPB")
    p.add_argument("--events", type=Path, default=Path("data/classified_crossings"))
    p.add_argument("--mag-dir", type=Path, default=MAG_DIR)
    p.add_argument("--output-root", type=Path, default=OUT_ROOT)
    p.add_argument("--run-name", default=None,
                   help="Subcarpeta de resultados; útil para pruebas")
    p.add_argument("--field", choices=FIELDS, default="b_total_nt")
    p.add_argument("--window-min", type=float, default=10.0)
    p.add_argument("--cadence-s", type=float, default=1.0)
    p.add_argument("--max-gap-s", type=float, default=1.5)
    p.add_argument("--min-segment-s", type=float, default=60.0)
    p.add_argument("--min-period-s", type=float, default=4.0)
    p.add_argument("--max-period-s", type=float, default=120.0)
    p.add_argument("--n-periods", type=int, default=48)
    p.add_argument("--detrend", choices=["linear", "constant", "none"], default="linear")
    p.add_argument("--exclude-interpolated", action="store_true")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--max-plots", type=int, default=12,
                   help="Cantidad máxima de escalogramas individuales (0 = ninguno)")
    p.add_argument("--min-side-points", type=int, default=20,
                   help="Mínimo de tiempos CWT válidos por lado y por período (default: 20)")
    args = p.parse_args()
    if args.window_min <= 0 or args.cadence_s <= 0 or args.max_gap_s <= 0:
        p.error("Las duraciones y la cadencia deben ser positivas")
    if args.n_periods < 3 or args.max_plots < 0:
        p.error("--n-periods debe ser >=3 y --max-plots >=0")
    if args.min_side_points < 1:
        p.error("--min-side-points debe ser >=1")
    if args.max_events is not None and args.max_events < 1:
        p.error("--max-events debe ser >=1")
    nyquist_period = 2.0 * args.cadence_s
    if args.min_period_s <= nyquist_period or args.max_period_s <= args.min_period_s:
        p.error("Elegir períodos >2*cadence-s y max-period-s > min-period-s")
    if args.max_gap_s < args.cadence_s:
        p.error("--max-gap-s debe ser >= --cadence-s")
    events = load_mpb_catalog(args.events)
    if args.max_events is not None:
        events = events.iloc[:args.max_events].copy()
    root = args.output_root
    if args.run_name:
        root = root / safe_name(args.run_name)
    elif args.field != "b_total_nt" or args.window_min != 10.0:
        root = root / safe_name(args.field + "_" + str(args.window_min) + "min")
    root.mkdir(parents=True, exist_ok=True)
    times_s = np.arange(-args.window_min * 30.0,
                         args.window_min * 30.0 + args.cadence_s / 2,
                         args.cadence_s)
    times_min = times_s / 60.0
    periods = np.geomspace(args.min_period_s, args.max_period_s, args.n_periods)
    plot_indices = set(np.linspace(0, len(events)-1,
                      num=min(len(events), args.max_plots), dtype=int).tolist())
    accumulated = np.zeros((len(periods), len(times_s)), dtype=float)
    counts = np.zeros_like(accumulated, dtype=int)
    summary = []
    by_period = []
    paired_by_period = []

    @lru_cache(maxsize=4)
    def day_cache(date_str):
        day = pd.Timestamp(date_str, tz="UTC")
        return read_day(day, args.mag_dir, args.field)

    for i, event in enumerate(events.itertuples(index=False)):
        print("[{}/{}] MPB {} {}".format(i+1, len(events), event.case_id,
                                         event.mpb_utc), flush=True)
        bs_near = bool(pd.notna(event.bs_utc) and
                       abs((event.bs_utc-event.mpb_utc).total_seconds()) <=
                       args.window_min * 30.0)
        common = dict(event_id=event.event_id, case_id=event.case_id,
                      group=event.group, mpb_utc=event.mpb_utc.isoformat(),
                      source_file=event.source_file,
                      bs_within_window=bs_near)
        try:
            mag = extract_mag_window(event.mpb_utc, args.mag_dir,
                                     args.field, args.window_min*30.0, day_cache)
            signal, power, avg_by_period, valid_by_period, result = cwt_event(
                mag, event.mpb_utc, args.field, times_s, periods,
                cadence_s=args.cadence_s, max_gap_s=args.max_gap_s,
                min_segment_s=args.min_segment_s,
                exclude_interpolated=args.exclude_interpolated,
                detrend=args.detrend)
            mask = np.isfinite(power)
            accumulated[mask] += power[mask]
            counts[mask] += 1
            summary.append(dict(**common, **result))
            before, after, n_before, n_after, delta = paired_power_by_period(
                power, times_s, min_side_points=args.min_side_points)
            for j, (period, avg, nvalid) in enumerate(zip(periods, avg_by_period, valid_by_period)):
                by_period.append(dict(**common, period_s=float(period),
                                      mean_normalized_power=float(avg) if np.isfinite(avg) else np.nan,
                                      n_valid_times=int(nvalid)))
                paired_by_period.append(dict(
                    **common, period_s=float(period),
                    power_before=float(before[j]) if np.isfinite(before[j]) else np.nan,
                    power_after=float(after[j]) if np.isfinite(after[j]) else np.nan,
                    n_valid_before=int(n_before[j]), n_valid_after=int(n_after[j]),
                    log10_after_over_before=float(delta[j]) if np.isfinite(delta[j]) else np.nan))
            if i in plot_indices:
                plot_map(times_min, periods, power,
                         "MPB | {} | {}".format(args.field, event.mpb_utc),
                         root / "figures" / "by_event" / (event.event_id + ".png"),
                         signal=signal)
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            summary.append(dict(**common, status="error",
                                error="{}: {}".format(type(exc).__name__, exc)))
            print("  AVISO:", exc, flush=True)

    pd.DataFrame(summary).to_csv(root / "mpb_wavelet_event_summary.csv", index=False)
    pd.DataFrame(by_period).to_csv(root / "mpb_wavelet_period_by_event.csv", index=False)
    pd.DataFrame(paired_by_period).to_csv(
        root / "mpb_wavelet_before_after_by_event.csv", index=False)
    mean = np.divide(accumulated, counts,
                     out=np.full_like(accumulated, np.nan),
                     where=counts > 0)
    rows = []
    for j, period in enumerate(periods):
        for k, minute in enumerate(times_min):
            rows.append(dict(period_s=float(period), time_relative_min=float(minute),
                             mean_normalized_power=float(mean[j, k]),
                             n_valid_events=int(counts[j, k])))
    pd.DataFrame(rows).to_csv(root / "mpb_wavelet_mean_map.csv", index=False)
    plot_map(times_min, periods, mean,
             "MPB | promedio de potencia wavelet normalizada entre eventos",
             root / "figures" / "mpb_mean_wavelet_power.png")
    with (root / "analysis_config.json").open("w", encoding="utf-8") as f:
        json.dump(dict(field=args.field, window_min=args.window_min,
                       cadence_s=args.cadence_s, max_gap_s=args.max_gap_s,
                       min_segment_s=args.min_segment_s, periods_s=periods.tolist(),
                       wavelet=WAVELET, detrend=args.detrend,
                       exclude_interpolated=args.exclude_interpolated,
                       min_side_points=args.min_side_points,
                       n_selected_events=len(events), n_individual_figures=len(plot_indices)),
                  f, indent=2)
    print("Resultados guardados en:", root)
    print("Sin consultas a BigQuery. Interpretar los mapas junto a n_valid_events.")


if __name__ == "__main__":
    main()
