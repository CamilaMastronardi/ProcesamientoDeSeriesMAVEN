#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MAVEN SWEA–MAG: mapas de Pearson r(E, ancho de ventana) cerca de BS / MPB.

Casos de entrada: catalogo *clasificado* por el usuario (NO cruces geometricos
Vignes, NO fechas inventadas). Acepta directamente los cuatro archivos
MAVEN_MPB_Data_Group*.csv originales: columnas yyyy-mm-dd, BS time,
MPB time, ThetaBn; se ignoran las filas totalmente vacias. NO se cambian
las fechas originales. Tambien acepta CSV en uno de estos formatos:

 A) Una fila por evento (long):
    case_id,event_type,datetime_utc
    caso_01,BS,2016-03-16T18:02:00Z
    caso_01,MPB,2016-03-16T18:14:00Z

 B) Una fila por caso (wide):
    case_id,bow_shock_datetime_utc,mpb_datetime_utc
    caso_01,2016-03-16T18:02:00Z,2016-03-16T18:14:00Z

Las columnas pueden configurarse mediante --case-col, --event-time-col,
--event-type-col, --bs-col, --mpb-col. Tiempos interpretados como UTC.

Cada mapa usa x = energia [eV] (escala log), y = ancho TOTAL de ventana
[minutos], centrada en cada evento. Pixel = Pearson r(log10(flujo angular
medio SWEA por canal), componente B interpolada a los tiempos SWEA).

SWEA se lee de BigQuery exclusivamente alrededor de los eventos, con la
reduccion angular de los flujos calculada dentro de BigQuery. MAG se lee
localmente de MAG_gap_treated y se interpola con las reglas de 02e:
  - coincidencia exacta del timestamp: valida si B es finito;
  - entre muestras tratadas contiguas: lineal solo con dt <= 1.5 s;
  - nunca extrapolar ni atravesar gaps rechazados;
  - calcular |B| a partir de las 3 componentes interpoladas.

Los coeficientes no definidos (insuficientes muestras / varianza cero) son
NaN, no cero. Para comparar regiones, ademas de mapas POR EVENTO se genera
la MEDIA ARITMETICA de las correlaciones por evento para BS y MPB, sin
mezclar observaciones entre eventos como si fueran independientes. Se
conserva la mediana en CSV como estadistica auxiliar, pero los mapas
principales presentan el promedio solicitado y el numero de casos validos.

Ventanas por defecto: 2, 3, ..., 60 minutos (59 duraciones en total).
La salida CSV se escribe progresivamente, y los Parquet SWEA-MAG cacheados se
reutilizan si incluyen toda la ventana requerida; NO usar --overwrite.
Dependencias: numpy, pandas, matplotlib, pyarrow, google-cloud-bigquery.
Ejemplos:
  python3 scripts/correlations/04a_boundary_energy_correlation_maps.py \\
      --events data/classified_crossings.csv --estimate-only
  python3 scripts/correlations/04a_boundary_energy_correlation_maps.py \\
      --events data/classified_crossings/ --max-scan-gib 150
  python3 scripts/correlations/04a_boundary_energy_correlation_maps.py \\
      --events data/classified_crossings.csv --plots-only
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
import glob
import copy
from datetime import date, timedelta
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from google.cloud import bigquery

SWEA_TABLE = 'maven-vignes-crossings.maven.swea_svy3d'
AXES_TABLE = 'maven-vignes-crossings.maven.swea_svy3d_axes'
MAG_DIR = Path('data/processed/MAG_gap_treated')
OUT_ROOT = Path('results/MAG_SWEA_boundary_correlation')
N_ENERGIES = 64
FLUX_COLS = [f'flux_e{k:02d}' for k in range(N_ENERGIES)]
FIELDS = ['b_total_nt','bx_ss_nt','by_ss_nt','bz_ss_nt']
MAX_INTERPOLATION_SPAN_S = 1.5
DEFAULT_WINDOWS_MIN = ','.join(str(x) for x in range(2, 61))
RAW_DATE_COL = 'yyyy-mm-dd'
RAW_BS_COL = 'BS time'
RAW_MPB_COL = 'MPB time'
SQL_FLUX = r'''WITH axes AS (
  SELECT source_file, ANY_VALUE(n_energy) AS n_energy,
      ANY_VALUE(n_elev) AS n_elev, ANY_VALUE(n_azim) AS n_azim
  FROM `{axes_table}` GROUP BY source_file
)
SELECT d.datetime_utc, d.source_file, d.quality,
       a.n_energy, a.n_elev, a.n_azim,
       ARRAY_LENGTH(d.diff_en_fluxes) AS flux_length,
       ARRAY(
           SELECT IF(COUNTIF(f IS NOT NULL AND NOT IS_NAN(f) AND NOT IS_INF(f) AND f>=0)>0,
              SAFE_DIVIDE(SUM(IF(f IS NOT NULL AND NOT IS_NAN(f) AND NOT IS_INF(f) AND f>=0, f, 0.0)),
                COUNTIF(f IS NOT NULL AND NOT IS_NAN(f) AND NOT IS_INF(f) AND f>=0)),
              CAST(NULL AS FLOAT64))
           FROM UNNEST(d.diff_en_fluxes) AS f WITH OFFSET AS off
           GROUP BY MOD(off, a.n_energy)
           ORDER BY MOD(off, a.n_energy)
       ) AS flux_by_energy
FROM `{swea_table}` AS d
LEFT JOIN axes AS a USING (source_file)
WHERE d.datetime_utc >= @start_ts AND d.datetime_utc < @end_ts
ORDER BY d.datetime_utc'''


def safe_name(text: object) -> str:
    s = re.sub(r'[^A-Za-z0-9_\-]+', '_', str(text)).strip('_')
    return (s or 'unnamed')[:90]


def stamp(value):
    """Parser estricto: naive timestamps en el catalogo se interpretan UTC."""
    try:
        t = pd.to_datetime(value, utc=True, errors='raise')
    except (TypeError, ValueError):
        return pd.NaT
    return t if isinstance(t, pd.Timestamp) else pd.NaT


def normalize_type(value):
    val = re.sub('[^a-z0-9]+','',str(value).strip().lower())
    if val in ('bs','bowshock','shock','bow'):
        return 'BS'
    if val in ('mpb','magneticpileupboundary','pileupboundary'):
        return 'MPB'
    raise ValueError('Tipo de frontera desconocido: ' + str(value) +
                     ' (usar BS o MPB)')


def get_column(df, preferred, alternatives):
    names = {str(col).strip().lower():col for col in df.columns}
    if preferred:
        if preferred not in df.columns:
            raise ValueError('No existe la columna pedida: '+ preferred)
        return preferred
    for name in alternatives:
        if name.lower() in names:
            return names[name.lower()]
    return None


def resolve_catalog_files(inputs):
    """CSV individuales, directorio de los cuatro grupos o comodines de shell."""
    selected = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            groups = sorted(path.glob('MAVEN_MPB_Data_Group*.csv'))
            selected.extend(groups if groups else sorted(path.glob('*.csv')))
        elif path.is_file():
            selected.append(path)
        else:
            selected.extend(Path(p) for p in sorted(glob.glob(str(item))))
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise FileNotFoundError('No se encontraron CSV de cruces en ' + ', '.join(map(str, inputs)))
    return selected


def raw_datetime(day, time_string, path, line, boundary):
    """Fecha y hora independientes; incluso 8:06:45. Interpretación UTC."""
    value = f'{str(day).strip()} {str(time_string).strip()}'
    ts = stamp(value)
    if pd.isna(ts):
        raise ValueError(f'Fecha/hora {boundary} inválida: {path.name}:{line} -> {value!r}')
    return ts


def extract_raw_records(path, raw):
    """CSV originales: una fila = un par BS/MPB clasificado."""
    group_match = re.search(r'Group\s*(\d+)', path.stem, flags=re.IGNORECASE)
    group = 'G' + group_match.group(1) if group_match else safe_name(path.stem)
    records = []
    catalog = []
    issues = []
    for ix, row in raw.iterrows():
        line = ix + 2  # encabezado en la línea 1
        day = str(row[RAW_DATE_COL]).strip()
        bs = str(row[RAW_BS_COL]).strip()
        mpb = str(row[RAW_MPB_COL]).strip()
        if not day and not bs and not mpb:
            issues.append({'issue':'blank_original_row','group':group,'source_file':path.name,
                'source_row':line,'event_type':'','event_time_utc':'','details':'Fila vacía ignorada'})
            continue
        if not all((day,bs,mpb)):
            raise ValueError(f'Fila incompleta en {path.name}:{line} -> {day}, {bs}, {mpb}')
        bs_time = raw_datetime(day,bs,path,line,'BS')
        mpb_time = raw_datetime(day,mpb,path,line,'MPB')
        case_id = f'{group}_row{line:04d}'
        theta = pd.to_numeric(row.get('ThetaBn (deg) (Cyril)', ''), errors='coerce')
        beta = pd.to_numeric(row.get('beta_proton (Halekas)', ''), errors='coerce')
        pdyn = pd.to_numeric(row.get('Pdyn_proton (Halekas)', ''), errors='coerce')
        sep = abs((mpb_time-bs_time).total_seconds()/60.0)
        catalog.append({'case_id':case_id, 'group':group, 'source_file':path.name,
            'source_row':line, 'date_utc':day, 'bs_time_original':bs,
            'mpb_time_original':mpb, 'bow_shock_datetime_utc':bs_time.isoformat(),
            'mpb_datetime_utc':mpb_time.isoformat(),
            'theta_bn_deg':theta, 'beta_proton':beta, 'pdyn_proton':pdyn,
            'bs_mpb_absolute_separation_minutes':sep})
        if sep > 360:
            issues.append({'issue':'bs_mpb_separation_over_6h','group':group,'source_file':path.name,
                'source_row':line,'event_type':'BS+MPB','event_time_utc':bs_time.isoformat(),
                'details':f'Separación absoluta de {sep:.1f} minutos; revisar sin modificar'})
        for etype, t in (('BS',bs_time),('MPB',mpb_time)):
            records.append({'case_id':case_id,'event_type':etype,'event_time_utc':t,
                'group':group,'source_file':path.name,'source_row':line,
                'theta_bn_deg':theta,'beta_proton':beta,'pdyn_proton':pdyn})
    return records, catalog, issues


def extract_generic_records(path, raw, args):
    """Compatibilidad con formatos genéricos long y wide del script anterior."""
    case_col = get_column(raw,args.case_col,['case_id','event_id','caso','orbit_id','id'])
    type_col = get_column(raw,args.event_type_col,['event_type','boundary','type','frontera','tipo'])
    time_col = get_column(raw,args.event_time_col,['datetime_utc','crossing_time','event_time','time_utc','timestamp'])
    bs_col = get_column(raw,args.bs_col,
        ['bow_shock_datetime_utc','bs_datetime_utc','bow_shock_time','bs_time','bowshock','bs'])
    mpb_col = get_column(raw,args.mpb_col,['mpb_datetime_utc','mpb_time','mpb'])
    if not ((type_col and time_col) or bs_col or mpb_col):
        raise ValueError('Catálogo no reconocido: ' + str(path) + '. '
            'Se espera yyyy-mm-dd + BS time + MPB time, o formato genérico long/wide.')
    group = safe_name(path.stem)
    records=[]
    for ix,row in raw.iterrows():
        row_group = row.get('group', '').strip() or group
        case_id = row[case_col].strip() if case_col and row[case_col].strip() else f'{row_group}_row{ix+2:04d}'
        source_row_raw = row.get('source_row', '').strip()
        source_row = int(source_row_raw) if source_row_raw.isdigit() else ix+2
        meta={'group':row_group,
              'source_file':row.get('source_file', '').strip() or path.name,
              'source_row':source_row,
              'theta_bn_deg':pd.to_numeric(row.get('theta_bn_deg', ''),errors='coerce'),
              'beta_proton':pd.to_numeric(row.get('beta_proton', ''),errors='coerce'),
              'pdyn_proton':pd.to_numeric(row.get('pdyn_proton', ''),errors='coerce')}
        if type_col and time_col:
            typ=str(row[type_col]).strip()
            tim=str(row[time_col]).strip()
            if not typ and not tim:continue
            t=stamp(tim)
            if pd.isna(t):raise ValueError(f'Fecha inválida en {path}:{ix+2}: {tim!r}')
            records.append({'case_id':case_id,'event_type':normalize_type(typ),
                            'event_time_utc':t,**meta})
        else:
            for typ,col in (('BS',bs_col),('MPB',mpb_col)):
                if col and row[col].strip():
                    t=stamp(row[col].strip())
                    if pd.isna(t):raise ValueError(f'Fecha {typ} inválida en {path}:{ix+2}')
                    records.append({'case_id':case_id,'event_type':typ,
                                    'event_time_utc':t,**meta})
    return records


def load_events(paths, args):
    """Crea eventos preservando las filas originales y un registro de auditoría."""
    files = resolve_catalog_files(paths)
    records=[]
    catalog=[]
    issues=[]
    for path in files:
        raw=pd.read_csv(path,dtype=str,keep_default_na=False,encoding='utf-8-sig')
        raw.columns=[c.strip() for c in raw.columns]
        if not len(raw):
            raise ValueError('Archivo vacío: '+str(path))
        if {RAW_DATE_COL,RAW_BS_COL,RAW_MPB_COL}.issubset(raw.columns):
            rec,cat,err=extract_raw_records(path,raw)
            records.extend(rec);catalog.extend(cat);issues.extend(err)
            print(f'  {path.name}: {len(cat)} filas clasificadas; {len(err)} avisos')
        else:
            rec=extract_generic_records(path,raw,args)
            records.extend(rec)
            print(f'  {path.name}: {len(rec)} eventos')
    if not records:
        raise RuntimeError('No hay eventos con fechas válidas.')
    df=pd.DataFrame(records)
    if args.start:
        start=date.fromisoformat(args.start)
        df=df[df.event_time_utc.dt.date>=start]
    if args.end:
        end=date.fromisoformat(args.end)
        df=df[df.event_time_utc.dt.date<=end]
    df=df.sort_values(['event_time_utc','event_type','case_id']).reset_index(drop=True)
    if args.max_events:
        df=df.iloc[:args.max_events].copy()
    if df.empty:
        raise RuntimeError('No hay eventos en el período seleccionado.')
    # No se eliminan anotaciones originales: preservamos las 569 filas de entrada.
    # Repeticiones del mismo cruce se identificarán y no se ponderarán dos veces
    # en el promedio sobre eventos.
    duplicates=df.duplicated(['event_type','event_time_utc'],keep=False)
    for row in df.loc[duplicates].itertuples():
        issues.append({'issue':'duplicated_boundary_timestamp', 'group':row.group,
            'source_file':row.source_file, 'source_row':row.source_row,
            'event_type':row.event_type,'event_time_utc':row.event_time_utc.isoformat(),
            'details':'Mismo tipo de frontera y mismo timestamp en otra fila del catálogo'})
    df['event_id']=[f"{i+1:04d}_{safe_name(r.case_id)}_{r.event_type}_{r.event_time_utc.strftime('%Y%m%dT%H%M%S')}"
                    for i,r in enumerate(df.itertuples())]
    return df, pd.DataFrame(catalog), pd.DataFrame(issues), files


def parse_windows(s):
    try:
        v=np.asarray([float(x.strip()) for x in s.split(',')],dtype=float)
    except ValueError as e:
        raise argparse.ArgumentTypeError('Usar intervalos en minutos separados por comas') from e
    if len(v)==0 or not np.all(np.isfinite(v)) or np.any(v<=0):
        raise argparse.ArgumentTypeError('Las duraciones deben ser positivas.')
    v=np.unique(v)
    return v


def day_intervals(events, max_minutes):
    """Rangos [start,end) a consultar por dia UTC; union por dia."""
    result={}
    half=pd.Timedelta(minutes=float(max_minutes)/2)
    for ts in events['event_time_utc']:
        start=ts-half; end=ts+half
        day=start.floor('D')
        while day<end:
            right=day+pd.Timedelta(days=1)
            a=max(start,day); b=min(end,right)
            if a<b:
                d=day.date()
                if d not in result:
                    result[d]=[a,b]
                else:
                    result[d]=[min(a,result[d][0]),max(b,result[d][1])]
            day=right
    return dict(sorted(result.items()))


def job_config(a,b,dry_run=False):
    return bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter('start_ts','TIMESTAMP',a.to_pydatetime()),
            bigquery.ScalarQueryParameter('end_ts','TIMESTAMP',b.to_pydatetime())],
        dry_run=dry_run,use_query_cache=not dry_run)


def validate_axes(client,table):
    rows=list(client.query(f'SELECT source_file, n_energy, n_elev, n_azim, energy_ev FROM `{table}`').result())
    if not rows:
        raise RuntimeError('No hay ejes SWEA en BigQuery')
    energy=np.asarray(rows[0].energy_ev,dtype=float)
    if energy.shape!=(64,) or not np.all(np.isfinite(energy)) or np.any(energy<=0):
        raise RuntimeError('Grilla energetica inicial invalida')
    seen=set()
    for row in rows:
        arr=np.asarray(row.energy_ev,dtype=float)
        if row.source_file in seen or int(row.n_energy)!=64 or int(row.n_elev)<1 or int(row.n_azim)<1 or arr.shape!=(64,) or not np.allclose(arr,energy,rtol=1e-6,atol=1e-6,equal_nan=False):
            raise RuntimeError('Ejes incompatibles o duplicados: '+str(row.source_file))
        seen.add(row.source_file)
    return energy


def query_fluxes(client,sql,a,b):
    job=client.query(sql,job_config=job_config(a,b))
    records=[]
    for row in job.result():
        shape=(row.n_energy,row.n_elev,row.n_azim)
        expected=int(row.n_energy)*int(row.n_elev)*int(row.n_azim) if all(v is not None for v in shape) else -1
        if expected<1 or int(row.n_energy)!=64 or row.flux_length!=expected or row.flux_by_energy is None or len(row.flux_by_energy)!=64:
            raise RuntimeError('Flux dimensionalmente invalido: '+str(row.source_file)+' '+str(row.datetime_utc))
        records.append((row.datetime_utc,row.source_file,row.quality,*row.flux_by_energy))
    df=pd.DataFrame(records,columns=['datetime_utc','source_file','quality']+FLUX_COLS)
    if not df.empty:
        df['datetime_utc']=pd.to_datetime(df['datetime_utc'],utc=True)
        df=df.sort_values('datetime_utc').reset_index(drop=True)
        if df.datetime_utc.duplicated().any():
            raise RuntimeError('Timestamp SWEA duplicado en '+str(a))
    return df


def mag_path(day,root):
    return root/f'{day.year:04d}'/f'{day.month:02d}'/f'mag_{day.isoformat()}.parquet'


def read_mag_neighborhood(day,root):
    paths=[mag_path(day+timedelta(days=k),root) for k in (-1,0,1)]
    if not paths[1].is_file():
        raise FileNotFoundError('MAG tratado faltante: '+str(paths[1]))
    cols=['datetime_utc','bx_ss_nt','by_ss_nt','bz_ss_nt','is_original','is_interpolated']
    frames=[pd.read_parquet(p,columns=cols) for p in paths if p.is_file()]
    out=pd.concat(frames,ignore_index=True)
    out['datetime_utc']=pd.to_datetime(out['datetime_utc'],utc=True)
    out=out.dropna(subset=['datetime_utc']).sort_values('datetime_utc')
    if out.datetime_utc.duplicated().any():
        raise RuntimeError('Timestamps MAG duplicados en '+str(day))
    out=out.reset_index(drop=True)
    original=out['is_original'].fillna(False).to_numpy(dtype=bool)
    filled=out['is_interpolated'].fillna(False).to_numpy(dtype=bool)
    if np.any(original==filled):
        raise RuntimeError('Flags MAG inconsistentes: '+str(day))
    return out


def interpolate_mag_at_swea(swea_times,mag,max_span_s=MAX_INTERPOLATION_SPAN_S):
    """Exact 02e interpolation logic: strict 1.5s span, no gap crossing."""
    times=pd.to_datetime(swea_times,utc=True).to_numpy(dtype='datetime64[ns]').astype('int64')
    n=len(times)
    values=np.full((n,3),np.nan)
    statuses=np.full(n,'unmatched',dtype=object)
    used_filled=np.zeros(n,dtype=bool)
    spans=np.full(n,np.nan)
    if n and not mag.empty:
        mt=pd.to_datetime(mag['datetime_utc'],utc=True).to_numpy(dtype='datetime64[ns]').astype('int64')
        vec=mag[['bx_ss_nt','by_ss_nt','bz_ss_nt']].to_numpy(dtype=float)
        flags=mag['is_interpolated'].to_numpy(dtype=bool)
        right=np.searchsorted(mt,times,side='left')
        within=right<len(mt)
        exact=np.zeros(n,dtype=bool)
        exact[within]=mt[right[within]]==times[within]
        idx=np.where(exact)[0]
        if len(idx):
            ok=idx[np.all(np.isfinite(vec[right[idx]]),axis=1)]
            values[ok]=vec[right[ok]]
            used_filled[ok]=flags[right[ok]]
            statuses[ok]=np.where(used_filled[ok],'exact_mag_gap_filled','exact_mag_original')
            spans[ok]=0.0
        idx=np.where(~exact & (right>0) & (right<len(mt)))[0]
        if len(idx):
            r=right[idx]; l=r-1; dt=mt[r]-mt[l]
            good=(dt>0)&(dt<=round(max_span_s*1e9))&np.all(np.isfinite(vec[l]),axis=1)&np.all(np.isfinite(vec[r]),axis=1)
            ids=idx[good]; li=l[good]; ri=r[good]
            if len(ids):
                w=((times[ids]-mt[li]).astype(float)/(mt[ri]-mt[li]).astype(float))[:,None]
                values[ids]=vec[li]+w*(vec[ri]-vec[li])
                spans[ids]=(mt[ri]-mt[li])/1e9
                used_filled[ids]=flags[li] | flags[ri]
                statuses[ids]=np.where(used_filled[ids],'linear_uses_mag_gap_filled','linear_from_mag_original')
    valid=np.all(np.isfinite(values),axis=1)
    total=np.full(n,np.nan)
    total[valid]=np.sqrt(np.sum(values[valid]**2,axis=1))
    out=pd.DataFrame(values,columns=['bx_ss_nt','by_ss_nt','bz_ss_nt'])
    out['b_total_nt']=total
    out['mag_sync_valid']=valid
    out['mag_sync_status']=statuses
    out['mag_uses_gap_filled_samples']=used_filled
    out['mag_interpolation_span_s']=spans
    return out


def cache_file(root,day):
    return root/'synced'/f'{day.year:04d}'/f'{day.month:02d}'/f'swea_mag_{day.isoformat()}.parquet'



def mag_fingerprint(day, root):
    """Firma de archivos de MAG usados para no reutilizar un cache viejo."""
    signature = []
    for offset in (-1, 0, 1):
        p = mag_path(day + timedelta(days=offset), root)
        if p.is_file():
            stat = p.stat()
            signature.append({'file': str(p.resolve()), 'bytes': stat.st_size,
                              'mtime_ns': stat.st_mtime_ns})
    return signature


def diagnose_mag_alignment(day, mag, swea_times=None, events=None):
    """Información local de cobertura y sincronización; sin acceso a BigQuery."""
    print('  --- DIAGNOSTICO MAG ' + str(day) + ' ---', flush=True)
    mag_times = pd.to_datetime(mag['datetime_utc'], utc=True)
    valid_field = np.all(np.isfinite(
        mag[['bx_ss_nt','by_ss_nt','bz_ss_nt']].to_numpy(dtype=float)
    ), axis=1)
    print(f'  MAG filas (ayer/hoy/manana): {len(mag):,}; vectores B finitos: {int(valid_field.sum()):,}', flush=True)
    if len(mag):
        print(f'  MAG primer y ultimo timestamp: {mag_times.iloc[0]} -> {mag_times.iloc[-1]}', flush=True)
        for col in ('is_original', 'is_interpolated'):
            print(f'  MAG {col}: {int(mag[col].fillna(False).sum()):,}', flush=True)

    if events is not None:
        for t in events:
            stamp_evt = pd.to_datetime(t, utc=True)
            half = pd.Timedelta(minutes=5)
            near = (mag_times >= stamp_evt-half) & (mag_times < stamp_evt+half)
            print(f'  MAG en ventana +/-5 min de {stamp_evt}: '
                  f'{int(near.sum()):,} filas, {int((near.to_numpy() & valid_field).sum()):,} B finitos',
                  flush=True)

    if swea_times is None or len(swea_times)==0 or len(mag)==0:
        return
    swea_ts = pd.to_datetime(swea_times,utc=True)
    st = pd.DatetimeIndex(swea_ts).to_numpy(dtype='datetime64[ns]').astype('int64')
    mt = pd.DatetimeIndex(mag_times).to_numpy(dtype='datetime64[ns]').astype('int64')
    right = np.searchsorted(mt,st,side='left')
    exact = np.zeros(len(st),dtype=bool)
    msk = right < len(mt)
    exact[msk] = mt[right[msk]] == st[msk]
    bracket = (~exact) & (right>0) & (right<len(mt))
    r = right[bracket]
    spans = (mt[r]-mt[r-1])/1e9
    ok_span = spans <= MAX_INTERPOLATION_SPAN_S
    ok_vector = valid_field[r] & valid_field[r-1]
    ok_exact = exact & valid_field[np.clip(right,0,len(mt)-1)]
    possible = int(ok_exact.sum()+np.count_nonzero(ok_span & ok_vector))
    other = np.minimum(
        np.abs(st-mt[np.clip(right,0,len(mt)-1)]),
        np.abs(st-mt[np.clip(right-1,0,len(mt)-1)]),
    ) / 1e9
    print(f'  SWEA timestamps cacheados: {len(st):,}; '
          f'primero/ultimo: {swea_ts.min()} -> {swea_ts.max()}', flush=True)
    print(f'  SWEA exactos en MAG: {int(exact.sum())}; '
          f'entre muestras MAG: {int(bracket.sum())}; '
          f'fuera del rango MAG: {int((~exact & ~bracket).sum())}', flush=True)
    if len(spans):
        print(f'  Pares MAG con dt<=1.5 s: {int(ok_span.sum())}/{len(spans)}; '
              f'pares con B finito: {int(ok_vector.sum())}/{len(spans)}', flush=True)
        print(f'  mediana y max dt entre MAG vecinos: '
              f'{np.median(spans):.4f}s / {np.max(spans):.4f}s', flush=True)
    if len(other):
        print(f'  Distancia al MAG mas cercano: mediana={np.median(other):.4f}s, '
              f'min={np.min(other):.4f}s, max={np.max(other):.4f}s', flush=True)
    print(f'  Sincronizaciones posibles segun reglas actuales: {possible:,}/{len(st):,}',flush=True)
    print('  --- FIN DIAGNOSTICO ---',flush=True)


def read_or_make_synced(day,span,client,sql,mag_root,out,overwrite,swea_table,axes_table):
    """Reutiliza el flujo SWEA cacheado al reparar o cambiar la ruta de MAG.

    Crucial: en la version original, si MAG no existia se guardaba un Parquet
    con todos los valores B=NaN que luego se reutilizaba como si fuera valido.
    Esta version no guarda sin MAG y puede reparar ese cache sin releer los
    arreglos de flujo desde BigQuery.
    """
    path=cache_file(out,day)
    meta_path=path.with_suffix('.json')
    a,b=span
    provenance={'version':2,'swea_table':swea_table,'axes_table':axes_table,
                'mag_root':str(mag_root.resolve()),
                'mag_fingerprint':mag_fingerprint(day,mag_root)}
    cached_swea=None
    if path.is_file() and meta_path.is_file() and not overwrite:
        try:
            meta=json.loads(meta_path.read_text())
            basic_ok=(meta.get('version') in (1,2) and
                      meta.get('swea_table')==swea_table and
                      meta.get('axes_table')==axes_table)
            start=stamp(meta['start_utc']);end=stamp(meta['end_utc'])
            if basic_ok and pd.notna(start) and pd.notna(end) and start<=a and end>=b:
                df=pd.read_parquet(path)
                if all(k in df.columns for k in ['datetime_utc','source_file','quality']+FLUX_COLS):
                    df['datetime_utc']=pd.to_datetime(df['datetime_utc'],utc=True)
                    cached_swea=df[['datetime_utc','source_file','quality']+FLUX_COLS].copy()
                    original_mag_root=(meta.get('mag_root')==provenance['mag_root'])
                    cache_mag_unchanged=(meta.get('version')==2 and
                           meta.get('mag_fingerprint')==provenance['mag_fingerprint'])
                    old_mag_valid=('mag_sync_valid' in df.columns and
                                   bool(df['mag_sync_valid'].fillna(False).any()))
                    if (original_mag_root and cache_mag_unchanged and old_mag_valid and
                        all(k in df.columns for k in FIELDS+[
                            'mag_sync_valid','mag_uses_gap_filled_samples'])):
                        print(f'  {day}: cache completo valido ({len(df):,} timestamps)',flush=True)
                        return df
                    print(f'  {day}: recuperando flujos SWEA del cache y resincronizando MAG',flush=True)
        except (ValueError,KeyError,TypeError,OSError):
            cached_swea=None
    # Si no hay MAG, fallar antes de consultar datos SWEA (o guardar un cache
    # silenciosamente inutil): mag_path() ya fue verificado en preflight.
    mag=read_mag_neighborhood(day,mag_root)
    swea=cached_swea if cached_swea is not None else query_fluxes(client,sql,a,b)
    if swea.empty:
        df=pd.DataFrame(columns=['datetime_utc','source_file','quality']+FLUX_COLS+FIELDS+
               ['mag_sync_valid','mag_sync_status','mag_uses_gap_filled_samples','mag_interpolation_span_s'])
    else:
        aligned=interpolate_mag_at_swea(swea['datetime_utc'],mag)
        df=pd.concat([swea.reset_index(drop=True),aligned.reset_index(drop=True)],axis=1)
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(path,index=False)
    meta_path.write_text(json.dumps({**provenance,'start_utc':a.isoformat(),
                                     'end_utc':b.isoformat()},indent=2))
    print(f'  {day}: SWEA={len(df):,}, MAG valido={int(df.mag_sync_valid.sum()):,}',flush=True)
    if len(df) and not bool(df.mag_sync_valid.any()):
        diagnose_mag_alignment(day,mag,df['datetime_utc'])
    return df


def pearson_map(df,event_time,windows_min,field,min_pairs,exclude_filled=False):
    """Pearson por canal y ventana, con pares validos de forma independiente."""
    n_w=len(windows_min)
    rmap=np.full((n_w,64),np.nan,dtype=float)
    nmap=np.zeros((n_w,64),dtype=int)
    if df.empty:
        return rmap,nmap
    tt=pd.to_datetime(df['datetime_utc'],utc=True)
    flux=df[FLUX_COLS].to_numpy(dtype=float)
    logs=np.full_like(flux,np.nan)
    good=np.isfinite(flux)&(flux>0)
    logs[good]=np.log10(flux[good])
    bb=pd.to_numeric(df[field],errors='coerce').to_numpy(dtype=float)
    ok_mag=np.isfinite(bb)
    if exclude_filled:
        ok_mag &= ~df['mag_uses_gap_filled_samples'].to_numpy(dtype=bool)
    for wi,width in enumerate(windows_min):
        half=pd.Timedelta(minutes=float(width)/2)
        # El limite derecho es abierto, como en la consulta BQ y el cache.
        mask=(tt>=event_time-half)&(tt<event_time+half)
        if not mask.any():
            continue
        x=logs[mask.to_numpy()]
        y=bb[mask.to_numpy()]
        goodrows=ok_mag[mask.to_numpy()]
        for j in range(64):
            keep=goodrows&np.isfinite(x[:,j])
            n=int(keep.sum())
            nmap[wi,j]=n
            if n<min_pairs:
                continue
            u=x[keep,j];v=y[keep]
            u=u-u.mean();v=v-v.mean()
            xx=float(np.dot(u,u));yy=float(np.dot(v,v))
            if xx<=1e-15 or yy<=1e-15:
                continue
            rmap[wi,j]=np.clip(np.dot(u,v)/np.sqrt(xx*yy),-1.,1.)
    return rmap,nmap


def channel_edges(energies):
    es=np.sort(np.asarray(energies,dtype=float))
    mids=np.sqrt(es[:-1]*es[1:])
    return np.r_[es[0]**2/mids[0],mids,es[-1]**2/mids[-1]]


def window_edges(ws):
    ws=np.asarray(ws,dtype=float)
    if len(ws)==1:
        return np.array([ws[0]*0.75,ws[0]*1.25])
    mids=(ws[:-1]+ws[1:])/2
    return np.r_[max(0,ws[0]-(mids[0]-ws[0])),mids,ws[-1]+ws[-1]-mids[-1]]


def plot_map(grid, energies, windows, title, filename, kind='r'):
    """Mapa con 64 canales visibles y un pixel por canal x ancho de ventana.

    En escala logaritmica las anchuras horizontales son iguales porque SWEA
    tiene energias casi geometricas. Los bordes sutiles delimitan cada canal.
    Gris = r indefinido / n=0. No se inventan correlaciones.
    """
    energies=np.asarray(energies,dtype=float).reshape(-1)
    windows=np.asarray(windows,dtype=float).reshape(-1)
    grid=np.asarray(grid,dtype=float)
    if len(energies)!=N_ENERGIES or grid.shape!=(len(windows),N_ENERGIES):
        raise ValueError(f'Mapa energetico debe tener ({len(windows)},64), recibido {grid.shape}')
    if not np.all(np.isfinite(energies)) or np.any(energies<=0) or np.unique(energies).size!=64:
        raise ValueError('Grilla energetica invalida, hay que conservar los 64 centros.')
    order=np.argsort(energies)
    if kind=='r':
        data=np.ma.masked_invalid(grid[:,order])
        cmapname='RdBu_r'
        norm=Normalize(vmin=-1,vmax=1)
        colorbar_label=r'Pearson medio $\langle r(\log_{10} J(E),B) \rangle$' if 'mean_r' in str(filename) else r'Pearson $r(\log_{10} J(E),B)$'
    else:
        data=np.ma.masked_where(~np.isfinite(grid[:,order])|(grid[:,order]<=0),grid[:,order])
        cmapname='viridis'
        norm=None
        colorbar_label='Cantidad de cruces con correlacion definida'
    # 64 valores explicitos en X. Una figura amplia permite leer cada energia
    # sin omitir canales, en lugar de depender del localizador log automatico.
    fig,ax=plt.subplots(figsize=(25.5,11.2 if len(windows)>28 else 8.7))
    cm=copy.copy(plt.get_cmap(cmapname))
    cm.set_bad('#d8d8d8')
    # Una celda horizontal por energia. Lineas finas para visualizar 64 bins.
    mesh=ax.pcolormesh(
        channel_edges(energies),window_edges(windows),data,
        cmap=cm,norm=norm,shading='flat',
        edgecolors=(0,0,0,0.24), linewidths=0.24,
        antialiased=False,
    )
    ax.set_xscale('log')
    # Mostrar explícitamente el valor energético de cada canal (no el índice).
    # Los 64 ticks coinciden con los centros físicos de los 64 rectángulos.
    from matplotlib.ticker import FixedLocator, NullLocator
    sorted_energies = np.sort(energies)
    ax.xaxis.set_major_locator(FixedLocator(sorted_energies))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f'{e:.5g}' for e in sorted_energies],
                       rotation=90, ha='center', va='top', fontsize=7.4)
    ax.tick_params(axis='x', which='major', length=3.5, pad=3)
    ax.set_xlim(channel_edges(energies)[[0, -1]])
    # Rotular las 59 ventanas completas volvería ilegible el eje Y.
    # Se dibujan TODOS los bins, pero solo se rotulan las escalas principales.
    if len(windows) > 28:
        ticks = [float(w) for w in windows if np.isclose(w % 2.0, 0.0, atol=1e-7)]
        if not np.isclose(ticks[0] if ticks else np.nan, windows[0]):
            ticks.insert(0, float(windows[0]))
        if not np.isclose(ticks[-1], windows[-1]):
            ticks.append(float(windows[-1]))
    else:
        ticks = list(windows)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{w:g}' for w in ticks])
    ax.set_ylabel('Duracion TOTAL alrededor del cruce [min]')
    ax.set_xlabel('Energia del canal SWEA [eV] — 64 valores (escala logaritmica)',labelpad=8)
    ax.set_title(title)
    fig.colorbar(mesh,ax=ax,label=colorbar_label,pad=0.015,aspect=30)
    ax.grid(False)
    fig.subplots_adjust(bottom=0.23,top=0.88,right=0.88,left=0.065)
    filename.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(filename,dpi=210)
    plt.close(fig)


def _summary_matrices(summary, energies, ws, value_col):
    """Devuelve promedio y cantidad en grilla de len(ws) x 64."""
    mat=np.full((len(ws),len(energies)),np.nan,dtype=float)
    count=np.zeros((len(ws),len(energies)),dtype=float)
    for row in summary.itertuples(index=False):
        wi=int(np.argmin(np.abs(ws-row.window_minutes)))
        if not np.isclose(ws[wi],row.window_minutes,rtol=0,atol=1e-8):
            raise ValueError('Ancho de ventana en CSV no corresponde a analysis_config.json')
        col=int(row.energy_index)
        if not 0<=col<64:
            raise ValueError('Indice energetico fuera de [0,63]: '+str(col))
        # Usar la columna con valor agregado explicito, no tomar la mediana.
        mat[wi,col]=float(getattr(row,value_col))
        count[wi,col]=float(row.n_valid_events)
    return mat,count


class StreamingSummaries:
    """Correlaciones por evento sin cargar los millones de filas del CSV en RAM.

    Se deduplican (frontera, timestamp) exactamente como en la v7.
    Se conserva la media no ponderada por cantidad de muestras SWEA.
    """
    def __init__(self):
        self.total = defaultdict(lambda: [0, 0.0, []])
        self.groups = defaultdict(lambda: [0, 0.0, []])
        self.seen_events = set()
        self.seen_group_events = set()

    def add_event(self, event, fields_and_maps, ws):
        unique = (event.event_type, event.event_time_utc.isoformat())
        total_once = unique not in self.seen_events
        group_key = (event.group,) + unique
        group_once = group_key not in self.seen_group_events
        self.seen_events.add(unique)
        self.seen_group_events.add(group_key)
        for field, grid in fields_and_maps:
            for wi, width in enumerate(ws):
                for j in range(N_ENERGIES):
                    r = float(grid[wi, j])
                    if total_once:
                        slot = self.total[(event.event_type, field, float(width), j)]
                        slot[0] += 1
                        if np.isfinite(r):
                            slot[1] += r
                            slot[2].append(r)
                    if group_once:
                        slot = self.groups[(event.group, event.event_type, field, float(width), j)]
                        slot[0] += 1
                        if np.isfinite(r):
                            slot[1] += r
                            slot[2].append(r)

    @staticmethod
    def to_frame(src, energy, with_group=False):
        result = []
        for key, (n_total, rsum, values) in src.items():
            group, etype, field, width, idx = ((key[0], *key[1:]) if with_group
                                                else (None, *key))
            n_valid = len(values)
            row = {'event_type':etype, 'field':field,
                   'window_minutes':float(width), 'energy_index':int(idx),
                   'energy_ev':float(energy[idx]),
                   'n_valid_events':int(n_valid),
                   'median_pearson_r':float(np.median(values)) if n_valid else np.nan,
                   'mean_pearson_r':rsum/n_valid if n_valid else np.nan,
                   'n_total_events':int(n_total)}
            if with_group:
                row = {'group':group, **row}
            result.append(row)
        return pd.DataFrame(result)


def plot_summaries_only(out, energies, ws, fields, plot_per_event=False):
    """Plotea desde resumenes compactos, sin cargar correlations_by_event.csv."""
    summary = pd.read_csv(out / 'correlations_by_boundary_summary.csv')
    for (etype, field), grp in summary.groupby(['event_type', 'field']):
        mat, count = _summary_matrices(grp, energies, ws, 'mean_pearson_r')
        title = (f'{etype} · PROMEDIO de Pearson r entre cruces clasificados\n'
                 f'{field} · media aritmética de r por cruce; sin ponderar por N')
        plot_map(mat, energies, ws, title,
                 out/'figures'/'summary'/f'{etype}_{field}_mean_r.png')
        plot_map(count, energies, ws,
                 f'{etype} · número de cruces con r definido\n{field}',
                 out/'figures'/'summary'/f'{etype}_{field}_n_cases.png', kind='n')
        valid = np.isfinite(mat).sum(axis=1)
        print(f'  {etype}/{field}: 64 canales, correlaciones definidas: '
              + ', '.join(f'{w:g}m={n}' for w, n in zip(ws, valid)), flush=True)

    group_file = out/'correlations_by_group_summary.csv'
    if group_file.is_file():
        groups = pd.read_csv(group_file)
        for (group, etype, field), grp in groups.groupby(['group','event_type','field']):
            mat, count = _summary_matrices(grp, energies, ws, 'mean_pearson_r')
            plot_map(mat, energies, ws,
                     f'{group} · {etype} · PROMEDIO de Pearson r entre cruces\n{field}',
                     out/'figures'/'by_group'/f'{group}_{etype}_{field}_mean_r.png')
            plot_map(count, energies, ws,
                     f'{group} · {etype} · número de cruces con r definido\n{field}',
                     out/'figures'/'by_group'/f'{group}_{etype}_{field}_n_cases.png',kind='n')
    if plot_per_event:
        plot_events_streaming(out, energies, ws, fields)


def plot_events_streaming(out, energies, ws, fields):
    """Si se solicita, graficar eventos uno por uno leyendo CSV en streaming."""
    path = out/'correlations_by_event.csv'
    wid = {round(float(w), 8): i for i, w in enumerate(ws)}
    def draw(identifier, etype, cid, time, grids):
        for field in fields:
            label=f'{etype} · {cid} · {time}\n{field} · Pearson de log10 flujo SWEA'
            plot_map(grids[field], energies, ws, label,
                     out/'figures'/'by_event'/etype/f'{identifier}_{field}.png')
    last = None
    grids = None
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            event = row['event_id']
            if event != last:
                if last is not None:
                    draw(last, etype, cid, time, grids)
                grids = {field:np.full((len(ws),64),np.nan) for field in fields}
                last=event
                etype=row['event_type']; cid=row['case_id']; time=row['event_time_utc']
            field=row['field']; index=int(row['energy_index'])
            if field in grids:
                wi=wid[round(float(row['window_minutes']),8)]
                grids[field][wi,index]=float(row['pearson_r']) if row['pearson_r'] else np.nan
    if last is not None:
        draw(last,etype,cid,time,grids)


def plot_from_results(out,energies,ws,fields,plot_per_event=False):
    """Construye medias BS y MPB; puede ejecutarse sin BigQuery (--plots-only)."""
    table=pd.read_csv(out/'correlations_by_event.csv')
    if table.empty:
        raise RuntimeError('No hay correlaciones por evento en CSV.')
    required={'event_type','event_time_utc','field','window_minutes','energy_index',
              'energy_ev','pearson_r','event_id','case_id'}
    if missing:=required-set(table.columns):
        raise RuntimeError('Faltan columnas en correlaciones por evento: '+str(missing))
    if len(energies)!=64 or table.energy_index.nunique()!=64:
        raise RuntimeError('El CSV no contiene exactamente los 64 canales de SWEA.')
    # Recalcular tablas a partir de resultados individuales guardados (en
    # --plots-only no se consulta BigQuery y funciona con ejecuciones antiguas).
    summary=summarize_per_boundary(table)
    summary.to_csv(out/'correlations_by_boundary_summary.csv',index=False)
    grouped=[]
    if 'group' in table:
        for group,gdf in table.groupby('group',sort=True):
            gs=summarize_per_boundary(gdf)
            gs.insert(0,'group',group)
            grouped.append(gs)
        if grouped:
            pd.concat(grouped,ignore_index=True).to_csv(
                out/'correlations_by_group_summary.csv',index=False)

    if plot_per_event:
        for eid,grp in table.groupby('event_id',sort=False):
            etype=grp['event_type'].iloc[0]
            cid=grp['case_id'].iloc[0]
            time=grp['event_time_utc'].iloc[0]
            for field in fields:
                mat=np.full((len(ws),64),np.nan)
                for row in grp.loc[grp['field']==field].itertuples():
                    wi=int(np.argmin(np.abs(ws-row.window_minutes)))
                    mat[wi,int(row.energy_index)]=row.pearson_r
                title=f'{etype} · {cid} · {time}\n{field} · Pearson de log10 flujo SWEA'
                plot_map(mat,energies,ws,title,
                         out/'figures'/'by_event'/etype/f'{eid}_{field}.png')

    for (etype,field),grp in summary.groupby(['event_type','field']):
        mat,count=_summary_matrices(grp,energies,ws,'mean_pearson_r')
        label=(f'{etype} · PROMEDIO de Pearson r entre cruces clasificados\n'
               f'{field} · media aritmetica de r por cruce; sin ponderar por N')
        plot_map(mat,energies,ws,label,
                 out/'figures'/'summary'/f'{etype}_{field}_mean_r.png')
        plot_map(count,energies,ws,
                 f'{etype} · numero de cruces con r definido\n{field}',
                 out/'figures'/'summary'/f'{etype}_{field}_n_cases.png',kind='n')
        # Cada ventana mantiene exactamente 64 canales, incluso los NaN.
        valid_by_window=np.isfinite(mat).sum(axis=1)
        print(f'  {etype}/{field}: 64 canales graficados; '
              f'canales con promedio definido por ventana: '
              + ', '.join(f'{w:g}m={n}' for w,n in zip(ws,valid_by_window)),flush=True)

    group_file=out/'correlations_by_group_summary.csv'
    if group_file.is_file():
        per_group=pd.read_csv(group_file)
        for (group,etype,field),grp in per_group.groupby(
                ['group','event_type','field'],sort=True):
            mat,count=_summary_matrices(grp,energies,ws,'mean_pearson_r')
            plot_map(mat,energies,ws,
                     f'{group} · {etype} · PROMEDIO de Pearson r entre cruces\n{field}',
                     out/'figures'/'by_group'/f'{group}_{etype}_{field}_mean_r.png')
            plot_map(count,energies,ws,
                     f'{group} · {etype} · numero de cruces con r definido\n{field}',
                     out/'figures'/'by_group'/f'{group}_{etype}_{field}_n_cases.png',kind='n')


def summarize_per_boundary(per_event):
    # Cruces repetidos en el catálogo no deben influir dos veces en el promedio.
    per_event=per_event.drop_duplicates(['event_type','event_time_utc','field','window_minutes','energy_index'])
    gb=per_event.groupby(['event_type','field','window_minutes','energy_index','energy_ev'],sort=True,dropna=False)
    out=gb.agg(n_valid_events=('pearson_r','count'),
               median_pearson_r=('pearson_r','median'),
               mean_pearson_r=('pearson_r','mean'),
               n_total_events=('event_id','nunique')).reset_index()
    # n_pairs = cantidad de puntos por evento; no mezclamos al estimar r.
    return out


def main():
    ap=argparse.ArgumentParser(description='Mapa energia × ancho de ventana para correlacion SWEA-MAG en BS y MPB')
    ap.add_argument('--events',nargs='+',type=Path,required=True,
                    help='Uno o varios CSV, o directorio con los CSV originales MAVEN_MPB_Data_Group*.csv')
    ap.add_argument('--case-col',default=None)
    ap.add_argument('--event-type-col',default=None)
    ap.add_argument('--event-time-col',default=None)
    ap.add_argument('--bs-col',default=None)
    ap.add_argument('--mpb-col',default=None)
    ap.add_argument('--start',default=None,help='YYYY-MM-DD: primera fecha de evento')
    ap.add_argument('--end',default=None,help='YYYY-MM-DD: ultima fecha de evento')
    ap.add_argument('--max-events',type=int,default=None,help='Prueba rapida con N eventos')
    ap.add_argument('--windows-min',default=DEFAULT_WINDOWS_MIN,
                    help='Duraciones TOTALES en minutos; por defecto 2 a 60 en pasos de 1')
    ap.add_argument('--min-pairs',type=int,default=6)
    ap.add_argument('--fields',nargs='+',choices=FIELDS,default=['b_total_nt'])
    ap.add_argument('--exclude-mag-gap-filled',action='store_true',
                    help='Control: no usar muestras MAG previamente reconstruidas')
    ap.add_argument('--swea-table',default=SWEA_TABLE)
    ap.add_argument('--axes-table',default=AXES_TABLE)
    ap.add_argument('--mag-dir',type=Path,default=MAG_DIR)
    ap.add_argument('--output-root',type=Path,default=OUT_ROOT)
    ap.add_argument('--run-name',default=None)
    ap.add_argument('--max-scan-gib',type=float,default=15.)
    ap.add_argument('--estimate-only',action='store_true')
    ap.add_argument('--catalog-only',action='store_true',help='Validar y normalizar los CSV sin acceder a BigQuery')
    ap.add_argument('--check-mag',action='store_true',help='Comprobar los Parquet MAG necesarios sin consultar BigQuery')
    ap.add_argument('--diagnose-mag',action='store_true',help='Auditar cobertura temporal MAG y cache SWEA sin BigQuery')
    ap.add_argument('--plot-per-event',action='store_true',help='Generar además un PNG por cada cruce (muchos gráficos)')
    ap.add_argument('--plots-only',action='store_true')
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    if args.min_pairs<3: ap.error('--min-pairs debe ser >=3')
    if args.max_events is not None and args.max_events<1: ap.error('--max-events debe ser >=1')
    if args.max_scan_gib<=0: ap.error('--max-scan-gib debe ser positivo')
    ws=parse_windows(args.windows_min)
    fields=list(dict.fromkeys(args.fields))
    run=safe_name(args.run_name or ('MAVEN_MPB_4groups' if len(args.events)>1 or args.events[0].is_dir() else args.events[0].stem))
    out=args.output_root/run
    if args.plots_only:
        energy=pd.read_csv(out/'energy_grid.csv').sort_values('energy_index')['energy_ev'].to_numpy()
        config=json.loads((out/'analysis_config.json').read_text())
        saved_ws=np.asarray(config['windows_min'],dtype=float)
        saved_fields=config['fields']
        plot_summaries_only(out,energy,saved_ws,saved_fields,args.plot_per_event)
        print('Mapas regenerados sin BigQuery:',out/'figures')
        return
    events, catalog, issues, source_files=load_events(args.events,args)
    day_spans=day_intervals(events,float(ws.max()))
    print('='*74)
    print('CORRELACIÓN LOCAL SWEA–MAG EN CRUCES CLASIFICADOS (v8: 59 VENTANAS DE 2 A 60 MIN + CSV PROGRESIVO)')
    print(f'Eventos: {len(events)}; BS={int((events.event_type=="BS").sum())}; MPB={int((events.event_type=="MPB").sum())}')
    print('Días UTC con ventanas:',len(day_spans))
    if len(catalog):
        print('Filas originales clasificadas (antes de filtros por fecha):',len(catalog))
        print('Por grupo:',catalog.groupby('group').size().to_dict())
    print('Eventos únicos por tipo y timestamp:',len(events.drop_duplicates(['event_type','event_time_utc'])))
    print('Avisos de catálogo:',len(issues))
    if args.catalog_only:
        out.mkdir(parents=True,exist_ok=True)
        events.to_csv(out/'events_selected.csv',index=False)
        if len(catalog):catalog.to_csv(out/'crossings_catalog_normalized.csv',index=False)
        issues.to_csv(out/'catalog_issues.csv',index=False)
        print('Catálogo y auditoría guardados sin BigQuery:',out)
        return
    # Verificar PRIMERO que existan los MAG. La version anterior generaba
    # mapas enteramente vacios y dejaba un cache con B=NaN cuando faltaban.
    missing_mag=[mag_path(d,args.mag_dir) for d in day_spans
                 if not mag_path(d,args.mag_dir).is_file()]
    print(f'MAG tratados presentes: {len(day_spans)-len(missing_mag)}/{len(day_spans)} días requeridos')
    if missing_mag:
        print('MAG faltantes (hasta 12):')
        for p in missing_mag[:12]: print('  ',p)
        if len(missing_mag)>12:print(f'  ... y {len(missing_mag)-12} días adicionales')
    if args.check_mag and not args.diagnose_mag:
        print('Chequeo MAG finalizado. No se consultó BigQuery.')
        return
    if args.diagnose_mag:
        for day in day_spans:
            p=mag_path(day,args.mag_dir)
            if not p.is_file():
                continue
            mag=read_mag_neighborhood(day,args.mag_dir)
            ev_times=[]
            for t in events.event_time_utc:
                if day in day_intervals(pd.DataFrame({'event_time_utc':[t]}),float(ws.max())):
                    ev_times.append(t)
            cached=cache_file(out,day)
            swea_times=None
            if cached.is_file():
                frame=pd.read_parquet(cached,columns=['datetime_utc'])
                swea_times=frame['datetime_utc']
            else:
                print('  SWEA cache no encontrado para '+str(day),flush=True)
            diagnose_mag_alignment(day,mag,swea_times,ev_times)
        print('Diagnóstico local finalizado. No se consultó BigQuery.')
        return
    if missing_mag and not args.estimate_only:
        raise FileNotFoundError(
            f'Faltan {len(missing_mag)} archivos MAG de {len(day_spans)} días. '
            'Revisá --mag-dir (debe apuntar a la carpeta que contiene los directorios YYYY/MM) '
            'o generá los Parquet con 01g_MAG_apply_gap_treatment.py. '
            'No se consulta BigQuery ni se crean caches sin MAG.')
    print('Anchos de ventana totales (min):',', '.join(f'{w:g}' for w in ws))
    print('Campos:',', '.join(fields))
    print('MAG: 1.5s max entre muestras tratadas; gaps sin rellenar: NaN')
    client=bigquery.Client(project=args.swea_table.split('.')[0])
    sql=SQL_FLUX.format(swea_table=args.swea_table,axes_table=args.axes_table)
    estimates={}
    for d, (a,b) in day_spans.items():
        estimates[d]=int(client.query(sql,job_config=job_config(a,b,True)).total_bytes_processed or 0)
    total_bytes=sum(estimates.values()); gib=total_bytes/1024**3
    print(f'BigQuery: lectura estimada total ~{gib:.3f} GiB ({len(day_spans)} consultas, una por día).')
    if args.estimate_only:
        print('Sin descargar datos (--estimate-only).')
        return
    if gib>args.max_scan_gib:
        raise RuntimeError(f'La estimación {gib:.3f} GiB supera --max-scan-gib {args.max_scan_gib:g}.')
    energy=validate_axes(client,args.axes_table)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'energy_index':np.arange(64),'energy_ev':energy}).to_csv(out/'energy_grid.csv',index=False)
    events.to_csv(out/'events_selected.csv',index=False)
    if len(catalog):catalog.to_csv(out/'crossings_catalog_normalized.csv',index=False)
    issues.to_csv(out/'catalog_issues.csv',index=False)
    config={'events':[str(p) for p in source_files],'fields':fields,'windows_min':ws.tolist(),
            'min_pairs':args.min_pairs,'exclude_mag_gap_filled':args.exclude_mag_gap_filled,
            'swea_table':args.swea_table,'axes_table':args.axes_table,
            'max_interpolation_span_seconds':MAX_INTERPOLATION_SPAN_S}
    (out/'analysis_config.json').write_text(json.dumps(config,indent=2))
    # Guardamos solo trozos de SWEA que intervienen en alguna ventana, una vez
    # por dia, y sincronizamos MAG a los timestamps SWEA sin resamplear SWEA.
    for d,span in day_spans.items():
        read_or_make_synced(d,span,client,sql,args.mag_dir,out,args.overwrite,args.swea_table,args.axes_table)
    # Escribir cada correlación inmediatamente, sin construir una lista de
    # >4 millones de diccionarios (que agotaría ~6 GiB de RAM).
    event_stats=[]
    accumulator=StreamingSummaries()
    out_csv=out/'correlations_by_event.csv'
    tmp_csv=out/'correlations_by_event.csv.tmp'
    columns=['event_id','case_id','group','source_file','source_row',
             'event_type','event_time_utc','field','window_minutes',
             'energy_index','energy_ev','pearson_r','n_valid_pairs']
    n_records=0
    n_defined=0
    with tmp_csv.open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=columns)
        writer.writeheader()
        for ei,event in enumerate(events.itertuples(index=False),1):
            t=event.event_time_utc
            lo=t-pd.Timedelta(minutes=float(ws.max())/2)
            hi=t+pd.Timedelta(minutes=float(ws.max())/2)
            segments=[]
            cur=lo.floor('D')
            while cur<hi:
                day_df=pd.read_parquet(cache_file(out,cur.date()))
                if len(day_df):
                    day_df['datetime_utc']=pd.to_datetime(day_df['datetime_utc'],utc=True)
                    segments.append(day_df[(day_df.datetime_utc>=lo)&(day_df.datetime_utc<hi)])
                cur+=pd.Timedelta(days=1)
            if segments:
                merged=pd.concat(segments,ignore_index=True).sort_values('datetime_utc')
                if merged.datetime_utc.duplicated().any():
                    raise RuntimeError('SWEA duplicado cerca del evento '+event.event_id)
            else:
                merged=pd.DataFrame()
            n_usable=int(merged['mag_sync_valid'].sum()) if len(merged) else 0
            n_filled=int(merged['mag_uses_gap_filled_samples'].sum()) if len(merged) else 0
            event_stats.append({'event_id':event.event_id,'case_id':event.case_id,
                                'group':event.group,'source_file':event.source_file,'source_row':event.source_row,
                                'event_type':event.event_type,'event_time_utc':t,
                                'n_swea_max_window':len(merged),'n_valid_mag_max_window':n_usable,
                                'n_uses_mag_gap_filled':n_filled})
            maps=[]
            for field in fields:
                grid,number=pearson_map(merged,t,ws,field,args.min_pairs,args.exclude_mag_gap_filled)
                maps.append((field,grid))
                for wi,width in enumerate(ws):
                    for j in range(N_ENERGIES):
                        rv=grid[wi,j]
                        valid=bool(np.isfinite(rv))
                        writer.writerow({'event_id':event.event_id,'case_id':event.case_id,
                            'group':event.group,'source_file':event.source_file,'source_row':event.source_row,
                            'event_type':event.event_type,'event_time_utc':t.isoformat(),
                            'field':field,'window_minutes':float(width),'energy_index':j,
                            'energy_ev':float(energy[j]),'pearson_r':float(rv) if valid else '',
                            'n_valid_pairs':int(number[wi,j])})
                        n_records+=1
                        n_defined+=int(valid)
            accumulator.add_event(event,maps,ws)
            print(f'[{ei}/{len(events)}] {event.event_type} {event.case_id} @ {t}: '
                  f'SWEA {len(merged)}, MAG valido {n_usable}, MAG ya reconstruido {n_filled}',flush=True)
    os.replace(str(tmp_csv),str(out_csv))
    if n_records==0:
        raise RuntimeError('No hay filas de resultado.')
    pd.DataFrame(event_stats).to_csv(out/'event_alignment_summary.csv',index=False)
    summary=StreamingSummaries.to_frame(accumulator.total,energy)
    summary.to_csv(out/'correlations_by_boundary_summary.csv',index=False)
    group_summary=StreamingSummaries.to_frame(accumulator.groups,energy,with_group=True)
    group_summary.to_csv(out/'correlations_by_group_summary.csv',index=False)
    plot_summaries_only(out,energy,ws,fields,args.plot_per_event)
    print('='*74)
    print('Correlaciones definidas:',n_defined,'/',n_records)
    print('Resultados:',out)
    print('Mapas por grupo: figures/by_group/*_mean_r.png; mapas globales BS/MPB: figures/summary/*_mean_r.png')
    if args.plot_per_event:print('Por evento: figures/by_event/BS y figures/by_event/MPB')
    else:print('Para mapas individuales, añadir --plots-only --plot-per-event (sin BigQuery)')
    print('Resumen entre casos: figures/summary/ (MEDIA de r y cantidad de casos)')
    print('Auditoría: catalog_issues.csv | Catálogo: crossings_catalog_normalized.csv')
    print('Importante: las ventanas más pequeñas tienen pocos puntos SWEA.')

if __name__=='__main__':
    main()
