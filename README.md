# Procesamiento de series temporales de MAVEN

Análisis de series temporales de la misión **Mars Atmosphere and Volatile EvolutioN (MAVEN)**, desarrollado para el trabajo final de **Procesamiento de Series Temporales (UBA)** y como parte del estudio de la interacción entre el viento solar y Marte.

El repositorio reúne procedimientos de control de calidad y preprocesamiento, análisis de cobertura espacial y correlaciones entre las mediciones del magnetómetro **MAG** y del analizador de electrones **SWEA**. El análisis de fronteras se concentra en el *bow shock* (**BS**) y la *magnetic pileup boundary* (**MPB**).

## Notebook integrador

El cuaderno **[00_Informe_integrador_MAVEN.ipynb](notebooks/00_Informe_integrador_MAVEN.ipynb)** reúne el trabajo completo hasta esta etapa: control de calidad de MAG, gaps y pseudogaps, política de interpolación y validación, SWEA, cobertura espacial y correlaciones MAG–SWEA para BS/MPB. Incluye los resultados numéricos ya archivados y celdas para visualizar los CSV y PNG existentes.

**No ejecuta consultas de BigQuery automáticamente.** En la sección **7.0** se puede elegir `MODO_CORRELACIONES = "solo_local"` (predeterminado: valida y audita el catálogo, carga resúmenes existentes y regenera gráficos locales), `"estimar"` (dry-run de BigQuery para SWEA) o `"completo"` (calcula Pearson BS/MPB en las 59 ventanas de 2–60 minutos, 64 energías SWEA; **consulta BigQuery y puede consumir cuota y generar costos**). Para usar los resultados en esta PC cuando faltan los CSV, primero cambiar a `"estimar"` y después, si es viable, a `"completo"`. La correlación utiliza el catálogo normalizado del repositorio y preserva los grupos G1–G4. La sección **7.2–7.7 genera y muestra directamente dentro de Jupyter** los mapas BS/MPB de Pearson y de cobertura, curvas por energía y duración y los mapas de los grupos G1–G4 (además de guardar los PNG en `informe_notebook/`); **requiere que el CSV de resultados Pearson exista**, obtenido desde BigQuery con el modo `"completo"` o recuperado de una ejecución anterior. En su sección **8.1.a**, si falta la nueva tabla pareada wavelet, ejecuta **localmente** el script CWT para los cruces MPB usando los Parquet MAG y luego reutiliza esos archivos en las ejecuciones siguientes. La opción `EJECUTAR_WAVELETS = "nunca"` desactiva este recálculo. Al finalizar se muestran las comparaciones antes/después y la cobertura, y se crea un ZIP con las wavelets y **también los resultados de Pearson si están disponibles**, sin incluir los Parquet ni el enorme CSV de correlaciones por evento. Para abrirlo, ejecutá `jupyter lab` desde la raíz del repositorio y seleccioná el notebook dentro de `notebooks/`.

## Organización

| Directorio | Contenido |
| --- | --- |
| `scripts/mag/` | Control de calidad MAG, caracterización y tratamiento de gaps y validación posterior. |
| `scripts/swea/` | Caracterización de la cadencia temporal de SWEA por energía. |
| `scripts/positions/` | Cobertura espacial y orbital de MAVEN. |
| `scripts/correlations/` | Correlaciones locales MAG–SWEA alrededor de cruces BS/MPB clasificados. |
| `scripts/wavelets/` | CWT Morlet de MAG en ventanas de 10 minutos alrededor de cada MPB clasificada. |
| `results/` | Tablas y figuras generadas por los análisis. |
| `data/` | Datos de entrada y archivos Parquet procesados, disponibles localmente según el análisis. |

Los scripts son ejecutables individuales; **no existe un único comando que ejecute toda la cadena**. Se recomienda ejecutarlos desde la raíz del repositorio para conservar las rutas relativas.

## Etapas del trabajo

### 1. Control de calidad y tratamiento de gaps de MAG

Los archivos en `scripts/mag/` cubren distintas etapas:

| Script | Propósito |
| --- | --- |
| `01_MAG_quality_control.py` | Resumen de calidad, cadencia y cobertura de MAG a partir de BigQuery. |
| `01b_MAG_gap_analysis.py` | Identificación y estadística de gaps internos en archivos originales. |
| `01c_MAG_boundary_analysis.py` | Continuidad temporal entre archivos y gaps en sus límites. |
| `01d_MAG_gap_interpolation_analysis.py` | Diagnóstico de gaps y variabilidad local; no modifica la serie. |
| `01e_MAG_pseudogap_validation.py` | Validación de técnicas de reconstrucción mediante pseudogaps. |
| `01f_MAG_gap_treatment_policy.py` | Clasificación de gaps reales y selección de los que pueden interpolarse. |
| `01g_MAG_apply_gap_treatment.py` | Aplicación de la política validada y generación de MAG tratado. |
| `01h_MAG_post_treatment_QC.py` | Verificación de integridad temporal, interpolación y trazabilidad. |

La serie tratada conserva los timestamps originales y agrega únicamente las muestras reconstruidas autorizadas por la política de tratamiento. No regulariza artificialmente toda la misión a una grilla de segundos enteros.

Para el análisis MAG–SWEA se espera encontrar los Parquet MAG en:

```text
data/processed/MAG_gap_treated/YYYY/MM/mag_YYYY-MM-DD.parquet
```

### 2. Caracterización de SWEA

`scripts/swea/02a_SWEA_cadence_by_energy.py` estudia la disponibilidad temporal de las mediciones de SWEA en cada canal de energía. La cadencia se estima a partir de las diferencias de timestamps válidos; no se impone una cadencia constante a todos los canales.

### 3. Cobertura espacial de MAVEN

- `scripts/positions/03a_MAVEN_position_spatial_coverage.py`: resumen de cobertura, altitud y proyecciones espaciales, incluyendo estadísticas por año.
- `scripts/positions/03b_MAVEN_frontal_by_month.py`: mapas de cobertura frontal Y–Z por mes y por año, utilizando posiciones proyectadas sobre ese plano (no un corte en X = 0).

### 4. Correlación MAG–SWEA alrededor de BS y MPB

Script principal:

```text
scripts/correlations/04a_boundary_energy_correlation_maps.py
```

**Objetivo:** estudiar la dependencia de la correlación entre el campo magnético y el flujo electrónico de SWEA con la energía y con la duración de la ventana temporal alrededor de cada cruce clasificado.

Para cada evento se calcula el coeficiente de Pearson entre una componente de MAG y el logaritmo decimal del flujo SWEA medio angular en cada energía:

```text
r(E, Δt) = corr( B(t), log10[J(E,t)] )
```

Por defecto se usa el módulo `|B|` (`b_total_nt`). Se pueden seleccionar también `bx_ss_nt`, `by_ss_nt` o `bz_ss_nt`.

**Configuración actual por defecto:**

- **59 ventanas de duración total:** 2, 3, 4, ..., 60 minutos.
- Ventanas centradas en el instante clasificado de cada BS o MPB.
- **64 canales de energía** de SWEA, representados en eV sobre un eje logarítmico.
- **Mínimo de 6 pares válidos** por coeficiente; correlaciones no definidas se guardan como NaN.
- Promedio aritmético **no ponderado por cantidad de muestras** de las correlaciones individuales, separado para BS y MPB.
- Desduplicación de eventos que comparten el mismo tipo de frontera y timestamp en los mapas promedio.
- Mapas adicionales con el número de cruces válidos por celda y resúmenes por grupo.

El script utiliza los **cruces identificados y clasificados manualmente** en los catálogos de eventos; no reemplaza estas etiquetas por intersecciones con un modelo geométrico de fronteras.

#### Datos necesarios

**Catálogo de cruces:** ya se incluye en el repositorio [`data/classified_crossings/crossings_catalog_normalized_4groups.csv`](data/classified_crossings/crossings_catalog_normalized_4groups.csv), con **569 casos** de los cuatro grupos manualmente clasificados (G1–G4) y las fechas BS/MPB en UTC. También se pueden utilizar los cuatro CSV originales `MAVEN_MPB_Data_Group*.csv` o catálogos personalizados en formatos long/wide. Se preservan las fechas y la procedencia del catálogo, y se registran advertencias sobre inconsistencias o duplicados. **Para evitar duplicar eventos, no mezclar el catálogo normalizado y los cuatro originales en un mismo análisis.**

**MAG:** archivos Parquet locales tratados, en `data/processed/MAG_gap_treated/`.

**SWEA:** tablas de BigQuery:

```text
maven-vignes-crossings.maven.swea_svy3d
maven-vignes-crossings.maven.swea_svy3d_axes
```

El script reduce angularmente los flujos SWEA en BigQuery y sincroniza MAG con los timestamps originales de SWEA. Para MAG se utiliza interpolación lineal únicamente entre muestras tratadas cuya separación no exceda **1,5 s**; no se extrapola ni se atraviesan gaps rechazados. El módulo `|B|` se calcula a partir de las tres componentes interpoladas.

#### Instalación y configuración

Desde la raíz del repositorio:

```bash
python3 -m pip install -r requirments.txt
```

> El archivo de dependencias actualmente se llama `requirments.txt` (con esa ortografía).

Para las consultas a BigQuery se necesitan credenciales de Google Cloud con acceso a las tablas. Una opción para configurar las credenciales locales, si se dispone de `gcloud`, es:

```bash
gcloud auth application-default login
```

Las consultas pueden consumir cuota y generar costos de BigQuery.

#### Ejecución

Validar el catálogo sin acceder a BigQuery:

```bash
python3 scripts/correlations/04a_boundary_energy_correlation_maps.py \
  --events data/classified_crossings/ \
  --catalog-only
```

Comprobar los archivos MAG locales:

```bash
python3 scripts/correlations/04a_boundary_energy_correlation_maps.py \
  --events data/classified_crossings/ \
  --check-mag
```

Estimar el volumen de datos a consultar:

```bash
python3 scripts/correlations/04a_boundary_energy_correlation_maps.py \
  --events data/classified_crossings/ \
  --estimate-only
```

Ejecutar el análisis de **2 a 60 minutos**, con paso de **1 minuto**:

```bash
python3 scripts/correlations/04a_boundary_energy_correlation_maps.py \
  --events data/classified_crossings/ \
  --max-scan-gib 150
```

El argumento `--max-scan-gib` es un **límite de seguridad interno del script**, no modifica las cuotas ni el presupuesto de Google Cloud. Su valor por defecto es 15 GiB; ajustarlo requiere considerar la estimación y la cuota disponible.

Para ejecutar una prueba limitada sin reemplazar los resultados del análisis principal:

```bash
python3 scripts/correlations/04a_boundary_energy_correlation_maps.py \
  --events data/classified_crossings/ \
  --max-events 10 \
  --run-name prueba_10_eventos \
  --max-scan-gib 150
```

Se pueden definir otras duraciones mediante `--windows-min`, expresadas **en minutos y separadas por comas**. Por ejemplo, `--windows-min 2,5,10,20,30,60`.

#### Reutilización de datos y resultados

La ejecución habitual guarda los datos SWEA–MAG sincronizados por día UTC en:

```text
results/MAG_SWEA_boundary_correlation/MAVEN_MPB_4groups/synced/YYYY/MM/
```

Si un Parquet cacheado cubre la ventana temporal solicitada y pasa las verificaciones de procedencia, el script puede reutilizarlo sin repetir la consulta SWEA. **No utilizar `--overwrite` para reanudar una ejecución.** Al ampliar el máximo de las ventanas puede ser necesario consultar nuevamente algunos días cuya cobertura cacheada sea insuficiente.

La ejecución escribe las correlaciones por evento progresivamente en un CSV temporal y lo sustituye por el CSV definitivo al finalizar. Esto evita acumular todas las filas individuales en RAM; **no implica que los cálculos de correlación parcialmente completados se reanuden a mitad de un evento**.

La carpeta de resultados principal es:

```text
results/MAG_SWEA_boundary_correlation/MAVEN_MPB_4groups/
├── analysis_config.json
├── energy_grid.csv
├── events_selected.csv
├── crossings_catalog_normalized.csv
├── catalog_issues.csv
├── event_alignment_summary.csv
├── correlations_by_event.csv
├── correlations_by_boundary_summary.csv
├── correlations_by_group_summary.csv
├── synced/
└── figures/
    ├── summary/
    │   ├── BS_b_total_nt_mean_r.png
    │   ├── BS_b_total_nt_n_cases.png
    │   ├── MPB_b_total_nt_mean_r.png
    │   └── MPB_b_total_nt_n_cases.png
    └── by_group/
```

Los mapas de `figures/summary/` representan el **promedio de Pearson entre cruces**, no la correlación de todos los puntos concatenados. Las figuras individuales pueden solicitarse con `--plot-per-event` o generarse posteriormente, usando los resultados guardados:

```bash
python3 scripts/correlations/04a_boundary_energy_correlation_maps.py \
  --events data/classified_crossings/ \
  --plots-only --plot-per-event
```

`--plots-only` no consulta BigQuery y usa la configuración guardada en `analysis_config.json`.

### 5. Wavelets de MAG alrededor de la MPB (10 minutos)

El script **[`05a_MPB_MAG_wavelet_10min.py`](scripts/wavelets/05a_MPB_MAG_wavelet_10min.py)** implementa una transformada wavelet continua (**CWT**) compleja Morlet sobre `|B|` (o `Bx`, `By`, `Bz`) alrededor de **cada cruce MPB clasificado manualmente**.

Se utiliza una ventana **total de 10 minutos**, desde **5 minutos antes hasta 5 minutos después** de la MPB. La CWT se calcula por defecto con 48 períodos logarítmicos entre **4 y 120 s** sobre una grilla de 1 s. Se elimina una tendencia lineal por segmento y se normaliza la potencia por la varianza del residuo para comparar cruces.

**Control de gaps y bordes:** el script toma los archivos MAG Parquet ya tratados, separa los tramos con discontinuidades mayores a 1,5 s, interpola solo dentro de tramos continuos y aplica una máscara aproximada del **cono de influencia (COI)** a cada segmento. No interpola gaps rechazados. Las muestras MAG previamente reconstruidas pueden excluirse con `--exclude-interpolated`.

Prueba limitada, con resultados en una subcarpeta independiente:

```bash
python3 scripts/wavelets/05a_MPB_MAG_wavelet_10min.py \
  --events data/classified_crossings/ \
  --max-events 5 --run-name prueba_5_mpb --max-plots 5
```

Análisis completo, con una selección de 12 escalogramas individuales:

```bash
python3 scripts/wavelets/05a_MPB_MAG_wavelet_10min.py \
  --events data/classified_crossings/ --max-plots 12
```

No es necesario BigQuery para esta etapa; sí se necesitan los **Parquet MAG locales**, el CSV de cruces (incluido en GitHub) y `PyWavelets` (ya incluido en `requirments.txt`).

Verificar la implementación sobre una onda sintética de período conocido y un gap artificial:

```bash
python3 -m unittest discover -s tests -p "test_mpb_wavelet.py" -v
```


**Salidas:** `results/MAG_MPB_wavelets_10min/` contiene `mpb_wavelet_event_summary.csv`, `mpb_wavelet_period_by_event.csv`, **`mpb_wavelet_before_after_by_event.csv`** (comparación pareada por período, mínimo 20 tiempos válidos por lado), `mpb_wavelet_mean_map.csv`, `analysis_config.json` y `figures/mpb_mean_wavelet_power.png` junto con ejemplos `figures/by_event/`. Al ejecutar todas las celdas del notebook se crea también `results/MAG_MPB_wavelets_10min/informe_notebook/` con tablas y figuras de cobertura, comparación pareada por período, bandas 4–12/12–30/30–60/60–120 s y sensibilidad a BS, además de **`MPB_wavelets_resultados_para_compartir.zip`** (figuras y CSV, sin Parquet MAG). La columna `n_valid_events` permite conocer cuántos cruces contribuyen a cada celda del mapa promedio; `bs_within_window` advierte cuándo también se encuentra el BS clasificado dentro de la ventana MPB.

El análisis de potencia wavelet es **exploratorio**: las transiciones de nivel del campo en la MPB pueden generar potencia amplia en escalas temporales sin que haya una onda física coherente. No se ha implementado un contraste de significancia frente a ruido rojo.

## Consideraciones metodológicas

- **Una correlación no demuestra causalidad.** Los cambios simultáneos del campo y del flujo pueden estar dominados por la transición entre regiones y no necesariamente por fluctuaciones acopladas.
- **Ventanas largas:** hasta 60 minutos alrededor de una frontera pueden incluir también la otra (BS o MPB). Actualmente **no se excluyen automáticamente** esos eventos; es necesario controlar esta situación antes de atribuir la correlación a una sola frontera.
- **Cobertura y significancia:** revisar `*_n_cases.png` y `n_valid_pairs` antes de interpretar patrones de `mean_r`, especialmente en energías con pocas mediciones válidas.
- **Promedio estadístico:** la media de los coeficientes individuales es descriptiva. No representa por sí sola un contraste de hipótesis ni una estimación de significancia estadística.
- **Datos y trazabilidad:** las series MAG tratadas conservan banderas de original/interpolado; puede repetirse el análisis excluyendo muestras MAG reconstruidas con `--exclude-mag-gap-filled`.

## Estado y próximas extensiones

El repositorio documenta las herramientas disponibles para preparar, caracterizar y comparar series temporales de MAVEN. El análisis CWT Morlet de MAG cerca de la MPB ya está implementado. Como próximas extensiones quedan el análisis de autocorrelación, la densidad espectral de potencia (Welch), STFT, la comparación de escalas entre MAG y SWEA y la correlación cruzada con retardos temporales. Estas extensiones todavía no están implementadas en el repositorio.
