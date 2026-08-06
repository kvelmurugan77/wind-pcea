# WindPCEA — Post-Construction Energy Yield Assessment

> **Portfolio site:** https://velmurugankaruppiah.github.io/wind-pcea/ · **Live sample report** included

A SCADA-based wind farm energy yield assessment tool in the style of commercial
(DNV-like) post-construction analyses. It performs the full workflow from raw
10-minute SCADA to a net annual energy yield with P-values, and produces a
self-contained HTML report, an Excel workbook and CSV exports.

**Highlights**
- ⚡ Full DNV-style pipeline: data QC → availability → power curve → wake → MCP → loss tree → Monte Carlo P-values
- 🏭 8 OEM SCADA profiles auto-detected: Vestas, Siemens Gamesa, Suzlon, Envision, Nordex, Goldwind, Inox, generic
- 📊 IEC 61400-12-1 binning, IEC 61400-26 availability categories, WindPRO-style wind rose
- 🌐 Long-term correction via user file or automatic NASA POWER (MERRA-2) fetch
- 📄 Self-contained HTML report (appendices with full calculation details) + Excel workbook + CSVs
- 🖥️ Web UI and CLI

## What it does (analysis pipeline)

| Step | Method |
|---|---|
| **Data QC** | SCADA resampling to 10-min; per-record operating-state classification (IEC 61400-26 style): operating, below cut-in, downtime, curtailment, derating, environmental, bad data, anemometer fault; capture-rate reporting |
| **Availability & losses** | Time-based and production-based availability per turbine; lost energy per category estimated from the warranted power curve at measured wind speed; downtime split by cause from SCADA status codes; monthly production table |
| **Power curve** | IEC 61400-12-1 0.5 m/s binning of operating, non-curtailed data; air-density correction to standard conditions; energy-weighted deviation vs the warranted curve per turbine and for the farm; degraded turbines are automatically highlighted |
| **Wake analysis** | Reference-turbine method: per 30° sector the least-waked turbines define the free-stream wind speed; per-turbine deficits per sector; wake energy loss from the warranted curve at free-stream vs nacelle speed |
| **Long-term correction (MCP)** | Sector-wise linear regression of site daily means on a long-term reference — user file, or NASA POWER (MERRA-2) reanalysis fetched automatically from latitude/longitude; long-term Weibull (shape from the measured record, scale adjusted to the long-term mean) |
| **LT gross AEP — two methods** | **Method A:** long-term Weibull × warranted power curve (integral). **Method B:** production regression — measured daily/monthly gross energy vs wind speed is regressed (E = c·v³ daily / E = a + b·WS monthly) and applied to the full long-term wind record, giving the LT energy series and annualised gross AEP. Both are computed, charted (scatter + fit + LT predicted series) and compared; Method A is primary by default (`lt_primary_method: method_b` switches to B for long PORs) |
| **Loss tree** | Gross AEP from the long-term Weibull × warranted curve; losses (availability, curtailment, derating, environmental, wake, turbine performance, electrical, other) applied multiplicatively; reconciliation of the modelled vs metered energy over the measurement period |
| **Uncertainty** | Monte Carlo (default 20,000 draws) of lognormal 1σ components → P50 / P75 / P90 / P99, 80% CI on P50, tornado chart of contributions |
| **Benchmark** | Assessed P50 vs pre-construction P50 (if provided) |

## Quick start

```bash
pip install -r requirements.txt

# 1) generate a realistic synthetic SCADA dataset (12 × 2.5 MW, Tamil Nadu climate)
python sample_data/generate_sample.py

# 2) run the assessment
python -m windpcea.cli --config sample_data/config.json \
    --scada sample_data/scada_sample.csv --outdir results

# 3) open the report
results/pceya_report.html
```

Outputs written to `results/`:

- `pceya_report.html` — self-contained report (inline CSS + charts, no internet needed)
- `pceya_results.xlsx` — all tables: summary, availability, monthly, power curve,
  wake sectors, MCP stats, long-term daily series, loss tree, uncertainty, P-values, QC
- `flagged_scada_10min.csv` — cleaned data with operating-state flags
- `farm_power_curve.csv`, `per_turbine_metrics.csv`

## GitHub Pages site

`docs/` contains a self-contained portfolio landing page and a full sample
assessment report. To publish on GitHub Pages:

1. Push this repository to GitHub (see below).
2. GitHub → repository **Settings → Pages** → *Source: Deploy from a branch*
   → branch `main`, folder `/docs` → Save.

The site will be live at
`https://<username>.github.io/wind-pcea/`.

Rebuild the site locally after regenerating the sample report:
`python scripts/build_docs.py`

## Web application

```bash
python app.py          # http://localhost:8000
```

Upload SCADA (CSV/XLSX, long or wide format), optionally a config JSON, warranted
power curve and long-term wind file, or run the bundled sample. The report is
viewed in the browser and all outputs can be downloaded.

## SCADA data format

Long format (preferred) — column names are matched fuzzily:

| column | description |
|---|---|
| `timestamp` | 10-min (or finer) records; resampled to 10-min automatically |
| `turbine_id` | turbine identifier (T01 …) |
| `power_kw` | active power (kW) |
| `wind_speed_mps` | nacelle wind speed (m/s) |
| `nacelle_dir_deg` (optional) | direction, 0–360° |
| `status_code` (optional) | operating state code (mapping in config `status_codes`) |
| `temp_c` (optional) | ambient temperature, enables air-density correction & environmental flags |
| `curtailment_flag` (optional) | explicit curtailment flag |

Wide format is also supported: `T01_power_kw`, `T01_wind_speed_mps`, …

## Configuration (JSON)

```json
{
  "farm_name": "My Farm",
  "latitude": 9.0, "longitude": 77.8,
  "hub_height_m": 100, "rotor_diameter_m": 136,
  "rated_power_kw": 2500, "cut_in_mps": 3.0, "cut_out_mps": 25.0,
  "warranted_power_curve": "warranted_power_curve.csv",
  "long_term_wind_file": "long_term_daily_ws.csv",
  "long_term_source": "auto",
  "air_density_correction": true, "air_pressure_kpa": 99.2,
  "electrical_loss_pct": 2.0, "other_loss_pct": 0.5,
  "preconstruction_p50_gwh": 92.0,
  "status_codes": {"operating": [100], "fault": [200], "maintenance": [300],
                   "grid": [400], "curtailment": [500], "environmental": [600]},
  "uncertainty_overrides_pct": {"wind_resource": 6.0},
  "mc_iterations": 20000
}
```

`long_term_source`: `auto` | `file` | `nasa_power` | `measured_only`.
Without a latitude/longitude or a long-term file, the measured record is used
with an increased wind-resource uncertainty (10 %).

## OEM SCADA compatibility

Exports from the major manufacturers are parsed automatically (column-name
aliases, date formats, units, status conventions). Set `"oem_profile"` in the
config to force a profile; otherwise it is auto-detected:

| OEM | Handled conventions |
|---|---|
| **Vestas** | `Timestamp`, `Turbine Name`, `Active Power (kW)`, `Wind Speed (m/s)`, `Nacelle Position (deg)`, text state codes (`Running`, `Fault`, `Maintenance`, `Grid Loss`…) |
| **Siemens Gamesa** | Compact headers (`ActivePower(kW)`, `WindSpeed(m/s)`, `NacellePosition`, `TurbineState`), numeric or text states |
| **Suzlon** | `Date Time`, `WTG No`, `Gen Active Power (kW)`, `Turbine Status`, dd/mm/yyyy dates |
| **Envision** | Semicolon-delimited CSVs, `dd.mm.yyyy` dates, European decimal commas, `Device Name` |
| **Nordex** | Separate `Date` + `Time` columns, `WEC`, `P-avg/V-avg/D-avg/T-avg` |
| **Goldwind** | Chinese headers (时间 / 机组号 / 有功功率 / 风速 / 机舱位置 / 机组状态) and Chinese statuses (运行, 故障, 维护, 限电…) |
| **Inox** | `Turbine ID`, `Active Power (MW)` (auto-scaled to kW) |
| **Generic** | Long or wide format (`T01_power_kw` …), units in column names, numeric/any text statuses |

Also handled for all profiles: metadata preamble rows before the header,
multi-sheet Excel (largest sheet used), 1/5/10/15-min or hourly data (resampled
to 10 min), power in MW auto-scaling, wind speed in km/h.

Test suite: `python tests/test_oem_scada.py` (generates realistic OEM-style
exports for all profiles and verifies parsing).

## Running the tool

**Windows EXE (no Python needed, OpenWind-style):**
- Download `WindPCEA.exe` from the latest build artifact: repo → **Actions** → *Build Windows EXE* → latest run → artifact download (or build it yourself with `build_exe.bat`).
- Double-click the EXE — it starts the local web app and opens your browser automatically.
- The GitHub Actions workflow builds the EXE in the cloud on every push.

**Web application:**
```bash
python app.py          # → http://localhost:8000
```
Upload SCADA (CSV/XLSX) + optional config JSON, warranted power curve and
long-term wind file — or click “Run with bundled sample data”. The report
opens in the browser; Excel and CSV outputs are downloadable.

**Command line:**
```bash
python -m windpcea.cli --config config.json --scada scada.csv --outdir results
```

**Try it immediately:**
```bash
python sample_data/generate_sample.py          # build a realistic 30 MW demo dataset
python -m windpcea.cli --config sample_data/config.json \
    --scada sample_data/scada_sample.csv --outdir results
```

**Inputs**
- SCADA file: long format (`timestamp, turbine_id, power_kw, wind_speed_mps,
  nacelle_dir_deg, status_code, temp_c, curtailment_flag`) or wide format
  (`T01_power_kw, T01_wind_speed_mps, …`); column names are matched fuzzily.
- Warranted power curve CSV (`wind_speed_mps, power_kw`) — optional; a generic
  curve is synthesised from the config otherwise.
- Long-term daily wind file (`date, ws_mps [, dir_deg]`) — optional; NASA
  POWER (MERRA-2) reanalysis is fetched automatically when latitude/longitude
  are configured.
- Config JSON — see `sample_data/config.json`; key fields: `status_codes`,
  `electrical_loss_pct`, `other_loss_pct`, `preconstruction_p50_gwh`,
  `uncertainty_overrides_pct`, `oem_profile`, `column_aliases`.

**Outputs** (written to the output directory)
- `pceya_report.html` — self-contained report (inline CSS, base64 charts,
  appendices with calculation details: A availability, B long-term yield,
  C how to run, D methodology)
- `pceya_results.xlsx` — summary + all analysis tables
- `flagged_scada_10min.csv`, `farm_power_curve.csv`, `per_turbine_metrics.csv`

## Notes & limitations

- This is an engineering tool implementing standard industry practice
  (IEC 61400-12-1 binning, IEC 61400-26 availability categories, MCP,
  Monte Carlo P-values). It is not a certified assessment.
- Memory: the loader streams CSVs in chunks, but the analysis pipeline needs
  the flagged data in memory (~1-2 GB per million 10-min records at float64).
  For very large exports set `"use_float32": true` in the config to halve this.
- The production-regression method (B) is most reliable with a multi-year
  period of record covering all seasons; for shorter records it is reported
  as an indicative cross-check.
- Wind speeds come from nacelle anemometry (rotor-affected); a met mast or
  lidar reference would refine the power curve and wake analysis.
- Losses measured over the SCADA period are assumed to carry forward to the
  long-term (stationarity).
- The uncertainty model uses lognormal multipliers with default 1σ values
  (wind resource ~3–10% depending on MCP quality, power curve 2%, wake from
  analysis, availability from analysis, …). Override via
  `uncertainty_overrides_pct` when better knowledge exists.
