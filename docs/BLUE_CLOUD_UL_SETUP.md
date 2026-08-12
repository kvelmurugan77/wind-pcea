# Blue Cloud — UL/OEPR-style Assessment Setup Guide

This guide sets up WindPCEA v1.5.2 to replicate the UL OEPR methodology
(Ref. PR-024699, UL Services Group, 12 July 2022) for the Blue Cloud wind
plant (43 × Vestas V126-3.45 MW, 148.35 MW, COD 2018, Texas/NM border).

## 1. Files you need

| File | Source | Notes |
|---|---|---|
| SCADA (CSV/XLSX) | CIP/Vestas/RWE | Nov 2018 → Mar 2022 (40 months), 10-min |
| Metered production | "Blue Cloud Metered Production.xlsx" | Used by UL as the net-energy basis |
| Warranted power curve | Vestas V126-3.45 MW | `wind_speed_mps, power_kw` → save as `vestas_v126_3450.csv` |
| Layout | UL OEPR Figure 8.2 (or OEM layout DWG) | `turbine, x, y` (metres) → save as `blue_cloud_layout.csv` (a placeholder is included; replace with the real coordinates) |

## 2. Configuration (already prepared: `sample_data/blue_cloud_config.json`)

```json
{
  "farm_name": "Blue Cloud Wind Project",
  "latitude": 34.039753, "longitude": -102.651647,
  "hub_height_m": 87.0, "rotor_diameter_m": 126.0,
  "rated_power_kw": 3450.0, "num_turbines": 43,
  "site_elevation_m": 1182.0,
  "warranted_power_curve": "vestas_v126_3450.csv",
  "layout_file": "blue_cloud_layout.csv",
  "long_term_source": "era5",
  "lt_primary_method": "ul",
  "excluded_months": ["2021-02", "2022-03"],
  "lt_density_correction": true,
  "future_losses": {"availability_pct": 3.0, "curtailment_pct": 1.4,
                    "electrical_pct": 2.0, "blade_degradation_pct": 1.5},
  "status_codes": {"operating": [0, 1], "fault": [4],
                   "maintenance": [], "grid": [],
                   "curtailment": [], "environmental": []}
}
```

Notes:
- `lt_primary_method: "ul"` → the LT gross uses the **UL/OEPR Step-1 monthly
  production regression** (30-day normalisation Eq. 3, density-corrected
  reference wind Eq. 4, OLS Eq. 5, de-normalisation Eq. 6).
- `excluded_months` reproduces UL's exclusions (Feb-2021 ice storm,
  Mar-2022 meter error). Add more if your data shows similar issues.
- `future_losses` reproduces UL's LT net stack (future availability,
  curtailment 1.4%, electrical, blade degradation).
- The **ERA5T reference** (2000–2025) is fetched automatically from the
  coordinates; UL used ERA-5 + MERRA-2. For an even closer match, supply a
  local ERA-5/MERRA-2 monthly file via `long_term_wind_file`.
- Check the status-code mapping against your SCADA (Envision-style exports
  use 0=standby, 1=operating, 4=fault — adjust if Vestas uses other codes).

## 3. Run

CLI:
```bash
python -m windpcea.cli --config sample_data/blue_cloud_config.json \
    --scada blue_cloud_scada.csv --outdir blue_cloud_results
```

Web app / EXE:
1. Launch WindPCEA (footer must read v1.5.2)
2. Fill farm name, rated power 3450, hub 87, lat 34.039753, lon -102.651647
3. Upload SCADA + config JSON (the file above) + warranted curve + layout
4. Run assessment → report opens; download HTML/Excel/CSV from the toolbar

## 4. How the output maps to UL's report

| UL OEPR value | Where it appears in the tool |
|---|---|
| LT gross annual production (631.3 GWh/yr) | §7 loss tree → Gross AEP (with `lt_primary_method: ul`) |
| Net P50 (607.1 GWh/yr) | §1 Key results → Net AEP P50 (future-loss stack applied) |
| Net capacity factor (46.7%) | §1 Key results → Capacity factor |
| P75 / P90 / P99 | §9 P-values (exceedance convention) |
| Energy-based availability (IEC 61400-26) | §3 Availability |
| Monthly gross normalisation (Eq. 3) | §7.2 UL/OEPR Step-1 table |
| MCP R² (MERRA-2 0.898 / ERA-5 0.939) | §7 MCP statistics (ERA5T R²) |
| Uncertainty (6.9% annual / 3.2% 10-yr) | §9 Uncertainty components |

## 5. DAKKS-readiness checklist

- [ ] OEM warranted power curve supplied (removes the generic-curve caveat)
- [ ] Real turbine layout used for the wake model (replace placeholder)
- [ ] Status-code mapping verified against the SCADA
- [ ] Excluded months documented (with reasons)
- [ ] Long-term reference source + period documented (ERA5T auto-fetch; UL used ERA-5 + MERRA-2)
- [ ] Future-loss assumptions documented and traceable to the config
- [ ] Appendix E traceability section included in the report (automatic)
