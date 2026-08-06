"""Robustness tests: the tool must never crash on pathological SCADA data.

Cases:
  1. Status codes that do not match the config -> everything classified
     as downtime (no operating records for wake / power-curve analysis).
  2. Missing wind-direction column.
  3. Zero/NaN power values everywhere (wrong units or dead channels).

Each case must produce a complete analysis and a report file.
Run:  python tests/test_robustness.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TMP = os.path.join(HERE, "_oem_tmp")
sys.path.insert(0, ROOT)

from windpcea import config as cfg_mod          # noqa: E402
from windpcea.analysis import run_analysis      # noqa: E402
from windpcea.report import build_html          # noqa: E402

SAMPLE = os.path.join(ROOT, "sample_data")


def base():
    d = pd.read_csv(os.path.join(SAMPLE, "scada_sample.csv"))
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d


def run_case(name, df, cfg_overrides=None):
    os.makedirs(TMP, exist_ok=True)
    p = os.path.join(TMP, f"case_{name}.csv")
    df.to_csv(p, index=False)
    out = os.path.join(TMP, f"out_{name}")
    os.makedirs(out, exist_ok=True)
    cfg = cfg_mod.load_config(os.path.join(SAMPLE, "config.json"))
    if cfg_overrides:
        cfg.update(cfg_overrides)
    r = run_analysis(cfg, p, outdir=out)
    html = build_html(r, out)
    assert os.path.exists(html) and os.path.getsize(html) > 10000, f"[{name}] report missing/small"
    wk = r["wake"]["sector_table"]
    assert "sector_deg" in wk.columns, f"[{name}] wake table lost its columns"
    print(f"  OK  {name:<32} rows={len(r['df']):>7} wake_rows={len(wk):>3} "
          f"P50={r['uncertainty']['p']['P50']:>9,.0f} MWh  report={os.path.basename(html)}")
    return r


def main():
    b = base()

    # 1) unknown status codes -> everything non-operating
    d1 = b.copy()
    d1["status_code"] = 999
    r1 = run_case("all_downtime", d1)
    assert (r1["df"]["flag"] != 0).all(), "[all_downtime] expected no operating rows"

    # 2) no direction column
    d2 = b.drop(columns=["nacelle_dir_deg"])
    run_case("no_direction", d2)

    # 3) zero power everywhere
    d3 = b.copy()
    d3["power_kw"] = 0.0
    run_case("zero_power", d3)

    # 4) all-NaN wind speed
    d4 = b.copy()
    d4["wind_speed_mps"] = np.nan
    run_case("no_wind", d4)

    # 5) tiny dataset (1 turbine, 2 days)
    d5 = b[b["turbine_id"] == "T01"].head(288)
    run_case("tiny", d5)

    # 6) non-standard column names (no 'power'/'wind' words) — fallback detection
    d6 = b.rename(columns={
        "timestamp": "Date/Time", "turbine_id": "Unit",
        "power_kw": "Output(MW)", "wind_speed_mps": "WS(m/s)",
        "nacelle_dir_deg": "Dir(deg)", "temp_c": "AmbT(C)",
        "status_code": "State"})
    d6["Output(MW)"] = d6["Output(MW)"] / 1000.0     # MW -> kW scaling
    run_case("custom_columns", d6)

    # 7) manual column mapping (truly arbitrary names)
    d7 = b.rename(columns={"timestamp": "TS", "turbine_id": "TID",
                           "power_kw": "PWR", "wind_speed_mps": "VSPD"})
    run_case("column_map_override", d7,
             {"column_map": {"timestamp": "TS", "turbine": "TID",
                             "power": "PWR", "ws": "VSPD"}})

    print("\nAll robustness tests passed ✓")


if __name__ == "__main__":
    main()
