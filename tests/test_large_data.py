"""Large-data tests: chunked streaming, memory-bounded loading.

  1. Equivalence: chunked read (small chunk size, forced multi-chunk)
     gives identical results to the default read.
  2. Stress: a ~1M-row SCADA file loads and analyses end-to-end.
  3. usecols filtering: a wide file with many junk channels loads correctly.

Run:  python tests/test_large_data.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TMP = os.path.join(HERE, "_oem_tmp")
sys.path.insert(0, ROOT)

from windpcea import scada  # noqa: E402
from windpcea import config as cfg_mod  # noqa: E402
from windpcea.analysis import run_analysis  # noqa: E402

SAMPLE = os.path.join(ROOT, "sample_data")


def main():
    os.makedirs(TMP, exist_ok=True)
    sample = os.path.join(SAMPLE, "scada_sample.csv")

    # ---- 1) chunked equivalence ------------------------------------------
    df1, _ = scada.load_scada(sample, chunksize=5_000)          # many chunks
    df2, _ = scada.load_scada(sample, chunksize=10_000_000)     # single chunk
    assert len(df1) == len(df2), f"row mismatch {len(df1)} vs {len(df2)}"
    assert df1["turbine"].nunique() == df2["turbine"].nunique()
    for c in ("power_kw", "ws"):
        a, b = df1[c].mean(), df2[c].mean()
        assert abs(a - b) < 1e-9 * max(1, abs(a)), f"mean mismatch {c}: {a} vs {b}"
    print(f"  OK  chunked equivalence       rows={len(df1):,}  chunks-5k == single-pass")

    # ---- 2) ~1.2M row stress test -----------------------------------------
    # (the loader streams with bounded memory; the analysis pipeline itself
    # needs ~1-2 GB RAM per million rows at float64 — enable "use_float32":
    # true in the config to halve that)
    d = pd.read_csv(sample)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    t0 = time.time()
    parts = []
    for i in range(6):
        p = d.copy()
        # contiguous copies (120-day span exactly) so the gap-filler has
        # nothing to interpolate and row counts stay identical
        p["timestamp"] = p["timestamp"] + pd.Timedelta(days=i * 121)
        parts.append(p)
    big = pd.concat(parts, ignore_index=True)
    big_path = os.path.join(TMP, "big_1m.csv")
    big.to_csv(big_path, index=False)
    print(f"  .. stress file: {len(big):,} rows, {os.path.getsize(big_path)/1e6:.0f} MB")

    t1 = time.time()
    cfg = cfg_mod.load_config(os.path.join(SAMPLE, "config.json"))
    r = run_analysis(cfg, big_path, outdir=os.path.join(TMP, "out_big"))
    elapsed = time.time() - t1
    assert len(r["df"]) == len(big), f"row mismatch: {len(r['df'])} vs {len(big)}"
    assert np.isfinite(r["losses"]["net_mwh"]) and r["losses"]["net_mwh"] > 0
    assert r["meta"]["read_info"]["chunks"] > 1, "large file should be streamed in chunks"
    print(f"  OK  ~1M-row stress            rows={len(big):,}  "
          f"load+analyse={elapsed:.0f}s  read={r['meta']['read_info']['method']}  "
          f"net P50={r['uncertainty']['p']['P50']:,.0f} MWh")

    # ---- 3) wide file with junk channels ---------------------------------
    ts = pd.date_range("2024-01-01", periods=1440, freq="10min")
    wide = pd.DataFrame({"Timestamp": ts})
    for tid in ["T01", "T02"]:
        sub = (d[d["turbine_id"] == tid].head(1440)
               .set_index("timestamp").reindex(ts))
        wide[f"{tid} Active Power (kW)"] = sub["power_kw"].values
        wide[f"{tid} Wind Speed (m/s)"] = sub["wind_speed_mps"].values
        wide[f"{tid} Nacelle Direction (deg)"] = sub["nacelle_dir_deg"].values
        for k in range(120):                      # junk channels (e.g. counters)
            wide[f"{tid} internal_param_{k}"] = np.arange(len(wide)) % 100
    wide_path = os.path.join(TMP, "wide_junk.csv")
    wide.to_csv(wide_path, index=False)
    dfw, _ = scada.load_scada(wide_path, chunksize=50_000)
    assert dfw["turbine"].nunique() == 2
    assert dfw["power_kw"].mean() > 100
    print(f"  OK  wide+junk (246 cols)      turbines={dfw['turbine'].nunique()}  "
          f"rows={len(dfw):,}  meanP={dfw['power_kw'].mean():.0f} kW")

    print("\nAll large-data tests passed OK")


if __name__ == "__main__":
    main()
