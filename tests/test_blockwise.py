"""Blockwise (out-of-core) engine tests.

  1. Equivalence: blockwise results match the in-memory pipeline on the
     sample dataset within tolerance.
  2. Block-boundary robustness: small blocks (30 days) still agree.

Run:  python tests/test_blockwise.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from windpcea import config as cfg_mod              # noqa: E402
from windpcea.analysis import run_analysis          # noqa: E402
from windpcea.blockwise import run_blockwise        # noqa: E402

SAMPLE = os.path.join(ROOT, "sample_data", "scada_sample.csv")


def main():
    cfg = cfg_mod.load_config(os.path.join(ROOT, "sample_data", "config.json"))

    t0 = time.time()
    r1 = run_analysis(cfg, SAMPLE)
    t1 = time.time()
    r2 = run_blockwise(cfg, SAMPLE, block_days=30)
    t2 = time.time()
    print(f"  .. in-memory {t1-t0:.1f}s | blockwise {t2-t1:.1f}s")

    def chk(name, a, b, tol_pct=1.0):
        rel = abs(a - b) / max(1e-9, abs(a)) * 100.0
        ok = rel <= tol_pct
        print(f"  {'OK ' if ok else 'FAIL'} {name:<28} {a:,.1f} vs {b:,.1f} ({rel:.2f}%)")
        assert ok, f"{name}: {a} vs {b}"

    chk("rows", r1["qc"]["rows"], r2["qc"]["rows"], 0.1)
    chk("E_actual_mwh", r1["energy"]["E_actual_mwh"], r2["energy"]["E_actual_mwh"], 0.5)
    chk("E_lost downtime", r1["energy"]["E_lost_mwh"][2], r2["energy"]["E_lost_mwh"][2], 2.0)
    chk("E_lost curtailment", r1["energy"]["E_lost_mwh"][3], r2["energy"]["E_lost_mwh"][3], 2.0)
    chk("time availability %", r1["availability"]["farm"]["time_avail_pct"],
        r2["availability"]["farm"]["time_avail_pct"], 1.0)
    chk("gross AEP (Method A)", r1["losses"]["gross_lt_mwh_method_a"],
        r2["losses"]["gross_lt_mwh_method_a"], 2.0)
    chk("net P50", r1["uncertainty"]["p"]["P50"], r2["uncertainty"]["p"]["P50"], 2.0)
    chk("CF %", r1["capacity_factor_pct"], r2["capacity_factor_pct"], 2.0)
    # wake is an estimate derived from per-block free-stream selection -> wider tol
    chk("wake loss %", r1["wake"]["wake_loss_pct"], r2["wake"]["wake_loss_pct"], 25.0)

    print("\nAll blockwise equivalence tests passed ✓")


if __name__ == "__main__":
    main()
