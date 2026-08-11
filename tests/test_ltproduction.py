"""UL/OEPR Step-1 method tests: 30-day normalization (Eq. 3), density
correction (Eq. 4), production regression (Eq. 5) and de-normalization
(Eq. 6)."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from windpcea import ltproduction as ltp  # noqa: E402


def test_normalization():
    """UL Eq. 3: monthly gross normalized to 30 days."""
    idx = pd.date_range("2024-01-01", periods=31, freq="D")   # 31-day month
    e = pd.Series(1.0, index=idx)                             # 1 MWh/day -> 31 MWh/month
    out = ltp.ul_monthly_energy(pd.DataFrame({
        "timestamp": idx, "turbine": "T1",
        "flag": 0, "expected_energy_kwh": e.values * 1000.0}), normalize_days=30.0)
    assert len(out) == 1
    assert abs(out.iloc[0] - 30.0) < 1e-6, f"Eq.3 failed: {out.iloc[0]} != 30"
    print("OK  Eq.3 30-day normalization: 31 MWh/31d -> 30.0 MWh/30d")


def test_density():
    """UL Eq. 4: Vn = Vobs * (rho_obs/rho0)^(1/3)."""
    ws = density_check = None
    rho = ltp.site_air_density_ratio(0.0, 15.0)      # sea level, 15C -> ~1.0
    assert abs(rho - 1.0) < 0.02, f"rho at std conditions: {rho}"
    vn = ltp.density_correct_ws(np.array([10.0]), 0.8)   # rho_obs/rho0 = 0.8
    expected = 10.0 * (0.8) ** (1 / 3)
    assert abs(vn[0] - expected) < 1e-9
    # colder (denser) air -> correction > 1
    vn_cold = ltp.density_correct_ws(np.array([10.0]), 1.1)
    assert vn_cold[0] > 10.0
    print("OK  Eq.4 density correction: 10 m/s at rho-ratio 0.8 -> %.4f m/s" % vn[0])


def test_regression_roundtrip():
    """Eq. 5 regression recovers a known linear relationship."""
    rng = np.random.default_rng(1)
    days = pd.date_range("2019-01-01", periods=40 * 30, freq="D")   # 40 months daily
    month_of = days.to_period("M")
    # monthly-varying wind (daily noise around it)
    base = 6.0 + 2.0 * np.sin(np.arange(month_of.nunique()) / 40 * 2 * np.pi)
    months_sorted = sorted(set(month_of.astype(str)))
    m2i = {m: i for i, m in enumerate(months_sorted)}
    month_idx = np.array([m2i[m] for m in month_of.astype(str)])
    ws_daily = base[month_idx] + rng.normal(0, 0.6, len(days))
    a_true, b_true = 300.0, 500.0
    energy_daily = (a_true * ws_daily + b_true) * (24.0 / 1000.0)   # kWh/day -> MWh/day
    df = pd.DataFrame({"timestamp": days, "turbine": "T1", "flag": 0,
                       "expected_energy_kwh": energy_daily * 1000.0})
    ref = pd.DataFrame({"ws": ws_daily, "temp_c": 15.0}, index=days)
    climate = {"lt_n_years": 5.0}
    cfg = {"excluded_months": [], "ul_normalize_days": 30.0,
           "lt_density_correction": False, "site_elevation_m": 0.0}
    out = ltp.ul_lt_assessment(cfg, df, climate, ref=ref)
    assert out is not None
    # monthly slope/intercept = daily a,b x 30 days x 24 h / 1000
    a_exp = a_true * 30.0 * 24.0 / 1000.0
    b_exp = b_true * 30.0 * 24.0 / 1000.0
    assert abs(out["a"] - a_exp) / a_exp < 0.15, f"slope {out['a']} vs {a_exp}"
    assert abs(out["b"] - b_exp) / abs(b_exp) < 0.5, f"intercept {out['b']} vs {b_exp}"
    assert out["r2"] > 0.8, f"R2 {out['r2']}"
    print(f"OK  Eq.5 regression recovered: a={out['a']:.0f} b={out['b']:.0f} R2={out['r2']:.3f}")


def test_exclusion():
    """Configured months are dropped from the regression."""
    idx = pd.date_range("2024-01-01", periods=240, freq="D")    # 8 months
    e = pd.Series(1.0, index=idx)
    df = pd.DataFrame({"timestamp": idx, "turbine": "T1", "flag": 0,
                       "expected_energy_kwh": e.values * 1000.0})
    rng2 = np.random.default_rng(2)
    ref = pd.DataFrame({"ws": 7.0 + rng2.normal(0, 1.2, 500)},
                       index=pd.date_range("2024-01-01", periods=500, freq="D"))
    climate = {"lt_n_years": 1.0}
    cfg = {"excluded_months": ["2024-02"], "ul_normalize_days": 30.0,
           "lt_density_correction": False, "site_elevation_m": 0.0}
    out = ltp.ul_lt_assessment(cfg, df, climate, ref=ref)
    assert out is not None
    assert out["n"] == 7, f"expected 7 months after exclusion, got {out["n"]}"
    assert out["excluded_months"] == ["2024-02"]
    print("OK  month exclusion: 8 months -> 7 (2024-02 dropped)")


if __name__ == "__main__":
    test_normalization()
    test_density()
    test_regression_roundtrip()
    test_exclusion()
    print("\nAll UL/OEPR Step-1 tests passed OK")
