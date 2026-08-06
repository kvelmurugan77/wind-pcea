"""Production-based long-term assessment (Method B — OEPR style).

In addition to the wind-speed MCP + warranted-curve method (Method A), this
module implements the classic production-regression approach used in
Operational Energy Production Reports:

    1. Daily variant: measured daily farm gross energy is regressed on the
       (MCP-predicted) daily site wind speed over the measured period; the
       fitted relationship is applied to the full long-term daily wind series
       to predict the long-term daily energy series.

    2. Monthly variant: measured monthly farm gross energy (30.44-day
       normalised, months with >= 20 days coverage) is regressed on the
       long-term reference monthly wind speed; the fit is applied to the full
       long-term monthly reference to predict the long-term energy series.

The long-term annual gross AEP is the annualised sum of the predicted series.
This directly answers "daily/monthly yield vs wind speed, then apply the
relationship to the long-term record" — cross-checked against Method A.
"""
import numpy as np
import pandas as pd
from scipy import stats

GROSS_FLAGS = (0, 1, 2, 3, 4, 5)   # operating, below cut-in, downtime,
                                   # curtailment, derating, environmental


def daily_gross_energy(df):
    """Daily farm gross energy (MWh) = warranted-curve energy at the measured
    wind speed over all plausible records (actual + all lost categories).
    This is the same gross definition used in the loss tree."""
    sub = df[df["flag"].isin(GROSS_FLAGS)]
    e = sub.groupby(sub["timestamp"].dt.date)["expected_energy_kwh"].sum() / 1000.0
    e.index = pd.to_datetime(e.index)
    return e


def monthly_gross_energy(df, min_days=20):
    """Monthly farm gross energy (MWh), normalised to a 30.44-day month.
    Months with fewer than min_days of records are dropped."""
    e = daily_gross_energy(df)
    mon = e.groupby(e.index.to_period("M")).agg(["sum", "count"])
    days = mon["count"]
    energy = mon["sum"] * 30.44 / days.clip(lower=1)
    energy = energy[days >= min_days]
    energy.index = energy.index.to_timestamp()   # month-start timestamps
    return energy


def fit_production(energy_series, wind_series, min_n=6, power_law=False):
    """Fit the production relationship E = f(WS) on the overlapping dates.

    power_law=False: linear regression E = a + b·WS (OEPR convention for the
    monthly variant).
    power_law=True : power-law regression E = c·WS^b (log-log OLS) — captures
    the E ∝ v³ physics and extrapolates far better to wind speeds outside the
    measured range (used for the daily variant).

    Returns dict with the parameters, r2, n and the predicted series over the
    full wind_series index, or None when there is not enough data.
    """
    j = pd.DataFrame({"E": energy_series, "WS": wind_series}).dropna()
    if len(j) < min_n:
        return None
    if power_law:
        # Physical model: below rated power the daily energy scales with the
        # third power of the daily mean wind speed (E = c * v^3). Fixing the
        # exponent at 3 (a single-coefficient fit) is far more robust than a
        # free log-log exponent, which becomes unstable on narrow wind ranges.
        m = (j["E"] > 0) & (j["WS"] > 0.5)
        j = j[m]
        if len(j) < min_n:
            return None
        c = float(np.exp(np.mean(np.log(j["E"]) - 3.0 * np.log(j["WS"]))))
        pred = pd.Series(c * wind_series.clip(lower=0.5) ** 3.0,
                         index=wind_series.index)
        ss_res = float(((j["E"] - c * j["WS"] ** 3.0) ** 2).sum())
        ss_tot = float(((j["E"] - j["E"].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"kind": "cubic", "c": c, "b": 3.0,
                "r2": r2, "n": int(len(j)),
                "predicted": pred.clip(lower=0.0)}
    res = stats.linregress(j["WS"], j["E"])
    pred = pd.Series(res.intercept + res.slope * wind_series,
                     index=wind_series.index)
    pred = pred.clip(lower=0.0)
    return {"kind": "linear", "a": float(res.intercept), "b": float(res.slope),
            "r2": float(res.rvalue ** 2), "n": int(len(j)),
            "predicted": pred}


def lt_production_assessment(cfg, df, climate, min_record_months=3.0):
    """Run both production-regression variants.

    Returns dict (or None when the record is too short):
      daily   : daily-energy vs daily-site-wind fit + LT predicted daily series
      monthly : monthly-energy vs monthly-reference-wind fit + LT predicted series
      primary : chosen method key + lt_gross_mwh (annualised LT gross AEP)
    """
    record_months = climate.get("record_months", 0.0)
    if record_months < min_record_months:
        return {"daily": None, "monthly": None, "primary": None,
                "record_months": record_months, "lt_gross_mwh": None,
                "note": f"record covers only {record_months:.1f} months "
                        f"(min {min_record_months:g} months required)"}
    out = {}
    lt_n_years = climate["lt_n_years"]

    # ---- daily variant (power-law fit, physics-correct) ------------------
    daily_e = daily_gross_energy(df)
    site_daily = climate["site_daily"]
    lt_daily = climate["lt_daily"]
    if len(daily_e.dropna()) >= 6:
        fit_d = fit_production(daily_e, site_daily, power_law=True)
        if fit_d is not None:
            lt_pred = fit_production(daily_e, lt_daily, power_law=True)
            if lt_pred is not None:
                annual_d = float(lt_pred["predicted"].sum() / lt_n_years)
                out["daily"] = {"fit": fit_d, "r2": fit_d["r2"],
                                "measured": daily_e, "wind": site_daily,
                                "lt_predicted": lt_pred["predicted"],
                                "lt_annual_gross_mwh": annual_d}

    # ---- monthly variant (linear, OEPR convention) -----------------------
    ref = climate.get("ref")
    if ref is not None and len(ref) >= 12:
        mon_e = monthly_gross_energy(df)
        if len(mon_e.dropna()) >= 6:
            ref_mon = ref["ws"].resample("ME").mean()
            ref_mon.index = ref_mon.index.to_period("M").to_timestamp()
            fit_m = fit_production(mon_e, ref_mon.reindex(mon_e.index))
            if fit_m is not None:
                lt_pred = fit_production(mon_e, ref_mon)
                if lt_pred is not None:
                    annual_m = float(lt_pred["predicted"].sum() / lt_n_years)
                    out["monthly"] = {"fit": fit_m, "r2": fit_m["r2"],
                                      "measured": mon_e,
                                      "wind": ref_mon.reindex(mon_e.index),
                                      "lt_predicted": lt_pred["predicted"],
                                      "lt_annual_gross_mwh": annual_m}

    if not out:
        return None

    # Primary selection: the production regression is only used as the
    # primary gross-AEP method when the measured record covers a full
    # seasonal cycle (>= 9 months); otherwise it is indicative only and the
    # Weibull x warranted-curve method (Method A) is used.
    record_months = climate.get("record_months", 0.0)
    seasonal_ok = record_months >= 9.0
    if seasonal_ok:
        if "monthly" in out and out["monthly"]["r2"] >= 0.5:
            primary = "monthly"
        elif "daily" in out and out["daily"]["r2"] >= 0.5:
            primary = "daily"
        else:
            primary = None
    else:
        primary = None
    out["primary"] = primary
    out["record_months"] = record_months
    out["lt_gross_mwh"] = out[primary]["lt_annual_gross_mwh"] if primary else None
    return out
