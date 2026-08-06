"""Power curve analysis (IEC 61400-12-1 style binning), Weibull fitting and AEP integration."""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import gamma


def weibull_fit(ws, max_k=6.0):
    """Fit a 2-parameter Weibull by the method of moments."""
    ws = np.asarray(ws, dtype=float)
    ws = ws[np.isfinite(ws) & (ws > 0)]
    if len(ws) < 5:
        return 7.0, 2.0
    mu = ws.mean()
    s = ws.std(ddof=1)
    if mu <= 0 or s <= 0:
        return 7.0, 2.0
    return weibull_fit_moments(mu, s)


def weibull_fit_moments(mu, sd, max_k=6.0):
    """Weibull (A, k) from the mean and standard deviation (method of moments)."""
    mu = float(mu)
    sd = float(sd)
    if mu <= 0 or sd <= 0:
        return 7.0, 2.0
    cv = sd / mu

    def f(k):
        return gamma(1 + 2 / k) / gamma(1 + 1 / k) ** 2 - (1 + cv ** 2)

    k = cv ** -1.086
    try:
        k = brentq(f, 0.35, max_k)
    except (ValueError, RuntimeError):
        pass
    A = mu / gamma(1 + 1 / k)
    return float(A), float(k)


def weibull_pdf(v, A, k):
    return (k / A) * (v / A) ** (k - 1) * np.exp(-(v / A) ** k)


def extend_curve(curve_df, rated, cut_out, vmax=45.0, bin_width=0.5):
    """Return (v_array, p_array) over a dense grid, with P=0 above cut-out."""
    v = np.arange(0.0, vmax + 1e-9, 0.1)
    base = np.interp(v, curve_df["bin_center"].values,
                     curve_df["mean_power"].fillna(0).values)
    p = np.where(v >= cut_out, 0.0, np.minimum(base, rated))
    return v, p


def interp_power_factory(v_arr, p_arr):
    return lambda ws: np.interp(np.asarray(ws, dtype=float), v_arr, p_arr)


def bin_power_curve(ws, p, bin_width=0.5, vmin=0.0, vmax=45.0, min_count=3,
                    density_ratio=None):
    """IEC-style binning of (wind speed, power) samples into width=0.5 m/s bins.

    If density_ratio is provided (air-density correction), power is normalised
    to standard air density: p_std = p / density_ratio.
    """
    ws = np.asarray(ws, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(ws) & np.isfinite(p)
    ws, p = ws[m], p[m]
    if density_ratio is not None:
        dr = np.asarray(density_ratio, dtype=float)[m]
        p = p / np.where(dr > 0.3, dr, 1.0)

    edges = np.arange(vmin, vmax + bin_width, bin_width)
    centers = edges[:-1] + bin_width / 2.0
    idx = np.digitize(ws, edges) - 1
    rows = []
    for b in range(len(centers)):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            rows.append((centers[b], 0, np.nan, np.nan, np.nan, np.nan))
            continue
        rows.append((centers[b], n, ws[sel].mean(), p[sel].mean(),
                     p[sel].std() if n > 1 else np.nan, p[sel].max()))
    out = pd.DataFrame(rows, columns=["bin_center", "count", "mean_ws",
                                      "mean_power", "std_power", "max_power"])
    out.loc[out["count"] < min_count, ["mean_ws", "mean_power"]] = np.nan
    return out


def energy_weighted_deviation(meas_curve, warr_curve, vmin=0.0, vmax=45.0):
    """Energy-weighted power curve deviation (%) over the measured wind distribution.

    dev = 100 * sum(n_b * (P_meas - P_warr)) / sum(n_b * P_warr)   over bins
    """
    m = meas_curve["mean_power"].notna() & (meas_curve["count"] > 0)
    v = meas_curve.loc[m, "bin_center"].values
    pm = meas_curve.loc[m, "mean_power"].values
    n = meas_curve.loc[m, "count"].values
    pw = np.interp(v, warr_curve["bin_center"].values,
                   warr_curve["mean_power"].fillna(0).values)
    num = (n * (pm - pw)).sum()
    den = (n * pw).sum()
    return 100.0 * num / den if den > 0 else 0.0


def aep_from_weibull(A, k, v_arr, p_arr, hours=8760.0):
    """Gross AEP (MWh) per turbine from a Weibull wind distribution and a power curve."""
    v = np.linspace(0.05, np.max(v_arr) - 0.05, 2000)
    pdf = weibull_pdf(v, A, k)
    p = np.interp(v, v_arr, p_arr)
    # trapezoidal integration — manual, so it works on any NumPy version
    # (np.trapz was removed in NumPy 2.0)
    integrand = pdf * p
    integral = float(np.sum((integrand[1:] + integrand[:-1]) * 0.5
                            * (v[1:] - v[:-1])))
    return hours * integral / 1000.0


def aep_from_timeseries(ws, v_arr, p_arr, dt_h):
    """Energy (MWh) implied by a wind-speed series through a power curve."""
    p = np.interp(np.asarray(ws, dtype=float), v_arr, p_arr)
    return float(np.nansum(p * dt_h) / 1000.0)


def air_density_ratio(temp_c, pressure_kpa=None):
    """Air density relative to standard (1.225 kg/m3 at 15 C, 101.325 kPa)."""
    T = np.asarray(temp_c, dtype=float) + 273.15
    p = pressure_kpa if pressure_kpa else 101.325
    rho = p * 1000.0 / (287.05 * T)
    return rho / 1.225
