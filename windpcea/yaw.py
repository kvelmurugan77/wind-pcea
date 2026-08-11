"""Yaw misalignment analysis from SCADA (DNV-style screening).

Uses the measured wind direction (e.g. Envision Amb_WindDir_Abs_Avg) and the
nacelle direction (Nac_Direction_Avg / Nac_Direc). The yaw error per record is

    yaw_error = wrap180(wind_dir - nacelle_dir)

For each turbine:
  * the mean yaw error (power-weighted) is a robust first estimate;
  * within narrow wind-speed bins the relative power (power / bin mean) is
    fitted against the yaw error with a quadratic; the vertex is the yaw
    offset that maximises power - i.e. the estimated yaw misalignment.
    Normalising within wind-speed bins removes any absolute power-curve
    mismatch (generic vs OEM curve), which would otherwise dominate the fit.

Output: per-turbine table (mean yaw error, fitted offset, R2, n, loss %)
+ pooled scatter for the chart.
"""
import numpy as np
import pandas as pd

FLAG_THRESHOLD_DEG = 5.0


def wrap180(deg):
    """Wrap angles to [-180, 180)."""
    deg = np.asarray(deg, dtype=float) % 360.0
    return np.where(deg > 180.0, deg - 360.0, deg)


def yaw_error_series(wind_dir, nacelle_dir):
    return wrap180(wind_dir - nacelle_dir)


def analyse_yaw(df, cfg, v_arr=None, p_arr=None, interp_power=None):
    """Per-turbine yaw misalignment screening.

    df must contain 'ws', 'power_kw', 'wind_dir_deg' and 'dir_deg' (nacelle).
    """
    if "wind_dir_deg" not in df.columns or "dir_deg" not in df.columns:
        return None
    op = df[(df["flag"] == 0) & df["ws"].notna() & df["power_kw"].notna()
            & df["wind_dir_deg"].notna() & df["dir_deg"].notna()].copy()
    if len(op) < 200:
        return None
    op["yaw_err"] = yaw_error_series(op["wind_dir_deg"].values,
                                     op["dir_deg"].values)

    rows = []
    samples = []
    for tid, g in op.groupby("turbine"):
        if len(g) < 100:
            continue
        m = (g["ws"] >= max(3.0, cfg.get("cut_in_mps", 3.0))) & (g["ws"] <= 12.0)
        sub = g[m].copy()
        if len(sub) < 80:
            continue
        # normalise power within 0.5 m/s wind-speed bins (removes curve mismatch)
        sub["ws_bin"] = np.floor(sub["ws"] * 2) / 2
        bin_mean = sub.groupby("ws_bin")["power_kw"].transform("mean")
        sub["pr"] = (sub["power_kw"] / bin_mean.clip(lower=1.0)).clip(0.3, 1.7)

        # robust primary estimate: power-weighted mean yaw error
        w = sub["power_kw"].clip(lower=0.0).values + 1.0
        mean_ye = float(np.average(sub["yaw_err"].values, weights=w))

        # quadratic fit: pr = a*ye^2 + b*ye + c
        ye = sub["yaw_err"].values
        pr = sub["pr"].values
        A = np.vstack([ye ** 2, ye, np.ones_like(ye)]).T
        try:
            coef, *_ = np.linalg.lstsq(A, pr, rcond=None)
        except np.linalg.LinAlgError:
            continue
        a, b, c = coef
        spread = float(ye.std())
        fitted = None
        r2 = 0.0
        if a < 0 and spread > 2.0:
            vertex = -b / (2.0 * a)
            if np.isfinite(vertex) and abs(vertex) <= 40:
                pr_pred = a * ye ** 2 + b * ye + c
                ss_res = float(((pr - pr_pred) ** 2).sum())
                ss_tot = float(((pr - pr.mean()) ** 2).sum())
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                if r2 >= 0.02:      # only trust a resolvable optimum
                    fitted = float(vertex)
        # final estimate: fitted offset when credible, else mean yaw error
        est = fitted if fitted is not None else float(mean_ye)
        pr0 = c
        pr_opt = (a * est ** 2 + b * est + c) if fitted is not None else pr0
        loss_pct = max(0.0, 100.0 * (pr_opt - pr0) / max(1e-9, pr0))
        rows.append({"turbine": tid,
                     "mean_yaw_error_deg": float(mean_ye),
                     "yaw_misalignment_deg": float(est),
                     "fit_r2": float(r2),
                     "yaw_spread_deg": float(spread),
                     "n": int(len(sub)),
                     "loss_pct": float(loss_pct),
                     "mean_power_ratio": float(pr.mean())})
        smp = sub[["ws", "yaw_err", "pr"]].copy()
        smp["turbine"] = tid
        samples.append(smp)
    if not rows:
        return None
    per = pd.DataFrame(rows).sort_values("yaw_misalignment_deg")
    pooled = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame()
    return {"per_turbine": per, "pooled": pooled,
            "count": int(len(op)),
            "flagged": per[per["yaw_misalignment_deg"].abs() > FLAG_THRESHOLD_DEG]}
