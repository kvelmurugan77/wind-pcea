"""Yaw error analysis from nacelle orientation vs wind direction.

Per-turbine yaw error estimate:

  offset(t) = angular difference (wind_direction - nacelle_direction),
  wrapped to [-180, +180).

The yaw error per turbine is the circular (power-weighted) mean of the
offset over operating records at usable wind speeds (>= 6 m/s), binned and
robustified. A positive yaw error means the nacelle is 'behind' the wind
(e.g. +10 deg = nacelle lags wind by 10 deg). The standard deviation and
the power-vs-offset curve are also computed (a dip around 0 offset with a
peak shifted away from 0 indicates a systematic yaw error).
"""
import numpy as np
import pandas as pd


def _wrap(deg):
    """Wrap angles to [-180, 180)."""
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


def circular_mean(deg):
    """Circular mean of an angle series (degrees)."""
    rad = np.deg2rad(np.asarray(deg, dtype=float))
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())))


def yaw_error_analysis(df, min_ws=6.0, bin_width=5.0):
    """Compute per-turbine yaw error + a farm offset histogram.

    Returns dict:
      per_turbine : DataFrame (turbine, yaw_error_deg, circular_std_deg,
                               n_records, mean_ws)
      histogram   : DataFrame (offset_bin_center, count) — farm aggregate
      power_vs_offset : DataFrame (offset_bin, mean_power_kw) per turbine
                        stored in per_turbine_pv dict
    """
    op = df[(df["flag"] == 0) & df["ws"].notna() & (df["ws"] >= min_ws)].copy()
    rows = []
    hist = {}
    pv_all = {}
    has_nac = "dir_deg" in op.columns and op["dir_deg"].notna().any()
    has_wind = "wind_dir_deg" in op.columns and op["wind_dir_deg"].notna().any()
    if has_nac and has_wind:
        for tid, g in op.groupby("turbine"):
            off = _wrap(g["wind_dir_deg"] - g["dir_deg"])
            m = np.isfinite(off)
            if m.sum() < 30:
                continue
            off_v = off[m]
            # power-weighted circular mean
            w = g.loc[m, "power_kw"].clip(lower=0).values + 1e-6
            rad = np.deg2rad(off_v)
            yaw = float(np.rad2deg(
                np.arctan2((np.sin(rad) * w).sum() / w.sum(),
                           (np.cos(rad) * w).sum() / w.sum())))
            circ_std = float(np.rad2deg(np.sqrt(-2 * np.log(
                np.clip(np.hypot((np.cos(rad) * w).sum() / w.sum(),
                                 (np.sin(rad) * w).sum() / w.sum()), 1e-9, 1.0)))))
            rows.append({"turbine": tid, "yaw_error_deg": round(yaw, 2),
                         "circular_std_deg": round(circ_std, 2),
                         "n_records": int(m.sum()),
                         "mean_ws": round(float(g.loc[m, "ws"].mean()), 2)})
            # farm histogram of offsets
            for o in off_v:
                b = int(np.floor((o + 180.0) / bin_width)) * bin_width - 180.0
                hist[b] = hist.get(b, 0) + 1
            # power vs offset bins (5-deg)
            bins = np.arange(-180, 180 + bin_width, bin_width)
            idx = np.digitize(off_v, bins) - 1
            pv = {}
            for i, pw in zip(idx, g.loc[m, "power_kw"].values):
                if 0 <= i < len(bins) - 1:
                    pv.setdefault(bins[i] + bin_width / 2, []).append(pw)
            pv_all[tid] = pd.DataFrame(
                {"offset_deg": [k for k in pv],
                 "mean_power_kw": [np.mean(v) for v in pv.values()]})
    per = pd.DataFrame(rows)
    hist_df = pd.DataFrame(sorted(hist.items()), columns=["offset_bin", "count"]) \
        if hist else pd.DataFrame(columns=["offset_bin", "count"])
    return {"per_turbine": per, "histogram": hist_df,
            "power_vs_offset": pv_all, "min_ws": min_ws}
