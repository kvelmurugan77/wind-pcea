"""Wake-loss analysis from nacelle anemometry.

Method (pragmatic, DNV-style when no met mast is available):
  * Per direction sector, the least-waked turbines (highest mean wind speed
    while operating) form a reference set; the free-stream wind speed per
    10-min interval is their mean (reference-turbine method). Turbines with
    anemometer faults are excluded.
  * The wake deficit of each turbine is 1 - ws_i / ws_free for that interval.
  * Deficits are aggregated by wind direction sector (default 30 deg).
  * Wake energy loss = sum over operating intervals of
    [P_curve(ws_free) - P_curve(ws_i)] * dt.

Caveat: nacelle anemometry is affected by rotor-induced flow; the resulting
losses are indicative, and a met-mast or lidar reference would refine them.
"""
import numpy as np
import pandas as pd


def free_stream_ws(df, exclude_turbines=None, min_ws=5.0, n_free=3, sector_width=30):
    """Free-stream wind speed per timestamp using the reference-turbine method.

    Per direction sector, the n_free least-waked turbines (highest mean wind
    speed while operating) form the reference; the free-stream wind speed is
    their mean per timestamp. This is far less biased than the raw maximum
    (which tracks the largest anemometer noise draw). Falls back to the max
    when no reference turbine is operating.
    """
    sub = df[(df["flag"] == 0) & df["ws"].notna()]
    if exclude_turbines:
        sub = sub[~sub["turbine"].isin(exclude_turbines)]
    if sub.empty:
        return pd.Series(dtype=float)

    # per-timestamp median direction -> sector
    if "dir_deg" in sub.columns and sub["dir_deg"].notna().any():
        dir_by_ts = sub.groupby("timestamp")["dir_deg"].median()
    else:
        dir_by_ts = pd.Series(0.0, index=pd.Index(sub["timestamp"].unique(), name="timestamp"))
    sec_by_ts = ((dir_by_ts // sector_width) * sector_width).astype(int)

    tmp = sub.copy()
    tmp["sector"] = tmp["timestamp"].map(sec_by_ts)
    means = tmp.groupby(["sector", "turbine"])["ws"].mean()
    free_set = {}
    for sec, g in means.groupby(level=0):
        # reference turbines = those within 1.5 % of the sector's highest mean
        # wind speed (robust against a waked turbine slipping into a fixed
        # top-N); in a wake-free sector the whole fleet qualifies
        best = g.max()
        sel = g[g >= best * (1.0 - 0.015)]
        free_set[int(sec)] = set(sel.index.get_level_values(1))

    fs_df = pd.DataFrame([{"sector": s, "turbine": t} for s, ts in free_set.items() for t in ts])
    tmp2 = tmp.merge(fs_df, on=["sector", "turbine"])
    free = tmp2.groupby("timestamp")["ws"].mean()

    # fallback for timestamps with no operating reference turbine
    missing = sub.loc[~sub["timestamp"].isin(free.index), "timestamp"].unique()
    if len(missing):
        m = sub[sub["timestamp"].isin(missing)].groupby("timestamp")["ws"].max()
        free = pd.concat([free, m])
    free[free < min_ws] = np.nan
    return free


def wake_analysis(df, v_arr, p_arr, interp_power, cfg, free):
    """Compute wake deficits and wake energy loss.

    Returns dict with sector table, per-turbine table, wake energy and loss %.
    Always returns tables with the standard columns, even when there is no
    usable operating data (empty tables instead of crashes).
    """
    rated = cfg["rated_power_kw"]
    out = {"sector_table": pd.DataFrame(columns=["sector_deg", "mean_deficit",
                                                 "n_samples"]),
           "per_turbine": pd.DataFrame(columns=["turbine", "mean_deficit",
                                                "n_samples", "wake_energy_mwh"]),
           "wake_energy_mwh": 0.0, "wake_loss_pct": 0.0, "n_intervals": 0}

    op = df[(df["flag"] == 0) & df["ws"].notna() & df["timestamp"].isin(free.index)]
    if op.empty or free is None or len(free) == 0:
        return out

    op = op.copy()
    op["ws_free"] = op["timestamp"].map(free)
    # the 0.1 m/s floor removes noise-only differences (reference-turbine
    # free stream is unbiased to ~0.5%); larger deficits are real wakes
    ok = op["ws_free"].notna() & (op["ws_free"] - op["ws"] > 0.1) & (op["ws_free"] > 6.0)
    op["deficit"] = np.where(ok, 1.0 - op["ws"] / op["ws_free"], 0.0)
    # air-density scaling keeps the wake term consistent with the expected
    # energy (which is density-corrected)
    dr = op["density_ratio"].clip(0.5, 1.15) if "density_ratio" in op.columns else 1.0
    op["wake_power_loss_kw"] = np.where(
        ok, np.maximum(0.0, interp_power(op["ws_free"].values) - interp_power(op["ws"].values))
        * dr, 0.0)

    out["n_intervals"] = int(ok.sum())
    out["wake_energy_mwh"] = float(op["wake_power_loss_kw"].sum() * df["dt_h"].iloc[0] / 1000.0)

    # sector aggregation (mean deficit over rows with a valid free stream)
    sw = float(cfg.get("sector_width_deg", 30))
    valid = op[op["ws_free"].notna() & (op["ws_free"] > 6.0)]
    if "dir_deg" in op.columns and op["dir_deg"].notna().any():
        valid = valid.copy()
        valid["sector"] = ((valid["dir_deg"] // sw) * sw).astype(int)
        tab = (valid.groupby("sector")["deficit"]
               .agg(["mean", "count"])
               .rename(columns={"mean": "mean_deficit", "count": "n_samples"}))
        tab.index.name = "sector_deg"
        out["sector_table"] = tab.reset_index()
    else:
        valid = valid.copy()
        valid["sector"] = 0
        out["sector_table"] = pd.DataFrame(
            {"sector_deg": [0], "mean_deficit": [valid["deficit"].mean()],
             "n_samples": [len(valid)]})

    # per-turbine mean deficit
    tab2 = (op.groupby("turbine")["deficit"]
              .agg(["mean", "count"])
              .rename(columns={"mean": "mean_deficit", "count": "n_samples"}))
    tab2["wake_energy_mwh"] = (op.groupby("turbine")["wake_power_loss_kw"].sum()
                               * df["dt_h"].iloc[0] / 1000.0)
    out["per_turbine"] = tab2.reset_index()
    return out


def wake_loss_share(wake_energy_mwh, gross_period_mwh):
    return 100.0 * wake_energy_mwh / gross_period_mwh if gross_period_mwh > 0 else 0.0
