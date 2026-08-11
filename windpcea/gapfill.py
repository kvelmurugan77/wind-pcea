"""Gap filling for SCADA time series (traceable, DNV-reporting friendly).

Rules (documented in the report so results are reproducible):
  * Short gaps  (1-6 intervals, i.e. <= 1 h at 10-min): linear interpolation
    per turbine between the bounding valid values.
  * Medium gaps (> 6 intervals up to 24 intervals / 4 h): imputed from the
    farm-average wind speed profile (normalised to the turbine's own level)
    for wind speed; power from the warranted curve at the imputed wind speed.
  * Longer gaps: NOT filled (would be speculation); flagged as missing and
    handled by coverage/availability accounting.

Every filled record is flagged (filled=1, method) and a summary table is
produced so the reader can audit exactly what was imputed.
"""
import numpy as np
import pandas as pd

MAX_INTERP = 6        # 1 h at 10-min
MAX_NEIGHBOR = 24     # 4 h at 10-min


def gap_fill(df, cfg, interp_power=None, max_interp=MAX_INTERP,
             max_neighbor=MAX_NEIGHBOR):
    """Fill missing wind-speed and power gaps per turbine.

    df: long-format flagged 10-min SCADA (needs timestamp, turbine, ws,
        power_kw). Returns (df_out, summary_df).
    """
    df = df.copy()
    if "filled" not in df.columns:
        df["filled"] = 0
        df["fill_method"] = ""

    dt_min = float(df["dt_h"].iloc[0]) * 60.0 if "dt_h" in df.columns else 10.0
    max_interp_n = max(1, int(round(max_interp / dt_min)))
    max_neighbor_n = max(max_interp_n + 1, int(round(max_neighbor / dt_min)))

    summary_rows = []

    for tid, g in df.groupby("turbine", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        mask_ws = g["ws"].isna()
        mask_p = g["power_kw"].isna()

        # ---- wind speed: linear interpolation for short gaps ----
        filled_ws = 0
        if mask_ws.any():
            interp = g["ws"].interpolate(method="linear", limit=max_interp_n,
                                         limit_direction="both")
            # only fill gaps fully inside valid data (no edge extrapolation)
            valid_idx = g["ws"].notna()
            n_valid = valid_idx.cumsum()
            from_start = valid_idx.idxmax() if valid_idx.any() else 0
            from_end = valid_idx[::-1].idxmax() if valid_idx.any() else len(g) - 1
            fill_ok = (g.index >= from_start) & (g.index <= from_end) & mask_ws
            g.loc[fill_ok, "ws"] = interp[fill_ok]
            g.loc[fill_ok, "filled"] = 1
            g.loc[fill_ok, "fill_method"] = "linear_interp"
            filled_ws = int(fill_ok.sum())

        # ---- medium gaps: farm-normalised neighbour imputation ----
        mask_ws = g["ws"].isna()
        filled_nb = 0
        if mask_ws.any():
            # running count of consecutive NaNs
            nans = mask_ws.astype(int)
            grp = (nans != nans.shift()).cumsum()
            lengths = nans.groupby(grp).transform("sum")
            med = (lengths > max_interp_n) & (lengths <= max_neighbor_n) & mask_ws
            if med.any():
                ts = g.loc[med, "timestamp"]
                # farm mean ws at those timestamps (all turbines)
                farm = df[df["timestamp"].isin(ts)].groupby("timestamp")["ws"].mean()
                # turbine level factor = median(ws_t / farm_ws) over valid rows
                j = g[["timestamp", "ws"]].merge(
                    farm.rename("farm"), on="timestamp", how="inner")
                ratio = (j["ws"] / j["farm"].replace(0, np.nan)).dropna()
                factor = float(ratio.median()) if len(ratio) else 1.0
                imputed = ts.map(farm) * factor
                g.loc[med, "ws"] = imputed.values
                g.loc[med, "filled"] = 1
                g.loc[med, "fill_method"] = "neighbour_impute"
                filled_nb = int(med.sum())

        # ---- power: from warranted curve at (filled) wind speed ----
        filled_p = 0
        if mask_p.any() and interp_power is not None:
            need = g["power_kw"].isna() & g["ws"].notna()
            g.loc[need, "power_kw"] = interp_power(g.loc[need, "ws"].values)
            g.loc[need, "filled"] = 1
            g.loc[need, "fill_method"] = g.loc[need, "fill_method"].replace("", "curve_estimate")
            filled_p = int(need.sum())

        summary_rows.append({"turbine": tid,
                             "ws_linear_filled": filled_ws,
                             "ws_neighbour_filled": filled_nb,
                             "power_curve_filled": filled_p,
                             "total_filled": filled_ws + filled_nb + filled_p})

        # write back
        df.loc[df["turbine"] == tid, "ws"] = g["ws"].values
        df.loc[df["turbine"] == tid, "power_kw"] = g["power_kw"].values
        df.loc[df["turbine"] == tid, "filled"] = g["filled"].values
        df.loc[df["turbine"] == tid, "fill_method"] = g["fill_method"].values

    summary = pd.DataFrame(summary_rows)
    return df, summary
