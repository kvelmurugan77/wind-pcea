"""Data quality control and operating-state classification for SCADA data.

Every 10-minute record is assigned a single `flag` code, following the spirit
of IEC 61400-26 availability categories:

    0 Operating            5 Environmental (icing / high-temperature derate)
    1 Below cut-in         6 Bad data (spikes, out of range)
    2 Downtime             7 Anemometer fault (frozen / erroneous wind speed)
    3 Curtailment          8 Missing / NaN
    4 Derated / partial load

When SCADA status codes are provided they take precedence; otherwise a
heuristic detector is used (sustained low power at usable wind speeds, etc.).
"""
import numpy as np
import pandas as pd

FLAG_NAMES = {0: "Operating", 1: "Below cut-in", 2: "Downtime",
              3: "Curtailment", 4: "Derated / partial load",
              5: "Environmental", 6: "Bad data", 7: "Anemometer fault",
              8: "Missing / NaN"}


def _status_to_flag(status, cfg):
    sc = cfg["status_codes"]
    operating = set(sc.get("operating", [1, 100]))
    mapping = {}
    for code, flags in [("fault", 2), ("maintenance", 2), ("grid", 2),
                        ("curtailment", 3), ("environmental", 5)]:
        for c in sc.get(code, []):
            mapping[int(c)] = flags
    v = int(status)
    if v in operating:
        return 0
    return mapping.get(v, 2)  # unknown codes -> downtime


def add_flags(df, cfg, v_arr, p_arr, interp_power):
    """Add flag + flag_reason columns to the long-format dataframe."""
    rated = float(cfg["rated_power_kw"])
    cut_in = float(cfg["cut_in_mps"])
    cut_out = float(cfg["cut_out_mps"])
    has_status = "status" in df.columns and df["status"].notna().any()
    if has_status and df["status"].nunique() < 2:
        has_status = False

    df = df.copy()
    # type hygiene: any column used in arithmetic must be numeric, otherwise
    # object-dtype cells (e.g. stray text in a numeric channel) would raise
    # 'can't multiply sequence by non-int of type float'
    for _c in ("power_kw", "ws", "dir_deg", "temp_c", "dt_h"):
        if _c in df.columns:
            df[_c] = pd.to_numeric(df[_c], errors="coerce")
    df["expected_power_kw"] = interp_power(df["ws"].values)
    df["flag"] = 8
    df["flag_reason"] = "Missing / NaN"

    valid = df["power_kw"].notna() & df["ws"].notna()
    df.loc[~valid, "flag_reason"] = "Missing / NaN"

    # bad data / spikes
    bad = valid & ((df["power_kw"] > 1.15 * rated) | (df["power_kw"] < 0)
                   | (df["ws"] < 0) | (df["ws"] > 45) | (df["ws"].isna()))
    df.loc[bad, "flag"] = 6
    df.loc[bad, "flag_reason"] = "Bad data / spike"

    # status-code classification (precedence over heuristics)
    if has_status:
        st_map = df["status"].dropna().apply(lambda s: _status_to_flag(s, cfg))
        df.loc[st_map.index, "flag"] = st_map.values
        df.loc[df["status"].notna(), "flag_reason"] = "SCADA status code"
        # keep spikes flagged as bad data even if status says operating
        df.loc[bad, "flag"] = 6
        df.loc[bad, "flag_reason"] = "Bad data / spike"

    # below cut-in (only if not already flagged worse than operating)
    below = (df["flag"] == 0) & (df["ws"] < cut_in)
    df.loc[below, "flag"] = 1
    df.loc[below, "flag_reason"] = "Below cut-in"

    # heuristic downtime: ws above cut-in+1 but power < 0.5% rated,
    # sustained for >= 1 hour (only where no status code governs)
    op = df["flag"] == 0
    low = op & (df["ws"] > cut_in + 1.0) & (df["power_kw"] < 0.005 * rated)
    df["_low"] = low.astype(int)
    dt_h = df["dt_h"].iloc[0] if "dt_h" in df.columns else 1.0 / 6.0
    min_consec = max(3, int(np.ceil(1.0 / dt_h)))
    roll = (df.groupby("turbine")["_low"]
              .rolling(min_consec, min_periods=min_consec).sum()
              .reset_index(level=0, drop=True))
    sustained = (roll >= min_consec) & df["_low"].astype(bool)
    df.loc[sustained, "flag"] = 2
    df.loc[sustained, "flag_reason"] = "Downtime (low power at usable wind)"

    # heuristic curtailment / derate: operating but well below expected power
    if "curt_flag" in df.columns:
        explicit_curt = (df["flag"] == 0) & (df["curt_flag"] == 1)
        df.loc[explicit_curt, "flag"] = 3
        df.loc[explicit_curt, "flag_reason"] = "Curtailment (SCADA flag)"

    op = df["flag"] == 0
    usable = op & (df["ws"] >= cut_in) & (df["ws"] < cut_out - 1.0)
    ratio = df["power_kw"] / df["expected_power_kw"].clip(lower=1.0)
    min_consec_c = max(3, int(np.ceil(0.5 / dt_h)))
    curt_c = usable & (ratio < 0.30) & (df["expected_power_kw"] > 0.15 * rated)
    der_c = usable & (ratio >= 0.30) & (ratio < 0.85) & (df["expected_power_kw"] > 0.15 * rated)
    for cond, flag, reason in [(curt_c, 3, "Curtailment (low output at usable wind)"),
                               (der_c, 4, "Derated / partial load")]:
        df["_c"] = cond.astype(int)
        roll = (df.groupby("turbine")["_c"]
                  .rolling(min_consec_c, min_periods=min_consec_c).sum()
                  .reset_index(level=0, drop=True))
        hit = (roll >= min_consec_c) & cond
        df.loc[hit, "flag"] = flag
        df.loc[hit, "flag_reason"] = reason
    df = df.drop(columns=["_c", "_low"])

    # environmental: high-temperature derate
    if "temp_c" in df.columns:
        hot = (df["flag"] == 0) & (df["temp_c"] > 39.0) & (ratio < 0.85)
        df.loc[hot, "flag"] = 5
        df.loc[hot, "flag_reason"] = "Environmental (high temperature)"
        ice = (df["flag"] == 0) & (df["temp_c"] < 1.0) & (ratio < 0.5)
        df.loc[ice, "flag"] = 5
        df.loc[ice, "flag_reason"] = "Environmental (icing)"

    # anemometer fault: wind speed frozen for >= 2 h at usable wind speed
    # while other turbines show real variation
    if df["turbine"].nunique() > 1:
        grp = df.groupby("turbine")["ws"]
        std5 = grp.rolling(12, min_periods=12).std().reset_index(level=0, drop=True)
        farm_spread = df.groupby("timestamp")["ws"].transform("std")
        frozen = ((std5 < 0.02 * df["ws"]) & (df["ws"] > 4.0)
                  & (farm_spread > 0.5) & (df["flag"] == 0))
        df.loc[frozen, "flag"] = 7
        df.loc[frozen, "flag_reason"] = "Anemometer fault (frozen)"

    df["flag"] = df["flag"].astype(int)
    df["energy_kwh"] = df["power_kw"] * df["dt_h"]
    df["expected_energy_kwh"] = df["expected_power_kw"] * df["dt_h"]
    return df


def flag_summary(df):
    """Counts of each flag across the farm."""
    vc = df["flag"].value_counts()
    out = pd.DataFrame({"flag": sorted(FLAG_NAMES),
                        "label": [FLAG_NAMES[f] for f in sorted(FLAG_NAMES)]})
    out["count"] = out["flag"].map(vc).fillna(0).astype(int)
    out["pct_of_records"] = 100.0 * out["count"] / max(1, len(df))
    return out[out["count"] > 0].reset_index(drop=True)
