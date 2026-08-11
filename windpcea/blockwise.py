"""Out-of-core (blockwise) analysis engine for very large SCADA datasets.

For files of 1 GB+ (tens of millions of 10-min records) the standard
in-memory pipeline can exceed available RAM. This module processes the data
in time blocks (all turbines per block, so the wake free-stream reference is
computed exactly as in the in-memory path):

  * the file is streamed in chunks; each chunk is standardised, filtered to
    the block time window and accumulated;
  * every block is resampled, quality-flagged and aggregated with vectorised
    operations into an accumulator (counts, sums, bin tables, daily series) —
    raw data is never held in memory as a whole;
  * the accumulator is turned into the same results structure as the
    in-memory pipeline, so reports / Excel / CLI work unchanged.
"""
import os

import numpy as np
import pandas as pd

from . import availability as avail_mod
from . import losses as losses_mod
from .powercurve import weibull_fit_moments
from . import ltproduction as ltp_mod
from . import mcp as mcp_mod
from . import oem as oem_mod
from . import powercurve as pc_mod
from . import qc as qc_mod
from . import scada as scada_mod
from . import uncertainty as unc_mod
from . import wake as wake_mod

PAD_DAYS = 2          # overlap at block edges so rolling-window flags survive


# --------------------------------------------------------------------------
# accumulator
# --------------------------------------------------------------------------
class Accumulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rows = 0
        self.flag_counts = {}
        self.E_actual_mwh = 0.0
        self.E_expected_mwh = 0.0
        self.E_lost = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
        self.downtime_split = {}
        self.per_turbine = {}
        self.monthly = {}
        self.monthly_downtime = {}
        self.monthly_perf = {}
        self.year_rows = {}
        self.bins = {}                 # key('farm'|tid) -> DataFrame parts
        self.bin_parts = []
        self.wake = {"energy_mwh": 0.0, "n_intervals": 0,
                     "sector_ds": {}, "sector_n": {}, "sector_e": {},
                     "turb_ds": {}, "turb_n": {}, "turb_e": {}}
        self.daily = {}                # date -> [gross_mwh, ws_sum, ws_n]
        self.ws_sum = 0.0
        self.ws_sumsq = 0.0
        self.ws_n = 0
        self.wind_rose = {}            # (dir5, ws1) -> count
        self.wind_monthly = {}         # month -> [sum, n]
        self.wind_diurnal = {}         # hour -> [sum, n]
        self.wind_hist = {}            # ws1 -> count
        self.pp_rows = []              # reservoir (list of small frames)
        self.pp_cap = 30000
        self.pp_n = 0

    # -- vectorised per-block aggregation ----------------------------------
    def add_block(self, dfb, tid):
        dt_h = float(dfb["dt_h"].iloc[0])
        self.rows += len(dfb)
        vc = dfb["flag"].value_counts()
        for k, v in vc.items():
            self.flag_counts[k] = self.flag_counts.get(k, 0) + int(v)

        prod = dfb["flag"].isin((0, 7))
        self.E_actual_mwh += float(dfb.loc[prod, "energy_kwh"].sum() / 1000.0)
        self.E_expected_mwh += float(dfb["expected_energy_kwh"].sum() / 1000.0)
        for fl in (2, 3, 4, 5):
            self.E_lost[fl] += float(dfb.loc[dfb["flag"] == fl,
                                             "expected_energy_kwh"].sum() / 1000.0)

        down = dfb[dfb["flag"] == 2]
        if len(down) and "status" in dfb.columns:
            sc = self.cfg["status_codes"]
            for code, label in [("fault", "Faults"), ("maintenance", "Maintenance"),
                                ("grid", "Grid outage")]:
                codes = set(sc.get(code, []))
                m = down["status"].isin(codes)
                self.downtime_split[label] = self.downtime_split.get(label, 0.0) + \
                    float(down.loc[m, "expected_energy_kwh"].sum() / 1000.0)

        pt = self.per_turbine.setdefault(tid, {"hours": 0.0, "downtime_h": 0.0,
                                               "curtailment_h": 0.0, "energy_mwh": 0.0,
                                               "downtime_loss_mwh": 0.0,
                                               "curtailment_loss_mwh": 0.0,
                                               "derating_loss_mwh": 0.0,
                                               "environmental_loss_mwh": 0.0})
        pt["hours"] += len(dfb) * dt_h
        pt["downtime_h"] += float((dfb["flag"] == 2).sum() * dt_h)
        pt["curtailment_h"] += float((dfb["flag"] == 3).sum() * dt_h)
        pt["energy_mwh"] += float(dfb.loc[prod, "energy_kwh"].sum() / 1000.0)
        pt["downtime_loss_mwh"] += float(dfb.loc[dfb["flag"] == 2,
                                                 "expected_energy_kwh"].sum() / 1000.0)
        pt["curtailment_loss_mwh"] += float(dfb.loc[dfb["flag"] == 3,
                                                    "expected_energy_kwh"].sum() / 1000.0)
        pt["derating_loss_mwh"] += float(dfb.loc[dfb["flag"] == 4,
                                                 "expected_energy_kwh"].sum() / 1000.0)
        pt["environmental_loss_mwh"] += float(dfb.loc[dfb["flag"] == 5,
                                                      "expected_energy_kwh"].sum() / 1000.0)

        # monthly aggregates
        mon = dfb["timestamp"].dt.to_period("M")
        for m, g in dfb.groupby(mon):
            key = str(m)
            mm = self.monthly.setdefault(key, {"energy_mwh": 0.0, "downtime_h": 0.0,
                                               "curtailment_h": 0.0, "total_h": 0.0})
            mm["energy_mwh"] += float(g.loc[g["flag"].isin((0, 7)),
                                            "energy_kwh"].sum() / 1000.0)
            mm["downtime_h"] += float((g["flag"] == 2).sum() * dt_h)
            mm["curtailment_h"] += float((g["flag"] == 3).sum() * dt_h)
            mm["total_h"] += len(g) * dt_h
            opm = g[g["flag"] == 0]
            mp = self.monthly_perf.setdefault(key, {"act": 0.0, "exp": 0.0})
            mp["act"] += float(opm["energy_kwh"].sum() / 1000.0)
            mp["exp"] += float(opm["expected_energy_kwh"].sum() / 1000.0)
            dm = g[g["flag"] == 2]
            if len(dm) and "status" in g.columns:
                sc = self.cfg["status_codes"]
                known = [c for k in ("fault", "maintenance", "grid") for c in sc.get(k, [])]
                for code, label in [("fault", "Faults"), ("maintenance", "Maintenance"),
                                    ("grid", "Grid outage")]:
                    codes = set(sc.get(code, []))
                    m2 = dm["status"].isin(codes)
                    self.monthly_downtime[(key, label)] = \
                        self.monthly_downtime.get((key, label), 0.0) + \
                        float(dm.loc[m2, "expected_energy_kwh"].sum() / 1000.0)
                other = dm[~dm["status"].isin(known)]
                self.monthly_downtime[(key, "Other")] = \
                    self.monthly_downtime.get((key, "Other"), 0.0) + \
                    float(other["expected_energy_kwh"].sum() / 1000.0)

        yr = dfb["timestamp"].dt.year
        for y, g in dfb.groupby(yr):
            yy = self.year_rows.setdefault(int(y), {"gross_mwh": 0.0,
                                                    "measured_mwh": 0.0, "hours": 0.0})
            yy["gross_mwh"] += float(g["expected_energy_kwh"].sum() / 1000.0)
            yy["measured_mwh"] += float(g.loc[g["flag"].isin((0, 7)),
                                              "energy_kwh"].sum() / 1000.0)
            yy["hours"] += len(g) * dt_h

        # operating-row statistics (vectorised)
        op = dfb[dfb["flag"] == 0]
        if len(op):
            self.ws_sum += float(op["ws"].sum())
            self.ws_sumsq += float((op["ws"] ** 2).sum())
            self.ws_n += int(len(op))
            self.bin_parts.append((tid, _bins_frame(op, self.cfg)))
            self._wind_accum(op)
            # pp reservoir
            self.pp_n += len(op)
            want = self.pp_cap - sum(len(p) for p in self.pp_rows)
            if want > 0:
                take = op.sample(min(len(op), want), random_state=7)
                self.pp_rows.append(take[["ws", "power_kw", "expected_power_kw"]])

        day = dfb["timestamp"].dt.date
        for d, g in dfb.groupby(day):
            key = pd.Timestamp(d)
            dd = self.daily.setdefault(key, [0.0, 0.0, 0])
            dd[0] += float(g["expected_energy_kwh"].sum() / 1000.0)
            gop = g[g["flag"] == 0]
            if len(gop):
                dd[1] += float(gop["ws"].sum())
                dd[2] += int(len(gop))

    def _wind_accum(self, op):
        ws = op["ws"].values
        hb = np.floor(ws).astype(int)
        for b, c in zip(*np.unique(hb[np.isfinite(ws)], return_counts=True)):
            self.wind_hist[int(b)] = self.wind_hist.get(int(b), 0) + int(c)
        if "dir_deg" in op.columns:
            dirs = op["dir_deg"].values
            m = np.isfinite(dirs) & np.isfinite(ws)
            keys = (np.floor(dirs[m] / 5).astype(int) * 5,
                    np.floor(ws[m]).astype(int))
            comb = keys[0] * 1000 + keys[1]
            for k, c in zip(*np.unique(comb, return_counts=True)):
                self.wind_rose[(int(k // 1000), int(k % 1000))] = \
                    self.wind_rose.get((int(k // 1000), int(k % 1000)), 0) + int(c)
        mm = op.groupby(op["timestamp"].dt.to_period("M"))["ws"].agg(["sum", "count"])
        for m, r in mm.iterrows():
            e = self.wind_monthly.setdefault(str(m), [0.0, 0])
            e[0] += float(r["sum"]); e[1] += int(r["count"])
        hh = op.groupby(op["timestamp"].dt.hour)["ws"].agg(["sum", "count"])
        for h, r in hh.iterrows():
            e = self.wind_diurnal.setdefault(int(h), [0.0, 0])
            e[0] += float(r["sum"]); e[1] += int(r["count"])

    # -- wake -----------------------------------------------------------------
    def add_wake_block(self, dfb_all, free, interp_power):
        dt_h = float(dfb_all["dt_h"].iloc[0])
        op = dfb_all[(dfb_all["flag"] == 0) & dfb_all["ws"].notna()
                     & dfb_all["timestamp"].isin(free.index)]
        if op.empty or free is None or len(free) == 0:
            return
        op = op.copy()
        op["ws_free"] = op["timestamp"].map(free)
        ok = op["ws_free"].notna() & (op["ws_free"] - op["ws"] > 0.1) \
            & (op["ws_free"] > 6.0)
        op["deficit"] = np.where(ok, 1.0 - op["ws"] / op["ws_free"], 0.0)
        dr = op["density_ratio"].clip(0.5, 1.15) if "density_ratio" in op.columns \
            else np.ones(len(op))
        loss = np.where(ok, np.maximum(0.0,
                                       interp_power(op["ws_free"].values)
                                       - interp_power(op["ws"].values)) * dr, 0.0)
        self.wake["energy_mwh"] += float((loss * dt_h).sum() / 1000.0)
        self.wake["n_intervals"] += int(ok.sum())
        v = op[op["ws_free"].notna() & (op["ws_free"] > 6.0)]
        if len(v):
            if "dir_deg" in v.columns:
                sec = (v["dir_deg"] // 30).astype(int) * 30
                for s, g in v.groupby(sec):
                    sk = int(s)
                    self.wake["sector_ds"][sk] = self.wake["sector_ds"].get(sk, 0.0) + \
                        float(g["deficit"].sum())
                    self.wake["sector_n"][sk] = self.wake["sector_n"].get(sk, 0) + len(g)
            for t, g in v.groupby("turbine"):
                self.wake["turb_ds"][t] = self.wake["turb_ds"].get(t, 0.0) + \
                    float(g["deficit"].sum())
                self.wake["turb_n"][t] = self.wake["turb_n"].get(t, 0) + len(g)
                self.wake["turb_e"][t] = self.wake["turb_e"].get(t, 0.0) + \
                    float((g["deficit"] * g["expected_power_kw"] * dt_h).sum() / 1000.0)


def _bins_frame(op, cfg):
    ws = op["ws"].values
    dr = op["density_ratio"].clip(0.3, 1.5).values if "density_ratio" in op.columns \
        else np.ones(len(op))
    p = op["power_kw"].values / dr
    bw = cfg["bin_width_mps"]
    idx = np.floor(ws / bw).astype(int)
    m = np.isfinite(ws) & np.isfinite(p) & (idx >= 0)
    out = pd.DataFrame({"bin": idx[m], "p": p[m], "ws": ws[m]})
    g = out.groupby("bin").agg(p_sum=("p", "sum"), ws_sum=("ws", "sum"), n=("ws", "size"))
    return g.reset_index()


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def scan_scada(path, profile_key, column_overrides, column_map):
    """Stream the file once to find turbines + time range. Only the timestamp
    and turbine columns are read (power/ws are NOT required here)."""
    columns, kw = scada_mod._sniff_csv(path)
    if profile_key in (None, "auto"):
        profile_key = oem_mod.detect_profile(columns, column_overrides)
    aliases = oem_mod.profile_aliases(profile_key, column_overrides)
    turb_aliases = oem_mod.all_aliases("turbine") + aliases["turbine"]
    ts_aliases = ["timestamp", "datetime", "date time", "time stamp"] + aliases["timestamp"]
    ts_col = turb_col = None
    for c in columns:
        nc = oem_mod.normalize_col_name(c)
        if ts_col is None and any(oem_mod.normalize_col_name(a) in nc for a in ts_aliases):
            ts_col = c
        if turb_col is None and any(oem_mod.normalize_col_name(a) in nc for a in turb_aliases):
            turb_col = c
    if column_map:
        ts_col = column_map.get("timestamp", ts_col)
        turb_col = column_map.get("turbine", turb_col)
    usecols = [c for c in (ts_col, turb_col) if c]
    usecols = list(dict.fromkeys(usecols))
    if not usecols:
        raise ValueError("Could not find timestamp/turbine columns in SCADA file")
    turbines = set()
    tmin, tmax = None, None
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000, **kw):
        if turb_col is not None and turb_col in chunk.columns:
            turbines.update(chunk[turb_col].astype(str).str.strip().dropna().unique())
        if ts_col is not None and ts_col in chunk.columns:
            ts = scada_mod._parse_dates(chunk[ts_col]).dropna()
            if len(ts):
                tmin = ts.min() if tmin is None else min(tmin, ts.min())
                tmax = ts.max() if tmax is None else max(tmax, ts.max())
    if not turbines or tmin is None:
        raise ValueError("Could not scan SCADA file (no turbine/timestamp columns)")
    return sorted(turbines), tmin, tmax, profile_key


def run_blockwise(cfg, scada_path, outdir=None, block_days=None):
    from .analysis import load_warranted_curve
    from . import config as cfg_mod

    cfg_mod.validate_config(cfg)   # numeric coercion + required fields
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    turbines, tmin, tmax, profile_key = scan_scada(
        scada_path, cfg.get("oem_profile", "auto"),
        cfg.get("column_aliases"), cfg.get("column_map") or None)
    n_turb = len(turbines)
    cfg["num_turbines"] = n_turb

    span_days = max(1.0, (tmax - tmin).days)
    if block_days is None:
        block_days = int(np.clip(2_000_000 / max(1, n_turb) / 144.0, 15, 365))
    blocks = []
    start = tmin
    step = pd.Timedelta(days=block_days)
    while start < tmax:
        end = min(start + step, tmax + pd.Timedelta(days=1))
        blocks.append((start, end))
        nxt = end - pd.Timedelta(days=PAD_DAYS)
        start = nxt if nxt > start else end   # always advance (no infinite loop)

    acc = Accumulator(cfg)
    warr, curve_note = load_warranted_curve(cfg)
    v_arr, p_arr = pc_mod.extend_curve(warr, float(cfg["rated_power_kw"]), float(cfg["cut_out_mps"]))
    interp_power = pc_mod.interp_power_factory(v_arr, p_arr)

    n_blocks = len(blocks)
    prev_end = None
    for bi, (bstart, bend) in enumerate(blocks):
        dfb_all = _load_block(scada_path, bstart - pd.Timedelta(days=PAD_DAYS),
                              bend, cfg, profile_key, v_arr, p_arr, interp_power)
        if dfb_all.empty:
            continue
        f7 = dfb_all.loc[dfb_all["flag"] == 7].groupby("turbine").size()
        tot = dfb_all.groupby("turbine").size()
        share = (f7 / tot).fillna(0.0)
        bad_anem = set(share[share > 0.02].index)
        free = wake_mod.free_stream_ws(dfb_all, exclude_turbines=bad_anem,
                                       n_free=int(cfg.get("n_free_turbines", 3)),
                                       sector_width=int(cfg.get("sector_width_deg", 30)))
        acc.add_wake_block(dfb_all, free, interp_power)
        # accumulate only the rows that belong to THIS block (blocks overlap
        # by PAD_DAYS for rolling-window continuity — do not double count)
        acc_start = bstart if prev_end is None else max(bstart, prev_end)
        for tid in turbines:
            dfb = dfb_all[dfb_all["turbine"] == tid]
            if dfb.empty:
                continue
            dfb = dfb[(dfb["timestamp"] >= acc_start) & (dfb["timestamp"] < bend)]
            if dfb.empty:
                continue
            acc.add_block(dfb, tid)
        prev_end = bend
        try:
            import resource as _res
            _mb = _res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            _mb = -1
        print(f"  block {bi+1}/{n_blocks} ({len(dfb_all):,} rows, "
              f"peak {_mb:.0f} MB, acc rows {acc.rows:,})")

    results = _finalize(cfg, acc, warr, v_arr, p_arr, interp_power, tmin, tmax,
                        n_turb, turbines, curve_note, outdir, n_blocks, profile_key)
    return results


def _load_block(path, start, end, cfg, profile_key, v_arr, p_arr, interp_power):
    columns, kw = scada_mod._sniff_csv(path)
    aliases = oem_mod.profile_aliases(profile_key, cfg.get("column_aliases"))
    usecols = list(cfg["column_map"].values()) if cfg.get("column_map") \
        else scada_mod._needed_columns(columns, aliases)
    chunk_rows = int(cfg.get("csv_chunk_rows", 250_000))
    parts = []
    ts_name = None
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_rows, **kw):
        # cheap time-window filter on the RAW chunk BEFORE full standardisation
        # (crucial for huge files: only block-relevant rows are transformed)
        if ts_name is None:
            for c in chunk.columns:
                nc = oem_mod.normalize_col_name(c)
                if any(oem_mod.normalize_col_name(a) in nc
                       for a in ["timestamp", "datetime", "date time", "time stamp"]):
                    ts_name = c
                    break
        if ts_name is not None and ts_name in chunk.columns:
            ts = scada_mod._parse_dates(chunk[ts_name])
            chunk = chunk.loc[(ts >= start) & (ts < end)]
        if chunk.empty:
            continue
        dfc, _ = scada_mod.standardize(chunk, profile_key,
                                       cfg.get("column_aliases"),
                                       column_map=cfg.get("column_map") or None)
        if dfc.empty:
            continue
        parts.append(dfc)
    if not parts:
        return pd.DataFrame()
    raw = pd.concat(parts, ignore_index=True)
    df = scada_mod.resample_10min(raw)
    if df.empty:
        return df
    dt = df.groupby("turbine")["timestamp"].diff().dt.total_seconds().median() / 3600.0
    df["dt_h"] = float(dt) if np.isfinite(dt) else 1.0 / 6.0
    density = None
    if cfg.get("air_density_correction") and "temp_c" in df.columns:
        df["density_ratio"] = pc_mod.air_density_ratio(
            df["temp_c"].values, cfg.get("air_pressure_kpa"))
        density = df["density_ratio"].values
    df = qc_mod.add_flags(df, cfg, v_arr, p_arr, interp_power)
    if density is not None:
        df["expected_power_kw"] = df["expected_power_kw"] * np.clip(density, 0.5, 1.15)
        df["expected_energy_kwh"] = df["expected_power_kw"] * df["dt_h"]
    mask7 = df["flag"] == 7
    if mask7.any():
        farm_avg = df.loc[df["flag"] == 0].groupby("timestamp")["expected_power_kw"].mean()
        df.loc[mask7, "expected_power_kw"] = df.loc[mask7, "timestamp"].map(farm_avg).fillna(0.0)
        df.loc[mask7, "expected_energy_kwh"] = df.loc[mask7, "expected_power_kw"] * df["dt_h"]
    return df


def _finalize(cfg, acc, warr, v_arr, p_arr, interp_power, tmin, tmax, n_turb,
              turbines, curve_note, outdir, n_blocks, profile_key):
    # ---- QC -------------------------------------------------------------
    rows = acc.rows
    expected_rows = int((tmax - tmin).total_seconds() / 600.0)
    flag_counts = pd.DataFrame(
        [{"flag": f, "label": qc_mod.FLAG_NAMES[f],
          "count": acc.flag_counts.get(f, 0),
          "pct_of_records": 100.0 * acc.flag_counts.get(f, 0) / max(1, rows)}
         for f in sorted(qc_mod.FLAG_NAMES) if acc.flag_counts.get(f, 0)])
    qc = {"flag_counts": flag_counts, "coverage_pct": 100.0 * rows / max(1, n_turb * expected_rows),
          "expected_rows_per_turbine": expected_rows, "rows": rows, "n_turbines": n_turb,
          "warnings": []}
    op_share = acc.flag_counts.get(0, 0) / max(1, rows)
    if op_share < 0.10:
        qc["warnings"].append(
            f"Only {100*op_share:.1f}% of records are classified as operating — the "
            "status-code mapping in the config may not match this file (check 'status_codes').")

    # ---- energy & availability -------------------------------------------
    energy = {"E_actual_mwh": acc.E_actual_mwh, "E_expected_mwh": acc.E_expected_mwh,
              "E_lost_mwh": acc.E_lost, "downtime_split": acc.downtime_split}
    per_df = pd.DataFrame([dict(turbine=tid, **v) for tid, v in acc.per_turbine.items()])
    per_df["time_avail_pct"] = 100.0 * (1 - per_df["downtime_h"] / per_df["hours"])
    denom = (per_df["energy_mwh"] + per_df["downtime_loss_mwh"]).clip(lower=1e-9)
    per_df["prod_avail_pct"] = 100.0 * per_df["energy_mwh"] / denom
    farm = {"time_avail_pct": float(per_df["time_avail_pct"].mean()),
            "prod_avail_pct": float(per_df["prod_avail_pct"].mean()),
            "energy_mwh": float(per_df["energy_mwh"].sum()),
            "downtime_loss_mwh": float(per_df["downtime_loss_mwh"].sum()),
            "curtailment_loss_mwh": float(per_df["curtailment_loss_mwh"].sum()),
            "derating_loss_mwh": float(per_df["derating_loss_mwh"].sum()),
            "environmental_loss_mwh": float(per_df["environmental_loss_mwh"].sum())}
    monthly = pd.DataFrame([dict(month=k, **v) for k, v in acc.monthly.items()])
    monthly["time_avail_pct"] = 100.0 * (1 - monthly["downtime_h"] / monthly["total_h"])
    monthly = monthly.sort_values("month").reset_index(drop=True)
    availability = {"per_turbine": per_df, "farm": farm, "monthly": monthly}

    # ---- power curve ------------------------------------------------------
    farm_curve = _merge_bins([p for t, p in acc.bin_parts], cfg)
    dev_pct = pc_mod.energy_weighted_deviation(farm_curve, warr)
    per_turb, per_turb_curves = [], []
    for tid in turbines:
        c = _merge_bins([p for t, p in acc.bin_parts if t == tid], cfg)
        d = pc_mod.energy_weighted_deviation(c, warr)
        pt = acc.per_turbine.get(tid, {})
        e_act = pt.get("energy_mwh", 0.0)
        e_lost = pt.get("downtime_loss_mwh", 0.0) + pt.get("curtailment_loss_mwh", 0.0) \
            + pt.get("derating_loss_mwh", 0.0) + pt.get("environmental_loss_mwh", 0.0)
        per_turb.append({"turbine": tid, "deviation_pct": d,
                         "performance_ratio": e_act / max(1e-6, e_act + e_lost),
                         "n_operating_rows": int(c["count"].sum())})
        per_turb_curves.append({"turbine": tid, "curve": c})
    power_curve = {"farm_curve": farm_curve, "warranted_curve": warr,
                   "deviation_pct": dev_pct, "per_turbine": pd.DataFrame(per_turb),
                   "per_turbine_curves": per_turb_curves,
                   "perf_energy_mwh": 0.0, "note": curve_note}

    # ---- wake ---------------------------------------------------------------
    sector_rows = [{"sector_deg": s, "mean_deficit": acc.wake["sector_ds"].get(s, 0.0)
                    / max(1, acc.wake["sector_n"].get(s, 0)),
                    "n_samples": acc.wake["sector_n"].get(s, 0)}
                   for s in sorted(set(acc.wake["sector_ds"]) | set(acc.wake["sector_n"]))]
    turb_rows = [{"turbine": t, "mean_deficit": acc.wake["turb_ds"].get(t, 0.0)
                  / max(1, acc.wake["turb_n"].get(t, 0)),
                  "n_samples": acc.wake["turb_n"].get(t, 0),
                  "wake_energy_mwh": acc.wake["turb_e"].get(t, 0.0)}
                 for t in acc.wake["turb_ds"]]
    wake = {"sector_table": pd.DataFrame(sector_rows,
                                         columns=["sector_deg", "mean_deficit", "n_samples"]),
            "per_turbine": pd.DataFrame(turb_rows,
                                        columns=["turbine", "mean_deficit", "n_samples",
                                                 "wake_energy_mwh"]),
            "wake_energy_mwh": acc.wake["energy_mwh"],
            "wake_loss_pct": 100.0 * acc.wake["energy_mwh"] / max(1e-9, acc.E_expected_mwh),
            "n_intervals": acc.wake["n_intervals"]}

    # ---- climate -----------------------------------------------------------
    daily = pd.Series({d: v[0] for d, v in acc.daily.items()}).sort_index()
    daily.index = pd.to_datetime(daily.index)
    site_daily = pd.Series({d: v[1] / max(1, v[2]) for d, v in acc.daily.items()}).sort_index()
    site_daily.index = pd.to_datetime(site_daily.index)
    df_light = pd.DataFrame({"timestamp": daily.index + pd.Timedelta(hours=12),
                             "ws": site_daily.values, "turbine": "T00", "flag": 0,
                             "dt_h": 24.0, "expected_energy_kwh": daily.values * 1000.0,
                             "energy_kwh": daily.values * 1000.0})
    climate = mcp_mod.long_term_climate(cfg, df_light, cache_dir=outdir)
    if acc.ws_n > 0:
        mu = acc.ws_sum / acc.ws_n
        var = max(0.0, acc.ws_sumsq / acc.ws_n - mu * mu)
        A_m, k_m = weibull_fit_moments(mu, np.sqrt(var))
        climate["meas_weibull"] = (A_m, k_m)
        lt_mean = climate["lt_mean_ws"]
        climate["lt_weibull"] = (A_m * lt_mean / max(0.5, mu), k_m)
        climate["meas_10min_mean_ws"] = mu
    production = ltp_mod.lt_production_assessment(cfg, df_light, climate)
    climate["production"] = production
    lt_A, lt_k = climate["lt_weibull"]

    # ---- losses ------------------------------------------------------------
    gross_lt_mwh_a = n_turb * pc_mod.aep_from_weibull(lt_A, lt_k, v_arr, p_arr)
    use_b = False
    if production is not None and production.get("lt_gross_mwh"):
        if cfg.get("lt_primary_method") == "method_b":
            use_b = True
        elif cfg.get("lt_primary_method") == "method_b_auto":
            r2 = production.get(production["primary"], {}).get("r2", 0) \
                if production.get("primary") else 0
            diff = abs(100.0 * (production["lt_gross_mwh"] - gross_lt_mwh_a) / gross_lt_mwh_a)
            use_b = production.get("record_months", 0) >= 12 and r2 >= 0.5 and diff <= 15.0
    if use_b:
        gross_lt_mwh = production["lt_gross_mwh"]
        lt_method_used = f"Production regression ({production['primary']})"
    else:
        gross_lt_mwh = gross_lt_mwh_a
        lt_method_used = "Weibull x warranted curve (Method A)"
    op_shortfall = float(sum(mp["exp"] - mp["act"] for mp in acc.monthly_perf.values()))
    perf_energy_mwh = op_shortfall - acc.wake["energy_mwh"]
    tree, net_mwh, recon = losses_mod.build_loss_tree(
        cfg, energy, wake["wake_energy_mwh"], perf_energy_mwh,
        acc.E_expected_mwh, gross_lt_mwh)
    unc = unc_mod.uncertainty_analysis(cfg, climate, tree, net_mwh,
                                       mc_iterations=cfg["mc_iterations"],
                                       seed=cfg["mc_seed"])
    cf = 100.0 * net_mwh * 1000.0 / (float(cfg["rated_power_kw"]) * n_turb * 8760.0)
    benchmark = None
    if cfg.get("preconstruction_p50_gwh"):
        pre = cfg["preconstruction_p50_gwh"] * 1000.0
        benchmark = {"preconstruction_p50_mwh": pre,
                     "assessment_p50_mwh": unc["p"]["P50"],
                     "ratio": unc["p"]["P50"] / pre,
                     "delta_pct": 100.0 * (unc["p"]["P50"] - pre) / pre}

    meta = {"farm_name": cfg["farm_name"], "turbines": turbines,
            "num_turbines": n_turb, "rated_power_kw": cfg["rated_power_kw"],
            "record_start": str(tmin), "record_end": str(tmax),
            "interval_h": 1.0 / 6.0, "rows": rows, "curve_note": curve_note,
            "oem_profile": oem_mod.display_name(profile_key),
            "oem_profile_key": profile_key,
            "read_info": {"method": f"blockwise out-of-core ({n_blocks} blocks)"}}
    results = {
        "cfg": cfg, "meta": meta, "df": _sample_frame(acc, tmin),
        "qc": qc, "power_curve": power_curve, "wake": wake, "energy": energy,
        "availability": availability, "climate": climate,
        "losses": {"tree": tree, "gross_lt_mwh": gross_lt_mwh,
                   "gross_lt_mwh_method_a": gross_lt_mwh_a,
                   "gross_lt_mwh_method_b": production["lt_gross_mwh"] if production else None,
                   "lt_method_used": lt_method_used, "net_mwh": net_mwh, "recon": recon},
        "uncertainty": unc, "benchmark": benchmark,
        "capacity_factor_pct": cf,
        "full_load_hours": net_mwh * 1000.0 / (float(cfg["rated_power_kw"]) * n_turb),
        "v_arr": v_arr, "p_arr": p_arr,
        "wind_stats": _wind_stats(acc),
        "yearly": pd.DataFrame([dict(year=y, **v) for y, v in acc.year_rows.items()]),
        "monthly_downtime": pd.DataFrame(
            [{"month": m, "cause": c, "loss_mwh": v}
             for (m, c), v in acc.monthly_downtime.items()]),
        "monthly_perf": pd.DataFrame(
            [{"month": m, "performance_ratio": v["act"] / max(1e-9, v["exp"])}
             for m, v in acc.monthly_perf.items()]),
    }
    if acc.pp_rows:
        results["wind_stats"]["pp_sample"] = pd.concat(acc.pp_rows, ignore_index=True)
    else:
        results["wind_stats"]["pp_sample"] = pd.DataFrame()
    return results


def _merge_bins(parts, cfg):
    if not parts:
        return pd.DataFrame(columns=["bin_center", "count", "mean_ws",
                                     "mean_power", "std_power", "max_power"])
    g = pd.concat(parts, ignore_index=True).groupby("bin").agg(
        p_sum=("p_sum", "sum"), ws_sum=("ws_sum", "sum"), n=("n", "sum"))
    bw = cfg["bin_width_mps"]
    out = pd.DataFrame({"bin_center": g.index * bw + bw / 2,
                        "count": g["n"],
                        "mean_ws": g["ws_sum"] / g["n"],
                        "mean_power": g["p_sum"] / g["n"],
                        "std_power": np.nan, "max_power": np.nan})
    out.loc[out["count"] < cfg["min_bin_count"], ["mean_ws", "mean_power"]] = np.nan
    return out.reset_index(drop=True)


def _wind_stats(acc):
    monthly = pd.DataFrame([{"month": m, "ws_mps": v[0] / max(1, v[1])}
                            for m, v in sorted(acc.wind_monthly.items())])
    diurnal = pd.DataFrame([{"hour": h, "ws_mps": v[0] / max(1, v[1])}
                            for h, v in sorted(acc.wind_diurnal.items())])
    hist = pd.DataFrame([{"bin_center": b + 0.5, "count": v}
                         for b, v in sorted(acc.wind_hist.items())])
    return {"rose": acc.wind_rose, "monthly_ws": monthly, "diurnal_ws": diurnal,
            "ws_hist": hist, "pp_sample": None}


def _sample_frame(acc, tmin):
    rose = acc.wind_rose
    rows = []
    for (d, w), c in rose.items():
        rows.extend([(d, w)] * min(c, 30))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["dir_deg", "ws"])
    df["timestamp"] = pd.Timestamp(tmin) + pd.to_timedelta(np.arange(len(df)) * 10, unit="m")
    df["turbine"] = "T00"
    df["power_kw"] = 0.0
    df["energy_kwh"] = 0.0
    df["expected_energy_kwh"] = 0.0
    df["flag"] = 0
    df["flag_reason"] = "sample"
    return df
