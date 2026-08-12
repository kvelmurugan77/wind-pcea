"""End-to-end post-construction energy yield assessment (PCEYA) pipeline.

run_analysis(cfg, scada_path) -> results dict with:

    df            flagged 10-min SCADA data (long format)
    meta          farm & record metadata
    qc            flag counts, coverage
    power_curve   measured vs warranted curve comparison
    wake          sector wake analysis
    energy        energy accounting (actual vs lost by category)
    availability  per-turbine & monthly availability
    climate       long-term wind climate (MCP) results
    losses        loss tree + gross/net AEP
    uncertainty   Monte Carlo P50/P75/P90/P99
    benchmark     actual vs pre-construction estimate
"""
import os

import numpy as np
import pandas as pd

from . import availability as avail_mod
from . import config as cfg_mod
from . import losses as losses_mod
from . import ltproduction as ltp_mod
from . import mcp as mcp_mod
from . import oem as oem_mod
from . import powercurve as pc_mod
from . import qc as qc_mod
from . import scada as scada_mod
from . import uncertainty as unc_mod
from . import yaw as yaw_mod
from . import wake as wake_mod


def load_warranted_curve(cfg):
    """Load the warranted power curve CSV or build a generic curve."""
    path = cfg.get("warranted_power_curve")
    if path and os.path.exists(path):
        try:
            raw = pd.read_csv(path)
            raw.columns = [str(c).strip().lower() for c in raw.columns]
            vcol = next((c for c in raw.columns if "wind" in c or "speed" in c), None)
            pcol = next((c for c in raw.columns if "power" in c), None)
            if vcol is None or pcol is None:
                raise ValueError("needs wind speed + power columns")
            df = pd.DataFrame({
                "bin_center": pd.to_numeric(raw[vcol], errors="coerce"),
                "mean_power": pd.to_numeric(raw[pcol], errors="coerce"),
            }).dropna().sort_values("bin_center")
            df["count"] = 0
            note = f"Warranted curve loaded from {os.path.basename(path)}"
            return df, note
        except Exception as e:
            raise ValueError(f"Could not read warranted power curve: {e}")
    # generic curve
    rated = float(cfg["rated_power_kw"])
    cut_in = float(cfg["cut_in_mps"])
    v = np.arange(0.0, 45.01, 0.5)
    p = rated * (1.0 - np.exp(-((np.maximum(v - cut_in, 0.0)) / 5.5) ** 3))
    p = np.where(v < cut_in, 0.0, p)
    p = np.where(v >= cfg["cut_out_mps"], 0.0, np.minimum(p, rated))
    df = pd.DataFrame({"bin_center": v, "mean_power": p, "count": 0})
    note = ("Generic warranted curve synthesised from configuration "
            "(no curve file supplied) - replace with the OEM curve for a real assessment")
    return df, note


def run_analysis(cfg, scada_path=None, outdir=None, scada_files=None):
    cfg_mod.validate_config(cfg)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    # ---------------- load & prepare data (single or multiple files) ------
    from . import multifile as multifile_mod
    data_sources = []
    overlap_warnings = []
    if scada_files and len(scada_files) > 1:
        df, data_sources, overlap_warnings = multifile_mod.load_multiple(
            scada_files,
            profile_key=cfg.get("oem_profile", "auto"),
            column_overrides=cfg.get("column_aliases"),
            column_map=cfg.get("column_map") or None,
            chunksize=int(cfg.get("csv_chunk_rows", 1_000_000)),
            use_float32=bool(cfg.get("use_float32", False)))
        profile_key = data_sources[0]["profile"] if data_sources else "generic"
    else:
        path = scada_path or (scada_files[0] if scada_files else None)
        if not path:
            raise ValueError("No SCADA input provided")
        df, profile_key = scada_mod.load_scada(
            path,
            profile_key=cfg.get("oem_profile", "auto"),
            column_overrides=cfg.get("column_aliases"),
            column_map=cfg.get("column_map") or None,
            chunksize=int(cfg.get("csv_chunk_rows", 1_000_000)),
            use_float32=bool(cfg.get("use_float32", False)))
        data_sources = [{"file": os.path.basename(path), "rows": int(len(df)),
                         "turbines": int(df["turbine"].nunique()),
                         "start": str(df["timestamp"].min()),
                         "end": str(df["timestamp"].max()),
                         "profile": profile_key}]
    if df.empty:
        raise ValueError("No valid SCADA records after loading")
    df["turbine"] = df["turbine"].astype(str)
    n_turb = int(df["turbine"].nunique())
    cfg["num_turbines"] = cfg.get("num_turbines") or n_turb
    if cfg["num_turbines"] != n_turb:
        cfg["num_turbines"] = n_turb

    warr, curve_note = load_warranted_curve(cfg)
    v_arr, p_arr = pc_mod.extend_curve(warr, float(cfg["rated_power_kw"]), float(cfg["cut_out_mps"]))
    interp_power = pc_mod.interp_power_factory(v_arr, p_arr)

    # air density correction (if temperature available)
    density = None
    if cfg.get("air_density_correction") and "temp_c" in df.columns:
        ratio = pc_mod.air_density_ratio(df["temp_c"].values, cfg.get("air_pressure_kpa"))
        df["density_ratio"] = ratio
        density = ratio

    df = qc_mod.add_flags(df, cfg, v_arr, p_arr, interp_power)
    # scale expected power by air density so loss accounting is density-consistent
    if density is not None:
        df["expected_power_kw"] = (df["expected_power_kw"] * np.clip(density, 0.5, 1.15))
        df["expected_energy_kwh"] = df["expected_power_kw"] * df["dt_h"]
    df = _gap_fill(df, cfg)   # DNV-style completeness: interpolate short gaps

    # impute expected power for anemometer-fault rows from the farm average
    # (the turbine kept producing; only its anemometer was frozen)
    mask7 = df["flag"] == 7
    if mask7.any():
        farm_avg = df.loc[df["flag"] == 0].groupby("timestamp")["expected_power_kw"].mean()
        imputed = df.loc[mask7, "timestamp"].map(farm_avg)
        df.loc[mask7, "expected_power_kw"] = imputed.fillna(0.0)
        df.loc[mask7, "expected_energy_kwh"] = df.loc[mask7, "expected_power_kw"] * df["dt_h"]

    meta = {
        "farm_name": cfg["farm_name"],
        "turbines": sorted(df["turbine"].unique().tolist()),
        "num_turbines": n_turb,
        "rated_power_kw": cfg["rated_power_kw"],
        "record_start": str(df["timestamp"].min()),
        "record_end": str(df["timestamp"].max()),
        "interval_h": float(df["dt_h"].iloc[0]),
        "rows": int(len(df)),
        "curve_note": curve_note,
        "oem_profile": oem_mod.display_name(profile_key),
        "oem_profile_key": profile_key,
        "read_info": df.attrs.get("read_info", {"method": "single pass", "chunks": 1}),
        "data_sources": data_sources,
        "deduped_rows": int(df.attrs.get("deduped_rows", 0)),
    }

    # ---------------- QC summary ----------------
    flag_counts = qc_mod.flag_summary(df)
    expected_rows = int((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 600.0)
    coverage_pct = 100.0 * len(df) / max(1.0, n_turb * expected_rows)
    qc = {"flag_counts": flag_counts, "coverage_pct": float(coverage_pct),
          "expected_rows_per_turbine": expected_rows,
          "rows": len(df), "n_turbines": n_turb,
          "warnings": _qc_warnings(df, flag_counts) + list(overlap_warnings)}

    # ---------------- energy accounting & availability ----------------
    energy = avail_mod.energy_accounting(df, cfg)
    availability = avail_mod.availability_analysis(df, cfg, energy)

    # ---------------- wind resource statistics (for DNV-style report) ----
    wind_stats = _wind_stats_from_df(df)

    # monthly downtime causes + monthly performance ratio + yearly table
    monthly_downtime = _monthly_downtime_causes(df, cfg)
    op = df[df["flag"] == 0]
    if len(op):
        op2 = op.copy()
        op2["_m"] = op2["timestamp"].dt.to_period("M").astype(str)
        mp = op2.groupby("_m").agg(act=("energy_kwh", "sum"),
                                   exp=("expected_energy_kwh", "sum"))
        monthly_perf = pd.DataFrame({"month": mp.index,
                                     "performance_ratio": mp["act"] / mp["exp"].clip(lower=1e-9)})
    else:
        monthly_perf = pd.DataFrame(columns=["month", "performance_ratio"])
    prod = df[df["flag"].isin((0, 7))]
    yearly = prod.groupby(prod["timestamp"].dt.year).agg(
        measured_mwh=pd.NamedAgg(column="energy_kwh", aggfunc=lambda s: s.sum() / 1000.0),
        hours=pd.NamedAgg(column="flag", aggfunc=lambda s: len(s) * df["dt_h"].iloc[0]),
    )
    gross_y = df.groupby(df["timestamp"].dt.year)["expected_energy_kwh"].agg(
        lambda s: s.sum() / 1000.0).rename("gross_mwh")
    yearly = yearly.join(gross_y).reset_index().rename(columns={"timestamp": "year"})

    # ---------------- wake analysis ----------------
    # only exclude turbines with a significant share of anemometer-fault rows
    # (sporadic low-wind false positives should not drop a turbine from the
    # free-stream pool; their rows are excluded individually anyway)
    f7 = df.loc[df["flag"] == 7].groupby("turbine").size()
    tot = df.groupby("turbine").size()
    f7_share = (f7 / tot).fillna(0.0)
    bad_anem = set(f7_share[f7_share > 0.02].index)
    free = wake_mod.free_stream_ws(df, exclude_turbines=bad_anem,
                                   n_free=int(cfg.get("n_free_turbines", 3)),
                                   sector_width=int(cfg.get("sector_width_deg", 30)))
    wake = wake_mod.wake_analysis(df, v_arr, p_arr, interp_power, cfg, free)
    valid_gross = df["flag"].isin((0, 1, 2, 3, 4, 5))
    gross_period_mwh = float(df.loc[valid_gross, "expected_energy_kwh"].sum() / 1000.0)
    wake["wake_loss_pct"] = wake_mod.wake_loss_share(wake["wake_energy_mwh"], gross_period_mwh)

    # layout-based wake model (Bastankhah-Gaussian) when a layout is supplied
    wake_layout = None
    if cfg.get("layout_file") and os.path.exists(cfg["layout_file"]):
        from . import wake_model as wm
        try:
            layout = wm.load_layout(cfg["layout_file"])
            if len(layout) > 1:
                rose = wm.rose_from_wind_stats(wind_stats.get("rose", {}))
                if len(rose) > 0:
                    wake_layout = wm.farm_wake_loss(
                        layout, rose, v_arr, p_arr,
                        D=float(cfg.get("rotor_diameter_m", 130)),
                        hub=float(cfg["hub_height_m"]),
                        TI=float(cfg.get("wake_ti", 0.10)),
                        ct_curve=None,
                        cut_in=float(cfg["cut_in_mps"]),
                        cut_out=float(cfg["cut_out_mps"]))
                    wake_layout["layout"] = layout
        except Exception as e:
            wake_layout = {"error": str(e)}
    wake["layout"] = wake_layout

    # ---------------- performance (power curve deviation) ----------------
    op = df[df["flag"] == 0]
    # turbine performance = shortfall at the turbine's own wind speed,
    # i.e. excluding the wake component (which is accounted separately)
    op_shortfall_mwh = float((op["expected_energy_kwh"] - op["energy_kwh"]).sum() / 1000.0)
    perf_energy_mwh = op_shortfall_mwh - wake["wake_energy_mwh"]

    # farm measured power curve (density-corrected power, operating data)
    if density is not None:
        op_curve_power = op["power_kw"] / np.clip(op["density_ratio"], 0.3, 1.5)
    else:
        op_curve_power = op["power_kw"]
    farm_curve = pc_mod.bin_power_curve(op["ws"].values, op_curve_power.values,
                                        bin_width=cfg["bin_width_mps"],
                                        min_count=cfg["min_bin_count"])
    dev_pct = pc_mod.energy_weighted_deviation(farm_curve, warr)
    per_turb = []
    per_turb_curves = []
    for tid, g in op.groupby("turbine"):
        g_p = g["power_kw"] / np.clip(g["density_ratio"], 0.3, 1.5) if density is not None else g["power_kw"]
        c = pc_mod.bin_power_curve(g["ws"].values, g_p.values,
                                   bin_width=cfg["bin_width_mps"],
                                   min_count=cfg["min_bin_count"])
        d = pc_mod.energy_weighted_deviation(c, warr)
        pr = float(g["energy_kwh"].sum() / max(1e-6, g["expected_energy_kwh"].sum()))
        per_turb.append({"turbine": tid, "deviation_pct": d, "performance_ratio": pr,
                         "n_operating_rows": len(g)})
        per_turb_curves.append({"turbine": tid, "curve": c})
    per_turb_df = pd.DataFrame(per_turb)
    power_curve = {"farm_curve": farm_curve, "warranted_curve": warr,
                   "deviation_pct": dev_pct, "per_turbine": per_turb_df,
                   "per_turbine_curves": per_turb_curves,
                   "perf_energy_mwh": perf_energy_mwh, "note": curve_note}

    # ---------------- long-term wind climate (MCP) ----------------
    climate = mcp_mod.long_term_climate(cfg, df, cache_dir=outdir)
    lt_A, lt_k = climate["lt_weibull"]

    # ---------------- gross & net AEP ----------------
    # Method A: long-term Weibull x warranted power curve
    gross_lt_mwh_a = n_turb * pc_mod.aep_from_weibull(lt_A, lt_k, v_arr, p_arr)
    # Method B: production regression (daily/monthly energy vs wind speed
    # applied to the long-term record) — the primary method when it works
    production = ltp_mod.lt_production_assessment(cfg, df, climate)
    climate["production"] = production
    # UL/OEPR Step-1 assessment (monthly production regression, Eq. 3-6)
    ul_lt = ltp_mod.ul_lt_assessment(cfg, df, climate, ref=climate.get("ref"))
    climate["ul_lt"] = ul_lt
    if production:
        # sanity check: production-based LT gross must be physically plausible
        # relative to the Weibull x curve method; flag degenerate fits
        for key in ("daily", "monthly"):
            v = production.get(key)
            if v is not None:
                ratio = v["lt_annual_gross_mwh"] / max(1.0, gross_lt_mwh_a)
                if not (0.3 <= ratio <= 2.0):
                    v["unreliable"] = True
                    v["unreliable_reason"] = (
                        f"LT gross {v['lt_annual_gross_mwh']:,.0f} MWh/yr is "
                        f"{ratio:.2f}x the Weibull x curve estimate "
                        f"({gross_lt_mwh_a:,.0f}) — degenerate fit, excluded "
                        f"from primary selection")
        usable = {k: v for k, v in production.items()
                  if isinstance(v, dict) and v.get("lt_annual_gross_mwh")
                  and not v.get("unreliable")}
        if production.get("primary") and usable:
            if production["primary"] not in usable:
                production["primary"] = "daily" if "daily" in usable else \
                    ("monthly" if "monthly" in usable else None)
        if not usable:
            production["lt_gross_mwh"] = None
            production["primary"] = None
            production.setdefault("note", "production regression excluded "
                                           "(unreliable fits)")
    lt_primary = cfg.get("lt_primary_method", "method_a")
    use_b = False
    if production is not None and production.get("lt_gross_mwh"):
        if lt_primary == "method_b":
            use_b = True
        elif lt_primary == "method_b_auto":
            fit_r2 = production.get(production["primary"], {}).get("r2", 0) \
                if production.get("primary") else 0
            diff_pct = abs(100.0 * (production["lt_gross_mwh"] - gross_lt_mwh_a)
                           / gross_lt_mwh_a)
            use_b = production.get("record_months", 0) >= 12 and fit_r2 >= 0.5 \
                and diff_pct <= 15.0
    use_ul = False
    if ul_lt is not None:
        if cfg.get("lt_primary_method") == "ul":
            use_ul = True
        elif cfg.get("lt_primary_method") == "method_b" and ul_lt["r2"] >= 0.5:
            use_ul = True
    if use_ul:
        gross_lt_mwh = ul_lt["lt_gross_mwh"]
        lt_method_used = "UL/OEPR monthly production regression (Step 1)"
    elif use_b:
        gross_lt_mwh = production["lt_gross_mwh"]
        lt_method_used = f"Production regression ({production['primary']})"
    else:
        gross_lt_mwh = gross_lt_mwh_a
        lt_method_used = "Weibull x warranted curve (Method A)"
    wake_used_mwh = wake["wake_energy_mwh"]
    wake_model_used = "SCADA reference-turbine"
    if wake.get("layout") and isinstance(wake["layout"], dict) \
            and "farm_loss_pct" in wake["layout"]:
        # layout-based (Bastankhah-Gaussian) wake loss applied to gross AEP
        if cfg.get("wake_model", "layout") in ("layout", "both"):
            wake_model_used = "Layout-based (Bastankhah-Gaussian)"
            # express the layout wake as an equivalent period energy so the
            # loss tree's % of gross equals the model's farm loss % exactly
            wake_used_mwh = wake["layout"]["farm_loss_pct"] / 100.0 * gross_period_mwh
    tree, net_mwh, recon = losses_mod.build_loss_tree(
        cfg, energy, wake_used_mwh, perf_energy_mwh, gross_period_mwh, gross_lt_mwh)
    wake["wake_model_used"] = wake_model_used
    # UL-style future loss stack (overrides measured losses for LT net)
    fl = cfg.get("future_losses") or {}
    if fl and any(v for v in fl.values()):
        fl_loss = 1.0
        for k in ("availability_pct", "curtailment_pct", "electrical_pct",
                  "blade_degradation_pct"):
            fl_loss *= (1.0 - float(fl.get(k, 0.0)) / 100.0)
        net_mwh = gross_lt_mwh * fl_loss
        recon["future_loss_stack"] = {k: float(fl.get(k, 0.0)) for k in
                                      ("availability_pct", "curtailment_pct",
                                       "electrical_pct", "blade_degradation_pct")}
        recon["future_loss_total_pct"] = 100.0 * (1.0 - fl_loss)

    cf = 100.0 * net_mwh * 1000.0 / (float(cfg["rated_power_kw"]) * n_turb * 8760.0)
    benchmark = None
    if cfg.get("preconstruction_p50_gwh"):
        pre = float(cfg["preconstruction_p50_gwh"]) * 1000.0
        benchmark = {"preconstruction_p50_mwh": pre,
                     "assessment_p50_mwh": None,  # filled after MC
                     "ratio": None, "delta_pct": None}

    # ---------------- uncertainty ----------------
    unc = unc_mod.uncertainty_analysis(cfg, climate, tree, net_mwh,
                                       mc_iterations=cfg["mc_iterations"],
                                       seed=cfg["mc_seed"])
    if benchmark:
        benchmark["assessment_p50_mwh"] = unc["p"]["P50"]
        benchmark["ratio"] = unc["p"]["P50"] / benchmark["preconstruction_p50_mwh"]
        benchmark["delta_pct"] = 100.0 * (unc["p"]["P50"] - benchmark["preconstruction_p50_mwh"]) \
            / benchmark["preconstruction_p50_mwh"]

    results = {
        "cfg": cfg, "meta": meta, "df": df, "qc": qc,
        "power_curve": power_curve, "wake": wake, "energy": energy,
        "availability": availability, "climate": climate,
        "losses": {"tree": tree, "gross_lt_mwh": gross_lt_mwh,
                   "gross_lt_mwh_method_a": gross_lt_mwh_a,
                   "gross_lt_mwh_method_b": production["lt_gross_mwh"] if production else None,
                   "lt_method_used": lt_method_used,
                   "net_mwh": net_mwh, "recon": recon},
        "uncertainty": unc, "benchmark": benchmark,
        "capacity_factor_pct": cf,
        "full_load_hours": net_mwh * 1000.0 / (float(cfg["rated_power_kw"]) * n_turb),
        "v_arr": v_arr, "p_arr": p_arr,
        "wind_stats": wind_stats,
        "monthly_downtime": monthly_downtime,
        "monthly_perf": monthly_perf,
        "yearly": yearly,
        "yaw": yaw_mod.analyse_yaw(df, cfg, v_arr, p_arr, interp_power),
        "per_turbine_losses": per_turbine_loss_matrix(df, wake, energy),
        "gap_stats": df.attrs.get("gap_stats", {"filled_rows": 0, "expected_rows": 0,
                                                "fill_pct": 0.0}),
        "traceability": build_traceability(cfg, meta, qc,
                                           df.attrs.get("gap_stats", {}),
                                           df.attrs.get("read_info", {})),
    }
    return results


def run_files(config_path, scada_path, outdir=None):
    cfg = cfg_mod.load_config(config_path)
    return run_analysis(cfg, scada_path, outdir=outdir)


def _wind_stats_from_df(df):
    """Wind-resource statistics for the report: rose histogram, monthly and
    diurnal mean wind speed, wind-speed distribution, P-P sample.

    Uses ALL valid wind speeds (excluding bad data / anemometer faults),
    INCLUDING below cut-in records, so the distribution and Weibull cover the
    full wind-speed range from 0 m/s (DNV practice). Gap-filled rows are
    included so the distribution represents the complete period.
    """
    op = df[(df["flag"].isin((0, 1))) | (df["gap_filled"] == 1)]  # operating + below cut-in + gap-filled
    op = op[op["ws"].notna()]
    ws = op["ws"]
    stats = {"rose": {}, "monthly_ws": pd.DataFrame(), "diurnal_ws": pd.DataFrame(),
             "ws_hist": pd.DataFrame(), "pp_sample": pd.DataFrame()}
    if len(op) == 0:
        return stats
    if "dir_deg" in op.columns:
        m = op["dir_deg"].notna() & op["ws"].notna()
        d5 = (op.loc[m, "dir_deg"] // 5 * 5).astype(int).values
        w1 = op.loc[m, "ws"].astype(int).values
        comb = d5 * 1000 + w1
        for k, c in zip(*np.unique(comb, return_counts=True)):
            stats["rose"][(int(k // 1000), int(k % 1000))] = int(c)
    mm = op.groupby(op["timestamp"].dt.to_period("M"))["ws"].mean()
    stats["monthly_ws"] = pd.DataFrame({"month": mm.index.astype(str),
                                        "ws_mps": mm.values})
    hh = op.groupby(op["timestamp"].dt.hour)["ws"].mean()
    stats["diurnal_ws"] = pd.DataFrame({"hour": hh.index, "ws_mps": hh.values})
    hist, edges = np.histogram(ws.values, bins=np.arange(0, 42, 1.0))
    stats["ws_hist"] = pd.DataFrame({"bin_center": edges[:-1] + 0.5, "count": hist})
    n = len(op)
    take = op.sample(min(30000, n), random_state=1)
    stats["pp_sample"] = take[["ws", "power_kw", "expected_power_kw"]].reset_index(drop=True)
    return stats


def _monthly_downtime_causes(df, cfg):
    down = df[df["flag"] == 2]
    rows = []
    if len(down) and "status" in df.columns:
        sc = cfg["status_codes"]
        down = down.copy()
        down["month"] = down["timestamp"].dt.to_period("M").astype(str)
        for code, label in [("fault", "Faults"), ("maintenance", "Maintenance"),
                            ("grid", "Grid outage")]:
            codes = set(sc.get(code, []))
            g = down[down["status"].isin(codes)].groupby("month")[
                "expected_energy_kwh"].sum() / 1000.0
            for m, v in g.items():
                rows.append({"month": m, "cause": label, "loss_mwh": float(v)})
        known = [c for k in ("fault", "maintenance", "grid") for c in sc.get(k, [])]
        g = down[~down["status"].isin(known)].groupby("month")[
            "expected_energy_kwh"].sum() / 1000.0
        for m, v in g.items():
            rows.append({"month": m, "cause": "Other", "loss_mwh": float(v)})
    return pd.DataFrame(rows)


def _qc_warnings(df, flag_counts):
    """Diagnostics when results look implausible (usually a status-code
    mapping mismatch or wrong units). Surfaced in the report & console."""
    warnings = []
    total = max(1, len(df))
    op_share = float((df["flag"] == 0).sum()) / total
    has_status = "status" in df.columns and df["status"].notna().any()
    if op_share < 0.10:
        if has_status:
            warnings.append(
                f"Only {100*op_share:.1f}% of records are classified as operating — the "
                "status-code mapping in the config may not match this file. Check "
                "'status_codes' (e.g. Envision uses operating=[0,1], fault=[4]).")
        else:
            warnings.append(
                f"Only {100*op_share:.1f}% of records are classified as operating — check "
                "that power is in kW (not MW) and wind speed in m/s.")
    else:
        down = 0
        if len(flag_counts):
            fc = flag_counts.set_index("flag")
            if 2 in fc.index:
                down = int(fc.loc[2, "count"])
        if has_status and down / total > 0.5:
            warnings.append(
                f"{100*down/total:.1f}% of records are downtime — the status-code mapping "
                "may be inverted or wrong for this file. Check 'status_codes'.")
    return warnings


# ---------------------------------------------------------------------------
# gap filling (DNV-style data completeness)
# ---------------------------------------------------------------------------
def _gap_fill(df, cfg, max_gap_h=6.0):
    """Reindex each turbine to the complete 10-min grid over the record
    period and interpolate wind speed / direction / temperature across gaps
    up to `max_gap_h`. Power is NOT interpolated (energy must come from real
    records). Filled rows are marked gap_filled=1, flag=8 with reason
    'Gap filled (interpolated)', and zero derived energy - so they never
    affect energy accounting or losses, but their wind speed feeds the
    wind-resource distribution and Weibull (full-period representativeness).
    """
    if df.empty:
        df["gap_filled"] = 0
        return df
    dt = pd.Timedelta(minutes=int(round(df["dt_h"].iloc[0] * 60)) or 10)
    df = df.copy()
    df["gap_filled"] = 0
    max_rows = max(1, int(round(max_gap_h * 60 / max(1, dt.total_seconds() / 60))))
    parts = []
    filled_rows = 0
    total_expected = 0
    for tid, g in df.groupby("turbine"):
        if len(g) < 10:
            parts.append(g)
            continue
        full = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq=dt)
        total_expected += len(full)
        g = g.set_index("timestamp").reindex(full)
        filled_mask = g["ws"].isna()
        n_filled = int(filled_mask.sum())
        for col in ("ws", "temp_c"):
            if col in g.columns:
                g[col] = g[col].interpolate(method="linear", limit=max_rows)
        for col in ("dir_deg", "wind_dir_deg"):
            if col in g.columns:
                rad = np.deg2rad(g[col].values)
                xs = pd.Series(np.cos(rad)).interpolate(limit=max_rows).values
                ys = pd.Series(np.sin(rad)).interpolate(limit=max_rows).values
                f = filled_mask.values
                g.loc[f, col] = np.rad2deg(np.arctan2(ys[f], xs[f])) % 360.0
        if "power_kw" in g.columns:
            g.loc[filled_mask, "power_kw"] = np.nan
        if "density_ratio" in g.columns:
            g.loc[filled_mask, "density_ratio"] = np.nan
        g["flag"] = g["flag"].fillna(8).astype(int)
        if "flag_reason" in g.columns:
            g["flag_reason"] = g["flag_reason"].fillna("Gap filled (interpolated)")
        g.loc[filled_mask, "flag"] = 8
        if "flag_reason" in g.columns:
            g.loc[filled_mask, "flag_reason"] = "Gap filled (interpolated)"
        for c in ("energy_kwh", "expected_energy_kwh", "expected_power_kw"):
            if c in g.columns:
                g.loc[filled_mask, c] = 0.0
        g["turbine"] = tid
        g["gap_filled"] = filled_mask.astype(int)
        g["dt_h"] = df["dt_h"].iloc[0]
        parts.append(g.reset_index().rename(columns={"index": "timestamp"}))
        filled_rows += n_filled
    df = pd.concat(parts, ignore_index=True).sort_values(
        ["turbine", "timestamp"]).reset_index(drop=True)
    df.attrs["gap_stats"] = {"filled_rows": int(filled_rows),
                             "expected_rows": int(total_expected),
                             "fill_pct": 100.0 * filled_rows / max(1, total_expected)}
    return df


def per_turbine_loss_matrix(df, wake, energy):
    """Per-WTG loss matrix (MWh and % of WTG gross) for every loss category."""
    gross_by = df.groupby("turbine")["expected_energy_kwh"].sum() / 1000.0
    rows = []
    for tid, g in df.groupby("turbine"):
        gross = float(gross_by.get(tid, 0.0)) or 1e-9
        row = {"turbine": tid}
        for fl, name in [(2, "downtime"), (3, "curtailment"), (4, "derating"),
                         (5, "environmental")]:
            e = float(g.loc[g["flag"] == fl, "expected_energy_kwh"].sum() / 1000.0)
            row[name + "_mwh"] = e
            row[name + "_pct"] = 100.0 * e / gross
        wk = float(wake["per_turbine"].set_index("turbine").reindex([tid])
                   ["wake_energy_mwh"].fillna(0.0).iloc[0]) if len(wake["per_turbine"]) else 0.0
        row["wake_mwh"] = wk
        row["wake_pct"] = 100.0 * wk / gross
        e_act = float(g.loc[g["flag"].isin((0, 7)), "energy_kwh"].sum() / 1000.0)
        row["energy_mwh"] = e_act
        row["gross_mwh"] = gross
        rows.append(row)
    return pd.DataFrame(rows)


def build_traceability(cfg, meta, qc, gap_stats, read_info):
    """DNV/DAKKS-style audit trail: inputs, processing, versions, QA."""
    from . import __version__
    return {
        "tool_version": __version__,
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "farm": meta["farm_name"],
        "record_period": [meta["record_start"], meta["record_end"]],
        "turbines": meta["num_turbines"],
        "read_method": read_info,
        "rows": qc["rows"],
        "coverage_pct": round(qc["coverage_pct"], 2),
        "gap_filling": gap_stats,
        "config_used": {k: v for k, v in cfg.items()
                        if k not in ("column_aliases",)},
        "standards": [
            "IEC 61400-12-1:2022 — power performance measurement, 0.5 m/s binning",
            "IEC 61400-26-1:2019 — availability categories & energy-based definitions",
            "DNV-RP-0126 — uncertainty & probability of exceedance (P-values)",
            "ISO/IEC 17025 / DAKKS-style traceability of inputs, processing and outputs",
        ],
        "methods": {
            "availability": "time-based A_T=(T_cal-T_down)/T_cal; energy-based "
                            "A_E=E_act/(E_act+E_down) with E_down=sum P_warr(v)*dt",
            "power_curve": "IEC 61400-12-1 binning, air-density corrected to 1.225 kg/m3",
            "wake": "reference-turbine free-stream proxy, 30-deg sectors",
            "lt_correction": "MCP sector-wise OLS vs ERA5T reanalysis (or user file)",
            "losses": "multiplicative loss tree on gross AEP",
            "uncertainty": "lognormal 1-sigma components, Monte Carlo, "
                           "Px = exceedance probability (P90 < P75 < P50)",
        },
    }
