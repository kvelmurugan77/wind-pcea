"""Multiple SCADA file support.

Concatenates several SCADA files (e.g. per-year exports, per-turbine-group
exports, multiple sites) into a single analysis dataset with:

  * per-file traceability metadata (file, rows, turbines, period, profile)
  * automatic de-duplication of overlapping (turbine, timestamp) records
    (keep the last file's value — later files win)
  * a per-file scan for period overlap warnings (DNV-style data QA)
"""
import os

import numpy as np
import pandas as pd

from . import scada


def scan_files(files, profile_key="auto", column_overrides=None, column_map=None):
    """Cheap per-file scan (timestamp + turbine columns only): period,
    turbines, overlap detection between files."""
    infos = []
    for f in files:
        try:
            columns, kw = scada._sniff_csv(f)
            ts_col = turb_col = None
            aliases = _aliases_for(columns, profile_key, column_overrides)
            for c in columns:
                nc = _norm(c)
                if ts_col is None and any(_norm(a) in nc for a in
                                          ["timestamp", "datetime", "date time", "time stamp"]):
                    ts_col = c
                if turb_col is None and any(_norm(a) in nc for a in
                                            ["turbine", "turbine id", "turbine_name",
                                             "assetnam", "device name", "wtg", "wec"]):
                    turb_col = c
            usecols = [c for c in (ts_col, turb_col) if c]
            tmin = tmax = None
            n_turb = 0
            for chunk in pd.read_csv(f, usecols=usecols, chunksize=500_000, **kw):
                if ts_col and ts_col in chunk.columns:
                    ts = scada._parse_dates(chunk[ts_col]).dropna()
                    if len(ts):
                        tmin = ts.min() if tmin is None else min(tmin, ts.min())
                        tmax = ts.max() if tmax is None else max(tmax, ts.max())
                if turb_col and turb_col in chunk.columns:
                    n_turb = max(n_turb, chunk[turb_col].nunique())
            infos.append({"file": os.path.basename(f), "start": str(tmin),
                          "end": str(tmax), "turbines": n_turb})
        except Exception as e:
            infos.append({"file": os.path.basename(f), "error": str(e),
                          "start": None, "end": None, "turbines": 0})
    # overlap warning
    warnings = []
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            a, b = infos[i], infos[j]
            if a.get("start") and b.get("start") and a.get("end") and b.get("end"):
                try:
                    if (pd.Timestamp(a["start"]) <= pd.Timestamp(b["end"])
                            and pd.Timestamp(b["start"]) <= pd.Timestamp(a["end"])):
                        warnings.append(
                            f"Files '{a['file']}' and '{b['file']}' overlap in time "
                            f"({a['start']}..{a['end']} vs {b['start']}..{b['end']}) - "
                            "duplicate records will be de-duplicated (later file wins).")
                except Exception:
                    pass
    return infos, warnings


def _norm(c):
    from . import oem
    return oem.normalize_col_name(c)


def _aliases_for(columns, profile_key, column_overrides):
    from . import oem
    if profile_key in (None, "auto"):
        profile_key = oem.detect_profile(columns, column_overrides)
    return oem.profile_aliases(profile_key, column_overrides)


def load_multiple(files, profile_key="auto", column_overrides=None, column_map=None,
                  chunksize=1_000_000, use_float32=False):
    """Load, concatenate and de-duplicate several SCADA files.

    Returns (combined_df, sources, overlap_warnings).
    """
    infos, warnings = scan_files(files, profile_key, column_overrides, column_map)
    frames = []
    sources = []
    for f in files:
        df, prof = scada.load_scada(f, profile_key=profile_key,
                                    column_overrides=column_overrides,
                                    column_map=column_map,
                                    chunksize=chunksize, use_float32=use_float32)
        frames.append(df)
        sources.append({
            "file": os.path.basename(f),
            "rows": int(len(df)),
            "turbines": int(df["turbine"].nunique()),
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
            "profile": prof,
        })

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["turbine", "timestamp"], keep="last")
    dedup = before - len(combined)
    combined = combined.sort_values(["turbine", "timestamp"]).reset_index(drop=True)
    dt = combined.groupby("turbine")["timestamp"].diff().dt.total_seconds().median() / 3600.0
    combined["dt_h"] = float(dt) if np.isfinite(dt) else 1.0 / 6.0
    combined.attrs["data_sources"] = sources
    combined.attrs["overlap_warnings"] = warnings
    combined.attrs["deduped_rows"] = int(dedup)
    combined.attrs["read_info"] = {"method": f"multi-file ({len(files)} files)",
                                   "chunks": len(files)}
    return combined, sources, warnings
