"""SCADA data loading, column standardisation and resampling.

Handles the export conventions of the major OEMs (Vestas, Siemens Gamesa,
Suzlon, Envision, Nordex, Goldwind, Inox ...) via the profiles in oem.py:

  * long format (timestamp, turbine, power, wind speed, ...) or wide format
    (T01_power_kw, T01_wind_speed_mps, ...) with units in the column names
  * European date formats (dd.mm.yyyy) and decimal commas
  * semicolon-delimited CSVs (common in Envision / Chinese OEM exports)
  * separate Date and Time columns (Nordex style)
  * status reported as text ("Running", "Fault", ...) or codes
  * power reported in MW instead of kW
  * Chinese column headers (有功功率, 风速, ...)
"""
import re

import numpy as np
import pandas as pd

from . import oem

TS_ALIASES = ["timestamp", "datetime", "date time", "time stamp", "date", "time"]

# fallback patterns for non-standard column names (most specific first)
POWER_FALLBACK = ["active power", "active_power", "activepower", "gen power",
                  "gen_power", "genpower", "power output", "power_output",
                  "poweroutput", "output", "power", "pavg", "kw", "mw"]
WS_FALLBACK = ["wind speed", "wind_speed", "windspeed", "windspd", "wind",
               "ws", "vavg", "v_avg"]


def _to_numeric_with_euro(df_col):
    """Numeric coercion that also understands '12,5' European decimals."""
    out = pd.to_numeric(df_col, errors="coerce")
    if out.isna().mean() > 0.3 and df_col.dtype == object:
        cleaned = df_col.astype(str).str.replace(",", ".", regex=False)
        out2 = pd.to_numeric(cleaned, errors="coerce")
        if out2.notna().sum() > out.notna().sum():
            out = out2
    return out


def _parse_dates(series):
    """Parse timestamps; prefers the interpretation that parses more values."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    s = series.astype(str).str.strip()
    d1 = pd.to_datetime(s, errors="coerce", dayfirst=False)
    d2 = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if d2.notna().sum() > d1.notna().sum():
        return d2
    return d1


def _find_col(df, aliases, exclude=None):
    for a in aliases:
        na = oem.normalize_col_name(a)
        for col in df.columns:
            if exclude and col in exclude:
                continue
            if na and na in oem.normalize_col_name(col):
                return col
    return None


def _split_wide_column(name, aliases):
    """For 'T01 Active Power (kW)': return (turbine_id, kind).

    Returns (None, None) if the column carries no turbine prefix. Also handles
    compact names like 'T01-P(kW)' / 'T01_WS' / 'T02风速' by stripping the
    trailing unit and matching a short alias at the end of the name.
    """
    norm = oem.normalize_col_name(name)
    pairs = []
    for kind in ("status", "temp", "dir", "ws", "power"):
        for a in aliases[kind]:
            na = oem.normalize_col_name(a)
            if len(na) >= 2:
                pairs.append((na, kind))
    pairs.sort(key=lambda x: -len(x[0]))   # longest alias first
    for na, kind in pairs:
        i = norm.find(na)
        if i > 0:                          # prefix exists
            tid = norm[:i].strip(" _-.:/")
            if tid:
                return tid, kind
    # compact fallback: strip trailing unit, match short alias at the end
    norm2 = re.sub(r"(kwh|mwh|kw|mw|kmh|km/h|m/s|deg|c)$", "", norm)
    for na, kind in [("power", "power"), ("p", "power"),
                     ("windspeed", "ws"), ("wind", "ws"), ("ws", "ws"), ("v", "ws"),
                     ("direction", "dir"), ("dir", "dir"), ("d", "dir"),
                     ("temperature", "temp"), ("temp", "temp"), ("t", "temp"),
                     ("status", "status"), ("state", "status")]:
        if norm2.endswith(na) and len(norm2) > len(na):
            tid = norm2[:-len(na)].strip(" _-.:/")
            if tid:
                return tid, kind
    return None, None


def standardize(df, profile_key="auto", column_overrides=None, column_map=None):
    """Standardize a raw dataframe into long format with known column names.

    column_map: optional dict of {'power': 'My Power Col', 'ws': ...,
    'turbine': ..., 'timestamp': ..., 'dir': ..., 'temp': ..., 'status': ...,
    'curtailment': ...} with the exact column names in the file. When given,
    those columns are used directly and override auto-detection.

    Returns (df_long, resolved_profile_key).
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if profile_key in (None, "auto"):
        profile_key = oem.detect_profile(df.columns, column_overrides)
    aliases = oem.profile_aliases(profile_key, column_overrides)

    # ---- explicit column mapping (user-provided) --------------------------
    if column_map:
        canon = {"timestamp": "timestamp", "turbine": "turbine_id", "power": "power_kw",
                 "ws": "wind_speed_mps", "dir": "nacelle_dir_deg", "temp": "temp_c",
                 "status": "status_code", "curtailment": "curt_flag"}
        for key, col in column_map.items():
            if not col or col not in df.columns or key not in canon:
                continue
            target = canon[key]
            src = df[col]
            if key == "power":
                vals = _to_numeric_with_euro(src)
                if "mw" in str(col).lower():
                    vals = vals * 1000.0
            elif key == "ws":
                vals = _to_numeric_with_euro(src)
                if "km/h" in str(col).lower() or "kmh" in str(col).lower():
                    vals = vals / 3.6
            elif key == "status":
                vals = _status_to_numeric(src)
            elif key == "curtailment":
                vals = (pd.to_numeric(src, errors="coerce") > 0).astype(int)
            elif key == "timestamp":
                vals = src            # parsed in the timestamp section below
            elif key == "turbine":
                vals = src.astype(str)  # keep turbine ids as strings
            else:
                vals = _to_numeric_with_euro(src)
            df[target] = vals
            if col != target and col in df.columns:
                df = df.drop(columns=[col])
        profile_key = "generic"
        aliases = oem.profile_aliases("generic", column_overrides)

    # ---- timestamp -----------------------------------------------------
    # prefer combined datetime columns; only fall back to separate
    # Date + Time columns (Nordex style) or a bare Date/Time column
    ts_col = _find_col(df, ["timestamp", "datetime", "date time", "time stamp"])
    if ts_col is None:
        ts_col = _find_col(df, aliases["timestamp"])   # e.g. Chinese 时间
    if ts_col is not None and any(a in oem.normalize_col_name(ts_col)
                                  for a in ("timestamp", "datetime")):
        df["timestamp"] = _parse_dates(df[ts_col])
        if ts_col != "timestamp":
            df = df.drop(columns=[ts_col])
    else:
        date_col = _find_col(df, ["date"])
        time_col = _find_col(df, ["time"], exclude={date_col} if date_col else None)
        if date_col is not None and time_col is not None and date_col != time_col:
            ts = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
            df["timestamp"] = _parse_dates(ts)
            df = df.drop(columns=[date_col, time_col])
        elif ts_col is not None:      # bare date/time column with full stamps
            df["timestamp"] = _parse_dates(df[ts_col])
            df = df.drop(columns=[ts_col])
        elif date_col is not None:
            df["timestamp"] = _parse_dates(df[date_col])
            df = df.drop(columns=[date_col])
        else:
            df["timestamp"] = pd.NaT

    # ---- turbine column (long format) ------------------------------------
    # find the status column first and exclude it from the turbine search,
    # otherwise "TurbineState" can match the turbine alias "turbine".
    # Turbine/status aliases are the union across ALL profiles so columns like
    # "Unit" are recognised no matter which profile was auto-detected.
    status_col = _find_col(df, oem.all_aliases("status") + aliases["status"])
    exclude_turb = {status_col} if status_col else None
    turb_col = _find_col(df, oem.all_aliases("turbine") + aliases["turbine"],
                         exclude=exclude_turb)
    is_long = turb_col is not None

    if not is_long:
        # wide format: T01_power_kw / T01 Active Power (kW) / T01_风速 ...
        per_turbine = {}
        unit_scale = {}
        for col in df.columns:
            if col == "timestamp":
                continue
            tid, kind = _split_wide_column(col, aliases)
            if tid:
                per_turbine.setdefault(tid, {})[col] = kind
                if "mw" in col.lower():
                    unit_scale[col] = "mw"
                elif "km/h" in col.lower() or "kmh" in col.lower():
                    unit_scale[col] = "kmh"

        if not per_turbine:
            # no prefixed columns: assume a single-turbine farm
            pcol = _find_col(df, aliases["power"])
            wcol = _find_col(df, aliases["ws"])
            if pcol and wcol:
                per_turbine = {"T01": {pcol: "power", wcol: "ws"}}
                if "mw" in pcol.lower():
                    unit_scale[pcol] = "mw"
            else:
                raise ValueError(
                    "Could not detect turbine columns. Expected long format "
                    "(turbine_id column) or wide format (T01_power_kw ...). "
                    f"Columns found: {list(df.columns[:15])}")

        frames = []
        for tid, cols in per_turbine.items():
            sub = pd.DataFrame({"timestamp": df["timestamp"]})
            for col, kind in cols.items():
                vals = _to_numeric_with_euro(df[col])
                if kind == "power":
                    if unit_scale.get(col) == "mw":
                        vals = vals * 1000.0
                    sub["power_kw"] = vals
                elif kind == "ws":
                    if unit_scale.get(col) == "kmh":
                        vals = vals / 3.6
                    sub["ws"] = vals
                elif kind == "dir":
                    sub["dir_deg"] = vals
                elif kind == "temp":
                    sub["temp_c"] = vals
                elif kind == "status":
                    sub["status"] = _status_to_numeric(vals)
                elif kind == "curt":
                    sub["curt_flag"] = (vals > 0).astype(int)
            sub["turbine"] = str(tid)
            frames.append(sub)
        df = pd.concat(frames, ignore_index=True)
    else:
        df["turbine"] = df[turb_col].astype(str).str.strip()
        df = df.drop(columns=[turb_col])
        pcol = _find_col(df, aliases["power"])
        for key, cands in [("power_kw", aliases["power"]), ("ws", aliases["ws"]),
                           ("dir_deg", aliases["dir"]), ("temp_c", aliases["temp"]),
                           ("status", aliases["status"])]:
            col = _find_col(df, cands)
            if col is not None:
                vals = _to_numeric_with_euro(df[col])
                if key == "power_kw":
                    if "mw" in col.lower():
                        vals = vals * 1000.0
                    df[key] = vals
                elif key == "ws":
                    if "km/h" in col.lower() or "kmh" in col.lower():
                        vals = vals / 3.6
                    df[key] = vals
                elif key == "status":
                    df[key] = _status_to_numeric(vals)
                else:
                    df[key] = vals
                if col != key:
                    df = df.drop(columns=[col])
        # explicit curtailment flag
        ccol = _find_col(df, ["curtail", "derat", "power limit", "limit"])
        if ccol is not None:
            df["curt_flag"] = (pd.to_numeric(df[ccol], errors="coerce") > 0).astype(int)
            df = df.drop(columns=[ccol])

    # ---- fallback detection for non-standard column names ------------------
    # (must run BEFORE the keep-list filter, which drops non-canonical names)
    if "power_kw" not in df.columns:
        pcol = _fallback_find(df, POWER_FALLBACK)
        if pcol:
            vals = _to_numeric_with_euro(df[pcol])
            if "mw" in pcol.lower():
                vals = vals * 1000.0
            df["power_kw"] = vals
            if pcol != "power_kw":
                df = df.drop(columns=[pcol])
    if "ws" not in df.columns:
        wcol = _fallback_find(df, WS_FALLBACK)
        if wcol:
            vals = _to_numeric_with_euro(df[wcol])
            if "km/h" in wcol.lower() or "kmh" in wcol.lower():
                vals = vals / 3.6
            df["ws"] = vals
            if wcol != "ws":
                df = df.drop(columns=[wcol])

    keep = [c for c in ["timestamp", "turbine", "power_kw", "ws", "dir_deg",
                        "temp_c", "status", "curt_flag"] if c in df.columns]
    df = df[keep].dropna(subset=["timestamp"]).sort_values(["turbine", "timestamp"])

    if "power_kw" not in df.columns:
        raise ValueError(
            "No power column found. Columns in file: "
            + ", ".join(str(c) for c in df.columns[:30])
            + ". Expected a column containing 'power' (kW or MW), e.g. 'Active Power (kW)'. "
              "If your file uses a custom name, use the Advanced column mapping "
              "(or 'column_map' in the config) to specify it.")
    if "ws" not in df.columns:
        raise ValueError(
            "No wind speed column found. Columns in file: "
            + ", ".join(str(c) for c in df.columns[:30])
            + ". Expected a column containing 'wind speed' (m/s). "
              "If your file uses a custom name, use the Advanced column mapping "
              "(or 'column_map' in the config) to specify it.")

    df["power_kw"] = pd.to_numeric(df["power_kw"], errors="coerce")
    df["ws"] = pd.to_numeric(df["ws"], errors="coerce")

    # power unit auto-detection: values look like MW (< 100) -> scale to kW
    pmax = df["power_kw"].max()
    if np.isfinite(pmax) and pmax < 100:
        df["power_kw"] = df["power_kw"] * 1000.0

    df["power_kw"] = df["power_kw"].clip(lower=-100.0)
    df["ws"] = df["ws"].clip(lower=0.0, upper=60.0)
    if "dir_deg" in df.columns:
        df["dir_deg"] = pd.to_numeric(df["dir_deg"], errors="coerce") % 360.0
    if "curt_flag" in df.columns:
        df["curt_flag"] = (df["curt_flag"] > 0).astype(int)
    return df.reset_index(drop=True), profile_key


def _status_to_numeric(series):
    """Map text status strings to numeric codes; pass numbers through."""
    if series.dtype != object:
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip().str.lower()
    mapped = s.map(oem.TEXT_STATUS)
    num = pd.to_numeric(series, errors="coerce")
    out = num.copy()
    out[mapped.notna()] = mapped[mapped.notna()]
    return out


def resample_10min(df):
    """Resample to a regular 10-minute grid (mean of physicals, last status)."""
    num_cols = [c for c in ["power_kw", "ws", "dir_deg", "temp_c"] if c in df.columns]
    df = df.set_index("timestamp").sort_index()

    agg = {c: "mean" for c in num_cols}
    if "status" in df.columns:
        agg["status"] = "last"
    if "curt_flag" in df.columns:
        agg["curt_flag"] = "max"

    out = df.groupby("turbine").resample("10min").agg(agg)
    out = out.dropna(subset=["power_kw", "ws"], how="all").reset_index()
    return out


def _locate_header_row(path):
    """Find the row that looks like a header when a file has preamble lines."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [f.readline() for _ in range(30)]
    except Exception:
        return 0
    keywords = ("timestamp", "date", "time", "power", "wind", "speed",
                "status", "turbine", "wtg", "wec")
    best, best_score = 0, 0
    for i, line in enumerate(lines):
        low = line.lower()
        score = sum(1 for k in keywords if k in low)
        if score > best_score:
            best, best_score = i, score
    return best


def _sniff_csv(path):
    """Read only the header row to find the delimiter and column names.

    Returns (columns, read_kwargs) where read_kwargs can be passed to
    pd.read_csv for the full (chunked) read. Handles comma / semicolon /
    tab / pipe delimiters and metadata preamble rows.
    """
    candidates = [("comma", {}), ("semicolon", {"sep": ";"}),
                  ("tab", {"sep": "\t"}), ("pipe", {"sep": "|"})]
    best, best_n = None, 0
    for _label, kw in candidates:
        try:
            hdr = pd.read_csv(path, nrows=0, **kw)
            n = len(hdr.columns)
            if n > best_n:
                best, best_n = kw, n
        except Exception:
            continue
    kw = best if best is not None else {}
    columns = list(pd.read_csv(path, nrows=0, **kw).columns)
    if len(columns) < 2:
        skip = _locate_header_row(path)
        columns = list(pd.read_csv(path, nrows=0, skiprows=skip, **kw).columns)
        kw = {**kw, "skiprows": skip}
    return columns, kw


def _matches(nc, aliases):
    """Substring match with a guard against too-short aliases (a single letter
    like 'p' must not match 'internal_param')."""
    for a in aliases:
        na = oem.normalize_col_name(a)
        if len(na) >= 2 and na in nc:
            return True
    return False


def _needed_columns(columns, aliases):
    """Only the columns needed for the analysis (memory saving on huge
    exports with hundreds of channels)."""
    needed = []
    ts_aliases = ["timestamp", "datetime", "date time", "time stamp"] + aliases["timestamp"]
    signal = (aliases["power"] + aliases["ws"] + aliases["dir"] + aliases["temp"]
              + aliases["status"] + ["curtail", "derat", "power limit", "limit"]
              + POWER_FALLBACK + WS_FALLBACK)
    for c in columns:
        nc = oem.normalize_col_name(c)
        if not nc:
            continue
        if _matches(nc, ts_aliases):
            needed.append(c)
            continue
        if _matches(nc, oem.all_aliases("turbine") + aliases["turbine"]):
            needed.append(c)
            continue
        if _matches(nc, signal):
            needed.append(c)
            continue
        tid, _kind = _split_wide_column(c, aliases)
        if tid:
            needed.append(c)
    return needed


def _fallback_find(df, patterns):
    """Find a column whose (normalised) name contains any pattern."""
    for p in patterns:
        np_ = oem.normalize_col_name(p)
        if not np_:
            continue
        for col in df.columns:
            if np_ in oem.normalize_col_name(col):
                return col
    return None


def _load_csv_chunked(path, profile_key, column_overrides, chunksize, use_float32,
                      column_map=None):
    """Stream a (possibly huge) CSV in chunks.

    Only the needed columns are read; each chunk is standardised to long
    format immediately, so memory stays bounded regardless of file size —
    essential for multi-year, multi-hundred-turbine exports.
    """
    columns, kw = _sniff_csv(path)
    if profile_key in (None, "auto"):
        profile_key = oem.detect_profile(columns, column_overrides)
    aliases = oem.profile_aliases(profile_key, column_overrides)
    if column_map:
        usecols = list(column_map.values())
    else:
        usecols = _needed_columns(columns, aliases)

    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, **kw)
    parts = []
    n_chunks = 0
    for chunk in reader:
        dfc, _ = standardize(chunk, profile_key, column_overrides, column_map=column_map)
        parts.append(dfc)
        n_chunks += 1
    if not parts:
        raise ValueError(f"CSV file {path} contains no data rows")
    df = pd.concat(parts, ignore_index=True)
    if use_float32:
        for c in ("power_kw", "ws", "dir_deg", "temp_c", "density_ratio"):
            if c in df.columns:
                df[c] = df[c].astype(np.float32)
    return df, profile_key, n_chunks


def load_scada(path, profile_key="auto", column_overrides=None, column_map=None,
               chunksize=1_000_000, use_float32=False):
    """Load a SCADA file (CSV / Parquet / Excel), standardise and resample.

    Large files are streamed in chunks so memory usage is bounded. Returns
    (df, resolved_profile_key).
    """
    lower = str(path).lower()
    read_info = {"method": "single pass", "chunks": 1, "input_rows": 0}
    if lower.endswith((".xlsx", ".xls")):
        try:
            sheets = pd.read_excel(path, sheet_name=None)
        except Exception:
            sheets = {0: pd.read_excel(path)}
        raw = max(sheets.values(), key=lambda s: len(s))
        read_info["input_rows"] = int(len(raw))
        try:
            df, resolved = standardize(raw, profile_key=profile_key,
                                       column_overrides=column_overrides,
                                       column_map=column_map)
        except ValueError:
            raise ValueError("Could not parse Excel SCADA file: expected a sheet "
                             "with timestamp, turbine and power/wind-speed columns")
    elif lower.endswith(".parquet"):
        try:
            df, resolved = standardize(pd.read_parquet(path), profile_key=profile_key,
                                       column_overrides=column_overrides,
                                       column_map=column_map)
        except ImportError:
            raise ValueError("Parquet support requires pyarrow: pip install pyarrow")
        read_info["method"] = "parquet"
    else:
        try:
            df, resolved, n_chunks = _load_csv_chunked(
                path, profile_key, column_overrides, chunksize, use_float32,
                column_map=column_map)
            read_info["method"] = f"streamed ({n_chunks} chunk{'s' if n_chunks > 1 else ''})"
            read_info["chunks"] = n_chunks
        except ValueError as e:
            raise ValueError(str(e))

    df = resample_10min(df)

    dt = df.groupby("turbine")["timestamp"].diff().dt.total_seconds().median() / 3600.0
    dt = float(dt) if np.isfinite(dt) else 1.0 / 6.0
    df["dt_h"] = dt
    df.attrs["read_info"] = read_info
    return df, resolved
