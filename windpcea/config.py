"""Configuration handling for the wind farm post-construction energy yield assessment tool.

The configuration follows industry (DNV-style) practice for post-construction
energy yield assessments (PCEYA). All fields are optional except where noted;
sensible defaults are provided.
"""
import json
import os

DEFAULTS = {
    "farm_name": "Wind Farm",
    "latitude": None,
    "longitude": None,
    "hub_height_m": 100.0,
    "rotor_diameter_m": 136.0,
    "rated_power_kw": 2500.0,
    "num_turbines": None,          # inferred from data if not given
    "cut_in_mps": 3.0,
    "cut_out_mps": 25.0,
    "turbine_id_pattern": None,    # regex to extract turbine id from columns (wide format)
    "oem_profile": "auto",         # auto | generic | vestas | sgre | suzlon | envision | nordex | goldwind | inox
    "column_aliases": {},          # optional extra column-name aliases per kind
    "column_map": {},              # exact column names: {"power": "Active Power (kW)", "ws": "...", "turbine": "...", "timestamp": "...", "dir": "...", "temp": "...", "status": "..."}
    "csv_chunk_rows": 1000000,     # rows per chunk when streaming large CSVs
    "use_float32": False,          # halve numeric memory (slightly lower precision)
    "large_file_mode": "auto",     # auto | true (blockwise out-of-core for 1GB+) | false
    "block_days": None,            # block size (days) for blockwise mode; None = auto
    "warranted_power_curve": None, # path to CSV: wind_speed_mps,power_kw
    "long_term_wind_file": None,   # path to CSV: date,ws_mps[,dir_deg]
    "long_term_source": "auto",    # auto | file | era5 | nasa_power | measured_only
    "era5_source": "open-meteo",   # open-meteo (ERA5T, free no key) | cds (needs CDS key)
    "era5_start_year": None,       # default: last full year - 24
    "era5_end_year": None,         # default: last full year
    "lt_primary_method": "method_a",  # method_a | method_b (production regression) | method_b_auto
    "nasa_power_start_year": 2001,
    "air_density_correction": True,
    "air_pressure_kpa": None,      # site mean pressure; if None assume 101.325 (sea level)
    "electrical_loss_pct": 2.0,
    "other_loss_pct": 0.5,
    "preconstruction_p50_gwh": None,
    "status_codes": {
        "operating": [1, 100],
        "fault": [200, 201, 202],
        "maintenance": [300],
        "grid": [400],
        "curtailment": [500],
        "environmental": [600],
    },
    "uncertainty_overrides_pct": {},   # e.g. {"power_curve": 3.0}
    "mc_iterations": 20000,
    "mc_seed": 42,
    "bin_width_mps": 0.5,
    "min_bin_count": 3,
    "sector_width_deg": 30,
}


def load_config(path=None, overrides=None):
    """Load a JSON config file and merge with defaults.

    overrides: dict of keys to override (used by the web UI).
    Numeric fields are coerced to float/int so string-typed values in a
    hand-edited JSON can never break arithmetic ("3450" * 1.15 would
    otherwise raise 'can't multiply sequence by non-int of type float').
    """
    cfg = dict(DEFAULTS)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        _deep_merge(cfg, user)
    if overrides:
        _deep_merge(cfg, overrides)
    _coerce_numeric(cfg)
    return cfg


_FLOAT_FIELDS = ["latitude", "longitude", "hub_height_m", "rotor_diameter_m",
                 "rated_power_kw", "cut_in_mps", "cut_out_mps",
                 "air_pressure_kpa", "electrical_loss_pct", "other_loss_pct",
                 "preconstruction_p50_gwh", "bin_width_mps", "sector_width_deg"]
_INT_FIELDS = ["num_turbines", "min_bin_count", "n_free_turbines",
               "csv_chunk_rows", "mc_iterations", "mc_seed",
               "nasa_power_start_year", "era5_start_year", "era5_end_year",
               "block_days"]
_BOOL_FIELDS = ["air_density_correction", "use_float32"]


def _coerce_numeric(cfg):
    """Coerce known numeric fields AND any numeric-looking string anywhere in
    the config (deep). This makes hand-edited configs with quoted numbers
    ('"rated_power_kw": "3450"') impossible to crash arithmetic with."""
    for k in _FLOAT_FIELDS:
        v = cfg.get(k)
        if v is not None and not isinstance(v, (int, float)):
            try:
                cfg[k] = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"config field '{k}' must be numeric, got: {v!r}")
    for k in _INT_FIELDS:
        v = cfg.get(k)
        if v is not None and not isinstance(v, int):
            try:
                cfg[k] = int(float(v))
            except (TypeError, ValueError):
                raise ValueError(f"config field '{k}' must be an integer, got: {v!r}")
    for k in _BOOL_FIELDS:
        v = cfg.get(k)
        if isinstance(v, str):
            cfg[k] = v.strip().lower() in ("1", "true", "yes", "on")
    _coerce_deep(cfg)
    return cfg


def _coerce_deep(node):
    """Recursively convert any string that is fully a number to float/int.
    Strings like 'auto', 'file', 'T01F', paths and names are untouched."""
    if isinstance(node, dict):
        for k in list(node.keys()):
            node[k] = _coerce_deep(node[k])
    elif isinstance(node, list):
        for i in range(len(node)):
            node[i] = _coerce_deep(node[i])
    elif isinstance(node, str):
        s = node.strip()
        if not s:
            return node
        s_num = s[1:] if s[0] in "+-" else s
        if s_num and s_num.replace(".", "", 1).replace("e", "", 1).isdigit() \
                and s_num.count(".") <= 1:
            try:
                f = float(node)
                if f == int(f) and "." not in s_num and "e" not in s_num.lower():
                    return int(f)
                return f
            except ValueError:
                pass
    return node


def _deep_merge(base, extra):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def validate_config(cfg):
    problems = []
    try:
        _coerce_numeric(cfg)
    except ValueError as e:
        problems.append(str(e))
    if not (cfg.get("rated_power_kw") and float(cfg["rated_power_kw"]) > 0):
        problems.append("rated_power_kw must be > 0")
    if not (float(cfg.get("cut_in_mps", 0)) < float(cfg.get("cut_out_mps", 1e9))):
        problems.append("cut_in_mps must be < cut_out_mps")
    if cfg.get("num_turbines") and int(cfg["num_turbines"]) < 1:
        problems.append("num_turbines must be >= 1")
    for key in ("electrical_loss_pct", "other_loss_pct"):
        v = float(cfg.get(key, 0))
        if v < 0 or v > 50:
            problems.append(f"{key} must be between 0 and 50 %")
    # coordinates are REQUIRED for a proper long-term assessment
    if not (cfg.get("latitude") and cfg.get("longitude")):
        if cfg.get("long_term_wind_file"):
            pass  # a user reference file replaces the need for coordinates
        elif cfg.get("long_term_source") == "measured_only":
            pass  # explicitly measured-only
        else:
            problems.append(
                "latitude and longitude are required (site coordinates) so the "
                "long-term reanalysis reference (ERA5T) can be fetched. Set them "
                "in the config or the web app, or provide 'long_term_wind_file', "
                "or set long_term_source='measured_only'.")
    if problems:
        raise ValueError("Configuration problems: " + "; ".join(problems))
    return cfg


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
