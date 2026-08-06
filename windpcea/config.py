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
    "long_term_source": "auto",    # auto | file | nasa_power | measured_only
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
    """
    cfg = dict(DEFAULTS)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        _deep_merge(cfg, user)
    if overrides:
        _deep_merge(cfg, overrides)
    return cfg


def _deep_merge(base, extra):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def validate_config(cfg):
    problems = []
    if not (cfg.get("rated_power_kw") and cfg["rated_power_kw"] > 0):
        problems.append("rated_power_kw must be > 0")
    if not (cfg.get("cut_in_mps", 0) < cfg.get("cut_out_mps", 1e9)):
        problems.append("cut_in_mps must be < cut_out_mps")
    if cfg.get("num_turbines") and cfg["num_turbines"] < 1:
        problems.append("num_turbines must be >= 1")
    for key in ("electrical_loss_pct", "other_loss_pct"):
        v = cfg.get(key, 0)
        if v < 0 or v > 50:
            problems.append(f"{key} must be between 0 and 50 %")
    if problems:
        raise ValueError("Configuration problems: " + "; ".join(problems))
    return cfg


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
