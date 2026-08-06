"""OEM SCADA column profiles for major wind turbine manufacturers.

Each OEM exports SCADA with its own column naming, status conventions, date
formats and units. This module stores the alias lists used for fuzzy column
matching, text-status mappings, and an auto-detection routine.

Supported profiles:
    vestas, sgre (Siemens Gamesa), suzlon, envision, nordex, goldwind,
    inox, generic

Aliases are matched case-insensitively as substrings of the (unit-stripped)
column name, so e.g. "Active Power (kW)" or "ActivePower" or "有功功率(kW)"
are all recognised.
"""
import re

# ---------------------------------------------------------------------------
# text status -> numeric code (mapped to the standard code set used by the
# QC module: 100 operating, 200 fault, 300 maintenance, 400 grid,
# 500 curtailment, 600 environmental)
# ---------------------------------------------------------------------------
TEXT_STATUS = {
    # operating / available
    "running": 100, "production": 100, "producing": 100, "generating": 100,
    "normal": 100, "operating": 100, "online": 100, "connected": 100,
    "standby": 100, "idle": 100, "waiting": 100,
    # faults
    "fault": 200, "error": 200, "alarm": 200, "trip": 200, "failure": 200,
    "failed": 200, "stalled": 200, "grid_fault": 200, "emergency": 200,
    # maintenance
    "maintenance": 300, "service": 300, "manual": 300, "manual_stop": 300,
    "maintenance_stop": 300, "stop": 300, "standstill": 300,
    # grid
    "grid": 400, "network": 400, "no_grid": 400, "grid_loss": 400,
    "grid_outage": 400, "disconnected": 400,
    # curtailment
    "curtail": 500, "curtailment": 500, "derate": 500, "derating": 500,
    "limited": 500, "power_limit": 500, "power_limited": 500,
    "load_limited": 500, "operator_limit": 500,
    # environmental
    "icing": 600, "ice": 600, "high_wind": 600, "high_temp": 600,
    "temp_derate": 600, "cutout": 600,
    # Chinese (Goldwind / Envision CN)
    "运行": 100, "正常": 100, "发电": 100, "并网": 100, "待机": 100,
    "故障": 200, "报警": 200, "停机": 300, "维护": 300, "手动": 300,
    "限电": 500, "限功率": 500, "降额": 500, "结冰": 600, "切出": 600,
}

# ---------------------------------------------------------------------------
# alias lists per profile (most specific first; matched as substrings)
# ---------------------------------------------------------------------------
OEM_PROFILES = {
    "generic": {
        "power": ["active power", "power", "p_avg", "pavg", "power_kw", "powerkw"],
        "ws": ["wind speed", "windspeed", "wind_speed", "ws"],
        "dir": ["nacelle direction", "nacelle position", "wind direction",
                "direction", "nacelle", "dir", "wd"],
        "temp": ["ambient temp", "ambient temperature", "temperature", "temp", "t_avg"],
        "status": ["turbine state", "state code", "status code", "status", "state",
                   "error code"],
        "turbine": ["turbine name", "turbine id", "turbine", "turbineid", "wtg",
                    "wtg id", "wtg name", "wec", "unit", "machine"],
        "timestamp": ["timestamp", "datetime", "date time", "time stamp", "date", "time"],
    },
    "vestas": {
        "power": ["active power", "power", "p_avg"],
        "ws": ["wind speed", "windspeed", "wind_speed"],
        "dir": ["nacelle position", "nacelle direction", "wind direction", "nacelle"],
        "temp": ["ambient temp", "ambient temperature", "temperature", "temp"],
        "status": ["turbine state", "state code", "state", "status"],
        "turbine": ["turbine name", "turbine id", "turbine", "wtg"],
        "timestamp": ["timestamp", "time", "date"],
    },
    "sgre": {   # Siemens Gamesa
        "power": ["activepower", "active power", "power"],
        "ws": ["windspeed", "wind speed", "ws"],
        "dir": ["nacelleposition", "nacelle position", "nacelle", "yaw"],
        "temp": ["ambienttemp", "ambient temperature", "temperature", "temp"],
        "status": ["turbinestate", "state", "status"],
        "turbine": ["turbinename", "turbine name", "wtg", "turbine"],
        "timestamp": ["timestamp", "datetime", "time"],
    },
    "suzlon": {
        "power": ["gen active power", "active power", "genpower", "power"],
        "ws": ["wind speed", "windspeed", "ws"],
        "dir": ["nacelle position", "nacelle direction", "wind direction", "direction"],
        "temp": ["ambient temperature", "temperature", "temp"],
        "status": ["turbine status", "status"],
        "turbine": ["turbine no", "turbine id", "turbine", "wtg no", "wtg"],
        "timestamp": ["date time", "time stamp", "timestamp", "date", "time"],
    },
    "envision": {
        "power": ["active power", "power", "grd_prod", "grid prod", "gridprod",
                  "grprod", "gprod", "prod"],
        "ws": ["wind speed", "windspeed", "amb_wind", "ws"],
        "dir": ["wind direction", "direction", "nac_direc", "nacelle dir", "wd"],
        "temp": ["temperature", "amb_tems", "nac_temp", "temp"],
        "status": ["status code", "sys_stats", "status", "state"],
        "turbine": ["device name", "device_name", "asset name", "assetnam",
                    "turbine id", "turbine", "asset"],
        "timestamp": ["timestamp", "datetime", "pctimest", "pc time", "time", "date"],
    },
    "nordex": {
        "power": ["p-avg", "p_avg", "p avg", "power"],
        "ws": ["v-avg", "v_avg", "wind speed", "windspeed"],
        "dir": ["d-avg", "d_avg", "nacelle", "direction", "dir"],
        "temp": ["t-avg", "t_avg", "temperature", "temp"],
        "status": ["status", "state"],
        "turbine": ["wec", "wec id", "wec no", "turbine", "wtg"],
        "timestamp": ["timestamp", "date", "time"],
    },
    "goldwind": {
        "power": ["有功功率", "active power", "power"],
        "ws": ["风速", "wind speed", "windspeed"],
        "dir": ["机舱位置", "机舱方位", "风向", "nacelle", "direction"],
        "temp": ["环境温度", "温度", "temperature", "temp"],
        "status": ["机组状态", "状态", "status"],
        "turbine": ["机组号", "机组编号", "风机编号", "turbine id", "turbine"],
        "timestamp": ["时间", "timestamp", "date", "time"],
    },
    "inox": {
        "power": ["active power", "gen power", "power"],
        "ws": ["wind speed", "windspeed", "ws"],
        "dir": ["nacelle direction", "wind direction", "direction", "nacelle"],
        "temp": ["ambient temperature", "temperature", "temp"],
        "status": ["turbine status", "status"],
        "turbine": ["turbine id", "turbine no", "wtg no", "wtg", "turbine"],
        "timestamp": ["date time", "time stamp", "timestamp", "date", "time"],
    },
}

DISPLAY_NAMES = {
    "vestas": "Vestas", "sgre": "Siemens Gamesa (SGRE)", "suzlon": "Suzlon",
    "envision": "Envision", "nordex": "Nordex", "goldwind": "Goldwind",
    "inox": "Inox Wind", "generic": "Generic",
}

_PROFILE_PRIORITY = ["vestas", "sgre", "suzlon", "envision", "nordex",
                     "goldwind", "inox", "generic"]


def normalize_col_name(name):
    """Lowercase, strip units in parentheses/brackets, collapse whitespace."""
    n = str(name).strip().lower()
    n = re.sub(r"[\(\[].*?[\)\]]", "", n)          # drop (kW), [m/s], (deg) ...
    n = re.sub(r"\s+", "", n)                       # collapse spaces
    return n


def match_kind(column, aliases):
    """Return the alias that matches `column`, or None."""
    norm = normalize_col_name(column)
    if not norm:
        return None
    for alias in aliases:
        a = normalize_col_name(alias)
        if a and a in norm:
            return alias
    return None


def profile_aliases(profile_key, overrides=None):
    """Alias lists for a profile, with optional user overrides merged on top."""
    key = profile_key if profile_key in OEM_PROFILES else "generic"
    aliases = {k: list(v) for k, v in OEM_PROFILES[key].items()}
    if overrides:
        for kind, extra in overrides.items():
            if kind in aliases:
                aliases[kind] = list(extra) + aliases[kind]
    return aliases


def all_aliases(kind):
    """Union of an alias kind across ALL profiles (used for the turbine and
    status columns, where a column like 'Unit' must be recognised regardless
    of which OEM profile was auto-detected)."""
    out = []
    for prof in OEM_PROFILES.values():
        out.extend(prof.get(kind, []))
    return list(dict.fromkeys(out))


def detect_profile(columns, overrides=None):
    """Pick the OEM profile whose aliases match the most columns."""
    cols = [normalize_col_name(c) for c in columns]
    best, best_score = "generic", 0
    for key in _PROFILE_PRIORITY:
        if key == "generic":
            continue
        aliases = profile_aliases(key, overrides)
        score = 0
        for col in cols:
            for kind in ("power", "ws", "dir", "temp", "status", "turbine", "timestamp"):
                if any(normalize_col_name(a) in col for a in aliases[kind]):
                    score += 1
                    break
        if score > best_score:
            best, best_score = key, score
    return best


def display_name(key):
    return DISPLAY_NAMES.get(key, key)
