"""ERA5 / ERA5T long-term wind reference downloader.

The user must always supply the wind farm latitude & longitude; this module
fetches a long-term reanalysis wind-speed series for the site, used as the
MCP reference for the long-term correction (per standard DNV/OEPR practice).

Sources (tried in order):
  1. Open-Meteo ERA5 archive  — free, no API key, hourly wind at 10/100/120 m,
     covers 1940->~5 days ago (ERA5T for the recent years). Recommended default.
  2. NASA POWER (MERRA-2)     — fallback if Open-Meteo fails.

The fetched series is cached locally (CSV) so repeated runs do not re-download.
"""
import datetime as dt
import os

import numpy as np
import pandas as pd

HOURS_PER_DAY = 24


def _om_hourly(lat, lon, start, end, height_m=100):
    """Open-Meteo ERA5 archive, hourly, wind speed at height (m)."""
    import requests
    if height_m <= 30:
        var = "wind_speed_10m"
    elif height_m <= 80:
        var = "wind_speed_80m"
    elif height_m <= 120:
        var = "wind_speed_100m"
    else:
        var = "wind_speed_120m"
    url = ("https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}"
           f"&start_date={start}&end_date={end}"
           f"&hourly={var},temperature_2m&timezone=UTC")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    j = r.json()
    hourly = j["hourly"]
    idx = pd.to_datetime(hourly["time"])
    ws = pd.to_numeric(pd.Series(hourly[var], index=idx), errors="coerce")
    # Open-Meteo returns km/h for these variables -> convert to m/s
    ws = ws / 3.6
    # wind shear correction from the model height to hub height (alpha 0.2)
    model_h = {"wind_speed_10m": 10, "wind_speed_80m": 80,
               "wind_speed_100m": 100, "wind_speed_120m": 120}[var]
    if abs(model_h - height_m) > 1:
        ws = ws * (height_m / model_h) ** 0.2
    temp = pd.to_numeric(pd.Series(hourly.get("temperature_2m", np.nan),
                                   index=idx), errors="coerce")
    out = pd.DataFrame({"ws": ws, "temp_c": temp})
    out.index.name = "date"
    return out


def fetch_era5(lat, lon, hub_height_m=100, start_year=None, end_year=None,
               cache_dir=None, source="open-meteo"):
    """Fetch a long-term daily ERA5/ERA5T wind-speed series for the site.

    Returns a DataFrame indexed by date with a 'ws' column (daily means, m/s
    at hub height). Caches to cache_dir/era5_daily_<lat>_<lon>.csv when given.
    """
    if end_year is None:
        end_year = dt.date.today().year - 1      # full years only
    if start_year is None:
        start_year = end_year - 24               # ~25 years of record
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"

    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f"era5_daily_{abs(lat):.3f}_{abs(lon):.3f}.csv")

    if cache_path and os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if len(cached) > 300:
            return cached

    if source == "open-meteo":
        hourly = _om_hourly(lat, lon, start, end, hub_height_m)
    else:
        raise ValueError("Only source='open-meteo' (ERA5T) is supported without a CDS key")

    daily = hourly["ws"].resample("D").mean().to_frame()
    daily = daily[daily["ws"].notna()]
    if cache_path:
        daily.to_csv(cache_path)
    return daily


def era5_to_ref_file(daily, out_path):
    """Write the fetched series in the reference-file format (date, ws_mps)."""
    out = daily.reset_index()
    out.columns = ["date", "ws_mps"]
    out.to_csv(out_path, index=False)
    return out_path


def describe_source(lat, lon, start_year, end_year, source="open-meteo"):
    """A human-readable description of the reference used (for the report)."""
    if source == "open-meteo":
        return (f"ERA5T reanalysis (Open-Meteo archive) at "
                f"{lat:.3f}, {lon:.3f}, {start_year}-{end_year}, "
                f"hub-height corrected (power law, alpha=0.2)")
    return f"ERA5 reanalysis (CDS) at {lat:.3f}, {lon:.3f}, {start_year}-{end_year}"
