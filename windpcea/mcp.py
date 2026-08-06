"""Long-term wind resource correction (Measure-Correlate-Predict).

Converts the short measurement period (SCADA record) into a long-term wind
climate at the site:

  1. Daily mean site wind speed from nacelle data (farm average).
  2. Long-term reference:
       - user-provided file (daily reanalysis / met mast time series), or
       - NASA POWER reanalysis (WS50M) fetched automatically for the
         configured latitude/longitude, or
       - fall back to the measured record only.
  3. Sector-wise linear regression (MCP) of site on reference.
  4. Predict the long-term site time series; fit a Weibull distribution.
"""
import os

import numpy as np
import pandas as pd


def build_site_daily(df):
    """Daily mean wind speed at the site (mean across operating turbines)."""
    sub = df[df["ws"].notna()]
    daily = sub.groupby(sub["timestamp"].dt.date)["ws"].mean()
    daily.index = pd.to_datetime(daily.index)
    return daily


def load_long_term_file(path):
    """Load a long-term daily reference file (CSV: date, ws_mps [, dir_deg])."""
    ref = pd.read_csv(path)
    ref.columns = [str(c).strip().lower() for c in ref.columns]
    date_col = next((c for c in ref.columns if "date" in c or "time" in c), ref.columns[0])
    ws_col = next((c for c in ref.columns if "ws" in c or "wind" in c or "speed" in c), None)
    if ws_col is None:
        raise ValueError(f"Long-term file {path} needs a wind speed column")
    out = pd.DataFrame({
        "date": pd.to_datetime(ref[date_col], errors="coerce"),
        "ws": pd.to_numeric(ref[ws_col], errors="coerce"),
    })
    dir_col = next((c for c in ref.columns if "dir" in c), None)
    if dir_col is not None:
        out["dir"] = pd.to_numeric(ref[dir_col], errors="coerce") % 360.0
    out = out.dropna(subset=["date", "ws"]).set_index("date").sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def fetch_nasa_power_daily(lat, lon, start="20010101", end=None):
    """Fetch daily 50-m wind speed from NASA POWER (MERRA-2) for a point."""
    import datetime as dt
    if end is None:
        end = dt.date.today().strftime("%Y%m%d")
    url = ("https://power.larc.nasa.gov/api/temporal/daily/point"
           f"?parameters=WS50M&community=RE&longitude={lon}&latitude={lat}"
           f"&start={start}&end={end}&format=JSON")
    import requests
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    raw = data["properties"]["parameter"]["WS50M"]
    idx = pd.to_datetime(list(raw.keys()), format="%Y%m%d")
    vals = pd.to_numeric(pd.Series(list(raw.values())), errors="coerce")
    out = pd.DataFrame({"ws": vals.values}, index=idx)
    out = out[(out["ws"] > 0)].dropna()
    return out


def run_mcp(site_daily, ref, ref_ws_col="ws", sector_width=30):
    """Sector-wise linear regression MCP.

    Returns statistics + the predicted long-term site daily series.
    """
    from scipy import stats as sstats
    joined = pd.DataFrame({"site": site_daily, "ref": ref[ref_ws_col]}).dropna()
    if joined.empty or len(joined) < 30:
        return None

    sw = int(sector_width)
    predicted = []
    rows = []
    if "dir" in ref.columns and ref["dir"].notna().any():
        ref_dir = ref["dir"].reindex(joined.index)
        joined["sector"] = (ref_dir // sw * sw).astype(int)
    else:
        joined["sector"] = 0

    for sector, grp in joined.groupby("sector"):
        if len(grp) < 30:
            continue
        slope, intercept, r, p, se = sstats.linregress(grp["ref"], grp["site"])
        lt_idx = ref.index if sector == 0 else ref.index[(ref["dir"] // sw * sw).astype(int) == sector]
        lt_pred = pd.Series(intercept + slope * ref.loc[lt_idx, ref_ws_col], index=lt_idx)
        lt_pred = lt_pred[lt_pred > 0]
        predicted.append(lt_pred)
        rows.append({"sector_deg": int(sector), "n_days": len(grp), "slope": slope,
                     "intercept": intercept, "r2": r ** 2})
    if not predicted:
        return None
    lt_series = pd.concat(predicted).sort_index()
    lt_series = lt_series[~lt_series.index.duplicated(keep="first")]
    lt_series[lt_series < 0] = 0
    r2s = np.array([r["r2"] for r in rows])
    r2 = float(np.average(r2s, weights=[r["n_days"] for r in rows]))
    return {"stats": pd.DataFrame(rows), "r2": r2, "lt_daily": lt_series}


def long_term_climate(cfg, df, cache_dir=None, prefer_file=True):
    """Run the long-term correction pipeline; return a results dict."""
    from .powercurve import weibull_fit

    site_daily = build_site_daily(df)
    method = "measured only (no long-term reference)"
    ref = None
    mcp = None
    lt_ws = None

    # 1) user-provided file
    if prefer_file and cfg.get("long_term_wind_file") and os.path.exists(cfg["long_term_wind_file"]):
        try:
            ref = load_long_term_file(cfg["long_term_wind_file"])
            mcp = run_mcp(site_daily, ref, sector_width=cfg.get("sector_width_deg", 30))
            if mcp is not None:
                method = "MCP against user long-term reference file"
        except Exception:
            ref = None

    # 1b) ERA5 / ERA5T reanalysis (preferred long-term reference)
    if mcp is None and cfg.get("long_term_source") in ("auto", "era5") \
            and cfg.get("latitude") is not None and cfg.get("longitude") is not None:
        try:
            from .era5 import fetch_era5, describe_source
            era = fetch_era5(
                cfg["latitude"], cfg["longitude"],
                hub_height_m=cfg.get("hub_height_m", 100),
                start_year=cfg.get("era5_start_year"),
                end_year=cfg.get("era5_end_year"),
                cache_dir=cache_dir,
                source=cfg.get("era5_source", "open-meteo"))
            ref = era.rename(columns={"ws": "ws"})
            mcp = run_mcp(site_daily, ref, sector_width=cfg.get("sector_width_deg", 30))
            if mcp is not None:
                method = describe_source(cfg["latitude"], cfg["longitude"],
                                         era.index.min().year, era.index.max().year,
                                         cfg.get("era5_source", "open-meteo"))
        except Exception:
            mcp = None
            ref = None

    # 2) NASA POWER reanalysis (fallback)
    if mcp is None and cfg.get("long_term_source") in ("auto", "nasa_power") \
            and cfg.get("latitude") is not None and cfg.get("longitude") is not None:
        try:
            cache = None
            if cache_dir:
                cache = os.path.join(cache_dir, "nasa_power_daily.csv")
            if cache and os.path.exists(cache):
                ref = load_long_term_file(cache)
            else:
                ref = fetch_nasa_power_daily(cfg["latitude"], cfg["longitude"],
                                             start=f"{int(cfg.get('nasa_power_start_year', 2001))}0101")
                if cache:
                    ref.reset_index().rename(columns={"index": "date"}).to_csv(cache, index=False)
            # correct 50 m -> hub height
            alpha = 0.20
            ref["ws"] = ref["ws"] * (float(cfg["hub_height_m"]) / 50.0) ** alpha
            ref = ref.rename(columns={"ws": "ws"})
            mcp = run_mcp(site_daily, ref, sector_width=cfg.get("sector_width_deg", 30))
            if mcp is not None:
                method = "MCP against NASA POWER (MERRA-2) reanalysis"
        except Exception:
            mcp = None

    # 3) fallback
    lt_ws = mcp["lt_daily"] if mcp is not None else None
    if lt_ws is None or len(lt_ws) < 365:
        lt_ws = site_daily
        method = "measured record only (increased wind-resource uncertainty)"

    # Long-term Weibull: the shape k is a site characteristic, so it is taken
    # from the measured record, and the scale A is adjusted to the long-term
    # mean wind speed. (Fitting directly to predicted daily means would
    # underestimate variance and inflate k.) The direct daily fit is kept for
    # reference only.
    from scipy.special import gamma
    meas_daily = site_daily
    meas_10min = df["ws"].dropna()
    meas_10min = meas_10min[meas_10min > 0.5]
    A_m, k_m = weibull_fit(meas_10min.values)
    lt_mean = float(lt_ws.mean()) if len(lt_ws.dropna()) else float("nan")
    A_lt = A_m * lt_mean / max(0.5, float(meas_daily.mean()))
    k_lt = k_m
    A_d, k_d = weibull_fit(lt_ws.values)
    if not np.isfinite(A_lt) or A_lt <= 0:
        A_lt, k_lt = 7.0, 2.0      # defensive defaults for pathological inputs

    result = {
        "method": method,
        "ref": ref,
        "mcp": mcp,
        "lt_daily": lt_ws,
        "lt_weibull": (float(A_lt), float(k_lt)),
        "lt_weibull_daily_fit": (float(A_d), float(k_d)),
        "lt_mean_ws": lt_mean,
        "lt_n_years": float(len(lt_ws) / 365.25),
        "site_daily": meas_daily,
        "meas_weibull": (float(A_m), float(k_m)),
        "meas_10min_mean_ws": float(meas_10min.mean()),
        "meas_mean_ws": float(meas_daily.mean()),
        "record_months": float((df["timestamp"].max() - df["timestamp"].min()).days / 30.44),
    }
    return result


def wind_resource_uncertainty_pct(climate):
    """Empirical 1-sigma wind-resource uncertainty (%) based on MCP quality."""
    if "measured" in climate["method"] or climate["mcp"] is None:
        return 10.0
    r2 = climate["mcp"]["r2"]
    yrs = climate["lt_n_years"]
    months = climate["record_months"]
    sigma = 3.0 + 5.0 * (1.0 - r2) + 1.5 * min(3.0, 24.0 / max(6.0, months)) + \
        1.0 * max(0.0, 5.0 - 0.5 * yrs)
    return float(np.clip(sigma, 2.5, 12.0))
