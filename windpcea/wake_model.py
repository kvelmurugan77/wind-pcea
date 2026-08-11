"""Layout-based wake modelling (DNV-grade).

Implements the Bastankhah & Porte-Agel (2014) Gaussian wake model — the
industry standard used by PyWake / FLORIS / modern WindPRO-class tools:

    sigma_y = k * x + sigma_y0                  (wake width grows downstream)
    sigma_y0 = 0.2 * D * sqrt(1 + 2*TI)         (initial wake width)
    k        = 0.3837 * TI + 0.003678           (wake growth rate)
    deficit(y) = (1 - sqrt(1 - CT/(8*(sigma/D)^2))) * exp(-y^2/(2*sigma^2))

  * rotor-averaged shadowing (Gaussian area average) for partial overlap
  * superposition: sqrt of sum of squares of individual deficits
  * CT from a default thrust-coefficient curve (config-overridable)
  * energy weighting from the measured wind rose (direction x speed bins)

Input: a layout CSV (turbine,x,y in metres) — e.g. from the OEM layout
drawing or a GIS export. Output: farm wake loss %, per-turbine losses and
a per-sector table, plus the free-stream/waked energy split.
"""
import numpy as np
import pandas as pd

# default thrust coefficient curve (typical modern 3-4 MW IEC class turbine)
DEFAULT_CT = pd.DataFrame({
    "v":  [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
           16.0, 18.0, 20.0, 22.0, 24.0, 25.0],
    "ct": [0.90, 0.90, 0.89, 0.87, 0.84, 0.80, 0.74, 0.67, 0.60, 0.53, 0.47,
           0.42, 0.33, 0.27, 0.22, 0.19, 0.16, 0.15],
})


def load_layout(path):
    """Load a layout CSV (turbine,x,y) and return a DataFrame [turbine,x,y]."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = next((c for c in df.columns if c in ("turbine", "turbine_id", "id", "wtg")), None)
    xcol = next((c for c in df.columns if c in ("x", "easting", "east")), None)
    ycol = next((c for c in df.columns if c in ("y", "northing", "north")), None)
    if tcol is None or xcol is None or ycol is None:
        raise ValueError("Layout CSV needs columns: turbine, x, y (metres)")
    out = pd.DataFrame({"turbine": df[tcol].astype(str).str.strip(),
                        "x": pd.to_numeric(df[xcol], errors="coerce"),
                        "y": pd.to_numeric(df[ycol], errors="coerce")}).dropna()
    return out.reset_index(drop=True)


def ct_at(ws, ct_curve=None):
    """Interpolated thrust coefficient at wind speed(s)."""
    curve = ct_curve if ct_curve is not None else DEFAULT_CT
    return float(np.interp(ws, curve["v"], curve["ct"]))


def _rotor_avg_gaussian(sigma, R):
    """Rotor-area average of a Gaussian centered at offset 0:
    (2*sigma^2/R^2) * (1 - exp(-R^2/(2*sigma^2)))."""
    if sigma <= 0:
        return 1.0
    t = R / (np.sqrt(2.0) * sigma)
    if t > 40:
        return 0.0
    return (2.0 * sigma ** 2 / R ** 2) * (1.0 - np.exp(-t ** 2))


def turbine_wake_deficits(layout, wind_dir_deg, ws, D, hub, TI=0.10,
                          ct_curve=None):
    """Effective wind-speed deficit (fraction) at every turbine for one
    (direction, speed) bin, using the Gaussian wake model with rotor-averaged
    partial shadowing and sqrt-of-sums superposition.

    wind_dir_deg: direction the wind comes FROM (meteorological convention).
    """
    n = len(layout)
    x = layout["x"].values.astype(float)
    y = layout["y"].values.astype(float)
    R = D / 2.0
    CT = ct_at(ws, ct_curve)
    sigma0 = 0.2 * D * np.sqrt(1 + 2 * TI)
    k = 0.3837 * TI + 0.003678

    # unit vector pointing DOWNWIND (from where the wind goes to)
    down = np.array([np.sin(np.deg2rad(wind_dir_deg)),
                     np.cos(np.deg2rad(wind_dir_deg))])
    deficits = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # vector from upstream i to downstream j
            dx = (x[j] - x[i]) * down[0] + (y[j] - y[i]) * down[1]
            dy = -(x[j] - x[i]) * down[1] + (y[j] - y[i]) * down[0]
            if dx <= 0.5 * D:            # not downstream of i
                continue
            sigma = k * (dx - 0.5 * D) + sigma0
            sD = sigma / D
            if sD < 1e-3:
                continue
            inner = 1.0 - CT / (8.0 * sD ** 2)
            if inner <= 0:
                centre_def = 1.0          # fully turbulent far wake
            else:
                centre_def = 1.0 - np.sqrt(inner)
            if centre_def <= 0:
                continue
            lateral = np.abs(dy) / (np.sqrt(2.0) * sigma)
            if lateral < 40:
                shadow = np.exp(-lateral ** 2) * _rotor_avg_gaussian(sigma, R)
            else:
                shadow = 0.0
            deficits[j] += (centre_def * shadow) ** 2
    return np.sqrt(deficits)              # sqrt of sum of squares


def farm_wake_loss(layout, rose, v_arr, p_arr, D, hub, TI=0.10,
                   ct_curve=None, cut_in=3.0, cut_out=25.0):
    """Farm wake loss (%) from the measured wind rose.

    rose: DataFrame [dir_deg, ws_bin, freq] — frequencies sum to 1.
    v_arr/p_arr: warranted power curve (m/s, kW) for energy weighting.

    Returns dict: farm_loss_pct, per_turbine (DataFrame), sector table,
    energy_free_mwh, energy_waked_mwh.
    """
    interp_p = np.interp
    n = len(layout)
    E_free = np.zeros(n)
    E_waked = np.zeros(n)
    E_free_total = 0.0
    E_waked_total = 0.0
    sector_rows = []

    for _, row in rose.iterrows():
        wdir = float(row["dir_deg"])
        ws = float(row["ws_bin"])
        f = float(row["freq"])
        if ws < cut_in or ws >= cut_out or f <= 0:
            continue
        defs = turbine_wake_deficits(layout, wdir, ws, D, hub, TI, ct_curve)
        veff = np.maximum(0.5, ws * (1.0 - np.clip(defs, 0.0, 0.95)))
        p_free = interp_p(np.full(n, ws), v_arr, p_arr)
        p_waked = interp_p(veff, v_arr, p_arr)
        e_free = float(p_free.sum() * f)
        e_waked = float(p_waked.sum() * f)
        E_free += p_free * f
        E_waked += p_waked * f
        E_free_total += e_free
        E_waked_total += e_waked
        sector_rows.append({"sector_deg": int(wdir),
                            "freq": f, "mean_deficit": float(np.mean(defs)),
                            "loss_pct": 100.0 * (e_free - e_waked) / max(1e-9, e_free)})

    per = pd.DataFrame({"turbine": layout["turbine"].values,
                        "x": layout["x"].values, "y": layout["y"].values,
                        "energy_free_mwh": E_free,
                        "energy_waked_mwh": E_waked})
    per["wake_loss_mwh"] = per["energy_free_mwh"] - per["energy_waked_mwh"]
    per["wake_loss_pct"] = 100.0 * per["wake_loss_mwh"] / per["energy_free_mwh"].clip(lower=1e-9)
    farm_loss = 100.0 * (E_free_total - E_waked_total) / max(1e-9, E_free_total)
    return {"farm_loss_pct": float(farm_loss),
            "per_turbine": per.sort_values("wake_loss_pct", ascending=False).reset_index(drop=True),
            "sector_table": pd.DataFrame(sector_rows),
            "energy_free_mwh": E_free_total, "energy_waked_mwh": E_waked_total}


def rose_from_wind_stats(rose_counts, sector_width=5.0):
    """Convert the measured (dir5, ws1) counter dict into a frequency
    DataFrame [dir_deg, ws_bin, freq]."""
    rows = []
    for (d, w), c in rose_counts.items():
        rows.append({"dir_deg": d + sector_width / 2.0, "ws_bin": w + 0.5, "freq": c})
    if not rows:
        return pd.DataFrame(columns=["dir_deg", "ws_bin", "freq"])
    df = pd.DataFrame(rows)
    df["freq"] = df["freq"] / df["freq"].sum()
    return df
