"""Generate a realistic synthetic SCADA dataset for a 12-turbine wind farm
(30 MW, Kayathar region, Tamil Nadu — SW monsoon dominated) plus a warranted
power curve and a long-term daily wind speed reference file.

Usage:  python generate_sample.py   (writes into this directory)
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- farm parameters ----------------
N_TURB = 12
RATED = 2500.0          # kW
CUT_IN, CUT_OUT = 3.0, 25.0
HUB, D = 100.0, 136.0   # m
LAT, LON = 9.0, 77.8    # Kayathar / Tiruppur region, Tamil Nadu
PERIOD = pd.date_range("2024-01-01", "2024-12-31 23:50", freq="10min")

rng = np.random.default_rng(20240805)

# ---------------- wind model (free stream at hub) ----------------
def weibull_sample(u, A, k):
    return A * (-np.log(np.clip(1.0 - u, 1e-9, 1 - 1e-9))) ** (1.0 / k)


def ar_series(n, rho, sigma, seed):
    r = np.random.default_rng(seed)
    x = np.zeros(n + 60)
    for t in range(1, len(x)):
        x[t] = rho * x[t - 1] + r.normal(0, sigma)
    return x[60:]


# master daily climate 2000-2024 (this becomes the long-term reference file,
# so MCP sees a realistic, well-correlated reanalysis-like dataset)
lt_dates = pd.date_range("2000-01-01", periods=int(
    (pd.Timestamp("2024-12-31") - pd.Timestamp("2000-01-01")).days) + 1, freq="D")
lt_season = 1.0 + 0.38 * np.cos(2 * np.pi * (lt_dates.dayofyear.values - 240) / 365.0)
x_daily = ar_series(len(lt_dates), 0.88, 1.0, 7)
u_daily = 1 - pd.Series(x_daily).rank(pct=True).values
ref_ws_daily = weibull_sample(u_daily, 8.1 * lt_season, 2.1)   # long-term reference

# site daily means for 2024 = reference 2024 with a measurement scatter
site_daily_ws = ref_ws_daily[lt_dates.year == 2024] * (1 + rng.normal(0, 0.10, 366))
site_daily_ws = pd.Series(site_daily_ws, index=lt_dates[lt_dates.year == 2024])

# expand daily means to 10-min with diurnal cycle + hourly autocorrelated turbulence
daily_rep = np.repeat(site_daily_ws.values, 144)[:len(PERIOD)]
hour_of_day = PERIOD.hour.values + PERIOD.minute.values / 60.0
diurnal = 1.0 + 0.14 * np.cos(2 * np.pi * (hour_of_day - 17) / 24.0)
n_h = int(np.ceil(len(PERIOD) / 6))
x = ar_series(n_h, 0.94, 0.9, 11)
x10 = np.repeat(x, 6)[:len(PERIOD)]
x10 = (x10 - x10.mean()) / x10.std()
ws_free = np.maximum(0.3, daily_rep * diurnal * (1 + 0.22 * x10))

# ---------------- direction (von Mises mixture) ----------------
def vm(theta, mu, kappa):
    return np.exp(kappa * np.cos(theta - mu)) / (2 * np.pi)

theta = np.deg2rad(np.arange(0, 360, 5))
weights = 0.45 * vm(theta, np.deg2rad(185), 3.0) + \
          0.30 * vm(theta, np.deg2rad(300), 2.0) + \
          0.25 * vm(theta, np.deg2rad(60), 2.0)
weights /= weights.sum()
dirs = np.random.choice(np.arange(0, 360, 5), size=len(PERIOD), p=weights).astype(float)
dirs += rng.normal(0, 6, len(PERIOD)) % 360

# ---------------- layout & wake model ----------------
# 3 rows x 4 columns, spacing 5D (east) x 4D (north)
xs = np.tile(np.arange(4) * 5 * D, 3)
ys = np.repeat(np.arange(3) * 4 * D, 4)
pos = np.stack([xs, ys], axis=1)


def wake_deficits(wdir):
    u = np.array([np.sin(np.deg2rad(wdir)), np.cos(np.deg2rad(wdir))])
    n = np.array([-u[1], u[0]])
    deficits = np.zeros(N_TURB)
    for i in range(N_TURB):
        for j in range(N_TURB):
            if i == j:
                continue
            d = (pos[j] - pos[i]) @ u
            l = abs((pos[j] - pos[i]) @ n)
            if d > 1.0 * D:
                sigma = (0.45 + 0.05 * d / D) * D     # wake spreads downstream
                deficits[j] += 0.22 * np.exp(-(l ** 2) / (2 * sigma ** 2)) \
                               * np.exp(-d / (5.0 * D))
    return np.clip(deficits, 0, 0.30)


# ---------------- power curve (same formula as the tool's generic curve) ----------------
def power_curve(v):
    p = RATED * (1.0 - np.exp(-(np.maximum(v - CUT_IN, 0.0) / 5.5) ** 3))
    p = np.where(v < CUT_IN, 0.0, p)
    return np.where(v >= CUT_OUT, 0.0, np.minimum(p, RATED))


perf = np.ones(N_TURB)
perf[6] = 0.94     # T07 degraded (e.g. blade soiling / pitch misalignment)
perf[11] = 0.97    # T12 slightly degraded

# ---------------- status event schedule ----------------
def rnd_events(n, t_min, t_max):
    starts = sorted(rng.choice(len(PERIOD), size=n, replace=False))
    out = []
    for s in starts:
        dur = rng.integers(int(t_min * 6), int(t_max * 6))
        out.append((s, min(s + dur, len(PERIOD) - 1)))
    return out

faults = {tid: rnd_events(rng.integers(2, 6), 2, 26) for tid in range(N_TURB)}
maint = {tid: rnd_events(rng.integers(1, 2), 8, 36) for tid in range(N_TURB)}
grid_out = rnd_events(6, 1, 10)
curtail = rnd_events(8, 3, 22)          # farm-wide grid curtailment
T03_curt = rnd_events(1, 40, 60)        # single-turbine curtailment

# ---------------- assemble records ----------------
rows = []
for i, tid in enumerate(range(1, N_TURB + 1)):
    t = PERIOD
    wdir = dirs.copy()
    wdir = np.mod(wdir + rng.normal(0, 4, len(t)), 360)
    wdir = np.minimum(wdir, 359.9)
    wdef = np.array([wake_deficits(d) for d in wdir[::6]])   # hourly -> 10min
    wdef = np.repeat(wdef, 6, axis=0)[:len(t)]
    ws_i = ws_free * (1 - wdef[:, i]) * (1 + rng.normal(0, 0.010, len(t)))
    ws_i = np.clip(ws_i, 0, 40)

    # temperature: seasonal + diurnal
    temp = (27 + 6.5 * np.sin(2 * np.pi * (PERIOD.dayofyear.values - 100) / 365)
            + 4.0 * np.sin(2 * np.pi * (PERIOD.hour.values - 14) / 24)
            + rng.normal(0, 1.2, len(t)))
    status = np.full(len(t), 100, dtype=int)
    # actual air density (site pressure ~99.2 kPa at ~200 m asl) -> power scales with rho
    rho_ratio = (99.2 * 1000.0 / (287.05 * (temp + 273.15))) / 1.225
    power = power_curve(ws_i) * rho_ratio * perf[i] * (1 + rng.normal(0, 0.025, len(t)))
    power = np.clip(power, 0, RATED * 1.02)

    for s, e in faults[i]:
        status[s:e] = 200
        power[s:e] = 0.0
    for s, e in maint[i]:
        status[s:e] = 300
        power[s:e] = 0.0
    for s, e in grid_out:
        status[s:e] = 400
        power[s:e] = 0.0
    for s, e in curtail:
        status[s:e] = 500
        power[s:e] = np.minimum(power[s:e], 0.30 * RATED)
    if i == 3:
        for s, e in T03_curt:
            status[s:e] = 500
            power[s:e] = np.minimum(power[s:e], 0.5 * RATED)
    hot = (temp > 39.0) & (status == 100)
    status[hot] = 600
    power[hot] *= 0.85
    cutout = ws_i >= CUT_OUT
    power[cutout] = 0.0

    if i == 4:  # T05 frozen anemometer for two weeks in March
        m = (PERIOD >= "2024-03-01") & (PERIOD < "2024-03-15")
        ws_i[m] = 6.2

    rows.append(pd.DataFrame({
        "timestamp": t, "turbine_id": f"T{tid:02d}",
        "power_kw": power, "wind_speed_mps": ws_i,
        "nacelle_dir_deg": wdir, "temp_c": temp, "status_code": status,
    }))

scada = pd.concat(rows, ignore_index=True)

# whole-farm data gap (capture-rate demo)
gap = (scada["timestamp"] >= "2024-06-10") & (scada["timestamp"] < "2024-06-13")
scada = scada[~gap].reset_index(drop=True)

# a few spike rows
for _ in range(3):
    r = rng.integers(0, len(scada))
    scada.loc[r, "power_kw"] = RATED * 4.0

scada.to_csv(os.path.join(HERE, "scada_sample.csv"), index=False)

# ---------------- warranted power curve ----------------
v = np.arange(0.0, 25.01, 0.5)
pd.DataFrame({"wind_speed_mps": v, "power_kw": power_curve(v)}).to_csv(
    os.path.join(HERE, "warranted_power_curve.csv"), index=False)

# ---------------- long-term daily reference file (2000-2024) ----------------
pd.DataFrame({"date": lt_dates, "ws_mps": ref_ws_daily}).to_csv(
    os.path.join(HERE, "long_term_daily_ws.csv"), index=False)

# ---------------- sample config ----------------
import json
cfg = {
    "farm_name": "Kayathar Hills Wind Farm (demo)",
    "latitude": LAT, "longitude": LON,
    "hub_height_m": HUB, "rotor_diameter_m": D,
    "rated_power_kw": RATED, "cut_in_mps": CUT_IN, "cut_out_mps": CUT_OUT,
    "warranted_power_curve": os.path.join(HERE, "warranted_power_curve.csv"),
    "long_term_wind_file": os.path.join(HERE, "long_term_daily_ws.csv"),
    "air_density_correction": True,
    "air_pressure_kpa": 99.2,
    "electrical_loss_pct": 2.0, "other_loss_pct": 0.5,
    "preconstruction_p50_gwh": 92.0,
    "status_codes": {"operating": [100], "fault": [200], "maintenance": [300],
                     "grid": [400], "curtailment": [500], "environmental": [600]},
}
with open(os.path.join(HERE, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

print("Sample dataset written to", HERE)
print("  scada_sample.csv            (12 turbines × 10-min, 2024)")
print("  warranted_power_curve.csv   (2.5 MW turbine)")
print("  long_term_daily_ws.csv      (2000-2024 daily)")
print("  config.json")
