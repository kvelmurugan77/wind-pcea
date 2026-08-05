"""Availability and lost-energy accounting (IEC 61400-26 style categories).

For every non-operating state we estimate the energy that would have been
produced, using the warranted power curve evaluated at the measured wind
speed: E_lost = sum( P_warr(ws) * dt ) over flagged intervals.

Categories:
  * Downtime     - faults, maintenance, grid outages (sub-split when status codes exist)
  * Curtailment  - grid/owner curtailment, low output at usable wind
  * Derating     - partial load operation
  * Environmental- icing / high-temperature derates
"""
import numpy as np
import pandas as pd

CATEGORY_LABELS = {2: "Downtime", 3: "Curtailment", 4: "Derating", 5: "Environmental"}


def energy_accounting(df, cfg):
    """Return dict with actual energy and per-category lost energy (MWh) over the period.

    Flag 7 (anemometer fault) rows still produced energy, so they count as
    production; only their wind speed is unreliable.
    """
    dt_h = df["dt_h"].iloc[0]
    producing = df["flag"].isin((0, 7))
    E_actual = float(df.loc[producing, "energy_kwh"].sum() / 1000.0)
    E_exp = float(df["expected_energy_kwh"].sum() / 1000.0)

    losses = {}
    for flag in (2, 3, 4, 5):
        sub = df[df["flag"] == flag]
        losses[flag] = float(sub["expected_energy_kwh"].sum() / 1000.0)

    # downtime sub-categories from status codes
    downtime_split = {}
    if "status" in df.columns and df["status"].notna().any():
        sub = df[df["flag"] == 2]
        sc = cfg["status_codes"]
        for code, label in [("fault", "Faults"), ("maintenance", "Maintenance"),
                            ("grid", "Grid outage")]:
            codes = set(sc.get(code, []))
            m = sub["status"].isin(codes)
            downtime_split[label] = float(sub.loc[m, "expected_energy_kwh"].sum() / 1000.0)
        other = losses[2] - sum(downtime_split.values())
        if other > 0.01:
            downtime_split["Other / unspecified"] = other

    return {"E_actual_mwh": E_actual, "E_expected_mwh": E_exp,
            "E_lost_mwh": losses, "downtime_split": downtime_split}


def availability_analysis(df, cfg, energy):
    """Per-turbine and farm-level availability metrics + monthly table."""
    dt_h = df["dt_h"].iloc[0]
    per = []
    for tid, g in df.groupby("turbine"):
        total_h = len(g) * dt_h
        down_h = (g["flag"] == 2).sum() * dt_h
        curt_h = (g["flag"] == 3).sum() * dt_h
        e_act = g.loc[g["flag"].isin((0, 7)), "energy_kwh"].sum() / 1000.0
        e_down = g.loc[g["flag"] == 2, "expected_energy_kwh"].sum() / 1000.0
        e_curt = g.loc[g["flag"] == 3, "expected_energy_kwh"].sum() / 1000.0
        e_der = g.loc[g["flag"] == 4, "expected_energy_kwh"].sum() / 1000.0
        e_env = g.loc[g["flag"] == 5, "expected_energy_kwh"].sum() / 1000.0
        time_avail = 100.0 * (total_h - down_h) / total_h if total_h else 100.0
        prod_avail = 100.0 * e_act / (e_act + e_down) if (e_act + e_down) > 0 else 100.0
        per.append({"turbine": tid, "hours": total_h, "downtime_h": down_h,
                    "curtailment_h": curt_h, "time_avail_pct": time_avail,
                    "prod_avail_pct": prod_avail, "energy_mwh": e_act,
                    "downtime_loss_mwh": e_down, "curtailment_loss_mwh": e_curt,
                    "derating_loss_mwh": e_der, "environmental_loss_mwh": e_env})
    per_df = pd.DataFrame(per)

    farm = {
        "time_avail_pct": float(per_df["time_avail_pct"].mean()),
        "prod_avail_pct": float(per_df["prod_avail_pct"].mean()),
        "energy_mwh": float(per_df["energy_mwh"].sum()),
        "downtime_loss_mwh": float(per_df["downtime_loss_mwh"].sum()),
        "curtailment_loss_mwh": float(per_df["curtailment_loss_mwh"].sum()),
        "derating_loss_mwh": float(per_df["derating_loss_mwh"].sum()),
        "environmental_loss_mwh": float(per_df["environmental_loss_mwh"].sum()),
    }

    # monthly table
    prod = df[df["flag"].isin((0, 7))]
    monthly = df.groupby(df["timestamp"].dt.to_period("M")).agg(
        downtime_h=pd.NamedAgg(column="flag", aggfunc=lambda s: (s == 2).sum() * dt_h),
        curtailment_h=pd.NamedAgg(column="flag", aggfunc=lambda s: (s == 3).sum() * dt_h),
        total_h=pd.NamedAgg(column="flag", aggfunc=lambda s: len(s) * dt_h),
    )
    monthly["energy_mwh"] = (prod.groupby(prod["timestamp"].dt.to_period("M"))
                             ["energy_kwh"].sum() / 1000.0)
    monthly["time_avail_pct"] = 100.0 * (1 - monthly["downtime_h"] / monthly["total_h"])
    monthly = monthly.reset_index()
    monthly["month"] = monthly["timestamp"].astype(str)
    monthly = monthly.drop(columns=["timestamp"])

    return {"per_turbine": per_df, "farm": farm, "monthly": monthly}
