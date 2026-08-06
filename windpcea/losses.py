"""Loss tree assembly (DNV-style) and net energy yield calculation.

Losses are computed as fractions of the gross energy over the measured period
and applied to the long-term gross AEP:

    Gross AEP  = N_turbines * 8760 h * integral( f_lt(v) * P_warr(v) dv )
    Net AEP    = Gross AEP * (1 - l_avail) * (1 - l_curtail) * ... * (1 - l_other)

A reconciliation check compares the modelled energy over the measured period
with the SCADA-measured energy.
"""
import numpy as np
import pandas as pd


def build_loss_tree(cfg, energy, wake_energy_mwh, perf_energy_mwh, gross_period_mwh, gross_lt_mwh):
    """Return (loss_tree_df, net_mwh, reconciliation_dict)."""
    def pct(x):
        return 100.0 * x / gross_period_mwh if gross_period_mwh > 0 else 0.0

    items = [
        ("Availability (downtime)", energy["E_lost_mwh"].get(2, 0.0), "Faults, maintenance & grid outages from SCADA"),
        ("Curtailment", energy["E_lost_mwh"].get(3, 0.0), "Grid / owner curtailment, low output at usable wind"),
        ("Derating / partial load", energy["E_lost_mwh"].get(4, 0.0), "Partial load operation"),
        ("Environmental", energy["E_lost_mwh"].get(5, 0.0), "Icing / high-temperature derates"),
        ("Wake", wake_energy_mwh, "Wake effects from nacelle-anemometry sector analysis"),
        ("Turbine performance", perf_energy_mwh, "Operating shortfall vs warranted power curve (can be negative)"),
    ]
    rows = []
    for name, e_mwh, desc in items:
        rows.append({"loss": name, "energy_mwh": e_mwh, "pct_of_gross": pct(e_mwh),
                     "description": desc})
    rows.append({"loss": "Electrical", "energy_mwh": np.nan,
                 "pct_of_gross": cfg["electrical_loss_pct"],
                 "description": "Transformer, collection & transmission losses (input)"})
    rows.append({"loss": "Other", "energy_mwh": np.nan,
                 "pct_of_gross": cfg["other_loss_pct"],
                 "description": "Miscellaneous losses (input)"})
    tree = pd.DataFrame(rows)
    measured_loss_pct = tree.loc[tree["energy_mwh"].notna(), "pct_of_gross"].sum()
    total_loss_pct = tree["pct_of_gross"].sum()
    net_mwh = gross_lt_mwh * (1.0 - total_loss_pct / 100.0)

    # reconciliation over measured period
    # measured energy is at turbine terminals, so only SCADA-derived losses
    # are applied; electrical & other (input) losses are excluded here
    modelled_net_period = gross_period_mwh * (1.0 - measured_loss_pct / 100.0)
    recon = {
        "gross_period_mwh": gross_period_mwh,
        "measured_mwh": energy["E_actual_mwh"],
        "modelled_net_period_mwh": modelled_net_period,
        "measured_loss_pct": measured_loss_pct,
        "total_loss_pct": total_loss_pct,
        "gap_pct": 100.0 * (modelled_net_period - energy["E_actual_mwh"]) / gross_period_mwh
        if gross_period_mwh else 0.0,
    }
    return tree, net_mwh, recon


def capacity_factor(net_mwh, cfg, hours=8760.0):
    rated_total_kw = float(cfg["rated_power_kw"]) * (int(cfg.get("num_turbines") or 1))
    return 100.0 * net_mwh * 1000.0 / (rated_total_kw * hours)
