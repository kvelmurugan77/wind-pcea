"""Uncertainty quantification: Monte Carlo simulation of net energy yield.

Each uncertainty component is expressed as a 1-sigma percentage of the net
energy and modelled with a lognormal multiplier. 10k-100k draws yield the
full distribution of net AEP, from which P50 / P75 / P90 / P99 and
confidence intervals are derived (industry-standard practice for P-values).
"""
import numpy as np
import pandas as pd

DEFAULT_UNCERTAINTIES_PCT = {
    "power_curve": 2.0,
    "availability": 0.6,
    "curtailment": 0.5,
    "wake": 1.5,
    "performance": 1.0,
    "electrical": 0.4,
    "environmental": 0.4,
    "other": 0.5,
}


def uncertainty_analysis(cfg, climate, tree, net_mwh, mc_iterations=20000, seed=42):
    """Monte Carlo P-values. Returns a results dict."""
    rng = np.random.default_rng(seed)

    comps = dict(DEFAULT_UNCERTAINTIES_PCT)
    comps["power_curve"] = _power_curve_uncertainty(tree)
    comps["availability"] = _availability_uncertainty(tree)
    comps["wake"] = _wake_uncertainty(tree)
    comps["performance"] = _performance_uncertainty(tree)
    comps["wind_resource"] = _wind_resource_uncertainty(cfg, climate)
    for k, v in cfg.get("uncertainty_overrides_pct", {}).items():
        if k in comps:
            comps[k] = float(v)

    names = {
        "wind_resource": "Wind resource (long-term)",
        "power_curve": "Power curve measurement",
        "availability": "Availability losses",
        "curtailment": "Curtailment losses",
        "wake": "Wake losses",
        "performance": "Turbine performance",
        "electrical": "Electrical losses",
        "environmental": "Environmental losses",
        "other": "Other",
    }
    df = pd.DataFrame([{"component": names[k], "sigma_pct": v,
                        "contribution_pct": None} for k, v in comps.items()])
    total_var = sum(v ** 2 for v in comps.values())
    df["contribution_pct"] = 100.0 * np.array([v ** 2 for v in comps.values()]) / total_var

    factors = np.ones(mc_iterations)
    for v in comps.values():
        factors *= np.exp(rng.normal(0.0, v / 100.0, mc_iterations))

    samples = net_mwh * factors
    # DNV / industry convention: Px = yield with x% probability of EXCEEDING.
    # P50 = median, P75 = 25th percentile, P90 = 10th percentile, P99 = 1st.
    p = {f"P{int(q)}": float(np.quantile(samples, 1.0 - q / 100.0))
         for q in (50, 75, 90, 99)}
    mean = float(samples.mean())
    sd = float(samples.std(ddof=1))
    p50_lo, p50_hi = [float(v) for v in np.quantile(samples, [0.10, 0.90])]
    p90_lo, p90_hi = [float(v) for v in np.quantile(samples, [0.05, 0.95])]

    return {"components": df, "samples": samples, "p": p, "mean": mean, "sd": sd,
            "p50_ci80": (p50_lo, p50_hi), "p90_ci90": (p90_lo, p90_hi),
            "deterministic_net_mwh": net_mwh}


def _wind_resource_uncertainty(cfg, climate):
    from .mcp import wind_resource_uncertainty_pct
    return wind_resource_uncertainty_pct(climate)


def _power_curve_uncertainty(tree):
    return 2.0


def _availability_uncertainty(tree):
    row = tree[tree["loss"] == "Availability (downtime)"]
    loss = row["pct_of_gross"].iloc[0] if len(row) else 5.0
    return float(np.clip(0.15 * loss, 0.3, 1.5))


def _wake_uncertainty(tree):
    row = tree[tree["loss"] == "Wake"]
    loss = row["pct_of_gross"].iloc[0] if len(row) else 5.0
    return float(np.clip(0.25 * loss, 0.5, 3.0))


def _performance_uncertainty(tree):
    return 1.0


def p_value_table(unc, net_mwh):
    rows = []
    for k in ("P50", "P75", "P90", "P99"):
        rows.append({"p_value": k, "net_aep_mwh": unc["p"][k],
                     "pct_of_deterministic": 100.0 * unc["p"][k] / net_mwh})
    rows.append({"p_value": "P50 80% CI", "net_aep_mwh": unc["p50_ci80"],
                 "pct_of_deterministic": None})
    return pd.DataFrame(rows)
