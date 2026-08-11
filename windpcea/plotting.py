"""Chart generation (matplotlib, Agg backend) -> base64 PNG for the HTML report."""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY = "#14365D"
TEAL = "#0E7C86"
ORANGE = "#E8871E"
GREEN = "#2E8B57"
RED = "#C0504D"
GREY = "#8A93A3"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#C9CFDA", "axes.grid": True, "grid.color": "#E5E9F0",
    "grid.linewidth": 0.7, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "font.size": 9.5, "legend.fontsize": 8.5,
})


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def wind_rose(dir_deg, ws, title="Wind rose", n_sectors=12,
              bins_speed=(0, 3, 6, 9, 12, 15, 20, 40)):
    """WindPRO-style wind rose.

    * 12 direction sectors (default), bars pointing outward from the centre;
      the bar length of each sector is its frequency of occurrence
    * wind-speed classes shown as colour-coded, stacked segments per sector
    * frequency grid circles with % labels and radial sector lines
    * cardinal compass labels, speed-class legend and mean-wind-speed note
    """
    from matplotlib.patches import Patch
    dirs = np.asarray(dir_deg, dtype=float)
    ws = np.asarray(ws, dtype=float)
    m = np.isfinite(dirs) & np.isfinite(ws)
    dirs, ws = dirs[m], ws[m]
    if len(ws) == 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "No valid wind data", ha="center", va="center")
        ax.axis("off")
        return fig

    n = int(n_sectors)
    sector = np.floor((dirs % 360.0) / (360.0 / n)).astype(int)
    sector = np.clip(sector, 0, n - 1)
    nbins = len(bins_speed) - 1
    freqs = np.zeros((n, nbins))
    for j in range(nbins):
        lo, hi = bins_speed[j], bins_speed[j + 1]
        sel = (ws >= lo) & (ws < hi)
        freqs[:, j] = np.bincount(sector[sel], minlength=n) / len(ws) * 100.0
    maxf = freqs.sum(axis=1).max()
    rmax = max(4.0, np.ceil((maxf + 1.0) / 2.0) * 2.0)

    colors = ["#9DC3E6", "#5B9BD5", "#2F6FB2", "#70AD47", "#FFD966",
              "#ED7D31", "#C00000"]
    colors = (colors * (nbins // len(colors) + 1))[:nbins]

    fig = plt.figure(figsize=(7.6, 6.8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    theta = (np.arange(n) + 0.5) * (2 * np.pi / n)
    width = (2 * np.pi / n) * 0.97
    for j in range(nbins):
        bottom = freqs[:, :j].sum(axis=1)
        ax.bar(theta, freqs[:, j], width=width, bottom=bottom, color=colors[j],
               edgecolor="white", linewidth=0.4, zorder=2)

    grid_ticks = np.arange(0, rmax + 1e-9, 2.0)
    ax.set_rgrids(grid_ticks, labels=[f"{t:g}%" for t in grid_ticks])
    ax.set_ylim(0, rmax)
    ax.set_rlabel_position(270)          # % labels on the left, WindPRO style
    ax.grid(color="#B9C4D0", linewidth=0.7)
    ax.set_thetagrids(np.arange(0, 360, 360 // n), [""] * n)
    ax.spines["polar"].set_color("#B9C4D0")

    compass = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S",
               225: "SW", 270: "W", 315: "NW"}
    for ang, lbl in compass.items():
        x, y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        ha = "center" if abs(x) < 0.35 else ("left" if x > 0 else "right")
        va = "center" if abs(y) < 0.35 else ("bottom" if y > 0 else "top")
        ax.text(np.deg2rad(ang), rmax * 1.10, lbl, ha=ha, va=va, fontsize=10,
                fontweight="bold", color="#33393F", zorder=5)

    ax.set_title(f"{title}\nN = {len(ws):,} valid records", pad=36, fontsize=11.5)

    labels = []
    for j in range(nbins):
        lo, hi = bins_speed[j], bins_speed[j + 1]
        labels.append(f"{lo:g}-{hi:g}" if hi < 1e9 else f"> {lo:g}")
    handles = [Patch(facecolor=colors[j], edgecolor="#666666", linewidth=0.4,
                     label=labels[j]) for j in range(nbins)]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.03, 1.02),
              frameon=True, framealpha=0.96, edgecolor="#B9C4D0",
              title="Wind speed (m/s)", fontsize=8.5, title_fontsize=9)
    calm = 100.0 * (ws < 0.5).mean()
    ax.text(1.03, 0.02,
            f"Mean wind speed: {ws.mean():.1f} m/s\nCalms: {calm:.1f}%",
            transform=ax.transAxes, fontsize=9, color="#445566", va="bottom")
    return fig


def power_curves(meas_curve, warr_curve, rated, title="Farm power curve vs warranted"):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.scatter(meas_curve["bin_center"], meas_curve["mean_power"], s=26,
               color=NAVY, label="Measured (binned, 0.5 m/s)", zorder=3)
    ax.plot(warr_curve["bin_center"], warr_curve["mean_power"], color=ORANGE,
            lw=2, label="Warranted power curve")
    ax.axhline(rated, color=GREY, ls="--", lw=1)
    ax.text(0.02, rated * 1.02, f"Rated {rated:,.0f} kW", fontsize=8, color=GREY)
    ax.set_xlim(0, 28); ax.set_ylim(0, rated * 1.18)
    ax.set_xlabel("Wind speed (m/s)"); ax.set_ylabel("Active power (kW)")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)
    return fig


def weibull_plot(ws_hist, weib, label_hist, title, A, k):
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.hist(ws_hist, bins=40, density=True, alpha=0.55, color=TEAL,
            label=label_hist)
    v = np.linspace(0, np.percentile(ws_hist, 99.9), 300)
    from .powercurve import weibull_pdf
    ax.plot(v, weibull_pdf(v, A, k), color=ORANGE, lw=2.2,
            label=f"Weibull fit (A={A:.2f} m/s, k={k:.2f})")
    ax.set_xlabel("Daily mean wind speed (m/s)"); ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.legend(frameon=False)
    return fig


def mcp_scatter(site_daily, ref_ws, r2, title="MCP: site vs long-term reference"):
    j = pd.DataFrame({"site": site_daily, "ref": ref_ws}).dropna()
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.scatter(j["ref"], j["site"], s=10, alpha=0.45, color=NAVY)
    if len(j) > 5:
        z = np.polyfit(j["ref"], j["site"], 1)
        x = np.linspace(j["ref"].min(), j["ref"].max(), 50)
        ax.plot(x, np.polyval(z, x), color=ORANGE, lw=2)
    ax.set_xlabel("Reference daily wind speed (m/s)")
    ax.set_ylabel("Site daily wind speed (m/s)")
    ax.set_title(f"{title}  (R\u00b2 = {r2:.2f})")
    ax.grid(True)
    return fig


def turbine_curves_overlay(per_turbine_curves, farm_curve, warr_curve, rated,
                           title="Per-turbine power curves vs warranted"):
    """Overlay of every turbine's binned curve (light) with the farm curve
    (bold) and the warranted curve (orange)."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for tc in per_turbine_curves:
        c = tc["curve"]
        m = c["mean_power"].notna()
        if m.sum() > 0:
            ax.plot(c.loc[m, "bin_center"], c.loc[m, "mean_power"],
                    color="#B9C4D0", lw=1.0, alpha=0.75,
                    label="Individual turbines" if tc is per_turbine_curves[0] else None)
    fm = farm_curve["mean_power"].notna()
    ax.plot(farm_curve.loc[fm, "bin_center"], farm_curve.loc[fm, "mean_power"],
            color=NAVY, lw=2.4, label="Farm measured curve")
    ax.plot(warr_curve["bin_center"], warr_curve["mean_power"], color=ORANGE,
            lw=2, label="Warranted power curve")
    ax.axhline(rated, color=GREY, ls="--", lw=1)
    ax.text(0.02, rated * 1.02, f"Rated {rated:,.0f} kW", fontsize=8, color=GREY)
    ax.set_xlim(0, 28); ax.set_ylim(0, rated * 1.18)
    ax.set_xlabel("Wind speed (m/s)"); ax.set_ylabel("Active power (kW)")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    return fig


def prod_scatter(energy, wind, fit, title, xlabel, ylabel,
                 energy_unit="MWh"):
    """Scatter of energy vs wind speed with the fitted regression curve."""
    j = pd.DataFrame({"E": energy, "WS": wind}).dropna()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.scatter(j["WS"], j["E"], s=14, alpha=0.6, color=TEAL)
    x = np.linspace(max(0.5, j["WS"].min()), j["WS"].max(), 80)
    if fit.get("kind") in ("power_law", "cubic"):
        y = fit["c"] * x ** fit["b"]
        label = f"E = {fit['c']:.2f}·WS^{fit['b']:.2f}"
    else:
        y = fit["a"] + fit["b"] * x
        label = f"E = {fit['b']:.3f}·WS {'+' if fit['a'] >= 0 else '−'} {abs(fit['a']):.1f}"
    ax.plot(x, y, color=ORANGE, lw=2, label=label)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (R² = {fit['r2']:.2f}, n = {fit['n']})")
    ax.legend(frameon=False)
    return fig


def lt_predicted_series(predicted, n_years, title="Long-term predicted energy",
                        ylabel="Energy (MWh)"):
    """Time series of the long-term predicted energy (daily or monthly)."""
    s = predicted
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    if len(s) > 600:
        ax.plot(s.index, s.rolling(30, center=True).mean(), color=NAVY, lw=1.2)
        ax.plot(s.index, s, color=TEAL, alpha=0.25, lw=0.5)
        ax.set_ylabel(ylabel)
    else:
        ax.plot(s.index, s, color=NAVY, lw=1.6, marker="o", ms=3)
        ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (mean {s.mean():,.0f} {ylabel} per period)")
    ax.set_xlabel("Year")
    return fig


def monthly_wind_bar(monthly_ws, title="Monthly mean wind speed (measured)"):
    if monthly_ws is None or len(monthly_ws) == 0:
        return _empty_fig("No monthly wind data")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(monthly_ws["month"].astype(str), monthly_ws["ws_mps"],
           color=TEAL, alpha=0.85, width=0.7)
    ax.axhline(monthly_ws["ws_mps"].mean(), color=ORANGE, ls="--", lw=1.4,
               label=f"Mean {monthly_ws['ws_mps'].mean():.2f} m/s")
    ax.set_ylabel("Wind speed (m/s)")
    ax.set_title(title)
    ax.legend(frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig


def diurnal_wind(diurnal_ws, title="Diurnal wind speed variation"):
    if diurnal_ws is None or len(diurnal_ws) == 0:
        return _empty_fig("No diurnal wind data")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(diurnal_ws["hour"], diurnal_ws["ws_mps"], color=NAVY, lw=2, marker="o", ms=4)
    ax.fill_between(diurnal_ws["hour"], diurnal_ws["ws_mps"], alpha=0.15, color=NAVY)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Wind speed (m/s)")
    ax.set_title(title)
    ax.set_xticks(range(0, 24, 2))
    return fig


def ws_distribution(ws_hist, weib, title="Wind speed distribution (10-min records)"):
    if ws_hist is None or len(ws_hist) == 0:
        return _empty_fig("No wind speed distribution data")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    total = ws_hist["count"].sum()
    ax.bar(ws_hist["bin_center"], 100.0 * ws_hist["count"] / total,
           width=1.0, color=TEAL, alpha=0.65, edgecolor="white")
    if weib:
        A, k = weib
        v = np.linspace(0.25, 40, 300)
        pdf = (k / A) * (v / A) ** (k - 1) * np.exp(-(v / A) ** k)
        ax.plot(v, 100.0 * pdf, color=ORANGE, lw=2.2,
                label=f"Weibull A={A:.2f}, k={k:.2f}")
    ax.set_xlabel("Wind speed (m/s)"); ax.set_ylabel("Frequency (%)")
    ax.set_title(title)
    ax.legend(frameon=False)
    return fig


def pp_plot(pp_sample, rated, title="P-P plot — measured vs predicted power (operating)"):
    if pp_sample is None or len(pp_sample) == 0:
        return _empty_fig("No P-P data")
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    ax.scatter(pp_sample["expected_power_kw"], pp_sample["power_kw"],
               s=2, alpha=0.18, color=TEAL, rasterized=True)
    lim = max(pp_sample["power_kw"].max(), pp_sample["expected_power_kw"].max(),
              rated) * 1.05
    ax.plot([0, lim], [0, lim], color=GREY, ls="--", lw=1.2,
            label="1:1 (perfect)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted power (warranted curve, kW)")
    ax.set_ylabel("Measured power (kW)")
    ax.set_title(title)
    ax.legend(frameon=False)
    return fig


def monthly_trend(df, value_col, title, ylabel, color=TEAL, fmt=".1f"):
    if df is None or len(df) == 0:
        return _empty_fig("No monthly data")
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.plot(df["month"].astype(str), df[value_col], color=color, lw=2, marker="o", ms=4)
    ax.set_ylabel(ylabel); ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig


def downtime_stacked(monthly_downtime, title="Downtime loss by cause and month"):
    if monthly_downtime is None or len(monthly_downtime) == 0:
        return _empty_fig("No downtime breakdown data")
    piv = monthly_downtime.pivot(index="month", columns="cause", values="loss_mwh").fillna(0)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    piv.plot(kind="bar", stacked=True, ax=ax, color=["#C0504D", "#E8871E",
                                                     "#5B9BD5", "#8A93A3"])
    ax.set_ylabel("Lost energy (MWh)"); ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig


def _empty_fig(text):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, text, ha="center", va="center", color="#8A93A3", fontsize=11)
    ax.axis("off")
    return fig


def yaw_histogram(hist_df, title="Yaw offset distribution (wind dir - nacelle)"):
    if hist_df is None or len(hist_df) == 0:
        return _empty_fig("No yaw offset data")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(hist_df["offset_bin"], hist_df["count"], width=5.0,
           color=NAVY, alpha=0.8, edgecolor="white")
    ax.axvline(0, color=ORANGE, lw=1.6, ls="--", label="0° (perfect alignment)")
    ax.set_xlabel("Offset (deg)"); ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_xlim(-45, 45)
    return fig


def wake_polar(sector_table, title="Mean wake deficit by sector"):
    """Polar chart of mean wake deficit per direction sector.

    Returns a placeholder figure when there is no wake data (empty table or
    missing columns) so the report never crashes on sparse data.
    """
    if (sector_table is None or len(sector_table) == 0
            or "sector_deg" not in sector_table.columns):
        fig, ax = plt.subplots(figsize=(5.6, 5.0))
        ax.text(0.5, 0.5, "Insufficient operating data\nfor wake analysis",
                ha="center", va="center", fontsize=12, color="#8A93A3")
        ax.axis("off")
        return fig
    fig = plt.figure(figsize=(5.6, 5.0))
    ax = fig.add_subplot(111, projection="polar")
    sw = sector_table["sector_deg"].diff().median() if len(sector_table) > 1 else 30.0
    theta = np.deg2rad(sector_table["sector_deg"] + sw / 2.0)
    r = sector_table["mean_deficit"] * 100.0
    if len(sector_table) > 1:
        # sort by angle for a closed polygon
        o = np.argsort(theta)
        theta, r = theta[o], r[o]
        theta = np.append(theta, theta[0]); r = np.append(r, r[0])
        ax.plot(theta, r, color=RED, lw=2)
        ax.fill(theta, r, color=RED, alpha=0.25)
    else:
        ax.bar(theta, r, width=2 * np.pi, color=RED, alpha=0.4)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_title(title, pad=18)
    return fig


def loss_waterfall(tree, gross_mwh, net_mwh, title="Loss tree (long-term AEP)"):
    labels = ["Gross AEP"] + [t["loss"] for t in tree] + ["Net AEP"]
    values = [gross_mwh] + [-t["pct_of_gross"] / 100.0 * gross_mwh for t in tree] + [0]
    running = [gross_mwh]
    for v in values[1:-1]:
        running.append(running[-1] + v)
    running.append(net_mwh)
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    x = np.arange(len(labels))
    for i in range(len(labels) - 1):
        bottom = min(running[i], running[i + 1])
        height = abs(running[i + 1] - running[i])
        # a LOSS reduces energy -> red; a GAIN (negative loss, e.g. curve
        # overperformance) increases energy -> green; ~0 -> grey
        color = RED if height > 0.0001 and running[i + 1] < running[i] else (
            GREEN if height > 0.0001 and running[i + 1] > running[i] else GREY)
        ax.bar(x[i], height, bottom=bottom, color=color, alpha=0.85, width=0.62)
        ax.text(x[i], running[i], f"{running[i]:,.0f}", ha="center", va="bottom",
                fontsize=8, color="#444")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=RED, label="Loss"),
                       Patch(facecolor=GREEN, label="Gain (overperformance)"),
                       Patch(facecolor=GREY, label="≈0")],
              loc="lower left", fontsize=8, frameon=False)
    ax.bar(x[-1], running[-1], bottom=0, color=NAVY, width=0.62)
    ax.text(x[-1], running[-1], f"{running[-1]:,.0f}", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8)
    ax.set_ylabel("Energy (MWh)")
    ax.set_title(title)
    ax.set_ylim(0, gross_mwh * 1.12)
    return fig


def tornado(components, title="Uncertainty contribution (1\u03c3, % of net AEP)"):
    df = components.sort_values("sigma_pct")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.barh(df["component"], df["sigma_pct"], color=TEAL, alpha=0.85)
    for i, v in enumerate(df["sigma_pct"]):
        ax.text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=8)
    ax.set_xlabel("1\u03c3 uncertainty (% of net AEP)")
    ax.set_title(title)
    return fig


def mc_histogram(samples, p, title="Monte Carlo net AEP distribution"):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(samples, bins=80, color=TEAL, alpha=0.6, density=True)
    for key, color in [("P50", NAVY), ("P75", ORANGE), ("P90", RED)]:
        ax.axvline(p[key], color=color, lw=1.8, ls="--",
                   label=f"{key}: {p[key]:,.0f} MWh")
    ax.set_xlabel("Net AEP (MWh)"); ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.legend(frameon=False)
    return fig
