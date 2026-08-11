"""Report generation: self-contained HTML report, Excel workbook, CSV exports."""
import datetime
import os

import numpy as np
import pandas as pd

from . import plotting as pl

NAVY = "#14365D"
TEAL = "#0E7C86"
ORANGE = "#E8871E"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _sensitivity_table(r):
    """One-at-a-time sensitivity of net P50 to key parameters."""
    from . import powercurve as pc_mod
    rows = []
    # force every value numeric — a single string in the loss tree would make
    # .sum() concatenate strings and break all arithmetic below
    tree = r["losses"]["tree"].copy()
    for _c in ("pct_of_gross", "energy_mwh"):
        if _c in tree.columns:
            tree[_c] = pd.to_numeric(tree[_c], errors="coerce").fillna(0.0)
    gross = float(r["losses"]["gross_lt_mwh"])
    net = float(r["losses"]["net_mwh"])
    p50 = float(r["uncertainty"]["p"]["P50"])
    total_loss = float(tree["pct_of_gross"].sum()) / 100.0

    # each loss category +/- 1 pp
    for _, row in tree.iterrows():
        li = float(row["pct_of_gross"]) / 100.0
        denom = 1.0 - li
        if denom <= 1e-6:
            continue
        d_net = net * 0.01 / denom          # +1pp loss -> net decreases by this
        rows.append({"parameter": f"Loss: {row['loss']} +1 pp",
                     "change": "+1 pp", "delta_p50_mwh": -d_net,
                     "delta_p50_pct": -100.0 * d_net / p50})

    # wind speed +/- 0.1 m/s
    A, k = r["climate"]["lt_weibull"]
    A = float(A); k = float(k)
    n_turb = int(r["meta"]["num_turbines"])
    v_arr, p_arr = r["v_arr"], r["p_arr"]
    gross_hi = n_turb * pc_mod.aep_from_weibull(A * (1 + 0.1 / A), k, v_arr, p_arr)
    gross_lo = n_turb * pc_mod.aep_from_weibull(A * (1 - 0.1 / A), k, v_arr, p_arr)
    net_hi = gross_hi * (1 - total_loss)
    net_lo = gross_lo * (1 - total_loss)
    rows.append({"parameter": "Long-term mean wind speed +0.1 m/s",
                 "change": "+0.1 m/s", "delta_p50_mwh": net_hi - net,
                 "delta_p50_pct": 100.0 * (net_hi - net) / p50})
    rows.append({"parameter": "Long-term mean wind speed −0.1 m/s",
                 "change": "−0.1 m/s", "delta_p50_mwh": net_lo - net,
                 "delta_p50_pct": 100.0 * (net_lo - net) / p50})

    # measurement period +- 25%
    if r["climate"].get("production") and r["climate"]["production"].get("lt_gross_mwh"):
        mb = float(r["climate"]["production"]["lt_gross_mwh"])
        for sgn, lbl in [(1.25, "+25%"), (0.75, "−25%")]:
            net_s = mb * sgn * (1 - total_loss)
            rows.append({"parameter": f"Production-regression gross {lbl}",
                         "change": lbl,
                         "delta_p50_mwh": net_s - net,
                         "delta_p50_pct": 100.0 * (net_s - net) / p50})
    return pd.DataFrame(rows)


def _conclusions_html(r, sens):
    """Auto-generated conclusions & recommendations (DNV-report style)."""
    avail = r["availability"]["farm"]
    pc = r["power_curve"]
    wk = r["wake"]
    cl = r["climate"]
    qc = r["qc"]
    pts = []
    recs = []

    # key finding bullets
    if r["benchmark"]:
        b = r["benchmark"]
        verdict = "above" if b["ratio"] >= 1 else "below"
        pts.append(f"Assessed net P50 energy yield is <b>{b['ratio']:.2f}×</b> "
                   f"({b['delta_pct']:+.1f}%) the pre-construction P50 — "
                   f"<b>{verdict}</b> the pre-construction estimate.")
    pts.append(f"Time-based availability is <b>{avail['time_avail_pct']:.2f}%</b> and "
               f"production-based availability <b>{avail['prod_avail_pct']:.2f}%</b>.")
    pts.append(f"Farm power curve deviation (energy-weighted) is "
               f"<b>{pc['deviation_pct']:+.2f}%</b> relative to the warranted curve.")
    pts.append(f"Wake loss is estimated at <b>{wk['wake_loss_pct']:.2f}%</b> of gross energy "
               f"from the nacelle-anemometry reference-turbine analysis.")
    if cl["mcp"] is not None:
        pts.append(f"Long-term correction (MCP) against "
                   f"{'a user reference' if 'user' in cl['method'] else 'NASA POWER reanalysis'}"
                   f" achieved R² = <b>{cl['mcp']['r2']:.2f}</b> over "
                   f"{cl['lt_n_years']:.1f} years.")
    pts.append(f"Total losses are <b>{_numeric_sum(r['losses']['tree']['pct_of_gross']):.2f}%</b> "
               f"of gross, giving net P50 = <b>{r['uncertainty']['p']['P50']:,.0f} MWh/yr</b> "
               f"(P90 = {r['uncertainty']['p']['P90']:,.0f} MWh/yr).")

    # recommendations
    if avail["time_avail_pct"] < 97.0:
        worst = (r["availability"]["per_turbine"]
                 .sort_values("downtime_loss_mwh", ascending=False)
                 .head(3)["turbine"].tolist())
        recs.append(f"Availability ({avail['time_avail_pct']:.1f}%) is below the typical 97% "
                    f"benchmark — prioritise investigation of turbines "
                    f"{', '.join(map(str, worst))}.")
    dev = pc["per_turbine"]
    if len(dev) and dev["deviation_pct"].max() > 2.0:
        worst = dev.sort_values("deviation_pct", ascending=False).head(3)
        recs.append(f"Turbines {', '.join(worst['turbine'].astype(str).tolist())} show "
                    f"energy-weighted deviations of "
                    f"{', '.join(f'{d:+.1f}%' for d in worst['deviation_pct'])} — "
                    f"recommend blade inspection / pitch alignment review.")
    if r["energy"]["E_lost_mwh"].get(3, 0) / max(1.0, r["energy"]["E_actual_mwh"]) > 0.02:
        recs.append(f"Curtailment ({r['energy']['E_lost_mwh'][3]:,.0f} MWh over the period, "
                    f"{100*r['energy']['E_lost_mwh'][3]/max(1.0, r['energy']['E_actual_mwh']):.1f}% "
                    f"of measured) is material — document grid/PPA curtailment for future "
                    f"revenue assessments.")
    if wk["wake_loss_pct"] > 4.0:
        recs.append(f"Wake losses ({wk['wake_loss_pct']:.1f}%) are significant — consider "
                    f"layout-aware wake modelling or wake-steering optimisation.")
    if cl["mcp"] is not None and cl["mcp"]["r2"] < 0.8:
        recs.append("MCP correlation is moderate (R² < 0.8) — a met mast or lidar campaign "
                    "would reduce the wind-resource uncertainty.")
    if qc["coverage_pct"] < 90.0:
        recs.append(f"SCADA capture rate is only {qc['coverage_pct']:.1f}% — request "
                    f"complete data from the operator to reduce uncertainty.")
    if not recs:
        recs.append("No material performance issues identified; the wind farm is performing "
                    "in line with expectations.")
    r["conclusions"] = {"findings": pts, "recommendations": recs}
    html = ["<h2>10&nbsp;&nbsp;Conclusions &amp; recommendations</h2>",
            "<h3>Key findings</h3><ul class='det'>"]
    html += [f"<li>{p}</li>" for p in pts]
    html.append("</ul><h3>Recommendations</h3><ul class='det'>")
    html += [f"<li>{rec}</li>" for rec in recs]
    html.append("</ul>")
    return "".join(html)


def _to_num(x):
    """Best-effort numeric cast (strings like '2.0' -> 2.0); None/NaN -> None."""
    if x is None:
        return None
    try:
        f = float(x)
        if np.isfinite(f):
            return f
        return None
    except (TypeError, ValueError):
        return x  # not numeric: keep for display


def _nfmt(x, nd=0, suffix="", plus=False):
    """Direct f-string formatting that never crashes on None/NaN/str."""
    x = _to_num(x)
    if x is None:
        return "–"
    if isinstance(x, str):
        return x + suffix
    return f"{x:+,.{nd}f}{suffix}" if plus else f"{x:,.{nd}f}{suffix}"


def _numeric_sum(s):
    """Sum that always returns a float — even if the series contains strings
    (pandas .sum() would otherwise concatenate and break formatting)."""
    try:
        return float(pd.to_numeric(s, errors="coerce").fillna(0.0).sum())
    except Exception:
        return 0.0


def _fmt(x, nd=0):
    x = _to_num(x)
    if x is None:
        return "–"
    if isinstance(x, str):
        return x
    return f"{x:,.{nd}f}"


def _pct(x, nd=1):
    x = _to_num(x)
    if x is None:
        return "–"
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}%"


def html_table(df, formats=None, caption=None, cls="tbl"):
    if df is None or len(df) == 0:
        return "<p class='muted'>No data.</p>"
    formats = formats or {}
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in formats and v is not None and isinstance(v, (int, float)) \
                    and np.isfinite(v) and not isinstance(v, bool):
                cells.append(f"<td>{formats[c](v)}</td>")
            elif isinstance(v, float) and not np.isfinite(v):
                cells.append("<td>–</td>")
            else:
                cells.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    cap = f"<caption>{caption}</caption>" if caption else ""
    return (f"<div class='tblwrap'><table class='{cls}'>{cap}"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def _img(b64, alt="chart", maxw="100%"):
    return (f"<div class='chart'><img src='data:image/png;base64,{b64}' "
            f"alt='{alt}' style='max-width:{maxw}'/></div>")


def _safe_df(r, key):
    """Return the dataframe for key, or an empty frame if missing."""
    v = r.get(key)
    return v if isinstance(v, pd.DataFrame) and len(v) > 0 else pd.DataFrame()


def _chart(fn, alt="chart", fallback="Chart unavailable"):
    """Render a chart; on any failure insert a note instead of crashing."""
    try:
        return _img(fn(), alt)
    except Exception:
        return f"<div class='note'><b>Note:</b> {fallback}</div>"


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------
CSS = """
:root{--navy:#14365D;--teal:#0E7C86;--orange:#E8871E;--ink:#22303f;--muted:#6b7787;
--line:#dde3ec;--bg:#f4f6fa;}
*{box-sizing:border-box} body{margin:0;font-family:'Segoe UI',system-ui,Arial,sans-serif;
color:var(--ink);background:var(--bg);line-height:1.45}
.wrap{max-width:1080px;margin:0 auto;padding:24px 20px 60px}
header.hero{background:linear-gradient(120deg,var(--navy) 0%,#1d4d7e 55%,var(--teal) 100%);
color:#fff;border-radius:14px;padding:26px 30px;margin-bottom:22px}
header.hero h1{margin:0 0 6px;font-size:24px;letter-spacing:.2px}
header.hero .sub{opacity:.92;font-size:13.5px}
.badge{display:inline-block;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.35);
padding:2px 10px;border-radius:20px;font-size:11.5px;margin-right:6px;margin-top:8px}
h2{color:var(--navy);font-size:17px;margin:30px 0 4px;padding-bottom:6px;border-bottom:2px solid var(--line)}
h3{color:var(--teal);font-size:14px;margin:18px 0 6px}
p.lead{font-size:13.5px;color:var(--muted);margin:2px 0 12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:14px 0}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted)}
.card .v{font-size:19px;font-weight:700;color:var(--navy);margin-top:2px}
.card .s{font-size:11px;color:var(--muted)}
.tblwrap{overflow-x:auto;margin:10px 0}
table.tbl{border-collapse:collapse;width:100%;background:#fff;font-size:12px;border:1px solid var(--line)}
table.tbl th{background:#eef2f8;color:var(--navy);text-align:left;padding:6px 9px;
border-bottom:1px solid var(--line);white-space:nowrap}
table.tbl td{padding:5px 9px;border-bottom:1px solid #eef1f6;white-space:nowrap}
table.tbl tr:last-child td{border-bottom:none}
table.tbl caption{caption-side:top;text-align:left;font-size:11.5px;color:var(--muted);padding:4px 0}
table.tbl tbody tr:nth-child(even){background:#fafbfd}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.grid2 .full{grid-column:1/-1}
@media(max-width:820px){.grid2{grid-template-columns:1fr}}
.chart{background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px;margin:10px 0;
text-align:center}
.chart img{border-radius:6px}
.note{background:#fff8ef;border:1px solid #f3ddb8;border-left:4px solid var(--orange);
border-radius:8px;padding:10px 14px;font-size:12.5px;margin:12px 0;color:#6b4a12}
.ok{background:#eef7f1;border-color:#cfe6d8;border-left-color:#2E8B57;color:#24543a}
.formula{background:#f7f9fc;border:1px solid var(--line);border-left:4px solid var(--teal);
border-radius:8px;padding:10px 14px;font-family:Consolas,'DejaVu Sans Mono',monospace;
font-size:12.5px;margin:8px 0;color:var(--ink);overflow-x:auto}
.formula b{color:var(--navy)}
ul.det{font-size:12.5px;padding-left:20px;margin:8px 0;color:var(--ink)}
ul.det li{margin:4px 0}
.foot{margin-top:36px;font-size:11px;color:var(--muted);text-align:center}
.tag{display:inline-block;background:var(--teal);color:#fff;font-size:10px;padding:2px 8px;
border-radius:4px;margin-left:6px;vertical-align:middle}
"""


def build_html(r, outdir):
    meta, cfg = r["meta"], r["cfg"]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    E = r["energy"]
    avail = r["availability"]["farm"]
    tree = r["losses"]["tree"]
    gross_lt, net = r["losses"]["gross_lt_mwh"], r["losses"]["net_mwh"]
    unc = r["uncertainty"]
    p = unc["p"]
    bm = r["benchmark"]

    parts = []
    # guarantee the loss tree is numeric BEFORE any .sum() — a string anywhere
    # would make pandas .sum() concatenate and crash every format below
    tree = tree.copy()
    for _c in ("pct_of_gross", "energy_mwh"):
        if _c in tree.columns:
            tree[_c] = pd.to_numeric(tree[_c], errors="coerce").fillna(0.0)
    parts.append(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>PCEYA — {meta['farm_name']}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style></head><body><div class="wrap">
<header class="hero">
<h1>Post-Construction Energy Yield Assessment</h1>
<div class="sub">{meta['farm_name']} &nbsp;•&nbsp; {meta['num_turbines']} × {cfg['rated_power_kw']:,.0f} kW &nbsp;•&nbsp;
SCADA period {meta['record_start']} → {meta['record_end']} &nbsp;•&nbsp; Report generated {now}</div>
<span class="badge">WindPCEA v1.0</span>
<span class="badge">IEC 61400-12-1 binning</span>
<span class="badge">IEC 61400-26 availability</span>
<span class="badge">MCP long-term correction</span>
<span class="badge">Monte Carlo P-values</span>
<div style="margin-top:14px">
  <a class="btn" href="./pceya_report.html" download style="display:inline-block;background:#fff;color:#14365D;font-weight:700;padding:8px 16px;border-radius:8px;text-decoration:none;font-size:13px;margin-right:8px">&#8681; Download report (HTML)</a>
  <a class="btn" href="./pceya_results.xlsx" download style="display:inline-block;background:#fff;color:#14365D;font-weight:700;padding:8px 16px;border-radius:8px;text-decoration:none;font-size:13px;margin-right:8px">&#8681; Excel workbook</a>
  <a class="btn" href="./flagged_scada_10min.csv" download style="display:inline-block;background:#fff;color:#14365D;font-weight:700;padding:8px 16px;border-radius:8px;text-decoration:none;font-size:13px;margin-right:8px">&#8681; Flagged SCADA (CSV)</a>
  <a class="btn" href="./gap_filled_scada_10min.csv" download style="display:inline-block;background:#fff;color:#14365D;font-weight:700;padding:8px 16px;border-radius:8px;text-decoration:none;font-size:13px;margin-right:8px">&#8681; Gap-filled SCADA (CSV)</a>
</div>
</header>""")

    # ---------------- executive summary ----------------
    parts.append("<h2>1&nbsp;&nbsp;Executive summary</h2>")
    parts.append(f"""<div class="cards">
<div class="card"><div class="k">Gross AEP (long-term)</div><div class="v">{_fmt(gross_lt)}</div>
<div class="s">MWh/yr, warranted curve</div></div>
<div class="card"><div class="k">Total losses</div><div class="v">{_pct(_to_num(_numeric_sum(tree['pct_of_gross'])))}</div>
<div class="s">of gross AEP</div></div>
<div class="card"><div class="k">Net AEP — P50</div><div class="v">{_fmt(p['P50'])}</div>
<div class="s">MWh/yr &nbsp;•&nbsp; 80% CI {_fmt(unc['p50_ci80'][0])}–{_fmt(unc['p50_ci80'][1])}</div></div>
<div class="card"><div class="k">Net AEP — P75</div><div class="v">{_fmt(p['P75'])}</div>
<div class="s">MWh/yr</div></div>
<div class="card"><div class="k">Net AEP — P90</div><div class="v">{_fmt(p['P90'])}</div>
<div class="s">MWh/yr</div></div>
<div class="card"><div class="k">Capacity factor</div><div class="v">{_pct(r['capacity_factor_pct'])}</div>
<div class="s">{_fmt(r['full_load_hours'])} full-load h</div></div>
<div class="card"><div class="k">Production availability</div><div class="v">{_pct(avail['prod_avail_pct'])}</div>
<div class="s">time-based {_pct(avail['time_avail_pct'])}</div></div>
<div class="card"><div class="k">Data coverage</div><div class="v">{_pct(r['qc']['coverage_pct'],0)}</div>
<div class="s">{_fmt(r['qc']['rows'])} records</div></div>
</div>""")

    # surface QC warnings prominently (mapping mismatches, implausible results)
    for _w in (r["qc"].get("warnings") or []):
        parts.append(f"<div class='note' style='border-left-color:#C0504D;background:#fdf0ef;"
                     f"color:#7a2e2a'><b>⚠ {_w}</b></div>")

    # DNV-style key results table
    kr = pd.DataFrame({
        "parameter": ["Gross energy yield (long-term, P50)",
                      "Total losses", "Net energy yield — P50", "Net energy yield — P75",
                      "Net energy yield — P90", "Net energy yield — P99",
                      "Capacity factor", "Full-load hours",
                      "Time-based availability", "Production-based availability",
                      "Power curve deviation (energy-weighted)", "Wake loss",
                      "Long-term mean wind speed", "Measurement period",
                      "Data capture rate"],
        "value": [_nfmt(gross_lt, 0, " MWh/yr"), _nfmt(_numeric_sum(tree['pct_of_gross']), 2, "%"),
                  _nfmt(p['P50'], 0, " MWh/yr"), _nfmt(p['P75'], 0, " MWh/yr"),
                  _nfmt(p['P90'], 0, " MWh/yr"), _nfmt(p['P99'], 0, " MWh/yr"),
                  _nfmt(r['capacity_factor_pct'], 1, "%"),
                  _nfmt(r['full_load_hours'], 0, " h/yr"),
                  _nfmt(avail['time_avail_pct'], 2, "%"), _nfmt(avail['prod_avail_pct'], 2, "%"),
                  _nfmt(r['power_curve']['deviation_pct'], 2, "%", plus=True),
                  _nfmt(r['wake']['wake_loss_pct'], 2, "%"),
                  _nfmt(r['climate']['lt_mean_ws'], 2, " m/s"),
                  f"{meta['record_start']} → {meta['record_end']}",
                  f"{r['qc']['coverage_pct']:.1f}%"]})
    parts.append(html_table(kr, caption="Key results — post-construction energy yield assessment"))

    if bm:
        color = "#2E8B57" if bm["ratio"] >= 1 else "#C0504D"
        parts.append(f"""<div class="note">
<b>Benchmark vs pre-construction estimate:</b> assessed P50 of {_fmt(bm['assessment_p50_mwh'])} MWh/yr
vs pre-construction P50 of {_fmt(bm['preconstruction_p50_mwh'])} MWh/yr
→ ratio <b style="color:{color}">{bm['ratio']:.2f}</b> ({bm['delta_pct']:+.1f}%).</div>""")

    parts.append(f"""<div class="note">
<b>Measurement-period reconciliation:</b> modelled energy over the SCADA period
({_fmt(r['losses']['recon']['modelled_net_period_mwh'])} MWh, SCADA-derived losses only) vs measured
({_fmt(r['losses']['recon']['measured_mwh'])} MWh)
→ gap {_pct(r['losses']['recon']['gap_pct'],2)} of gross. The small residual arises from
anemometer-fault imputation and the free-stream proxy; the loss tree is otherwise consistent
with the metered production.</div>""")

    # ---------------- data & QC ----------------
    parts.append("<h2>2&nbsp;&nbsp;Data overview &amp; quality control</h2>")
    parts.append(f"""<p class="lead">SCADA data is resampled to a 10-minute grid, one record per turbine.
Each record is classified into an operating state (IEC 61400-26 style). Records that cannot be
explained as genuine production are excluded from the power curve analysis but are accounted for
as energy losses.</p>""")
    parts.append(f"""<div class="note ok"><b>SCADA format:</b> OEM profile <b>{meta['oem_profile']}</b>
{'(auto-detected)' if cfg.get('oem_profile') in (None, 'auto') else '(as configured)'} — long or wide
format, {meta['interval_h']*60:g}-min records resampled to 10 min. Read method: {meta['read_info']['method']}.
Supported OEM exports: Vestas, Siemens Gamesa (SGRE), Suzlon, Envision, Nordex, Goldwind, Inox and generic
column conventions, including text status codes, European date/decimal formats, semicolon-delimited files
and MW units. Large datasets are streamed in chunks with only the needed columns loaded, so multi-GB /
multi-year / multi-hundred-turbine exports are handled within bounded memory.</div>""")
    parts.append(f"""<div class="cards">
<div class="card"><div class="k">Turbines</div><div class="v">{meta['num_turbines']}</div>
<div class="s">{', '.join(meta['turbines'][:6])}{'…' if meta['num_turbines']>6 else ''}</div></div>
<div class="card"><div class="k">Recording interval</div><div class="v">{meta['interval_h']*60:g} min</div>
<div class="s">resampled to 10 min</div></div>
<div class="card"><div class="k">Records</div><div class="v">{_fmt(r['qc']['rows'])}</div>
<div class="s">{_fmt(r['qc']['expected_rows_per_turbine'])} expected per turbine</div></div>
<div class="card"><div class="k">Capture rate</div><div class="v">{_pct(r['qc']['coverage_pct'],1)}</div>
<div class="s">of expected 10-min records</div></div></div>""")
    parts.append(html_table(
        r["qc"]["flag_counts"].rename(columns={"label": "state", "count": "records",
                                               "pct_of_records": "share"}),
        formats={"share": lambda v: _pct(v, 2)},
        caption="Operating-state classification of all SCADA records"))

    q = r["qc"]["flag_counts"]
    n_bad = int(q.loc[q["flag"].isin([6, 7]), "count"].sum())
    parts.append(f"""<div class="grid2"><div class="full">
{_chart(lambda: pl.fig_to_b64(pl.wind_rose(r['df']['dir_deg'].dropna(), r['df']['ws'].dropna(),
     title="Wind rose — all valid records")), 'Wind rose', 'No valid wind data for wind rose')}
</div></div>""")
    if n_bad:
        parts.append(f"<div class='note ok'><b>Data quality:</b> {_fmt(n_bad)} records flagged as bad data or "
                     f"anemometer faults were excluded from curve fitting and wake analysis.</div>")

    # ---- wind resource (DNV-style section 2.1) --------------------------
    ws_stats = r.get("wind_stats") or {}
    parts.append("<h3>2.1&nbsp;&nbsp;Wind resource — measured period</h3>")
    parts.append(f"""<div class="grid2">
<div>{_chart(lambda: pl.fig_to_b64(pl.monthly_wind_bar(ws_stats.get('monthly_ws'))),
    'Monthly wind', 'No monthly wind data')}</div>
<div>{_chart(lambda: pl.fig_to_b64(pl.diurnal_wind(ws_stats.get('diurnal_ws'))),
    'Diurnal wind', 'No diurnal wind data')}</div>
<div class="full">{_chart(lambda: pl.fig_to_b64(pl.ws_distribution(
    ws_stats.get('ws_hist'), r['climate'].get('meas_weibull'))), 'WS distribution',
    'No wind distribution data')}</div>
</div>""")
    if "yearly" in r and r["yearly"] is not None and len(r["yearly"]) > 1:
        parts.append(html_table(
            r["yearly"], formats={"year": lambda v: f"{v:g}",
                                  "gross_mwh": lambda v: _fmt(v, 0),
                                  "measured_mwh": lambda v: _fmt(v, 0),
                                  "hours": lambda v: _fmt(v, 0)},
            caption="Yearly production summary (gross = warranted-curve energy at measured wind)"))

    # ---- 2.2 gap filling (traceable) ----
    gf = r.get("gap_fill") or {}
    parts.append("<h3>2.2&nbsp;&nbsp;Data gap filling</h3>")
    if gf and gf.get("summary") is not None and len(gf["summary"]):
        gfs = gf["summary"]
        parts.append(
            "<p class=\"lead\">Missing wind-speed / power records were imputed per turbine "
            "with the following traceable rules, so every filled value can be audited:</p>"
            "<ul class=\"det\">"
            "<li><b>Short gaps (max 1 h):</b> linear interpolation of wind speed between the "
            "bounding valid records; power from the warranted curve at the interpolated wind speed.</li>"
            "<li><b>Medium gaps (more than 1 h up to 4 h):</b> wind speed imputed from the "
            "farm-average wind speed at the same timestamps, scaled by the turbine's own median "
            "ratio; power from the warranted curve.</li>"
            "<li><b>Longer gaps (more than 4 h):</b> not filled (would be speculation) — "
            "accounted via data coverage and availability metrics instead.</li>"
            "</ul>"
            f"<p class=\"lead\"><b>{gf.get('n_filled', 0):,} records filled in total.</b> "
            "The column <code>fill_method</code> in the exported gap-filled CSV identifies each "
            "imputed value (<code>linear_interp</code> / <code>neighbour_impute</code> / "
            "<code>curve_estimate</code>).</p>")
        parts.append(html_table(
            gfs,
            formats={"ws_linear_filled": lambda v: _fmt(v, 0),
                     "ws_neighbour_filled": lambda v: _fmt(v, 0),
                     "power_curve_filled": lambda v: _fmt(v, 0),
                     "total_filled": lambda v: _fmt(v, 0)},
            caption="Gap-filling summary per turbine"))

    # ---------------- availability ----------------
    parts.append("<h2>3&nbsp;&nbsp;Availability &amp; lost-energy accounting</h2>")
    ds = E["downtime_split"] or {}
    ds_html = "".join(f"<tr><td>{k}</td><td style='text-align:right'>{_fmt(v)} MWh</td></tr>"
                      for k, v in sorted(ds.items(), key=lambda x: -x[1])) if ds else ""
    parts.append(f"""<div class="cards">
<div class="card"><div class="k">Time-based availability</div><div class="v">{_pct(avail['time_avail_pct'])}</div>
<div class="s">mean across turbines</div></div>
<div class="card"><div class="k">Production-based availability</div><div class="v">{_pct(avail['prod_avail_pct'])}</div>
<div class="s">energy-based (excl. curtailment)</div></div>
<div class="card"><div class="k">Downtime loss</div><div class="v">{_fmt(E['E_lost_mwh'].get(2,0))}</div>
<div class="s">MWh over period</div></div>
<div class="card"><div class="k">Curtailment loss</div><div class="v">{_fmt(E['E_lost_mwh'].get(3,0))}</div>
<div class="s">MWh over period</div></div></div>""")
    parts.append(f"""<div class="grid2">
<div>{html_table(
    pd.DataFrame([{"metric": "Energy produced (SCADA)", "mwh": E["E_actual_mwh"]},
                  {"metric": "Downtime loss (faults/maintenance/grid)", "mwh": E["E_lost_mwh"].get(2,0)},
                  {"metric": "Curtailment loss", "mwh": E["E_lost_mwh"].get(3,0)},
                  {"metric": "Derating / partial load", "mwh": E["E_lost_mwh"].get(4,0)},
                  {"metric": "Environmental losses", "mwh": E["E_lost_mwh"].get(5,0)}]),
    formats={"mwh": lambda v: _fmt(v)},
    caption="Energy balance over the measurement period (MWh)")}
{html_table(pd.DataFrame([{"category": k, "loss_mwh": v} for k, v in ds.items()]),
    formats={"loss_mwh": lambda v: _fmt(v)},
    caption="Downtime split by cause (from SCADA status codes)") if ds_html else ""}
</div><div>
{html_table(r["availability"]["per_turbine"].round(2),
    formats={"hours": lambda v: _fmt(v,0), "downtime_h": lambda v: _fmt(v,0),
             "curtailment_h": lambda v: _fmt(v,0), "time_avail_pct": lambda v: _pct(v,2),
             "prod_avail_pct": lambda v: _pct(v,2), "energy_mwh": lambda v: _fmt(v,1),
             "downtime_loss_mwh": lambda v: _fmt(v,1),
             "curtailment_loss_mwh": lambda v: _fmt(v,1),
             "derating_loss_mwh": lambda v: _fmt(v,1),
             "environmental_loss_mwh": lambda v: _fmt(v,1)},
    caption="Per-turbine availability &amp; losses")}
</div></div>""")
    parts.append(f"""<div class="note"><b>How availability is calculated</b>
<ul class="det">
<li><b>Time-based availability</b> — fraction of calendar time the turbine was available to produce:<br>
<span class="formula">A<sub>T</sub> = (T<sub>calendar</sub> − T<sub>downtime</sub>) / T<sub>calendar</sub> × 100&nbsp;%</span>
T<sub>downtime</sub> = hours classified as <i>faults + maintenance + grid outages</i> (SCADA status codes,
or the sustained low-power detector where codes are absent). Below cut-in, curtailment, derating and
environmental records are <i>available but not producing</i> and do not count against time-based availability.</li>
<li><b>Production (energy)-based availability</b> — fraction of the energy that could have been produced:<br>
<span class="formula">A<sub>E</sub> = E<sub>actual</sub> / (E<sub>actual</sub> + E<sub>lost,downtime</sub>) × 100&nbsp;%
&nbsp;&nbsp;with&nbsp;&nbsp; E<sub>lost,downtime</sub> = Σ<sub>downtime</sub> P<sub>warr</sub>(v<sub>i</sub>) · Δt<sub>i</sub></span>
E<sub>lost,downtime</sub> is estimated by evaluating the warranted power curve at the measured wind speed of
every downtime record (air-density adjusted). Curtailment is excluded from the denominator, in line with
IEC 61400-26-1 production-based availability. <b>In this run:</b> A<sub>T</sub> = {_pct(avail['time_avail_pct'],2)},
A<sub>E</sub> = {_pct(avail['prod_avail_pct'],2)}. Full derivation in Appendix A.</li>
</ul></div>""")
    parts.append(html_table(
        r["availability"]["monthly"], formats={"energy_mwh": lambda v: _fmt(v,1),
                                   "downtime_h": lambda v: _fmt(v,0),
                                   "curtailment_h": lambda v: _fmt(v,0),
                                   "total_h": lambda v: _fmt(v,0),
                                   "time_avail_pct": lambda v: _pct(v,1)},
        caption="Monthly production &amp; time-based availability"))
    parts.append(f"""<div class="grid2">
<div>{_chart(lambda: pl.fig_to_b64(pl.monthly_trend(
    r['availability']['monthly'], 'time_avail_pct',
    'Monthly time-based availability', 'Availability (%)')), 'Availability trend',
    'Availability trend unavailable')}</div>
<div>{_chart(lambda: pl.fig_to_b64(pl.downtime_stacked(
    _safe_df(r, 'monthly_downtime'))),
    'Downtime stacked chart', 'Downtime breakdown unavailable')}</div>
</div>""")

    # ---------------- power curve ----------------
    pc = r["power_curve"]
    parts.append("<h2>4&nbsp;&nbsp;Power curve analysis</h2>")
    parts.append(f"""<p class="lead">The measured farm power curve is derived from operating,
non-curtailed 10-min records binned into 0.5 m/s wind-speed bins (IEC 61400-12-1),
air-density corrected to standard conditions. The energy-weighted deviation from the
warranted curve is computed over the measured operating wind distribution.</p>
<div class="note"><b>Energy-weighted power curve deviation (farm): {_pct(pc['deviation_pct'])}</b>
<span class="tag">{"underperformance" if pc['deviation_pct'] > 0 else "overperformance"}</span><br>
{pc['note']}</div>""")
    parts.append(_chart(lambda: pl.fig_to_b64(pl.power_curves(
        pc["farm_curve"], pc["warranted_curve"], cfg["rated_power_kw"])), "Power curve",
        "Power curve chart unavailable (insufficient operating data)"))
    if pc.get("per_turbine_curves"):
        parts.append(_chart(lambda: pl.fig_to_b64(pl.turbine_curves_overlay(
            pc["per_turbine_curves"], pc["farm_curve"], pc["warranted_curve"],
            cfg["rated_power_kw"])), "Turbine curves overlay",
            "Per-turbine curve overlay unavailable"))
    # binned comparison table: measured vs warranted per 0.5 m/s bin
    pw_arr = np.interp(pc["farm_curve"]["bin_center"],
                       pc["warranted_curve"]["bin_center"],
                       pc["warranted_curve"]["mean_power"])
    bt = pd.DataFrame({
        "wind_speed_bin": pc["farm_curve"]["bin_center"],
        "records": pc["farm_curve"]["count"],
        "measured_kw": pc["farm_curve"]["mean_power"].round(1),
        "warranted_kw": pw_arr.round(1),
        "delta_kw": (pc["farm_curve"]["mean_power"] - pw_arr).round(1),
    })
    bt["delta_pct"] = (100.0 * bt["delta_kw"] / bt["warranted_kw"].clip(lower=1)).round(1)
    bt = bt[(bt["wind_speed_bin"] >= cfg["cut_in_mps"]) & (bt["wind_speed_bin"] <= 25.5)]
    parts.append(html_table(
        bt[bt["records"] > 0],
        formats={"records": lambda v: _fmt(v, 0),
                 "measured_kw": lambda v: _fmt(v, 0),
                 "warranted_kw": lambda v: _fmt(v, 0),
                 "delta_kw": lambda v: f"{v:+.0f}",
                 "delta_pct": lambda v: f"{v:+.1f}%"},
        caption="Farm power curve — measured vs warranted per 0.5 m/s bin "
                "(positive delta = underperformance)"))
    parts.append(html_table(
        pc["per_turbine"].round(4),
        formats={"deviation_pct": lambda v: _pct(v, 2),
                 "performance_ratio": lambda v: f"{v:.4f}",
                 "n_operating_rows": lambda v: _fmt(v, 0)},
        caption="Per-turbine energy-weighted power curve deviation &amp; operating performance ratio "
                "(ratio of actual to expected energy while operating; <1 = shortfall)"))
    # P-P plot + monthly performance ratio (DNV-style turbine performance)
    pp = (r.get("wind_stats") or {}).get("pp_sample")
    parts.append(f"""<div class="grid2">
<div>{_chart(lambda: pl.fig_to_b64(pl.pp_plot(pp, cfg['rated_power_kw'])),
    'P-P plot', 'P-P plot unavailable (insufficient operating data)')}</div>
<div>{_chart(lambda: pl.fig_to_b64(pl.monthly_trend(
    _safe_df(r, 'monthly_perf'), 'performance_ratio',
    'Monthly performance ratio (actual/expected while operating)',
    'Performance ratio', color=ORANGE, fmt='.3f')), 'Monthly performance',
    'Monthly performance unavailable')}</div>
</div>""")

    # ---------------- wake ----------------
    wk = r["wake"]
    parts.append("<h2>5&nbsp;&nbsp;Wake analysis</h2>")
    parts.append(f"""<p class="lead">The free-stream wind speed per interval is estimated with the reference-turbine
method (mean of the least-waked turbines per direction sector; anemometer-faulted turbines excluded). The wake deficit
per turbine per sector is 1 − ws<sub>i</sub>/ws<sub>free</sub>; the wake energy loss is the difference
between the warranted-curve energy at free-stream and at nacelle speed over operating intervals.</p>""")
    if wk["sector_table"].empty:
        parts.append(f"""<div class="note"><b>Wake analysis could not be performed:</b> no usable operating
records were found (check that the SCADA status-code mapping in the config matches your OEM codes, and that
power/wind-speed values are in kW and m/s). Wake loss is set to 0% and wake uncertainty is widened accordingly.</div>""")
    else:
        parts.append(f"""<div class="note"><b>Wake loss: {_pct(wk['wake_loss_pct'])} of gross ({_fmt(wk['wake_energy_mwh'])} MWh over period)</b>
<br>Without a met mast / lidar reference the free-stream proxy is indicative; layout-aware wake modelling
would refine this value.</div>""")
    parts.append(f"""<div class="grid2">
<div>{_chart(lambda: pl.fig_to_b64(pl.wake_polar(wk["sector_table"])), 'Wake polar',
        "Wake polar unavailable (insufficient data)")}</div>
<div>{html_table(wk["sector_table"].round(4),
    formats={"mean_deficit": lambda v: _pct(v*100,2), "n_samples": lambda v: _fmt(v,0)},
    caption="Mean wake deficit by direction sector")}
{html_table(wk["per_turbine"].sort_values("mean_deficit", ascending=False).round(4),
    formats={"mean_deficit": lambda v: _pct(v*100,2), "n_samples": lambda v: _fmt(v,0),
             "wake_energy_mwh": lambda v: _fmt(v,1)},
    caption="Per-turbine mean wake deficit")}</div></div>""")

    # ---------------- yaw error analysis (5.1) ----------------
    yw = r.get("yaw") or {}
    parts.append("<h3>5.1&nbsp;&nbsp;Yaw error analysis (wind direction vs nacelle)</h3>")
    if yw and yw.get("per_turbine") is not None and len(yw["per_turbine"]):
        parts.append(
            "<p class=\"lead\">The yaw error per turbine is the power-weighted circular mean "
            "of (wind direction - nacelle direction) over operating records at wind speeds "
            f"&ge; {yw.get('min_ws', 6.0)} m/s. A positive value means the nacelle lags the wind; "
            "a systematic non-zero yaw error reduces energy capture (cosine loss ~ 1 - cos(yaw)) "
            "and is a candidate for yaw optimisation.</p>")
        parts.append(
            "<div class=\"grid2\">"
            f"<div>{_chart(lambda: pl.fig_to_b64(pl.yaw_histogram(yw.get('histogram'))), 'Yaw histogram', 'Yaw histogram unavailable')}</div>"
            "<div>" + html_table(
                yw["per_turbine"],
                formats={"yaw_error_deg": lambda v: f"{v:+.2f}&deg;",
                         "circular_std_deg": lambda v: f"{v:.2f}&deg;",
                         "n_records": lambda v: _fmt(v, 0),
                         "mean_ws": lambda v: f"{v:.2f}"},
                caption="Per-turbine yaw error (power-weighted circular mean)") + "</div>"
            "</div>")
        worst = yw["per_turbine"].reindex(
            yw["per_turbine"]["yaw_error_deg"].abs().sort_values(ascending=False).index).head(3)
        if len(worst) and worst["yaw_error_deg"].abs().max() > 3.0:
            parts.append(
                "<div class=\"note\"><b>Yaw finding:</b> turbines "
                + ", ".join(str(t) for t in worst["turbine"].tolist())
                + " show yaw errors of "
                + ", ".join(f"{v:+.1f}&deg;" for v in worst["yaw_error_deg"].tolist())
                + " - candidate yaw alignment / LiDAR campaigns.</div>")

    # ---------------- long-term climate ----------------
    cl = r["climate"]
    parts.append("<h2>6&nbsp;&nbsp;Long-term wind climate &amp; MCP</h2>")
    parts.append(f"""<p class="lead">The short measurement record is correlated with a long-term reference
({cl['method']}, {cl['lt_n_years']:.1f} years) using sector-wise linear regression (Measure-Correlate-Predict),
and a long-term Weibull wind distribution is fitted to the predicted long-term daily wind speeds.</p>""")
    if cl["mcp"] is not None:
        parts.append(f"""<div class="cards">
<div class="card"><div class="k">MCP R²</div><div class="v">{cl['mcp']['r2']:.2f}</div>
<div class="s">energy-weighted across sectors</div></div>
<div class="card"><div class="k">Measured mean wind speed</div><div class="v">{cl['meas_mean_ws']:.2f} m/s</div>
<div class="s">daily means, nacelle anemometry</div></div>
<div class="card"><div class="k">Long-term mean wind speed</div><div class="v">{cl['lt_mean_ws']:.2f} m/s</div>
<div class="s">predicted via MCP</div></div>
<div class="card"><div class="k">Long-term Weibull</div><div class="v">A={cl['lt_weibull'][0]:.2f}, k={cl['lt_weibull'][1]:.2f}</div>
<div class="s">fitted to LT daily means</div></div></div>""")
    parts.append(f"""<div class="grid2">
<div>{_chart(lambda: pl.fig_to_b64(pl.weibull_plot(cl['site_daily'].values, cl['lt_weibull'],
    'Measured record (daily means)', 'Measured &amp; long-term wind distributions',
    cl['lt_weibull'][0], cl['lt_weibull'][1])), 'Weibull', 'Wind distribution chart unavailable')}</div>
<div>{_chart(lambda: pl.fig_to_b64(pl.mcp_scatter(cl['site_daily'], cl['ref']['ws'] if cl['ref'] is not None else cl['site_daily'],
    cl['mcp']['r2'] if cl['mcp'] else 0.0)), 'MCP', 'MCP chart unavailable')}</div></div>""")
    if cl["mcp"] is not None:
        parts.append(html_table(cl["mcp"]["stats"].round(4),
                                formats={"n_days": lambda v: _fmt(v,0), "slope": lambda v: f"{v:.4f}",
                                         "intercept": lambda v: f"{v:.4f}", "r2": lambda v: f"{v:.3f}"},
                                caption="MCP regression statistics per sector"))

    # ---- production-based LT assessment (Method B) ---------------------
    prod = cl.get("production")
    parts.append("<h3>6.1&nbsp;&nbsp;Production-based long-term assessment (energy vs wind regression)</h3>")
    parts.append("""<p class="lead">In addition to the Weibull × warranted-curve method (Section 7),
the measured <b>gross energy vs wind-speed relationship</b> is established directly — daily farm gross
energy vs daily site wind speed, and (when ≥ 6 months are available) monthly farm gross energy vs the
long-term reference monthly wind speed. The fitted relationship is then applied to the full long-term
record to predict the long-term energy series; its annualised sum is the long-term gross AEP.</p>""")
    if prod is None or (not prod.get("daily") and not prod.get("monthly")):
        note = (prod or {}).get("note", "insufficient measured data (need ≥ 3 months "
                                         "with valid wind and energy)")
        parts.append(f"""<div class="note"><b>Production-based assessment not available:</b> {note}
The Weibull × warranted-curve method (Method A) is used for the gross AEP.</div>""")
    else:
        if prod.get("daily"):
            d = prod["daily"]
            if d["fit"]["kind"] in ("power_law", "cubic"):
                eq = f"E<sub>day</sub> = {d['fit']['c']:.2f} × WS<sup>{d['fit']['b']:.2f}</sup>"
            else:
                eq = (f"E<sub>day</sub> = {d['fit']['b']:.3f} × WS "
                      f"{'−' if d['fit']['a'] < 0 else '+'} {abs(d['fit']['a']):.0f}")
            parts.append(f"""<div class="grid2">
<div>{_chart(lambda: pl.fig_to_b64(pl.prod_scatter(
    d['measured'], d['wind'], d['fit'], 'Daily gross energy vs daily site wind',
    'Daily site wind speed (m/s)', 'Daily farm gross energy (MWh)')), 'Daily energy vs wind',
    'Daily scatter unavailable')}</div>
<div>{_chart(lambda: pl.fig_to_b64(pl.lt_predicted_series(
    d['lt_predicted'], cl['lt_n_years'], 'Long-term predicted daily energy',
    'Energy (MWh/day)')), 'LT predicted daily energy', 'LT series unavailable')}</div></div>
<div class="note"><b>Daily production regression:</b> {eq} MWh, R² = {d['fit']['r2']:.2f}
(n = {d['fit']['n']} days) → LT annual gross <b>{_fmt(d['lt_annual_gross_mwh'])} MWh/yr</b>
(applied to {cl['lt_n_years']:.1f} years of long-term daily wind).{(' <b style="color:#C0504D">' + d['unreliable_reason'] + '</b>') if d.get('unreliable') else ''}</div>""")
        if "monthly" in prod:
            m = prod["monthly"]
            parts.append(f"""<div class="grid2">
<div>{_chart(lambda: pl.fig_to_b64(pl.prod_scatter(
    m['measured'], m['wind'], m['fit'], 'Monthly gross energy vs reference wind',
    'Monthly reference wind speed (m/s)', 'Monthly farm gross energy (MWh, 30-day norm.)')), 'Monthly energy vs wind',
    'Monthly scatter unavailable')}</div>
<div>{_chart(lambda: pl.fig_to_b64(pl.lt_predicted_series(
    m['lt_predicted'], cl['lt_n_years'], 'Long-term predicted monthly energy',
    'Energy (MWh/month)')), 'LT predicted monthly energy', 'LT monthly series unavailable')}</div></div>
<div class="note"><b>Monthly production regression:</b> E<sub>m</sub> = {m['fit']['b']:.3f} × WS_ref
{'−' if m['fit']['a'] < 0 else '+'} {abs(m['fit']['a']):.0f} MWh (30-day normalised),
R² = {m['fit']['r2']:.2f} (n = {m['fit']['n']} months) → LT annual gross
<b>{_fmt(m['lt_annual_gross_mwh'])} MWh/yr</b>.{(' <b style="color:#C0504D">' + m['unreliable_reason'] + '</b>') if m.get('unreliable') else ''}</div>""")
        used = (f"{_fmt(gross_lt)} MWh/yr ({r['losses']['lt_method_used']})"
                if prod.get("lt_gross_mwh") else
                f"{_fmt(gross_lt)} MWh/yr ({r['losses']['lt_method_used']}) — "
                f"Method B indicative only (record covers {prod.get('record_months', 0):.1f} months "
                f"of a full seasonal cycle; < 9 months)")
        parts.append(f"""<div class="note ok"><b>Method comparison:</b> Weibull × warranted curve
(Method A) = {_fmt(r['losses']['gross_lt_mwh_method_a'])} MWh/yr · production regression
(Method B{', ' + prod['primary'] if prod.get('primary') else ''}) =
{_fmt(prod.get('lt_gross_mwh') or r['losses']['gross_lt_mwh_method_b'])} MWh/yr ·
<b>used for the loss tree: {used}</b>.</div>""")

    # ---------------- loss tree ----------------
    parts.append("<h2>7&nbsp;&nbsp;Loss tree &amp; net energy yield</h2>")
    parts.append(f"""<p class="lead">Gross AEP (long-term) = {_fmt(gross_lt)} MWh/yr is obtained by the
<b>{r['losses']['lt_method_used']}</b> method (see §6.1 for the production-regression method and its
cross-check against the Weibull × warranted-curve method). Losses measured over the SCADA period
(availability, curtailment, wake, performance, …) plus electrical and other losses are applied to give
the net energy yield.</p>""")
    parts.append(f"""<div class="note"><b>How the long-term energy yield is calculated</b>
<ul class="det">
<li><b>1. Long-term wind climate (MCP):</b> daily site-mean wind speed is regressed on the long-term
reference per 30° sector (OLS); the long-term daily series is predicted and converted to a Weibull
distribution — shape k from the measured record, scale A adjusted to the long-term mean wind speed.</li>
<li><b>2. Gross AEP — two methods, cross-checked:</b><br>
<span class="formula">Method A (curve):&nbsp; E<sub>gross</sub> = 8760 h × ∫<sub>0</sub><sup>∞</sup> f(v; A, k) · P<sub>warr</sub>(v) dv
&nbsp;→ {_fmt(r['losses']['gross_lt_mwh_method_a'])} MWh/yr</span>
<span class="formula">Method B (production regression, §6.1):&nbsp; E<sub>day/m</sub> = a + b·WS applied to the
long-term wind record &nbsp;→ {_fmt(r['losses']['gross_lt_mwh_method_b'])} MWh/yr</span>
Method B is used as primary when its fit is adequate (R² ≥ 0.5 monthly / daily fallback); both are
always reported and compared.</li>
<li><b>3. Loss tree:</b> each loss l<sub>i</sub> is applied multiplicatively:<br>
<span class="formula">E<sub>net</sub> = E<sub>gross</sub> × (1 − l<sub>avail</sub>) × (1 − l<sub>curt</sub>) × (1 − l<sub>derate</sub>)
× (1 − l<sub>env</sub>) × (1 − l<sub>wake</sub>) × (1 − l<sub>perf</sub>) × (1 − l<sub>elec</sub>) × (1 − l<sub>other</sub>)</span>
Availability, curtailment, derating, environmental, wake and performance losses are measured from the
SCADA period; electrical and other losses are inputs.</li>
<li><b>4. P-values:</b> Monte Carlo ({cfg['mc_iterations']:,} draws) applies lognormal 1σ uncertainty
components; P50/P75/P90/P99 are quantiles of the resulting net-AEP distribution.</li>
</ul>
<b>In this run:</b> gross {_fmt(gross_lt)} MWh/yr ({r['losses']['lt_method_used']})
→ {_pct(_to_num(_numeric_sum(tree['pct_of_gross'])))} losses
→ deterministic net {_fmt(unc['deterministic_net_mwh'])} MWh/yr → P50 {_fmt(p['P50'])} MWh/yr.
Full derivation in Appendix B.</div>""")
    parts.append(_chart(lambda: pl.fig_to_b64(pl.loss_waterfall(
        tree.to_dict("records"), gross_lt, net)), "Loss tree", "Loss tree chart unavailable"))
    parts.append(
        "<h3>7.1&nbsp;&nbsp;Loss calculation details</h3>"
        "<p class=\"lead\">Every loss category in the waterfall above is quantified as follows "
        "(all values are air-density corrected; energy = power x interval):</p>"
        "<ul class=\"det\">"
        "<li><b>Availability (downtime):</b> E_down = sum of warranted-curve power at the measured "
        "wind speed over all records classified as fault / maintenance / grid outage. "
        "A_E = E_act / (E_act + E_down).</li>"
        "<li><b>Curtailment:</b> E_curt = sum of (P_warr(v) - P_act) over curtailment-flagged "
        "records, or records producing below 30% of expected at usable wind speeds.</li>"
        "<li><b>Derating / partial load:</b> E_der = sum of (P_warr(v) - P_act) over records "
        "producing at 30-85% of expected at usable wind speeds (derate states).</li>"
        "<li><b>Environmental:</b> E_env = sum of (P_warr(v) - P_act) over records flagged "
        "icing / high-temperature derate.</li>"
        "<li><b>Wake:</b> E_wake = sum of [P_warr(v_free) - P_warr(v_i)] x dt over operating "
        "records, where v_free is the reference-turbine free-stream wind speed and v_i the "
        "turbine's nacelle wind speed (deficits aggregated per 30&deg; sector).</li>"
        "<li><b>Turbine performance:</b> operating shortfall of actual vs warranted-curve energy "
        "at the turbine's own wind speed, minus the wake component (to avoid double counting). "
        "A negative value means the fleet outperforms the warranted curve.</li>"
        "<li><b>Electrical / Other:</b> input assumptions (transformer, collection, "
        "transmission losses; miscellaneous), applied multiplicatively as configured.</li>"
        "</ul>"
        "<p class=\"lead\">All losses are expressed as % of gross energy over the measured "
        "period and applied multiplicatively to the long-term gross AEP; the reconciliation "
        "note in the executive summary quantifies the residual gap between modelled and "
        "measured energy.</p>")
    parts.append(html_table(
        tree, formats={"energy_mwh": lambda v: (_fmt(v) if v is not None else "input"),
                       "pct_of_gross": lambda v: _pct(v, 2)},
        caption="Loss tree — losses as % of gross AEP"))

    # ---- WTG-level loss stack ----
    per = r["availability"]["per_turbine"]
    gross_p = float(r["losses"]["recon"]["gross_period_mwh"]) or 1.0
    wtg_rows = []
    wk_map = {}
    if len(r["wake"]["per_turbine"]):
        wk_map = dict(zip(r["wake"]["per_turbine"]["turbine"],
                          r["wake"]["per_turbine"]["wake_energy_mwh"]))
    for _, row in per.iterrows():
        tid = row["turbine"]
        d = float(row.get("downtime_loss_mwh", 0))
        c = float(row.get("curtailment_loss_mwh", 0))
        de = float(row.get("derating_loss_mwh", 0))
        e = float(row.get("environmental_loss_mwh", 0))
        wk = float(wk_map.get(tid, 0.0))
        wtg_rows.append({"turbine": tid,
                         "downtime_pct": 100.0 * d / gross_p,
                         "curtailment_pct": 100.0 * c / gross_p,
                         "derating_pct": 100.0 * de / gross_p,
                         "environmental_pct": 100.0 * e / gross_p,
                         "wake_pct": 100.0 * wk / gross_p,
                         "total_loss_pct": 100.0 * (d + c + de + e + wk) / gross_p})
    wtg = pd.DataFrame(wtg_rows)
    wtg_fmt = {k: (lambda v: _pct(v, 2)) for k in wtg.columns if k != "turbine"}
    parts.append(html_table(
        wtg.round(3), formats=wtg_fmt,
        caption="WTG-level loss categorisation (% of period gross energy) - "
                "availability, curtailment, derating, environmental and wake "
                "losses per turbine"))

    # ---------------- uncertainty ----------------
    parts.append("<h2>8&nbsp;&nbsp;Uncertainty &amp; P-values</h2>")
    parts.append(f"""<p class="lead">A Monte Carlo simulation ({cfg['mc_iterations']:,} draws) combines the
1σ uncertainty of each component (lognormal multipliers) to produce the distribution of net annual
energy production. P-values are quantiles of that distribution.</p>""")
    parts.append(f"""<div class="grid2">
<div>{_chart(lambda: pl.fig_to_b64(pl.mc_histogram(unc['samples'], p)), 'MC histogram', 'Histogram unavailable')}</div>
<div>{_chart(lambda: pl.fig_to_b64(pl.tornado(unc['components'])), 'Tornado', 'Tornado chart unavailable')}</div></div>""")
    pv = pd.DataFrame([
        {"P-value": "P50 (median)", "Net AEP (MWh/yr)": p["P50"], "of deterministic": f"{100*p['P50']/net:.1f}%"},
        {"P-value": "P75", "Net AEP (MWh/yr)": p["P75"], "of deterministic": f"{100*p['P75']/net:.1f}%"},
        {"P-value": "P90", "Net AEP (MWh/yr)": p["P90"], "of deterministic": f"{100*p['P90']/net:.1f}%"},
        {"P-value": "P99", "Net AEP (MWh/yr)": p["P99"], "of deterministic": f"{100*p['P99']/net:.1f}%"},
    ])
    parts.append(html_table(
        pv,
        formats={"Net AEP (MWh/yr)": lambda v: f"{v:,.0f}",
                 "of deterministic": lambda v: f"{v:.1f}%"},
        caption="P-values of net annual energy yield"))
    mc_mean = float(np.mean(unc["samples"]))
    mc_median = float(np.median(unc["samples"]))
    mc_sd = float(unc["sd"])
    audit_note = (
        f"<div class='note'><b>Monte Carlo audit:</b> {cfg['mc_iterations']:,} draws, "
        f"lognormal multipliers. Sample mean = {_fmt(mc_mean)} MWh/yr, sample median = "
        f"{_fmt(mc_median)} MWh/yr, deterministic net = {_fmt(net)} MWh/yr, "
        f"1σ = {_fmt(mc_sd)} MWh/yr ({100.0*mc_sd/net:.1f}%). A lognormal distribution has "
        f"median = deterministic value; the difference ({100.0*(mc_median-net)/net:+.2f}%) "
        f"is sampling noise. Combined 1σ = "
        f"{_pct(float(np.sqrt((unc['components']['sigma_pct']**2).sum())))}.</div>"
    )
    parts.append(audit_note)

    parts.append(html_table(
        unc["components"],
        formats={"sigma_pct": lambda v: _pct(v, 2), "contribution_pct": lambda v: _pct(v, 1)},
        caption="Uncertainty components (1σ) and share of total variance"))

    # ---------------- DNV-style: sensitivity & conclusions ----------------
    sens = _sensitivity_table(r)
    parts.append("<h2>9&nbsp;&nbsp;Sensitivity analysis</h2>")
    parts.append("""<p class="lead">Sensitivity of the net P50 yield to the key parameters,
computed by perturbing each parameter while holding the others at their base values.</p>""")
    parts.append(html_table(sens, formats={"delta_p50_mwh": lambda v: f"{v:+,.0f}",
                                           "delta_p50_pct": lambda v: f"{v:+.2f}%"},
                            caption="One-at-a-time sensitivity of net P50 energy yield"))
    parts.append(_conclusions_html(r, sens))

    # ---------------- appendices ----------------
    total_h = float(r["availability"]["per_turbine"]["hours"].sum())
    down_h = float(r["availability"]["per_turbine"]["downtime_h"].sum())
    e_act = E["E_actual_mwh"]
    e_down = E["E_lost_mwh"].get(2, 0.0)
    a_e_implied = 100.0 * e_act / (e_act + e_down) if (e_act + e_down) > 0 else float("nan")
    lt = r["climate"]
    combined_sigma = float(np.sqrt((unc["components"]["sigma_pct"] ** 2).sum()))

    parts.append("""<h2>Appendix A — Availability: calculation details</h2>
<p class="lead">Definitions follow IEC 61400-26-1 availability categories and standard PCEYA
practice. Every 10-min SCADA record is first classified into one of three groups:</p>
<ul class="det">
<li><b>Unavailable (downtime):</b> faults, maintenance and grid outages — from SCADA status codes
when present (config <code>status_codes</code>), otherwise from the heuristic detector
(power &lt; 0.5% of rated at wind speeds above cut-in + 1 m/s, sustained ≥ 1 h).</li>
<li><b>Available but not producing:</b> below cut-in, curtailment, derating / partial load and
environmental derates. These count as <i>available</i>; their lost energy is quantified separately
so it can be studied (and claimed, e.g. grid curtailment) rather than masked.</li>
<li><b>Bad data / anemometer faults:</b> excluded from both availability metrics and from power-curve
and wake fitting.</li>
</ul>
<div class="formula"><b>Time-based availability (per turbine):</b>
A<sub>T</sub> = (T<sub>calendar</sub> − T<sub>downtime</sub>) / T<sub>calendar</sub> × 100&nbsp;%<br>
<b>Production (energy) based availability (per turbine):</b>
A<sub>E</sub> = E<sub>actual</sub> / (E<sub>actual</sub> + E<sub>lost,downtime</sub>) × 100&nbsp;%<br>
E<sub>lost,downtime</sub> = Σ<sub>downtime records</sub> P<sub>warr</sub>(v<sub>i</sub>, ρ<sub>i</sub>) · Δt<sub>i</sub>
&nbsp;— the warranted power curve at the measured wind speed, air-density corrected, integrated over
the downtime records.</div>
<p class="lead">Farm availability = mean of the per-turbine values. Curtailed time is <i>available</i>
for A<sub>T</sub> and excluded from A<sub>E</sub> (IEC 61400-26-1 production-based availability).</p>""")
    parts.append(f"""<p class="lead"><b>Worked example (this run):</b> farm calendar time
ΣT<sub>cal</sub> = {_fmt(total_h)} h, downtime ΣT<sub>down</sub> = {_fmt(down_h)} h
→ <b>A<sub>T</sub> = {_pct(avail['time_avail_pct'],2)}</b>. Measured energy
E<sub>act</sub> = {_fmt(e_act)} MWh, downtime loss E<sub>down</sub> = {_fmt(e_down)} MWh
→ aggregate <b>A<sub>E</sub> = {_pct(a_e_implied,2)}</b> (farm mean of per-turbine values:
{_pct(avail['prod_avail_pct'],2)}).</p>""")

    parts.append(f"""<h2>Appendix B — Long-term energy yield: calculation details</h2>
<p class="lead"><b>Step 0 — Long-term wind climate (MCP).</b> Daily farm-mean wind speeds from nacelle
anemometry (mean across operating turbines) are regressed per 30° direction sector on the long-term
reference ({lt['method']}, {lt['lt_n_years']:.1f} years) by ordinary least squares. The regressions map
the reference onto the site for the full long-term period, giving a long-term daily site wind series.
The long-term Weibull distribution is then</p>
<div class="formula">k<sub>LT</sub> = k<sub>measured record</sub>
&nbsp;&nbsp;(&nbsp;shape is a site characteristic, taken from the measured 10-min record&nbsp;)<br>
A<sub>LT</sub> = A<sub>measured</sub> × μ<sub>LT</sub> / μ<sub>measured</sub>
&nbsp;&nbsp;(&nbsp;scale adjusted so the distribution matches the long-term mean wind speed&nbsp;)</div>
<p class="lead"><b>Step 1 — Gross AEP, Method A (Weibull × warranted curve).</b> Per turbine per year,
numerically integrated (trapezoidal, 0.05 m/s grid) from cut-in to cut-out with P = 0 above cut-out:</p>
<div class="formula">E<sub>gross,1</sub> = 8760 h × ∫<sub>0</sub><sup>∞</sup> f(v; A<sub>LT</sub>, k<sub>LT</sub>) · P<sub>warr</sub>(v) dv
&nbsp;&nbsp;&nbsp;f(v; A, k) = (k/A)·(v/A)<sup>k−1</sup>·e<sup>−(v/A)<sup>k</sup></sup></div>
<p class="lead"><b>Step 1b — Gross AEP, Method B (production regression, §6.1).</b> The measured
<b>gross daily energy</b> (warranted-curve energy at measured wind over all plausible records =
actual + downtime + curtailment + derating + environmental) is regressed on the daily site wind
speed; with ≥ 6 months of data the <b>gross monthly energy</b> (30.44-day normalised) is regressed
on the long-term reference monthly wind speed. The fitted relationship is applied to the full
long-term wind record and annualised:</p>
<div class="formula">E<sub>gross,LT</sub> = Σ<sub>LT</sub> (a + b·WS<sub>t</sub>) / N<sub>years</sub></div>
<p class="lead"><b>Step 2 — Losses.</b> SCADA-derived losses (as % of gross energy over the measured
period) plus the electrical/other inputs are applied multiplicatively to the chosen gross AEP:</p>
<div class="formula">E<sub>net</sub> = E<sub>gross</sub> × Π<sub>i</sub> (1 − l<sub>i</sub>) &nbsp; with
l<sub>i</sub> = availability, curtailment, derating, environmental, wake, performance, electrical, other</div>
<p class="lead"><b>Step 3 — P-values.</b> A Monte Carlo simulation ({cfg['mc_iterations']:,} draws)
applies a lognormal multiplier to each component (1σ<sub>i</sub>); combined
1σ = √(Σ σ<sub>i</sub>²) = {_pct(combined_sigma,2)}. P50 equals the deterministic net yield
(lognormal median), P90 ≈ P50 · e<sup>−1.282σ</sup>, P75 ≈ P50 · e<sup>−0.674σ</sup>.</p>""")
    parts.append(f"""<p class="lead"><b>Worked example (this run):</b> A<sub>LT</sub> =
{lt['lt_weibull'][0]:.2f} m/s, k<sub>LT</sub> = {lt['lt_weibull'][1]:.2f} → gross
{_fmt(gross_lt)} MWh/yr; total losses {_pct(_to_num(_numeric_sum(tree['pct_of_gross'])))} → deterministic net
{_fmt(unc['deterministic_net_mwh'])} MWh/yr → <b>P50 = {_fmt(p['P50'])} MWh/yr</b>,
P75 = {_fmt(p['P75'])}, P90 = {_fmt(p['P90'])}, P99 = {_fmt(p['P99'])}.</p>""")

    parts.append("""<h2>Appendix C — How to run this tool</h2>
<ul class="det">
<li><b>Web application:</b> <code>python app.py</code> → open <code>http://localhost:8000</code>,
upload the SCADA file (CSV/XLSX, long or wide format) plus optional config JSON, warranted power
curve and long-term wind file — or click “Run with bundled sample data”. The report opens in the
browser; the Excel workbook and CSVs can be downloaded.</li>
<li><b>Command line:</b> <code>python -m windpcea.cli --config config.json --scada scada.csv
--outdir results</code></li>
<li><b>Inputs:</b> SCADA export (OEM profiles auto-detected: Vestas, Siemens Gamesa, Suzlon,
Envision, Nordex, Goldwind, Inox — including text status codes, European date/decimal formats,
semicolon-delimited files, MW units and Chinese headers), warranted power curve CSV
(<code>wind_speed_mps, power_kw</code>), optional long-term daily wind file (<code>date, ws_mps</code>)
or automatic NASA POWER reanalysis from latitude/longitude, and a JSON config for status-code
mapping, losses and uncertainty overrides.</li>
<li><b>Outputs:</b> <code>pceya_report.html</code> (this report), <code>pceya_results.xlsx</code>,
<code>flagged_scada_10min.csv</code>, <code>farm_power_curve.csv</code>,
<code>per_turbine_metrics.csv</code>.</li>
</ul>""")

    parts.append("""<h2>Appendix D — Data processing, traceability &amp; DNV/DAKKS alignment</h2>
<ul class="det">
<li><b>Input traceability:</b> every report records the SCADA file, OEM profile detected,
recording interval, record period, coverage, and the read method (streamed / blockwise). The
flagged 10-min dataset (<code>flagged_scada_10min.csv</code>) carries per-record operating-state
flags and reasons; the gap-filled export adds <code>fill_method</code> for every imputed value
(§2.2).</li>
<li><b>QC decisions:</b> spikes, out-of-range values, frozen anemometers, text-in-numeric channels
are coerced/flagged with documented rules; excluded records are reported in the QC table and never
silently dropped from loss accounting.</li>
<li><b>DNV-style PCEYA alignment:</b> IEC 61400-12-1 binning (0.5 m/s, min-count, density
correction); IEC 61400-26 availability categories; MCP long-term correction (sector-wise linear
regression) against ERA5T/NASA POWER or a user reference; loss tree applied multiplicatively;
Monte Carlo P-values with lognormal components; sensitivity analysis; reconciliation of modelled
vs metered energy.</li>
<li><b>DAKKS-readiness notes:</b> for accreditation-style audits, keep (1) the original SCADA
files, (2) the config JSON used, (3) the warranted power curve source, (4) the long-term reference
provenance (ERA5T version/period or met-mast), and (5) this report's parameter values — the
pipeline is deterministic given (1)-(4), so results can be independently reproduced.</li>
<li><b>Known limitations (documented):</b> nacelle anemometry instead of a met mast for wind
speed; heuristics where status codes are absent; wake estimate uses the reference-turbine method;
losses assumed stationary into the long term.</li>
</ul>

<h2>Appendix E — Methodology, assumptions &amp; limitations</h2>
<ul class="lead" style="padding-left:18px">
<li><b>Data preparation:</b> SCADA resampled to 10-min averages; records classified per IEC 61400-26-style
operating states (operating / below cut-in / downtime / curtailment / derating / environmental / bad data /
anemometer fault). OEM export conventions are handled automatically (see Appendix C).</li>
<li><b>Availability:</b> time-based and production-based availability per turbine; lost energy from the
warranted power curve at measured wind speed (Appendix A).</li>
<li><b>Power curve:</b> IEC 61400-12-1 0.5 m/s binning of operating, non-curtailed data; air-density
correction to 1.225 kg/m³; energy-weighted deviation vs the warranted curve.</li>
<li><b>Wake:</b> free-stream wind speed per interval from the reference-turbine method (mean of the
least-waked turbines per 30° sector, anemometer-faulted turbines excluded); deficits per turbine and
sector; wake energy = warranted-curve energy at free-stream minus nacelle speed.</li>
<li><b>Long-term correction:</b> sector-wise linear-regression MCP against a user file or NASA POWER
(MERRA-2) reanalysis; long-term Weibull with shape from the measured record (Appendix B).</li>
<li><b>Loss tree:</b> losses applied multiplicatively to gross AEP; electrical and other losses are user inputs.</li>
<li><b>Uncertainty:</b> lognormal 1σ components, Monte Carlo sampling; P50/P75/P90/P99 quantiles.</li>
<li><b>Limitations:</b> nacelle anemometry (not a met mast) for wind speed; heuristic state detection
where status codes are absent; losses measured over the SCADA period are assumed stationary into the
long term; results should be refined with met-mast/lidar data, site layout and OEM curves for
certification-grade work.</li>
</ul>""")

    parts.append(f"""<div class="foot">Generated by WindPCEA — post-construction energy yield assessment.
This tool implements standard industry practice (IEC 61400-12-1, IEC 61400-26, MCP, Monte Carlo P-values)
in the style of commercial post-construction analyses. It is an engineering aid, not a certified assessment.
</div></div></body></html>""")

    path = os.path.join(outdir, "pceya_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


# --------------------------------------------------------------------------
# Excel export
# --------------------------------------------------------------------------
def export_excel(r, outdir):
    path = os.path.join(outdir, "pceya_results.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        # summary
        tree = r["losses"]["tree"]
        unc = r["uncertainty"]
        p = unc["p"]
        rows = [
            ("Farm", r["meta"]["farm_name"]), ("Turbines", r["meta"]["num_turbines"]),
            ("Rated power (kW)", r["cfg"]["rated_power_kw"]),
            ("SCADA period", f"{r['meta']['record_start']} → {r['meta']['record_end']}"),
            ("Data coverage (%)", round(r["qc"]["coverage_pct"], 1)),
            ("Time-based availability (%)", round(r["availability"]["farm"]["time_avail_pct"], 2)),
            ("Production-based availability (%)", round(r["availability"]["farm"]["prod_avail_pct"], 2)),
            ("Energy-weighted power curve deviation (%)", round(r["power_curve"]["deviation_pct"], 2)),
            ("Wake loss (% of gross)", round(r["wake"]["wake_loss_pct"], 2)),
            ("Total losses (% of gross)", round(tree["pct_of_gross"].sum(), 2)),
            ("Gross AEP (MWh/yr)", round(r["losses"]["gross_lt_mwh"], 0)),
            ("Net AEP deterministic (MWh/yr)", round(r["losses"]["net_mwh"], 0)),
            ("Net AEP P50 (MWh/yr)", round(p["P50"], 0)),
            ("Net AEP P75 (MWh/yr)", round(p["P75"], 0)),
            ("Net AEP P90 (MWh/yr)", round(p["P90"], 0)),
            ("Net AEP P99 (MWh/yr)", round(p["P99"], 0)),
            ("Uncertainty 1σ (MWh/yr)", round(unc["sd"], 0)),
            ("Capacity factor (%)", round(r["capacity_factor_pct"], 1)),
            ("Full-load hours (h/yr)", round(r["full_load_hours"], 0)),
            ("Long-term correction method", r["climate"]["method"]),
            ("Long-term mean wind speed (m/s)", round(r["climate"]["lt_mean_ws"], 2)),
            ("Long-term Weibull A (m/s)", round(r["climate"]["lt_weibull"][0], 2)),
            ("Long-term Weibull k", round(r["climate"]["lt_weibull"][1], 2)),
            ("Gross AEP - Method A (Weibull x curve, MWh/yr)", round(r["losses"]["gross_lt_mwh_method_a"], 0)),
            ("Gross AEP - Method B (production regression, MWh/yr)",
             round(r["losses"]["gross_lt_mwh_method_b"], 0) if r["losses"]["gross_lt_mwh_method_b"] else None),
            ("LT method used", r["losses"]["lt_method_used"]),
        ]
        if r["benchmark"]:
            b = r["benchmark"]
            rows += [("Pre-construction P50 (MWh/yr)", round(b["preconstruction_p50_mwh"], 0)),
                     ("Assessment/Pre-construction ratio", round(b["ratio"], 3))]
        pd.DataFrame(rows, columns=["Parameter", "Value"]).to_excel(xw, sheet_name="Summary", index=False)

        r["availability"]["per_turbine"].to_excel(xw, sheet_name="Availability", index=False)
        r["availability"]["monthly"].to_excel(xw, sheet_name="Monthly", index=False)
        pc = pd.DataFrame({
            "bin_center": r["power_curve"]["farm_curve"]["bin_center"],
            "measured_mean_power_kw": r["power_curve"]["farm_curve"]["mean_power"],
            "measured_count": r["power_curve"]["farm_curve"]["count"],
            "warranted_power_kw": np.interp(r["power_curve"]["farm_curve"]["bin_center"],
                                            r["power_curve"]["warranted_curve"]["bin_center"],
                                            r["power_curve"]["warranted_curve"]["mean_power"]),
        })
        pc.to_excel(xw, sheet_name="PowerCurve", index=False)
        r["power_curve"]["per_turbine"].to_excel(xw, sheet_name="PerTurbineDeviation", index=False)
        r["wake"]["sector_table"].to_excel(xw, sheet_name="WakeSectors", index=False)
        r["wake"]["per_turbine"].to_excel(xw, sheet_name="WakePerTurbine", index=False)
        if r["climate"]["mcp"] is not None:
            r["climate"]["mcp"]["stats"].to_excel(xw, sheet_name="MCPStats", index=False)
        pd.DataFrame({"date": r["climate"]["lt_daily"].index,
                      "ws_mps": r["climate"]["lt_daily"].values}).to_excel(xw, sheet_name="LongTermDaily", index=False)
        tree.to_excel(xw, sheet_name="LossTree", index=False)
        unc["components"].to_excel(xw, sheet_name="Uncertainty", index=False)
        pd.DataFrame([{"P-value": k, "NetAEP_MWh": v} for k, v in p.items()]).to_excel(
            xw, sheet_name="PValues", index=False)
        r["qc"]["flag_counts"].to_excel(xw, sheet_name="QC", index=False)
    return path


def export_csvs(r, outdir):
    out = []
    df = r["df"][["timestamp", "turbine", "power_kw", "ws", "energy_kwh",
                  "expected_energy_kwh", "flag", "flag_reason"]
                 + (["dir_deg"] if "dir_deg" in r["df"] else [])
                 + (["temp_c"] if "temp_c" in r["df"] else [])]
    p1 = os.path.join(outdir, "flagged_scada_10min.csv")
    df.to_csv(p1, index=False)
    out.append(p1)
    p2 = os.path.join(outdir, "farm_power_curve.csv")
    r["power_curve"]["farm_curve"].to_csv(p2, index=False)
    out.append(p2)
    p3 = os.path.join(outdir, "per_turbine_metrics.csv")
    r["power_curve"]["per_turbine"].to_csv(p3, index=False)
    out.append(p3)
    gf = r.get("gap_fill") or {}
    if gf and gf.get("df") is not None and len(gf["df"]):
        p4 = os.path.join(outdir, "gap_filled_scada_10min.csv")
        keep = [c for c in ["timestamp", "turbine", "power_kw", "ws", "dir_deg",
                            "temp_c", "status", "curt_flag", "flag", "flag_reason",
                            "filled", "fill_method"] if c in gf["df"].columns]
        gf["df"][keep].to_csv(p4, index=False)
        out.append(p4)
    return out


def console_summary(r):
    tree = r["losses"]["tree"]
    p = r["uncertainty"]["p"]
    lines = [
        f"Post-Construction Energy Yield Assessment — {r['meta']['farm_name']}",
    ]
    for _w in (r["qc"].get("warnings") or []):
        lines.append(f"  ! WARNING: {_w}")
    lines += [
        f"  Period: {r['meta']['record_start']} → {r['meta']['record_end']}  "
        f"({r['meta']['num_turbines']} turbines, {r['meta']['interval_h']*60:g} min data)",
        f"  Coverage: {r['qc']['coverage_pct']:.1f}%   "
        f"Time-based availability: {r['availability']['farm']['time_avail_pct']:.2f}%   "
        f"Production-based: {r['availability']['farm']['prod_avail_pct']:.2f}%",
        f"  Power curve deviation (energy-weighted): {r['power_curve']['deviation_pct']:+.2f}%   "
        f"Wake loss: {r['wake']['wake_loss_pct']:.2f}%",
        f"  Long-term: {r['climate']['method']}  →  "
        f"Weibull A={r['climate']['lt_weibull'][0]:.2f} m/s, k={r['climate']['lt_weibull'][1]:.2f}",
        f"  Gross AEP: {r['losses']['gross_lt_mwh']:,.0f} MWh/yr ({r['losses']['lt_method_used']}; "
        f"Method A cross-check: {r['losses']['gross_lt_mwh_method_a']:,.0f})   "
        f"Total losses: {_numeric_sum(tree['pct_of_gross']):.2f}%",
        f"  Net AEP: P50 = {p['P50']:,.0f}   P75 = {p['P75']:,.0f}   "
        f"P90 = {p['P90']:,.0f}   P99 = {p['P99']:,.0f} MWh/yr",
        f"  Capacity factor: {r['capacity_factor_pct']:.1f}%",
    ]
    if r["benchmark"]:
        b = r["benchmark"]
        lines.append(f"  Benchmark: P50 = {b['ratio']:.2f} × pre-construction "
                     f"({b['delta_pct']:+.1f}%)")
    return "\n".join(lines)
