"""Build the GitHub Pages portfolio site (docs/).

- docs/index.html        : landing page (self-contained, inline CSS, embedded charts)
- docs/sample-report.html: full sample PCEYA report (self-contained)

Usage:  python scripts/build_docs.py
"""
import base64
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
IMGS = os.path.join(DOCS, "images")


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def img_tag(path, alt, maxw="100%"):
    return (f"<img src=\"data:image/png;base64,{b64(path)}\" "
            f"alt=\"{alt}\" style=\"max-width:{maxw};border-radius:8px;"
            f"border:1px solid var(--line)\"/>")


def main():
    os.makedirs(IMGS, exist_ok=True)

    # copy the sample report into the site
    shutil.copy(os.path.join(ROOT, "results", "pceya_report.html"),
                os.path.join(DOCS, "sample-report.html"))

    css = """
:root{--navy:#14365D;--teal:#0E7C86;--orange:#E8871E;--ink:#22303f;--muted:#6b7787;--line:#dde3ec;--bg:#f4f6fa}
*{box-sizing:border-box}
body{margin:0;font-family:'Segoe UI',system-ui,Arial,sans-serif;color:var(--ink);background:var(--bg);line-height:1.55}
.hero{background:linear-gradient(120deg,var(--navy),#1d4d7e 55%,var(--teal));color:#fff;padding:52px 24px 44px;text-align:center}
.hero h1{margin:0;font-size:34px;letter-spacing:.3px}
.hero p{max-width:760px;margin:12px auto 0;opacity:.93;font-size:15px}
.badges{margin-top:18px}
.badge{display:inline-block;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.4);padding:4px 14px;border-radius:22px;font-size:12px;margin:4px 5px}
.cta{margin-top:26px}
.btn{display:inline-block;background:#fff;color:var(--navy);font-weight:700;padding:12px 26px;border-radius:9px;text-decoration:none;font-size:14px;margin:5px}
.btn.alt{background:transparent;color:#fff;border:1.5px solid #fff}
.wrap{max-width:1000px;margin:0 auto;padding:0 20px 70px}
h2{color:var(--navy);font-size:21px;margin:44px 0 6px}
h2 .num{color:var(--teal)}
p.lead{color:var(--muted);font-size:14px;margin:4px 0 16px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:16px 0}
.card{background:#fff;border:1px solid var(--line);border-radius:11px;padding:16px 18px}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--muted);font-weight:700}
.card .v{font-size:22px;font-weight:800;color:var(--navy);margin-top:3px}
.card .s{font-size:11.5px;color:var(--muted)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:780px){.grid2{grid-template-columns:1fr}}
.step{background:#fff;border:1px solid var(--line);border-radius:11px;padding:16px 20px;margin:12px 0}
.step b{color:var(--navy)}
.step code{background:#eef2f8;padding:2px 7px;border-radius:5px;font-size:12.5px}
code.block{display:block;background:#0f2038;color:#cfe3ff;padding:14px 18px;border-radius:9px;margin:10px 0;font-size:13px;white-space:pre-wrap}
table{width:100%;border-collapse:collapse;background:#fff;font-size:13px;border:1px solid var(--line);margin:10px 0}
th{background:#eef2f8;color:var(--navy);text-align:left;padding:8px 12px}
td{padding:7px 12px;border-top:1px solid #eef1f6}
.foot{margin-top:50px;text-align:center;color:var(--muted);font-size:12px}
.tag{display:inline-block;background:var(--teal);color:#fff;font-size:10.5px;padding:2px 9px;border-radius:4px;margin-left:5px;vertical-align:middle}
"""

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WindPCEA — Post-Construction Energy Yield Assessment for Wind Farms</title>
<style>{css}</style></head><body>

<div class="hero">
  <h1>🌬️ WindPCEA</h1>
  <p><b>Post-Construction Energy Yield Assessment (PCEYA)</b> for wind farms — a SCADA-driven
  analysis toolkit in the style of commercial (DNV-like) assessments: data QC, availability &
  loss accounting, IEC power-curve analysis, wake analysis, long-term correction (MCP),
  loss tree and Monte Carlo P-values.</p>
  <div class="badges">
    <span class="badge">IEC 61400-12-1</span><span class="badge">IEC 61400-26</span>
    <span class="badge">Measure-Correlate-Predict</span><span class="badge">Monte Carlo P50/P75/P90</span>
    <span class="badge">8 OEM profiles</span><span class="badge">Python • pandas • scipy</span>
  </div>
  <div class="cta">
    <a class="btn" href="https://github.com/kvelmurugan77/wind-pcea/releases/latest/download/WindPCEA.exe">⬇ Download WindPCEA.exe (Windows)</a>
    <a class="btn" href="sample-report.html">View full sample report →</a>
    <a class="btn alt" href="https://github.com/kvelmurugan77/wind-pcea">Source code</a>
  </div>
  <p style="opacity:.8;font-size:12.5px;margin-top:10px">Double-click the EXE — no Python needed. It starts locally and opens your browser.</p>
</div>

<div class="wrap">

<h2><span class="num">1.</span> What it does</h2>
<p class="lead">Give it 10-minute SCADA data and it produces a complete post-construction energy
yield assessment — the same workflow a DNV-style consultant would run.</p>
<div class="cards">
  <div class="card"><div class="k">Large SCADA datasets</div><div class="s">Streams multi-GB CSVs in chunks (only needed columns loaded); 1M rows analysed end-to-end in ~10 s; Parquet support; hundreds of turbines</div></div>
  <div class="card"><div class="k">Data QC</div><div class="s">10-min resampling, operating-state classification (IEC 61400-26), bad-data &amp; frozen-anemometer detection, capture rate</div></div>
  <div class="card"><div class="k">Availability</div><div class="s">Time- &amp; energy-based availability per turbine, lost energy by cause, downtime split, monthly table</div></div>
  <div class="card"><div class="k">Power curve</div><div class="s">IEC 61400-12-1 0.5 m/s binning, air-density correction, energy-weighted deviation vs warranted curve, degraded-turbine detection</div></div>
  <div class="card"><div class="k">Wake analysis</div><div class="s">Reference-turbine free-stream method, per-sector deficits, wake energy loss</div></div>
  <div class="card"><div class="k">Long-term correction</div><div class="s">Sector-wise linear-regression MCP against user file or NASA POWER (MERRA-2); LT gross AEP cross-checked by production regression (daily/monthly energy vs wind speed)</div></div>
  <div class="card"><div class="k">Loss tree &amp; P-values</div><div class="s">Gross → net AEP with full loss tree, Monte Carlo P50/P75/P90/P99, 80% CI, tornado of uncertainties</div></div>
</div>

<h2><span class="num">2.</span> Sample results (30 MW demo farm, Tamil Nadu)</h2>
<div class="cards">
  <div class="card"><div class="k">Gross AEP</div><div class="v">98,975</div><div class="s">MWh/yr (long-term)</div></div>
  <div class="card"><div class="k">Net AEP — P50</div><div class="v">91,610</div><div class="s">MWh/yr</div></div>
  <div class="card"><div class="k">P90</div><div class="v">102,537</div><div class="s">MWh/yr</div></div>
  <div class="card"><div class="k">Capacity factor</div><div class="v">34.8%</div><div class="s">2.5 MW × 12 turbines</div></div>
  <div class="card"><div class="k">Availability</div><div class="v">98.3%</div><div class="s">time-based • 98.0% production-based</div></div>
  <div class="card"><div class="k">OEM SCADA</div><div class="v">8 profiles</div><div class="s">Vestas • SGRE • Suzlon • Envision • Nordex • Goldwind • Inox</div></div>
</div>

<h2><span class="num">3.</span> Charts from the analysis</h2>
<div class="grid2">
  <div>{img_tag(os.path.join(IMGS,'wind_rose.png'),'Wind rose')}</div>
  <div>{img_tag(os.path.join(IMGS,'power_curve.png'),'Power curve')}</div>
  <div>{img_tag(os.path.join(IMGS,'loss_tree.png'),'Loss tree')}</div>
  <div>{img_tag(os.path.join(IMGS,'mc_hist.png'),'Monte Carlo')}</div>
  <div>{img_tag(os.path.join(IMGS,'wake_polar.png'),'Wake polar')}</div>
  <div>{img_tag(os.path.join(IMGS,'tornado.png'),'Tornado')}</div>
  <div>{img_tag(os.path.join(IMGS,'prod_daily.png'),'Daily energy vs wind')}</div>
  <div>{img_tag(os.path.join(IMGS,'lt_daily.png'),'LT predicted energy')}</div>
</div>

<h2><span class="num">4.</span> Run it yourself</h2>
<div class="step"><b>Web app</b> — <code>python app.py</code> → open <code>http://localhost:8000</code>,
upload your SCADA file (any OEM format) and get the report in seconds.</div>
<div class="step"><b>Command line</b> — one demo command:
<code class="block">pip install -r requirements.txt
python -m windpcea.cli --config sample_data\\config.json --scada sample_data\\scada_sample.csv --outdir results</code>
Open <code>results/pceya_report.html</code> — a complete self-contained assessment report
(report HTML, Excel workbook with 12 sheets, and cleaned SCADA CSV are generated).</div>

<h2><span class="num">5.</span> OEM SCADA compatibility</h2>
<table>
<tr><th>OEM</th><th>Conventions handled automatically</th></tr>
<tr><td><b>Vestas</b></td><td><code>Turbine Name</code>, <code>Active Power (kW)</code>, <code>Nacelle Position</code>, text state codes (Running, Fault, …)</td></tr>
<tr><td><b>Siemens Gamesa</b></td><td>Compact headers <code>ActivePower(kW)</code>, <code>WindSpeed(m/s)</code>, <code>TurbineState</code></td></tr>
<tr><td><b>Suzlon</b></td><td><code>Date Time</code>, <code>WTG No</code>, <code>Gen Active Power</code>, dd/mm/yyyy dates</td></tr>
<tr><td><b>Envision</b></td><td>Semicolon CSVs, European decimal commas, <code>dd.mm.yyyy</code> dates, <code>Device Name</code></td></tr>
<tr><td><b>Nordex</b></td><td>Separate <code>Date</code>+<code>Time</code>, <code>WEC</code>, <code>P-avg/V-avg/D-avg/T-avg</code></td></tr>
<tr><td><b>Goldwind</b></td><td>Chinese headers &amp; statuses (时间, 机组号, 有功功率, 运行, 故障…)</td></tr>
<tr><td><b>Inox</b></td><td><code>Turbine ID</code>, <code>Active Power (MW)</code> auto-scaled to kW</td></tr>
<tr><td><b>Generic</b></td><td>Long or wide format (<code>T01_power_kw</code>…), units in names, any status conventions</td></tr>
</table>

<h2><span class="num">6.</span> Tech stack</h2>
<p class="lead">Python 3.11+, pandas, numpy, scipy (Weibull fits, regressions), matplotlib
(charts), openpyxl (Excel export), Flask (web UI). Fully offline-capable except the optional
NASA POWER long-term data fetch.</p>

<div class="foot">WindPCEA — engineering tool implementing standard industry practice
(IEC 61400-12-1 / IEC 61400-26 / MCP / Monte Carlo). Not a certified assessment.
<br>© 2026 Velmurugan Karuppiah — MIT License</div>
</div></body></html>"""

    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("docs/index.html written:", os.path.getsize(os.path.join(DOCS, "index.html")) // 1024, "KB")
    print("docs/sample-report.html written:",
          os.path.getsize(os.path.join(DOCS, "sample-report.html")) // 1024, "KB")


if __name__ == "__main__":
    main()
