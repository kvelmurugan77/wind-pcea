"""WindPCEA web application — upload SCADA, run a post-construction energy
yield assessment, view the report and download outputs.

Run:  python app.py   (serves on http://0.0.0.0:8000)
or as a standalone Windows EXE (double-click; browser opens automatically).
"""
import json
import logging
import os
import re
import shutil
import socket
import sys
import threading
import uuid
import webbrowser

from flask import Flask, jsonify, request, send_from_directory

from windpcea import config as cfg_mod
from windpcea.analysis import run_analysis
from windpcea.blockwise import run_blockwise
from windpcea.report import build_html, export_csvs, export_excel

FROZEN = bool(getattr(sys, "frozen", False))
if FROZEN:
    # standalone EXE: bundled files live in _MEIPASS, user data next to the
    # EXE / in %LOCALAPPDATA%
    BASE = os.path.dirname(sys.executable)
    BUNDLED = getattr(sys, "_MEIPASS", BASE)
    RUNS = os.path.join(os.environ.get("LOCALAPPDATA", BASE), "WindPCEA", "runs")
    logging.basicConfig(filename=os.path.join(BASE, "windpcea.log"),
                        level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
else:
    BASE = os.path.dirname(os.path.abspath(__file__))
    BUNDLED = BASE
    RUNS = os.path.join(BASE, "runs")
SAMPLE = os.path.join(BUNDLED, "sample_data")
os.makedirs(RUNS, exist_ok=True)


def _find_port(start=8000):
    for p in range(start, start + 10):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return start

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB uploads

CSS = """
:root{--navy:#14365D;--teal:#0E7C86;--orange:#E8871E;--ink:#22303f;--muted:#6b7787;--line:#dde3ec}
*{box-sizing:border-box}
body{margin:0;font-family:'Segoe UI',system-ui,Arial,sans-serif;color:var(--ink);background:#f4f6fa;line-height:1.5}
.hero{background:linear-gradient(120deg,var(--navy),#1d4d7e 55%,var(--teal));color:#fff;padding:26px 30px}
.hero h1{margin:0;font-size:22px}
.hero p{margin:6px 0 0;opacity:.9;font-size:13px}
.wrap{max-width:980px;margin:22px auto;padding:0 18px 60px}
.panel{background:#fff;border:1px solid var(--line);border-radius:12px;padding:20px 24px;margin-bottom:16px}
h2{color:var(--navy);font-size:15px;margin:0 0 12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:760px){.grid2{grid-template-columns:1fr}}
label{display:block;font-size:12px;color:var(--muted);margin:10px 0 4px;font-weight:600}
input[type=text],input[type=number]{width:100%;padding:8px 10px;border:1px solid #c9d2e0;border-radius:7px;font-size:13px}
input[type=file]{width:100%;padding:8px;border:1px dashed #c9d2e0;border-radius:7px;font-size:12px;background:#fafbfd}
.btn{display:inline-block;background:var(--teal);color:#fff;border:none;padding:11px 22px;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer}
.btn.ghost{background:#eef2f8;color:var(--navy);border:1px solid var(--line)}
.btn:hover{opacity:.9}
#status{margin:14px 0;font-size:13px;color:var(--muted);display:none}
#status .spin{display:inline-block;width:14px;height:14px;border:2px solid #c9d2e0;border-top-color:var(--teal);border-radius:50%;animation:sp 0.8s linear infinite;vertical-align:-2px;margin-right:8px}
@keyframes sp{to{transform:rotate(360deg)}}
.hint{font-size:11.5px;color:var(--muted);margin-top:4px}
.badge{display:inline-block;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.35);padding:2px 10px;border-radius:20px;font-size:11px;margin-right:6px}
.foot{margin-top:26px;font-size:11.5px;color:var(--muted);text-align:center}
"""

INDEX_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>WindPCEA — Post-Construction Energy Yield Assessment</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>__CSS__</style></head><body>
<div class="hero">
<h1>WindPCEA — Post-Construction Energy Yield Assessment</h1>
<p>SCADA-based assessment in the style of commercial (DNV-like) analyses: data QC, availability &amp; loss accounting,
IEC power curve analysis, wake analysis, long-term correction (MCP), loss tree and Monte Carlo P50/P75/P90.</p>
<span class="badge">IEC 61400-12-1</span><span class="badge">IEC 61400-26</span><span class="badge">MCP</span><span class="badge">Monte Carlo P-values</span>
</div>
<div class="wrap">

<div class="panel">
<h2>Run the assessment</h2>
<div class="grid2">
<div>
<label>SCADA data file (CSV / XLSX) *</label>
<input type="file" id="scada" accept=".csv,.xlsx,.xls">
<div class="hint">10-min (or finer) records. Long format: timestamp, turbine_id, power_kw, wind_speed_mps,
nacelle_dir_deg, status_code, temp_c … or wide format: T01_power_kw, T01_wind_speed_mps …</div>
<label>Config JSON (optional)</label>
<input type="file" id="cfgfile" accept=".json">
<div class="hint">Farm parameters, warranted curve path, status-code mapping, uncertainty overrides …</div>
</div>
<div>
<label>Warranted power curve (optional)</label>
<input type="file" id="curve" accept=".csv">
<div class="hint">Columns: wind_speed_mps, power_kw. If omitted a generic curve is generated from the config.</div>
<label>Long-term daily wind file (optional)</label>
<input type="file" id="ltfile" accept=".csv">
<div class="hint">Columns: date, ws_mps [, dir_deg]. If omitted, NASA POWER (MERRA-2) reanalysis is fetched
automatically when latitude/longitude are configured.</div>
</div>
</div>
<div class="grid2">
<div>
<label>Farm name</label><input type="text" id="farmname" value="My Wind Farm">
<label>Rated power per turbine (kW)</label><input type="number" id="rated" value="2500">
<label>Hub height (m)</label><input type="number" id="hub" value="100">
</div>
<div>
<label>Latitude * (required)</label><input type="text" id="lat" placeholder="e.g. 9.00">
<div class="hint">Site coordinates are required — the tool downloads 25 years of
ERA5T reanalysis wind for the long-term correction.</div>
<label>Longitude * (required)</label><input type="text" id="lon" placeholder="e.g. 77.80">
<label>Electrical losses (%)</label><input type="number" id="elec" value="2.0">
</div>
</div>
<details style="margin-top:18px">
<summary style="cursor:pointer;font-size:12.5px;color:var(--teal);font-weight:600">⚙ Advanced: manual column mapping (only if auto-detection fails)</summary>
<div class="grid2" style="margin-top:8px">
<div>
<label>Power column</label><input type="text" id="power_col" placeholder="e.g. Active Power (kW)">
<label>Wind speed column</label><input type="text" id="ws_col" placeholder="e.g. Wind Speed (m/s)">
<label>Turbine column</label><input type="text" id="turbine_col" placeholder="e.g. Turbine Name">
<label>Timestamp column</label><input type="text" id="ts_col" placeholder="e.g. Date/Time">
</div>
<div>
<label>Direction column (optional)</label><input type="text" id="dir_col" placeholder="e.g. Nacelle Position">
<label>Temperature column (optional)</label><input type="text" id="temp_col" placeholder="e.g. Ambient Temp">
<label>Status column (optional)</label><input type="text" id="status_col" placeholder="e.g. State / Status Code">
<label>Curtailment flag (optional)</label><input type="text" id="curt_col" placeholder="e.g. Curtailment Flag">
</div>
</div>
<div class="hint">Type the EXACT column names from your file. Only fill the ones that failed
auto-detection — the rest are detected automatically.</div>
</details>
<div style="margin-top:16px">
<button class="btn" id="run" onclick="runAnalyze(false)">Run assessment</button>
<button class="btn ghost" id="runsample" onclick="runAnalyze(true)" style="margin-left:10px">Run with bundled sample data</button>
</div>
<div id="status"></div>
</div>

<div class="foot">WindPCEA v1.0 — engineering aid implementing standard industry practice; not a certified assessment.</div>
</div>
<script>
async function runAnalyze(sample){
  const st = document.getElementById('status');
  st.style.display = 'block';
  st.innerHTML = '<span class="spin"></span>Running analysis (QC, availability, power curve, wake, MCP, Monte Carlo)… this takes a few seconds.';
  const fd = new FormData();
  fd.append('sample', sample ? '1' : '0');
  if(!sample){
    fd.append('farmname', document.getElementById('farmname').value);
    fd.append('rated', document.getElementById('rated').value);
    fd.append('hub', document.getElementById('hub').value);
    fd.append('lat', document.getElementById('lat').value);
    fd.append('lon', document.getElementById('lon').value);
    fd.append('elec', document.getElementById('elec').value);
    for(const [k,id] of [['scada','scada'],['cfgfile','cfgfile'],['curve','curve'],['ltfile','ltfile']]){
      const f = document.getElementById(id).files[0];
      if(f) fd.append(k, f);
    }
    for(const id of ['power_col','ws_col','turbine_col','ts_col','dir_col','temp_col','status_col','curt_col']){
      const v = document.getElementById(id).value.trim();
      if(v) fd.append(id, v);
    }
  }
  try{
    const r = await fetch('/analyze', {method:'POST', body: fd});
    const j = await r.json();
    if(!r.ok){ st.innerHTML = '⚠ ' + (j.error || 'Analysis failed') +
      '<br><span style="font-size:12px">Tip: if column detection failed, open ⚙ Advanced ' +
      'above, type the exact column names from your file, and retry.</span>'; return; }
    st.innerHTML = '<span class="spin"></span>Done — opening report…';
    window.location.href = j.report_url;
  }catch(e){ st.innerHTML = '⚠ Network error: ' + e; }
}
</script></body></html>"""


@app.route("/")
def index():
    return INDEX_HTML.replace("__CSS__", CSS)


def _save_upload(f, run_dir, name):
    if f and f.filename:
        f.save(os.path.join(run_dir, name))
        return os.path.join(run_dir, name)
    return None


@app.route("/analyze", methods=["POST"])
def analyze():
    is_sample = request.form.get("sample") == "1"
    run_id = uuid.uuid4().hex[:10]
    run_dir = os.path.join(RUNS, run_id)
    os.makedirs(run_dir, exist_ok=True)

    try:
        if is_sample:
            scada_path = os.path.join(SAMPLE, "scada_sample.csv")
            cfg = cfg_mod.load_config(os.path.join(SAMPLE, "config.json"))
            outdir = run_dir
        else:
            scada_path = _save_upload(request.files.get("scada"), run_dir, "scada.csv")
            if not scada_path:
                return jsonify({"error": "SCADA file is required"}), 400
            cfg_path = _save_upload(request.files.get("cfgfile"), run_dir, "config.json")
            cfg = cfg_mod.load_config(cfg_path) if cfg_path else cfg_mod.load_config(None)
            curve = _save_upload(request.files.get("curve"), run_dir, "warranted_power_curve.csv")
            lt = _save_upload(request.files.get("ltfile"), run_dir, "long_term_daily_ws.csv")
            overrides = {
                "farm_name": request.form.get("farmname") or cfg.get("farm_name"),
            }
            rated = request.form.get("rated")
            hub = request.form.get("hub")
            lat = request.form.get("lat")
            lon = request.form.get("lon")
            elec = request.form.get("elec")
            if rated:
                overrides["rated_power_kw"] = float(rated)
            if hub:
                overrides["hub_height_m"] = float(hub)
            # coordinates are REQUIRED unless a long-term file is supplied
            if lat:
                overrides["latitude"] = float(lat)
            if lon:
                overrides["longitude"] = float(lon)
            if not lat or not lon:
                if not lt and not request.files.get("cfgfile"):
                    return jsonify({"error":
                        "Latitude and longitude are required — the tool needs "
                        "the site coordinates to download the ERA5T long-term "
                        "wind reference for the energy-yield assessment."}), 400
            if elec:
                overrides["electrical_loss_pct"] = float(elec)
            if curve:
                overrides["warranted_power_curve"] = curve
            if lt:
                overrides["long_term_wind_file"] = lt
                overrides["long_term_source"] = "file"
            # advanced: manual column mapping (only used when the user filled
            # in at least one field)
            colmap = {}
            for key, field in [("power", "power_col"), ("ws", "ws_col"),
                               ("turbine", "turbine_col"), ("timestamp", "ts_col"),
                               ("dir", "dir_col"), ("temp", "temp_col"),
                               ("status", "status_col"), ("curtailment", "curt_col")]:
                v = request.form.get(field, "").strip()
                if v:
                    colmap[key] = v
            if colmap:
                overrides["column_map"] = colmap
            cfg = cfg_mod.load_config(None, overrides)
            outdir = run_dir

        # very large files -> blockwise out-of-core mode (bounded memory)
        large = cfg.get("large_file_mode", "auto")
        fsize_mb = (os.path.getsize(scada_path) / 1e6) if os.path.exists(scada_path) else 0
        use_blockwise = large is True or (
            large == "auto" and fsize_mb >= 900)
        if use_blockwise:
            results = run_blockwise(cfg, scada_path, outdir=outdir)
        else:
            results = run_analysis(cfg, scada_path, outdir=outdir)
        build_html(results, outdir)
        export_excel(results, outdir)
        export_csvs(results, outdir)
        with open(os.path.join(run_dir, "config_used.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        return jsonify({"run_id": run_id,
                        "report_url": f"/report/{run_id}",
                        "summary": _quick_summary(results)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _quick_summary(r):
    p = r["uncertainty"]["p"]
    return {"farm": r["meta"]["farm_name"], "gross_mwh": round(r["losses"]["gross_lt_mwh"]),
            "p50": round(p["P50"]), "p75": round(p["P75"]), "p90": round(p["P90"]),
            "availability": round(r["availability"]["farm"]["time_avail_pct"], 2)}


@app.route("/report/<run_id>")
def report(run_id):
    path = os.path.join(RUNS, run_id, "pceya_report.html")
    if not os.path.exists(path):
        return "Report not found", 404
    with open(path, encoding="utf-8") as f:
        return f.read()


@app.route("/download/<run_id>/<filename>")
def download(run_id, filename):
    if not re.match(r"^[\w.\-]+$", filename):
        return "invalid", 400
    return send_from_directory(os.path.join(RUNS, run_id), filename, as_attachment=True)


if __name__ == "__main__":
    if FROZEN:
        port = _find_port()
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
        logging.info(f"WindPCEA started on http://127.0.0.1:{port}")
        print(f"WindPCEA running at http://127.0.0.1:{port} (log: windpcea.log)")
        app.run(host="127.0.0.1", port=port, threaded=True)
    else:
        print("WindPCEA web app on http://0.0.0.0:8000")
        app.run(host="0.0.0.0", port=8000, threaded=True)
