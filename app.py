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

from windpcea import __version__ as WIND_PCEA_VERSION
from windpcea import config as cfg_mod
from windpcea import scada as scada_mod
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
<div id="updateBanner" style="display:none;background:#fff8ef;border:1px solid #f3ddb8;color:#6b4a12;padding:10px 18px;font-size:13px;text-align:center"></div>
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
<label>SCADA data files (CSV / XLSX) *</label>
<input type="file" id="scada" accept=".csv,.xlsx,.xls" multiple>
<div class="hint">Select one or several files (per-year exports, per-turbine groups) - they are
concatenated automatically with de-duplication and per-file traceability.
10-min (or finer) records. Long format: timestamp, turbine_id, power_kw, wind_speed_mps,
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
<button class="btn ghost" id="selftest" onclick="selfTest()" style="margin-left:10px">Self-test app</button>
<button class="btn ghost" id="previewbtn" onclick="previewFile()" style="margin-left:10px">Preview my file</button>
<div id="stResult"></div>
<div id="pvResult" style="margin-top:12px"></div>
<script>
async function previewFile(){
  const el = document.getElementById('pvResult');
  const f = document.getElementById('scada').files[0];
  if(!f){ el.innerHTML = '<span class="hint">Select a SCADA file first.</span>'; return; }
  el.innerHTML = '<span class="spin"></span>Reading file header…';
  const fd = new FormData(); fd.append('scada', f);
  try{
    const r = await fetch('/preview', {method:'POST', body: fd});
    const j = await r.json();
    if(!r.ok){ el.innerHTML = '⚠ ' + j.error; return; }
    let h = '<div class="note ok"><b>v' + j.version + ' — what the tool sees in your file</b><br>' +
      'Detected profile: <b>' + j.profile + '</b><br><b>Columns:</b> ' + j.columns.join(' | ') +
      '<br><b>First rows:</b><br><pre style="font-size:11px;overflow-x:auto;background:#f7f9fc;padding:8px;border-radius:6px;margin:6px 0">' +
      j.rows.slice(0,3).map(rr => rr.join(' | ')).join('<br>') + '</pre>';
    h += '<b>Suggested mapping:</b> ';
    for(const [k,v] of Object.entries(j.suggested)){ h += '<b>' + k + '</b>→' + v + '  '; }
    h += '<br><span style="font-size:12px">If power is missing or wrong, fill the ⚙ Advanced fields below and retry.</span></div>';
    // auto-fill advanced mapping from suggestions
    const map = {'power_col':'power','ws_col':'ws','turbine_col':'turbine','ts_col':'timestamp','dir_col':'dir','temp_col':'temp','status_col':'status'};
    for(const [field, key] of Object.entries(map)){
      if(j.suggested[key] && !document.getElementById(field).value){
        document.getElementById(field).value = j.suggested[key];
      }
    }
    el.innerHTML = h;
  }catch(e){ el.innerHTML = '⚠ ' + e; }
}
</script>
<script>
async function selfTest(){
  const el = document.getElementById('stResult');
  el.innerHTML = '<span class="spin"></span>Running self-test…';
  try{
    const r = await fetch('/selftest');
    const j = await r.json();
    if(r.ok){ el.innerHTML = '✅ v' + j.version + ' self-test OK — P50 ' + j.p50.toLocaleString() + ' MWh/yr (' + j.rows.toLocaleString() + ' rows in ' + j.seconds + 's)'; }
    else { el.innerHTML = '⚠ v' + j.version + ' self-test FAILED: ' + j.error; }
  }catch(e){ el.innerHTML = '⚠ self-test error: ' + e; }
}
</script>
</div>
<div id="status"></div>
</div>

<div class="foot">WindPCEA v__VER__ — engineering aid implementing standard industry practice; not a certified assessment.</div>
</div>
<script>
async function checkUpdate(){
  try{
    const r = await fetch('/api/version');
    const j = await r.json();
    const b = document.getElementById('updateBanner');
    if(j.latest && j.latest !== j.current){
      b.style.display = 'block';
      b.innerHTML = '⚠ You are running <b>v' + j.current + '</b> but <b>v' + j.latest +
        '</b> is available — <a href="' + j.update_url + '" target="_blank" style="color:#6b4a12;font-weight:700">download the update</a> ' +
        'to get the latest fixes (footer shows your version).';
    }
  }catch(e){}
}
checkUpdate();
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
    for(const [k,id] of [['cfgfile','cfgfile'],['curve','curve'],['ltfile','ltfile']]){
      const f = document.getElementById(id).files[0];
      if(f) fd.append(k, f);
    }
    for(const f of document.getElementById('scada').files){
      fd.append('scada', f);
    }
    for(const id of ['power_col','ws_col','turbine_col','ts_col','dir_col','temp_col','status_col','curt_col']){
      const v = document.getElementById(id).value.trim();
      if(v) fd.append(id, v);
    }
  }
  try{
    const r = await fetch('/analyze', {method:'POST', body: fd});
    const j = await r.json();
    if(!r.ok){
      let tb = '';
      if(j.trace && j.trace.length){ tb = '<br><pre style="font-size:11px;background:#f7f9fc;border:1px solid #dde3ec;padding:8px;border-radius:6px;overflow-x:auto;color:#22303f">' + j.trace.join('<br>') + '</pre>'; }
      let ct = '';
      if(j.config_types){ ct = '<br><span style="font-size:11px;color:#6b7787">config: ' + JSON.stringify(j.config_types) + '</span>'; }
      st.innerHTML = '⚠ v' + (j.version||'?') + ': ' + (j.error || 'Analysis failed') + ct + tb +
        '<br><span style="font-size:12px">Tip: if column detection failed, open ⚙ Advanced ' +
        'above, type the exact column names from your file, and retry.</span>';
      return;
    }
    st.innerHTML = '<span class="spin"></span>Done — opening report…';
    window.location.href = j.report_url;
  }catch(e){ st.innerHTML = '⚠ Network error: ' + e; }
}
</script></body></html>"""


@app.route("/")
def index():
    return INDEX_HTML.replace("__CSS__", CSS).replace("__VER__", WIND_PCEA_VERSION)


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

    # NOTE: scada_paths must be defined in EVERY branch — it is referenced
    # below for multi-file mode, and Python treats it as a local because it
    # is assigned in the upload branch. If the demo branch skipped it, the
    # "Run demo" button crashed with UnboundLocalError (v1.5.3 bug).
    scada_paths = []
    cfg = {}
    try:
        if is_sample:
            scada_path = os.path.join(SAMPLE, "scada_sample.csv")
            scada_paths = [scada_path]
            cfg = cfg_mod.load_config(os.path.join(SAMPLE, "config.json"))
            outdir = run_dir
        else:
            scada_files_up = request.files.getlist("scada")
            if not scada_files_up:
                return jsonify({"error": "SCADA file is required"}), 400
            scada_paths = []
            for idx, f in enumerate(scada_files_up):
                p = _save_upload(f, run_dir, f"scada_{idx}_{f.filename or 'in'}")
                if p:
                    scada_paths.append(p)
            if not scada_paths:
                return jsonify({"error": "SCADA file could not be read — "
                                         "the upload was empty or failed."}), 400
            scada_path = scada_paths[0]
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
            # merge the form overrides ON TOP of the uploaded config file
            # (previously load_config(None, ...) silently discarded the file)
            cfg = cfg_mod.load_config(cfg_path, overrides)
            outdir = run_dir

        # very large files -> blockwise out-of-core mode (bounded memory)
        large = cfg.get("large_file_mode", "auto")
        fsize_mb = (os.path.getsize(scada_path) / 1e6) if os.path.exists(scada_path) else 0
        use_blockwise = large is True or (
            large == "auto" and fsize_mb >= 900)
        if use_blockwise:
            results = run_blockwise(cfg, scada_path, outdir=outdir,
                                    scada_files=scada_paths if len(scada_paths) > 1 else None)
        else:
            results = run_analysis(cfg, scada_path, outdir=outdir,
                                   scada_files=scada_paths if len(scada_paths) > 1 else None)
        report_path = build_html(results, outdir)
        # inject a download toolbar (HTML report + Excel) into the report page
        try:
            with open(report_path, encoding="utf-8") as f:
                html = f.read()
            toolbar = (
                '<div style="position:fixed;top:0;left:0;right:0;z-index:999;'
                'background:#14365D;color:#fff;padding:8px 16px;font-size:13px;'
                'display:flex;gap:12px;align-items:center;box-shadow:0 2px 8px rgba(0,0,0,.25)">'
                '<b>WindPCEA report</b> '
                f'<a href="/download/{run_id}/pceya_report.html" style="color:#fff">Download report (HTML)</a> '
                f'<a href="/download/{run_id}/pceya_results.xlsx" style="color:#fff">Download Excel</a> '
                f'<a href="/download/{run_id}/flagged_scada_10min.csv" style="color:#fff">Download flagged data (CSV)</a>'
                '</div>')
            if "<body>" in html:
                html = html.replace("<body>", "<body>" + toolbar, 1)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html)
        except Exception:
            pass
        export_excel(results, outdir)
        export_csvs(results, outdir)
        with open(os.path.join(run_dir, "config_used.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        return jsonify({"run_id": run_id,
                        "report_url": f"/report/{run_id}",
                        "summary": _quick_summary(results)})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        last = tb.strip().splitlines()[-15:]
        # report the actual types of every config field used in arithmetic
        types = {}
        for _k in ("rated_power_kw", "hub_height_m", "cut_in_mps", "cut_out_mps",
                   "air_pressure_kpa", "electrical_loss_pct", "other_loss_pct",
                   "preconstruction_p50_gwh", "mc_iterations", "mc_seed",
                   "bin_width_mps", "min_bin_count", "latitude", "longitude"):
            _v = cfg.get(_k)
            types[_k] = f"{type(_v).__name__}:{_v!r}" if _v is not None else "unset"
        return jsonify({"error": str(e), "trace": last,
                        "config_types": types,
                        "version": WIND_PCEA_VERSION}), 500


def _quick_summary(r):
    p = r["uncertainty"]["p"]
    return {"farm": r["meta"]["farm_name"], "gross_mwh": round(r["losses"]["gross_lt_mwh"]),
            "p50": round(p["P50"]), "p75": round(p["P75"]), "p90": round(p["P90"]),
            "availability": round(r["availability"]["farm"]["time_avail_pct"], 2)}


@app.route("/api/version")
def api_version():
    """Report the running version and the latest release on GitHub, so the
    UI can warn the user when they are on an outdated build."""
    import requests as _rq
    latest = None
    try:
        j = _rq.get("https://api.github.com/repos/kvelmurugan77/wind-pcea/releases/latest",
                    timeout=10).json()
        latest = j.get("tag_name", "").lstrip("v")
    except Exception:
        latest = None
    return jsonify({"current": WIND_PCEA_VERSION, "latest": latest,
                    "update_url": "https://github.com/kvelmurugan77/wind-pcea/releases/latest"})


@app.route("/preview", methods=["POST"])
def preview():
    """Show the uploaded file's columns + first rows, and what the tool
    would map them to. Removes all guessing about column names."""
    import pandas as pd
    files = request.files.getlist("scada")
    f = files[0] if files else None
    if not f:
        return jsonify({"error": "no file"}), 400
    path = os.path.join(RUNS, "_preview_tmp.csv")
    os.makedirs(RUNS, exist_ok=True)
    f.save(path)
    try:
        cols, kw = scada_mod._sniff_csv(path)
        raw = pd.read_csv(path, nrows=6, **kw)
        rows = raw.fillna("").astype(str).values.tolist()
        # suggested mapping using the same logic the analysis uses
        from windpcea import oem as oem_mod
        profile = oem_mod.detect_profile(cols)
        aliases = oem_mod.profile_aliases(profile)
        suggest = {}
        for key, cands in [("power", aliases["power"] + ["grd_prod", "prod", "active power"]),
                           ("ws", aliases["ws"] + ["amb_wind", "wind"]),
                           ("turbine", aliases["turbine"] + ["assetnam"]),
                           ("timestamp", aliases["timestamp"] + ["pctimest"]),
                           ("dir", aliases["dir"] + ["nac_direc"]),
                           ("temp", aliases["temp"] + ["amb_tems"]),
                           ("status", aliases["status"] + ["sys_stats"])]:
            for c in cols:
                nc = oem_mod.normalize_col_name(c)
                if any(oem_mod.normalize_col_name(a) in nc for a in cands):
                    suggest[key] = c
                    break
        return jsonify({"columns": cols, "rows": rows,
                        "profile": profile, "suggested": suggest,
                        "version": WIND_PCEA_VERSION})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/selftest")
def selftest():
    """Run the bundled sample end-to-end; proves the app itself works."""
    import time as _t
    try:
        cfg = cfg_mod.load_config(os.path.join(SAMPLE, "config.json"))
        _t0 = _t.time()
        r = run_analysis(cfg, os.path.join(SAMPLE, "scada_sample.csv"))
        return jsonify({"ok": True, "version": WIND_PCEA_VERSION,
                        "p50": round(r["uncertainty"]["p"]["P50"]),
                        "seconds": round(_t.time() - _t0, 1),
                        "rows": r["qc"]["rows"]})
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "version": WIND_PCEA_VERSION,
                        "error": str(e),
                        "trace": traceback.format_exc().strip().splitlines()[-15:]}), 500


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
