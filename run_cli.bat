@echo off
REM WindPCEA - demo run via command line (Windows)
cd /d %~dp0
pip install -r requirements.txt
python -m windpcea.cli --config sample_data\config.json --scada sample_data\scada_sample.csv --outdir results
echo.
echo Report: results\pceya_report.html
pause
