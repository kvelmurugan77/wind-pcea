@echo off
REM WindPCEA - start the web application (Windows)
cd /d %~dp0
pip install -r requirements.txt
python app.py
pause
