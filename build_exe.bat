@echo off
REM Build WindPCEA as a standalone Windows EXE (requires Python once).
REM Usage: double-click, or run from a terminal.
cd /d %~dp0
echo Installing dependencies (first run only)...
pip install -r requirements.txt pyinstaller
echo Building WindPCEA.exe ...
pyinstaller --noconfirm --clean --onefile --name WindPCEA ^
  --add-data "sample_data;sample_data" ^
  --hidden-import matplotlib.backends.backend_agg ^
  app.py
echo.
echo Done! Your EXE is at: dist\WindPCEA.exe
echo Copy WindPCEA.exe anywhere - double-click to run (no Python needed).
pause
