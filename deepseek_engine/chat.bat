@echo off
rem Double-click to launch the Nori chat UI in your browser.
cd /d "%~dp0"
python -m dse.chat %*
pause
