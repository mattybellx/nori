@echo off
rem Double-click to launch the nori chat UI in your browser.
cd /d "%~dp0"
python -m dse.chat %*
pause
