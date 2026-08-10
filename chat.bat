@echo off
rem Double-click to launch the nori chat UI in your browser.
cd /d "%~dp0deepseek_engine"
python -m dse.chat %*
pause
