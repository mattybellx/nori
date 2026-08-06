@echo off
rem Double-click to launch the Nori chat UI in your browser.
cd /d "%~dp0deepseek_engine"
python -m dse.chat %*
pause
