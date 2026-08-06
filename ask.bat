@echo off
rem Double-click to open the Nori terminal Q&A (ask + audit + stats).
cd /d "%~dp0deepseek_engine"
python -m dse.ask %*
pause
