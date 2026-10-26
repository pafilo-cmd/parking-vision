@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app_parking.py
) else (
    python app_parking.py
)
pause
