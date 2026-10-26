@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" calibrar_plazas.py
) else (
    python calibrar_plazas.py
)
pause
