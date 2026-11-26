@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Primero instala el entorno siguiendo README.md.
  pause
  exit /b 1
)
start "" "http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app/app.py --server.address 127.0.0.1
