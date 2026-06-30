@echo off
setlocal
py -3.10 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pytest -q
echo.
echo Setup selesai. Edit config.yaml lalu jalankan: python run_project.py validate
endlocal
