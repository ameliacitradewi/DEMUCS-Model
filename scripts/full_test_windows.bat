@echo off
call .venv\Scripts\activate.bat
python run_project.py separate-dataset --split test
python run_project.py evaluate-dataset --split test
python run_project.py report --split test
