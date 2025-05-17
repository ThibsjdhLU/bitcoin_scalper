@echo off
set PYTHONPATH=%PYTHONPATH%;%CD%
set WATCHFILES_IGNORE_PATTERNS=logs/*
python app/main.py 