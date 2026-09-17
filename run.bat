@echo off
title Signal Analyzer Pro
echo ============================================================
echo   Signal Analyzer Pro - Production Version
echo ============================================================
echo.
echo   Starting application...
echo   All features ready (ML models pre-trained)
echo.
:: Ensure working directory is the folder where this script is located
if "%~dp0" neq "" cd /d "%~dp0"
if not exist "main.py" (
    if exist "d:\quick share\Tarang\main.py" cd /d "d:\quick share\Tarang"
)

:: Prevent Python from creating .pyc cache files
set PYTHONDONTWRITEBYTECODE=1

py -3.11 -B main.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start application
    echo.
    echo Troubleshooting:
    echo   1. Check if Python 3.11 is installed: py -3.11 --version
    echo   2. Check if PyQt6 is installed: py -3.11 -m pip list ^| findstr PyQt6
    echo   3. Install dependencies: py -3.11 -m pip install -r requirements.txt
    echo.
    pause
) else (
    echo Application closed normally.
)
pause
