@echo off
echo ============================================================
echo   SwarmLLM Setup and Run
echo ============================================================
echo.

cd /d "%~dp0"

:: Create venv if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo.
)

:: Activate venv
call .venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet
echo.

:: Run interactive setup + launch
python scripts\setup_run.py

echo.
pause
