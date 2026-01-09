@echo off
REM NCGN Cognitive Cockpit - Quick Start Script for Windows

echo.
echo ================================================================================
echo   NCGN COGNITIVE COCKPIT - STARTING...
echo ================================================================================
echo.

REM Check if virtual environment is activated
python -c "import sys; sys.exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)" 2>nul
if errorlevel 1 (
    echo [WARNING] Virtual environment not activated!
    echo.
    echo Please activate your environment first:
    echo   ncgn_env\Scripts\activate
    echo.
    echo Then run this script again.
    echo.
    pause
    exit /b 1
)

REM Check if Flask is installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo [ERROR] Flask not installed!
    echo.
    echo Installing dashboard dependencies...
    pip install flask flask-socketio python-socketio eventlet werkzeug
    echo.
)

REM Check if required directories exist
if not exist "templates" (
    echo [ERROR] templates directory not found!
    echo Please ensure you're in the NCGN project directory.
    pause
    exit /b 1
)

if not exist "static" (
    echo [ERROR] static directory not found!
    pause
    exit /b 1
)

REM Display system info
echo [INFO] Checking system...
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA Available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.

echo [INFO] Starting NCGN Cognitive Cockpit...
echo.
echo ================================================================================
echo   Dashboard will be available at: http://localhost:5000
echo   Press Ctrl+C to stop
echo ================================================================================
echo.

REM Start dashboard
python ncgn_dashboard.py

pause

