@echo off
setlocal
echo ===================================================
echo   NSCK Training Mode: PONG ONLY
echo ===================================================
cd /d "%~dp0python"

echo [1/2] Launching Brain...
start "NSCK BRAIN" cmd /k "py -3.11 python_server.py"

timeout /t 3

echo [2/2] Launching Pong...
start "NSCK PONG" cmd /k "py -3.11 pong_ui.py"

echo.
echo Training Pong...
endlocal
