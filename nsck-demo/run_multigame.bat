@echo off
setlocal
echo ===================================================
echo   NSCK Multi-Game System (Snake + Pong)
echo ===================================================
echo.

:: 0. Dependency Check
echo [0/3] Checking Dependencies (OpenCV)...
py -3.11 -m pip install opencv-python --user >nul

cd /d "%~dp0python"

echo [1/3] Launching Brain Server...
start "NSCK BRAIN" cmd /k "py -3.11 python_server.py"

echo Waiting for brain to initialize...
timeout /t 5

echo [2/3] Launching Snake Client...
start "NSCK SNAKE" cmd /k "py -3.11 snake_ui.py"

echo [3/3] Launching Pong Client...
start "NSCK PONG" cmd /k "py -3.11 pong_ui.py"

echo.
echo System Active! Watch the "NSCK BRAIN" window for logs.
pause
endlocal
