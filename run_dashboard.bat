@echo off
echo ===================================================
echo NSCK Unified Dashboard Launcher
echo ===================================================
echo.
echo Using Python at: C:\Users\Asta\AppData\Local\Programs\Python\Python311\python.exe
echo.

C:\Users\Asta\AppData\Local\Programs\Python\Python311\python.exe launch_dashboard.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Dashboard crashed or failed to start.
    pause
)
