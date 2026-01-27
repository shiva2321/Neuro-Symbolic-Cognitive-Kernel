@echo off
setlocal
echo Starting Validation Experiment (Training Phase)...
cd /d "%~dp0python"

echo Launching Server (Logging to d:\NGCN\server_training.log)...
start "NSCK BRAIN" cmd /k "py -3.11 -u python_server.py > d:\NGCN\server_training.log 2>&1"

timeout /t 5

echo Launching Snake...
start "NSCK SNAKE" cmd /k "py -3.11 snake_ui.py"

echo Launching Pong...
start "NSCK PONG" cmd /k "py -3.11 pong_ui.py"

echo Launching Dashboard...
start "NSCK DASHBOARD" cmd /k "py -3.11 dashboard.py"

echo Experiment Started.
endlocal
