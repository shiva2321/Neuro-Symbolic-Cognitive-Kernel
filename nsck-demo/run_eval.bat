@echo off
setlocal
echo Starting Validation Experiment (EVALUATION Phase)...
cd /d "%~dp0python"

echo Launching Server in EVAL MODE (Logging to d:\NGCN\server_eval.log)...
start "NSCK BRAIN" cmd /k "py -3.11 -u python_server.py --eval > d:\NGCN\server_eval.log 2>&1"

timeout /t 5

echo Launching Snake...
start "NSCK SNAKE" cmd /k "py -3.11 snake_ui.py"

echo Launching Pong...
start "NSCK PONG" cmd /k "py -3.11 pong_ui.py"

echo Launching Dashboard...
start "NSCK DASHBOARD" cmd /k "py -3.11 dashboard.py"

echo Evaluation Experiment Started.
endlocal
