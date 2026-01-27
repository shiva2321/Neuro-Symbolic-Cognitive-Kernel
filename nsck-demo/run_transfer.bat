@echo off
cd /d "%~dp0"
echo ===================================================
echo   NSCK TRANSFER EXPERIMENT (FROZEN PONG)
echo ===================================================
echo.
echo This experiment tests "Neuro-Symbolic Transfer".
echo Phase 1: Train on Snake (Teacher Guided)
echo Phase 2: Switch to Pong (SNN Weights FROZEN)
echo.
echo EXPECTATION:
echo 1. SNN will have high entropy (Confusion) on Pong (New Task).
echo 2. VSA will detect Confusion and rescue immediately.
echo 3. The system will play Pong COMPETENTLY despite the SNN being untrained/frozen.
echo.

cd python

echo Launching Server in TRANSFER MODE...
:: --freeze-pong: Disables backprop during Pong
:: --eval: No, we want VSA to be ON.
:: We need a standard run but with freeze-pong flag.
start "NSCK BRAIN (Transfer)" cmd /k "py -3.11 -u python_server.py --freeze-pong > d:\NGCN\server_transfer.log 2>&1"

timeout /t 5

echo Launching Dashboard...
start "NSCK Dashboard" cmd /k "py -3.11 dashboard.py"

echo Launching Snake (Training Phase - 1 minute)...
start "Snake Game" cmd /k "py -3.11 simulation.py --game snake --speed 20"

timeout /t 60

echo.
echo SWITCHING TO PONG (Transfer Test)...
echo Terminating Snake...
taskkill /FI "WINDOWTITLE eq Snake Game" /F

echo Launching Pong...
start "Pong Game" cmd /k "py -3.11 simulation.py --game pong --speed 20"

echo.
echo Observe Dashboard: Pong should have Agreement ~100% (due to VSA) but SNN Loss high/Entropy high?
echo Actually, Agreement tracks (Teacher == Student).
echo Check Log for [VSA RESCUE] tags. They should be frequent.
echo.
pause
