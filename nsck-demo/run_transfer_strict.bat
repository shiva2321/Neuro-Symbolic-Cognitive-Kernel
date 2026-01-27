@echo off
cd /d "%~dp0"
echo ===================================================
echo   NSCK STRICT TRANSFER TEST (SINK OR SWIM)
echo ===================================================
echo.
echo This validates that VSA Transfer is REAL and not just Teacher Forcing.
echo.
echo Phase 1: Train on Snake (Teacher Allowed) - 30 seconds
echo Phase 2: Switch to Pong (Teacher DISABLED, SNN Weights FROZEN)
echo.
echo EXPECTATION:
echo 1. Pong SNN is confused (High Entropy)
echo 2. Teacher is OFF (Assessment of pure System 1 + System 2 capability)
echo 3. The paddle SHOULD tracked the ball using VSA Predicates.
echo.

cd python

echo Launching Server in NO-TEACHER TRANSFER MODE...
:: --freeze-pong: SNN doesn't learn Pong.
:: --no-teacher: If System fails, we DO NOT fall back to Oracle. Evaluation is honest.
start "NSCK BRAIN (Strict)" cmd /k "py -3.11 -u python_server.py --freeze-pong --no-teacher > d:\NGCN\server_strict.log 2>&1"

timeout /t 5

echo Launching Dashboard...
start "NSCK Dashboard" cmd /k "py -3.11 dashboard.py"

echo Launching Snake (Reasoning Warmup)...
start "Snake Game" cmd /k "py -3.11 simulation.py --game snake --speed 20"

timeout /t 30

echo.
echo SWITCHING TO PONG (The True Test)...
echo Terminating Snake...
taskkill /FI "WINDOWTITLE eq Snake Game" /F

echo Launching Pong (No Teacher)...
start "Pong Game" cmd /k "py -3.11 simulation.py --game pong --speed 20"

echo.
echo Monitor the Dashboard "Score" graph.
echo If the score (rally length) increases, the system is truly reasoning.
echo.
pause
