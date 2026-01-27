@echo off
echo STARTING VERIFIED TRANSFER EXPERIMENT...
cd python

:: 1. Start Verification Listener (Logs Metrics)
start "Verifier" cmd /c "python verify_transfer_stats.py > verification_results.txt 2>&1"

:: 2. Start Server in STRICT MODE (No Teacher, Frozen Pong)
start "NSCK Server" python python_server.py --no-teacher --freeze-pong

echo Waiting for Server...
timeout /t 3

:: 3. Start Snake (Reasoning Check)
start "Snake Game" python snake_ui.py
echo Snake Running for 10s...
timeout /t 10
taskkill /F /FI "WINDOWTITLE eq Snake Game*"
taskkill /F /IM python.exe /FI "WINDOWTITLE eq NSCK Snake*" 

:: 4. Start Pong (Transfer Check)
echo STARTING PONG (TRANSFER TEST)...
start "Pong Game" python pong_ui.py
echo Pong Running for 30s...
timeout /t 30

:: 5. Cleanup
echo CLEANING UP...
taskkill /F /IM python.exe
echo Experiment Complete. Check Verifier Window for Results.
pause
