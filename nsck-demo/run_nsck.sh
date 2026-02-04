#!/bin/bash

# NSCK System Launcher
# Handles cleanup and startup of the Brain and Dashboard

# 1. Kill existing processes
echo "[LAUNCHER] Cleaning up old processes..."
pkill -f "python_server.py"
pkill -f "web_dashboard.py"
pkill -f "maze_ui.py"
pkill -f "snake_ui.py"
pkill -f "pong_ui.py"
sleep 2

# Trap CTRL+C and Exit to kill background processes
trap "kill 0" EXIT

# 2. Start Brain Server (Background)
echo "[LAUNCHER] Starting Brain (python_server.py)..."
PYTHONPATH=. /home/shivam/Downloads/Node_network/venv/bin/python /home/shivam/Downloads/Node_network/nsck-demo/python/python_server.py > brain.log 2>&1 &
BRAIN_PID=$!
echo "[LAUNCHER] Brain PID: $BRAIN_PID"

# Wait for ZMQ ports to bind
sleep 3

# 3. Start Dashboard (Foreground)
echo "[LAUNCHER] Starting Dashboard (web_dashboard.py)..."
PYTHONPATH=. /home/shivam/Downloads/Node_network/venv/bin/python /home/shivam/Downloads/Node_network/nsck-demo/python/web_dashboard.py

# Wait for dashboard to finish (if it crashes, we exit and kill brain)
wait $BRAIN_PID
