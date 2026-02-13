import zmq
import json
import time
import subprocess
import sys
import os
from pathlib import Path
import pytest


@pytest.mark.skip(reason="Integration test requires launching python_server + open ZMQ ports; skipped by default in CI/local unit runs.")
def test_server_interaction():
    repo_root = Path(__file__).resolve().parents[1]
    nsck_demo_dir = repo_root

    # 1. Start Server in Background with --no-teacher (to trigger A2C logic)
    server_process = subprocess.Popen(
        [sys.executable, "python/python_server.py", "--no-teacher", "--eval"],
        cwd=str(nsck_demo_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    context = zmq.Context()
    push_sock = context.socket(zmq.PUSH)
    push_sock.connect("tcp://127.0.0.1:5565")
    
    sub_sock = context.socket(zmq.SUB)
    sub_sock.connect("tcp://127.0.0.1:5566")
    sub_sock.setsockopt_string(zmq.SUBSCRIBE, "")
    
    # IMPORTANT: don't stall forever if server didn't start.
    sub_sock.setsockopt(zmq.RCVTIMEO, 5000)  # ms
    sub_sock.setsockopt(zmq.LINGER, 0)

    print(">> Waiting for server wakeup...")
    time.sleep(2)

    try:
        # 2. Send Fake Snake State (Step 1)
        print(">> Sending Step 1 (Snake)...")
        msg1 = {
            "game": "snake",
            "session_id": "test_session_1",
            "state": {
                "head": [5, 5],
                "food": [5, 2] # Up
            },
            "reward": 0.0,
            "done": False,
            "image": "", # Server handles blank
            "score": 0
        }
        push_sock.send_json(msg1)
        
        # 3. Receive Response (with timeout)
        try:
            topic, cmd = sub_sock.recv_string().split(":")
        except zmq.Again:
            # Print server output to help diagnose startup failures
            try:
                outs, errs = server_process.communicate(timeout=1)
            except Exception:
                outs, errs = "", ""
            raise AssertionError(
                "Timed out waiting for server response on tcp://127.0.0.1:5566. "
                f"Server may not have started.\nSTDOUT:\n{outs}\nSTDERR:\n{errs}"
            )

        print(f"<< Received: {topic}:{cmd}")
        assert topic == "SNAKE"
        assert cmd in ["UP", "DOWN", "LEFT", "RIGHT"]
        
        # 4. Send Fake Snake State (Step 2 - Reward)
        print(">> Sending Step 2 (Reward +1)...")
        msg2 = {
            "game": "snake",
            "session_id": "test_session_1",
            "state": {
                "head": [5, 4], # Moved Up
                "food": [5, 2]
            },
            "reward": 1.0, # Got closer
            "done": False,
            "image": "",
            "score": 0
        }
        push_sock.send_json(msg2)
        
        # 5. Receive Response (with timeout)
        try:
            topic2, cmd2 = sub_sock.recv_string().split(":")
        except zmq.Again:
            try:
                outs, errs = server_process.communicate(timeout=1)
            except Exception:
                outs, errs = "", ""
            raise AssertionError(
                "Timed out waiting for second server response.\n"
                f"STDOUT:\n{outs}\nSTDERR:\n{errs}"
            )

        print(f"<< Received: {topic2}:{cmd2}")
        print(">> Success! Server processed 2 steps without crashing.")

    finally:
        server_process.terminate()
        try:
            server_process.wait(timeout=2)
        except Exception:
            server_process.kill()

        push_sock.close()
        sub_sock.close()
        context.term()


if __name__ == "__main__":
    test_server_interaction()
