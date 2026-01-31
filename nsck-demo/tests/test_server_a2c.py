import zmq
import json
import time
import subprocess
import sys
import os

def test_server_interaction():
    # 1. Start Server in Background with --no-teacher (to trigger A2C logic)
    server_process = subprocess.Popen(
        [sys.executable, "python/python_server.py", "--no-teacher", "--eval"],
        cwd="d:\\NGCN\\nsck-demo",
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
    
    print(">> Waiting for server wakeup...")
    time.sleep(5) # Give it time to load model
    
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
        
        # 3. Receive Response
        topic, cmd = sub_sock.recv_string().split(":")
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
        
        # 5. Receive Response
        topic2, cmd2 = sub_sock.recv_string().split(":")
        print(f"<< Received: {topic2}:{cmd2}")
        
        print(">> Success! Server processed 2 steps without crashing.")
        
    except Exception as e:
        print(f"FAILED: {e}")
        # Print server output
        outs, errs = server_process.communicate(timeout=1)
        print("SERVER STDOUT:", outs)
        print("SERVER STDERR:", errs)
    finally:
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    test_server_interaction()
