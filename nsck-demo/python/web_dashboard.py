
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import zmq
import threading
import json
import os
import time

# --- Configuration ---
app = Flask(__name__, static_folder="../web", template_folder="../web")
app.config['SECRET_KEY'] = 'nsck_agi_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Process Tracker
active_processes = {}

def stop_active_game():
    """Helper to stop currently running game process."""
    if 'game' in active_processes and active_processes['game']:
        p = active_processes['game']
        try:
            print(f"Stopping process {p.pid}...")
            p.terminate()
            try:
                p.wait(timeout=1)
            except subprocess.TimeoutExpired:
                print("Force killing...")
                p.kill()
        except Exception as e:
            print(f"Error stopping process: {e}")
        finally:
            active_processes['game'] = None

# --- ZMQ Setup (Subscriber to Brain) ---
# We reuse the same ports as the legacy dashboard for compatibility
# PUB/SUB from Server (5557 or 5556? Check dashboard.py)
# Legacy dashboard used: 
# self.sub_sock = self.context.socket(zmq.SUB)
# self.sub_sock.connect("tcp://localhost:5557") 
# PUSH to Server: 
# self.push_sock = self.context.socket(zmq.PUSH)
# self.push_sock.connect("tcp://localhost:5558")

ZMQ_SUB_VISUALS = 5566
ZMQ_SUB_DATA = 5567
ZMQ_PUSH_PORT = 5565

context = zmq.Context()
push_sock = context.socket(zmq.PUSH)
push_sock.connect(f"tcp://localhost:{ZMQ_PUSH_PORT}")

# Helper to send commands to Brain
def send_command(cmd_type, payload=None):
    msg = {"type": cmd_type}
    if payload:
        msg.update(payload)
    try:
        push_sock.send_json(msg)
    except Exception as e:
        print(f"ZMQ Error: {e}")

# --- Background Thread for ZMQ Listener ---
def zmq_listener():
    """
    Listens to the Brain's broadcast and forwards it to the Web Client via SocketIO.
    """
    # Visuals Socket
    sub_visuals = context.socket(zmq.SUB)
    sub_visuals.connect(f"tcp://localhost:{ZMQ_SUB_VISUALS}")
    sub_visuals.setsockopt_string(zmq.SUBSCRIBE, "")
    
    # Data Socket
    sub_data = context.socket(zmq.SUB)
    sub_data.connect(f"tcp://localhost:{ZMQ_SUB_DATA}")
    sub_data.setsockopt_string(zmq.SUBSCRIBE, "") # Listen to everything (STATS, TELEMETRY, LOG)
    
    # Poller
    poller = zmq.Poller()
    poller.register(sub_visuals, zmq.POLLIN)
    poller.register(sub_data, zmq.POLLIN)
    
    print(f"FRONTEND: Listening on ports {ZMQ_SUB_VISUALS} (Visuals) and {ZMQ_SUB_DATA} (Data)...")
    
    while True:
        try:
            socks = dict(poller.poll(100)) # 100ms timeout
            
            if sub_visuals in socks and socks[sub_visuals] == zmq.POLLIN:
                raw_msg = sub_visuals.recv_string()
                # Expecting "GAME:JSON"
                parts = raw_msg.split(":", 1)
                if len(parts) == 2:
                    game, json_str = parts
                    try:
                         data = json.loads(json_str)
                         socketio.emit('game_update', data)
                    except: pass 
                
            if sub_data in socks and socks[sub_data] == zmq.POLLIN:
                topic_str = sub_data.recv_string()
                # Topic format: "TOPIC:JSON"
                parts = topic_str.split(":", 1)
                if len(parts) == 2:
                    topic, json_str = parts
                    try:
                        payload = json.loads(json_str)
                        if topic == "TELEMETRY":
                            socketio.emit('telemetry_update', payload)
                        elif topic == "BRAIN_STATE":
                            socketio.emit('brain_update', payload)
                            socketio.emit('log_entry', {'source': 'BRAIN', 'message': payload.get('message', str(payload))})
                        elif topic == "LOG":
                            socketio.emit('log_entry', payload)
                        elif topic == "STATS": # Legacy compatibility
                            socketio.emit('telemetry_update', payload)
                        elif topic == "STATS": # Legacy compatibility
                            socketio.emit('telemetry_update', payload)
                        elif topic == "CHAT_RESPONSE":
                            socketio.emit('chat_response', payload)
                    except json.JSONDecodeError:
                        pass
                        
            # Handle Visuals separately since logic is tricky
            if sub_visuals in socks and socks[sub_visuals] == zmq.POLLIN:
                 raw_msg = sub_visuals.recv_string()
                 # Expecting "GAME:JSON"
                 parts = raw_msg.split(":", 1)
                 if len(parts) == 2:
                     game, json_str = parts
                     try:
                         data = json.loads(json_str)
                         socketio.emit('game_update', data)
                     except: pass

        except Exception as e:
            print(f"Listener Error: {e}")

# Start Listener
t = threading.Thread(target=zmq_listener, daemon=True)
t.start()

# --- Routes ---
@app.route('/')
def index():
    return send_from_directory('../web', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../web', path)

# --- SocketIO Events (Commands from UI) ---
@socketio.on('connect')
def handle_connect():
    print("Client Connected")
    emit('status', {'msg': 'Connected to NSCK Mission Control'})

@socketio.on('start_game')
def on_start_game(data):
    print(f"CMD: Start Game {data['game']}")
    # In legacy, dashboard spawned the process. 
    # Ideally, PythonServer should spawn it, or we have a launcher.
    # For now, we assume the server can handle "cmd": "launch_game" if implemented,
    # OR we implement a simple process spawner here if needed.
    # Plan: Send command to PythonServer -> PythonServer spawns UI? 
    # PythonServer (server.py) doesn't have process spawning logic usually.
    # Dashboard.py had it.
    
    # Simple fix: We spawn it here for now, or send to a "Launcher Service".
    # Let's send a command to the "Server" if it supports it, OR we spawn 'python snake_ui.py'
    # For safety/simplicity in this phase: We spawn here.
    import subprocess
    
    # STOP existing game first!
    stop_active_game()
    
    if data['game'] == 'snake':
        proc = subprocess.Popen(["python", "snake_ui.py"], cwd=os.getcwd())
        active_processes['game'] = proc
    elif data['game'] == 'pong':
        proc = subprocess.Popen(["python", "pong_ui.py"], cwd=os.getcwd())
        active_processes['game'] = proc
    elif data['game'] == 'maze':
        proc = subprocess.Popen(["python", "maze_ui.py"], cwd=os.getcwd())
        active_processes['game'] = proc

@socketio.on('stop_game')
def on_stop_game(data=None):
    print("CMD: Stop Game")
    stop_active_game()
    print("Game Process Terminated.")

@socketio.on('system_command')
def on_system_command(data):
    """
    Generic commands: sleep, reset, etc.
    """
    cmd = data.get('cmd')
    if cmd == 'sleep':
        send_command("admin", {"cmd": "force_sleep"})
    elif cmd == 'wake':
        send_command("admin", {"cmd": "wake_up"})

@socketio.on('submit_chat')
def on_submit_chat(data):
    """
    Handle user chat input.
    """
    text = data.get('text')
    if text:
        print(f"CHAT OUT: {text}")
        # Send to Brain via ZMQ Push
        # Use type "CHAT_INPUT" to differentiate
        try:
            push_sock.send_json({
                "type": "CHAT_INPUT", 
                "text": text,
                "teach_mode": data.get('teach', False)
            })
        except Exception as e:
            print(f"Chat Send Error: {e}")

if __name__ == '__main__':
    print("NSCK WEB DASHBOARD: Starting on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
