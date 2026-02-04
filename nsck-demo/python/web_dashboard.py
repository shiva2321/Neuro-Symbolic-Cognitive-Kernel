
from flask import Flask, render_template, send_from_directory, send_file, Response
from flask_socketio import SocketIO, emit
import zmq
import threading
import json
import json


import os
import time
import subprocess
import io
import zipfile
import sys

# --- Configuration ---
app = Flask(__name__, static_folder="../web", template_folder="../web")
app.config['SECRET_KEY'] = 'nsck_agi_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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
                print(f"Force killing process {p.pid}...")
                p.kill()
                p.wait()
        except Exception as e:
            print(f"Error stopping process: {e}")
            # Last ditch effort
            try:
                 os.kill(p.pid, 9)
            except: pass
        finally:
            active_processes['game'] = None
            emit('game_status', {'active': False}, broadcast=True)

# --- ZMQ Setup (Subscriber to Brain) ---
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
    # Create NEW context sockets for this thread (ZMQ safety)
    # Actually, sharing context is thread-safe, but sockets are NOT. 
    # We must create sockets INSIDE the thread.
    local_ctx = zmq.Context()
    
    # Visuals Socket
    sub_visuals = local_ctx.socket(zmq.SUB)
    sub_visuals.connect(f"tcp://localhost:{ZMQ_SUB_VISUALS}")
    sub_visuals.setsockopt_string(zmq.SUBSCRIBE, "")
    sub_visuals.setsockopt(zmq.CONFLATE, 1) # Keep only latest message
    
    # Data Socket
    sub_data = local_ctx.socket(zmq.SUB)
    sub_data.connect(f"tcp://localhost:{ZMQ_SUB_DATA}")
    sub_data.setsockopt_string(zmq.SUBSCRIBE, "") 
    
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
                parts = raw_msg.split(":", 1)
                if len(parts) == 2:
                    game, json_str = parts
                    try:
                         data = json.loads(json_str)
                         socketio.emit('game_update', data)
                    except: pass 
                
            if sub_data in socks and socks[sub_data] == zmq.POLLIN:
                topic_str = sub_data.recv_string()
                # print(f"DEBUG_ZMQ: Received {topic_str[:50]}...") # excessive logs?
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
                        elif topic == "WORKSPACE":
                            socketio.emit('workspace_update', payload)
                        elif topic == "CAUSAL":
                            socketio.emit('causal_update', payload)
                        elif topic == "VIS":
                            # Handle Visual packets arriving on Data port (from server patch)
                            # print(f"DEBUG_VIS: Emitting game_update (len={len(json_str)})")
                            socketio.emit('game_update', payload)
                        elif topic == "CHAT_RESPONSE":
                            socketio.emit('chat_response', payload)
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            print(f"Listener Error: {e}")
            time.sleep(1) 
            
        time.sleep(0.01) # Yield


# Start Listener (Threading)
t = threading.Thread(target=zmq_listener, daemon=True)
t.start()

# --- Routes ---
@app.route('/')
def index():
    return send_from_directory('../web', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../web', path)

@app.route('/export_logs')
def export_logs():
    """Aggregates system logs and traces into a single text file download."""
    try:
        # Collect relevant files
        files_to_export = ["training_log.csv"]
        
        # We'll stream the content into a single text buffer
        buffer = io.StringIO()
        buffer.write(f"NSCK SYSTEM LOG EXPORT - {time.ctime()}\n")
        buffer.write("="*50 + "\n\n")
        
        # 1. Training Log (CSV)
        if os.path.exists("training_log.csv"):
            buffer.write("--- TRAINING TELEMETRY (training_log.csv) ---\n")
            with open("training_log.csv", 'r') as f:
                buffer.write(f.read())
            buffer.write("\n\n")
            
        # 2. Latest Run Log (Txt) - Find most recent run_log_*.txt
        log_files = [f for f in os.listdir('.') if f.startswith('run_log_') and f.endswith('.txt')]
        if log_files:
            latest_log = max(log_files, key=os.path.getctime)
            buffer.write(f"--- LATEST RUN LOG ({latest_log}) ---\n")
            with open(latest_log, 'r') as f:
                buffer.write(f.read())
            buffer.write("\n\n")
            
        # 3. Causal Dump (If available)
        if os.path.exists("causal_dump.json"):
             buffer.write("--- CAUSAL GRAPH SNAPSHOT (causal_dump.json) ---\n")
             with open("causal_dump.json", 'r') as f:
                 buffer.write(f.read())
             buffer.write("\n\n")

        # Convert to BytesIO for Flask send_file
        mem_file = io.BytesIO()
        mem_file.write(buffer.getvalue().encode('utf-8'))
        mem_file.seek(0)
        
        return send_file(
            mem_file,
            as_attachment=True,
            download_name=f'nsck_logs_{int(time.time())}.txt',
            mimetype='text/plain'
        )
    except Exception as e:
        return f"Error exporting logs: {e}", 500

# --- SocketIO Events (Commands from UI) ---
@socketio.on('connect')
def handle_connect():
    print("Client Connected")
    emit('status', {'msg': 'Connected to NSCK Mission Control'})
    emit('log_entry', {'source': 'SYSTEM', 'message': 'Client Connected to Mission Control'})

@socketio.on('start_game')
def on_start_game(data):
    print(f"CMD: Start Game {data['game']}")
    stop_active_game()
    
    # Robustly find script path relative to THIS file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    game_script = os.path.join(base_dir, f"{data['game']}_ui.py")
    
    if os.path.exists(game_script):
        # Use sys.executable to ensure we use the same venv python
        # Set cwd to base_dir so scripts can find their assets
        proc = subprocess.Popen([sys.executable, game_script], cwd=base_dir)
        active_processes['game'] = proc
        
        emit('log_entry', {'source': 'SYSTEM', 'message': f'Launched {data["game"].upper()} Process'})
        emit('game_status', {'active': True, 'game': data['game']}, broadcast=True)
    else:
        emit('log_entry', {'source': 'ERROR', 'message': f'Script {game_script} not found'})

@socketio.on('stop_game')
def on_stop_game(data=None):
    print("CMD: Stop Game")
    stop_active_game()
    # Also tell Brain to stop expecting input
    send_command("stop_game")
    emit('log_entry', {'source': 'SYSTEM', 'message': 'Game Process Terminated'})

@socketio.on('admin_command')
def on_admin_command(data):
    """Unified Admin Handler."""
    cmd = data.get('cmd')
    print(f"CMD: Admin {cmd}")
    
    payload = {"cmd": cmd}
    
    # Pass through additional args
    if 'device' in data: payload['device'] = data['device']
    if 'mode' in data: payload['mode'] = data['mode']
    if 'epochs' in data: payload['epochs'] = data['epochs']
    
    send_command("admin", payload)
    emit('log_entry', {'source': 'USER', 'message': f'Sent Admin Command: {cmd}'})

@socketio.on('train_char')
def on_train_char(data):
    """
    Handle handwriting training data.
    data = { 'image': base64_string, 'label': 'A', 'mode': 'handwritten' }
    """
    # For now, we trigger the 'train_char' admin command which uses dataset
    # Future: Actually accept the image and train on it (Requires updating server to accept image payload)
    # The server has `run_char_train`, which currently pulls from dataset.
    # We will trigger that for "General Training".
    
    # If the user drew something, we'd need to send that image.
    # Implementing "Interactive Training" would require a new message type on server.
    # For this overhaul, we'll map the button to the 'train_char' command we saw in server.
    
    print(f"CMD: Train Char")
    txt = data.get('text', None) 
    send_command("admin", {"cmd": "train_char", "epochs": 1, "text": txt})
    emit('log_entry', {'source': 'USER', 'message': 'Initiated Character Training'})

@socketio.on('submit_chat')
def on_submit_chat(data):
    text = data.get('text')
    if text:
        print(f"CHAT OUT: {text}")
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
