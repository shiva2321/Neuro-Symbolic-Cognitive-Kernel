
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
import signal

# --- Configuration ---
app = Flask(__name__, static_folder="../web", template_folder="../web")
app.config['SECRET_KEY'] = 'nsck_agi_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Process Tracker
active_processes = {} # { 'snake': proc, 'pong': proc, ... }
process_lock = threading.Lock()

def stop_active_game(game=None):
    """Helper to stop specific game or all processes."""
    with process_lock:
        if game:
            targets = [game] if game in active_processes else []
        else:
            targets = list(active_processes.keys())
            
        for g in targets:
            p = active_processes.get(g)
            if p:
                try:
                    print(f"[DEBUG] Stopping {g} process {p.pid} (poll={p.poll()})...")
                    
                    # Windows-specific robust kill (Tree Kill)
                    if sys.platform == "win32":
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
                    else:
                        p.terminate()
                    
                    # Cleanup handle
                    p.poll() 
                except Exception as e:
                    print(f"Error stopping {g}: {e}")
                finally:
                    if g in active_processes:
                        del active_processes[g]
    
    if not game: # Global stop
        emit('game_status', {'active': False, 'all': True}, broadcast=True)
        send_command("stop_game")
    else:
        emit('game_status', {'active': False, 'game': game}, broadcast=True)
        # We can also signal the brain to go dormant if this was the only task?
        # For now, per the user's "Stop will stop the game play", the brain should probably stop too?
        # Actually, let's just emit the game status. The brain gating is global.
        # User said: "STOP all will also stop brain". "STOP will stop game play".
        # So per-task stop is process only.

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
                    game = game.lower() # Normalize to lowercase
                    try:
                         data = json.loads(json_str)
                         # Send to specific room for high-fidelity view
                         socketio.emit('game_update', data, room=game)
                         # Also broadcast to main dashboard for "visual overview"
                         socketio.emit('game_update_global', data)
                    except: pass 
                
            if sub_data in socks and socks[sub_data] == zmq.POLLIN:
                topic_str = sub_data.recv_string()
                parts = topic_str.split(":", 1)
                if len(parts) == 2:
                    topic, json_str = parts
                    try:
                        payload = json.loads(json_str)
                        game = payload.get("game", payload.get("task", "global")).lower() # Normalize
                        
                        if topic == "TELEMETRY":
                            socketio.emit('telemetry', payload, room=game)
                            socketio.emit('telemetry_global', payload)
                        elif topic == "BRAIN_STATE":
                            socketio.emit('brain_update', payload) # Global
                        elif topic == "LOG":
                            socketio.emit('log_entry', payload, room=game)
                            socketio.emit('log_entry_global', payload)
                        elif topic == "WORKSPACE":
                            socketio.emit('workspace_update', payload, room=game)
                            socketio.emit('workspace_update_global', payload)
                        elif topic == "CAUSAL":
                            socketio.emit('causal_update', payload, room=game)
                        elif topic == "CHAT_RESPONSE":
                            socketio.emit('chat_response', payload, room=game)
                            socketio.emit('chat_response_global', payload)
                        elif topic == "VIS":
                            # Route visual data to game rooms and global summary
                            socketio.emit('game_update', payload, room=game)
                            socketio.emit('game_update_global', payload)
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

@app.route('/task/<game>')
def task_view(game):
    return send_from_directory('../web', 'task_view.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../web', path)

@app.route('/export_logs')
def export_logs():
    """Generates a comprehensive, structured AGI report."""
    try:
        from flask import request
        target_task = request.args.get('task')
        scope = target_task.upper() if target_task else "GLOBAL"
        
        buffer = io.StringIO()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        buffer.write(f"{'='*60}\n")
        buffer.write(f" NSCK AGI SYSTEM REPORT - {scope}\n")
        buffer.write(f" Generated: {timestamp}\n")
        buffer.write(f"{'='*60}\n\n")

        # 1. Telemetry Section
        buffer.write(f">> [SECTION 1] PERFORMANCE TELEMETRY ({scope})\n")
        buffer.write("-" * 80 + "\n")
        
        telemetry_rows = []
        
        # Try CSV first
        if os.path.exists("training_log.csv"):
            try:
                with open("training_log.csv", 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        if target_task:
                            tt = target_task.lower()
                            telemetry_rows = [l.strip().split(',') for l in lines[1:] if f",{tt}," in l.lower()]
                        else:
                            telemetry_rows = [l.strip().split(',') for l in lines[1:]]
            except: pass

        # Fallback to SQLite if CSV is empty/missing
        if not telemetry_rows and os.path.exists("nsck_logs.db"):
            try:
                import sqlite3
                conn = sqlite3.connect("nsck_logs.db")
                curr = conn.cursor()
                if target_task:
                    curr.execute("SELECT timestamp, step, task, value, value, value, value FROM telemetry WHERE task = ? ORDER BY timestamp DESC LIMIT 100", (target_task.lower(),))
                else:
                    curr.execute("SELECT timestamp, step, task, value, value, value, value FROM telemetry ORDER BY timestamp DESC LIMIT 100")
                telemetry_rows = curr.fetchall()
                conn.close()
            except: pass

        if telemetry_rows:
            import datetime
            header = ["TIME", "STEP", "TASK", "AGG_ACT", "LOSS", "CONF", "SCORE"]
            buffer.write(f"{header[0]:<20} | {header[1]:<6} | {header[2]:<6} | {header[3]:<7} | {header[4]:<7} | {header[5]:<6} | {header[6]:<7}\n")
            buffer.write("-" * 80 + "\n")
            
            for row in telemetry_rows[-100:]:
                if len(row) < 7: continue
                try:
                    iso_time = datetime.datetime.fromtimestamp(float(row[0])).strftime('%H:%M:%S')
                    buffer.write(f"{iso_time:<20} | {row[1]:<6} | {row[2]:<6} | {float(row[3]):<7.2f} | {float(row[4]):<7.4f} | {float(row[5]):<6.3f} | {float(row[6]):<7.3f}\n")
                except:
                    buffer.write(f"{'ERR':<20} | {row[1]:<6} | {row[2]:<6} | {'-':<7} | {'-':<7} | {'-':<6} | {'-':<7}\n")
        else:
            buffer.write("No telemetry data available for this task.\n")
        buffer.write("\n")

        # 2. Cognitive Trace & Decisions (from nsck_session.txt)
        if os.path.exists("nsck_session.txt"):
            with open("nsck_session.txt", 'r', encoding='utf-8') as f:
                all_logs = f.readlines()

            # 2a. Cognitive Trace (Thoughts)
            buffer.write(f">> [SECTION 2] COGNITIVE TRACE: STATE, DRIVES & REASONING\n")
            buffer.write("-" * 80 + "\n")
            
            # Filter for specific task if requested
            if target_task:
                tt = target_task.upper()
                # Find thoughts for this task specifically (includes [WON], [STATE], [DRIVES], [COMPETITION])
                thoughts = [l.strip() for l in all_logs if "[THOUGHT]" in l and f":{tt}]" in l]
                if not thoughts: # Fallback to generic if no enriched tags yet
                    thoughts = [l.strip() for l in all_logs if "[THOUGHT]" in l and f" {tt} " in l.upper()]
                for l in thoughts[-300:]: buffer.write(f"{l}\n")
            else:
                thoughts = [l.strip() for l in all_logs if "[THOUGHT]" in l]
                for l in thoughts[-300:]: buffer.write(f"{l}\n")
            buffer.write("\n")

            # 2b. Decisions & Actions
            buffer.write(f">> [SECTION 3] DECISION LOG: ACTIONS & TEACHER INPUTS\n")
            buffer.write("-" * 80 + "\n")
            if target_task:
                tt = target_task.upper()
                decisions = [l.strip() for l in all_logs if "[DECISION]" in l and f" {tt} " in l.upper()]
                for l in decisions[-150:]: buffer.write(f"{l}\n")
            else:
                decisions = [l.strip() for l in all_logs if "[DECISION]" in l]
                for l in decisions[-150:]: buffer.write(f"{l}\n")
            buffer.write("\n")

            # 3. System Logs
            buffer.write(f">> [SECTION 4] SYSTEM & SERVER LOGS (GLOBAL CONTEXT)\n")
            buffer.write("-" * 80 + "\n")
            sys_logs = [l.strip() for l in all_logs if "[SERVER]" in l or "[SYSTEM]" in l or "[ADMIN]" in l or "[CHAT]" in l]
            for l in sys_logs[-150:]: buffer.write(f"{l}\n")
            buffer.write("\n")

        # 4. Causal Insights
        if os.path.exists("causal_dump.json"):
             buffer.write(f">> [SECTION 5] CAUSAL ANALYTICS\n")
             buffer.write("-" * 50 + "\n")
             try:
                 with open("causal_dump.json", 'r') as f:
                     data = json.load(f)
                     if target_task:
                         tk = target_task.lower()
                         if tk in data:
                             for link in data[tk][:50]:
                                 buffer.write(f" * {link.get('cause')} -> {link.get('effect')} (Strength: {link.get('strength', 0):.2f})\n")
                     else:
                         for task, links in data.items():
                             buffer.write(f"Task {task.upper()}:\n")
                             for link in links[:10]:
                                 buffer.write(f"   - {link.get('cause', 'UNK')} -> {link.get('effect', 'UNK')} (Str: {link.get('strength', 0):.2f})\n")
             except: pass
             buffer.write("\n")

        buffer.write(f"{'='*60}\n")
        buffer.write(" END OF AUTOMATED AGI REPORT\n")
        buffer.write(f"{'='*60}\n")
        
        filename = f"nsck_{target_task.lower() if target_task else 'global'}_report.txt"
        return Response(
            buffer.getvalue(),
            mimetype="text/plain",
            headers={"Content-Disposition": f"attachment;filename={filename}"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Export Error: {e}", 500

# Helper to clean up dead processes (zombies)
def cleanup_dead_processes():
    """Removes processes that have terminated from the dictionary."""
    with process_lock:
        to_remove = []
        for g, proc in active_processes.items():
            if proc.poll() is not None: # Process has exited
                to_remove.append(g)
        for g in to_remove:
            del active_processes[g]

# --- SocketIO Events (Commands from UI) ---
@socketio.on('connect')
def handle_connect():
    print("Client Connected")
    emit('status', {'msg': 'Connected to NSCK Mission Control'})

@socketio.on('join_task')
def on_join_task(data):
    game = data.get('game')
    if game:
        from flask_socketio import join_room
        join_room(game)
        print(f"Client joined room: {game}")

@socketio.on('start_game')
def on_start_game(data):
    game = data['game']
    print(f"CMD: Start Game {game}")
    
    # Clean up first
    cleanup_dead_processes()
    
    with process_lock:
        if game in active_processes:
            emit('log_entry', {'source': 'SYSTEM', 'message': f'{game.upper()} already running (PID {active_processes[game].pid}).'})
            return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    game_script = os.path.join(base_dir, f"{game}_ui.py")
    
    if os.path.exists(game_script):
        proc = subprocess.Popen([sys.executable, game_script], cwd=base_dir)
        
        with process_lock:
            active_processes[game] = proc
        
        emit('log_entry', {'source': 'SYSTEM', 'message': f'Launched {game.upper()} Process (PID {proc.pid})'})
        emit('game_status', {'active': True, 'game': game}, broadcast=True)
        # Ensure brain is active when a task starts
        send_command("admin", {"cmd": "start_game"})
    else:
        emit('log_entry', {'source': 'ERROR', 'message': f'Script {game_script} not found'})


@socketio.on('stop_game')
def on_stop_game(data=None):
    game = data.get('game') if data else None
    print(f"CMD: Stop Game {game if game else 'ALL'}")
    stop_active_game(game)
    if not game:
        send_command("stop_game") # Tell brain to stop all
    emit('log_entry', {'source': 'SYSTEM', 'message': f'Terminated {game if game else "ALL"}'})

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
    def dashboard_signal_handler(sig, frame):
        print(f"\n[DASHBOARD] Signal {sig} received. Shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, dashboard_signal_handler)
    signal.signal(signal.SIGTERM, dashboard_signal_handler)

    print("NSCK WEB DASHBOARD: Starting on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
