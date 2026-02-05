
import subprocess
import time
import sys
import os
import signal

# Configuration
PYTHON_EXE = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SERVER_SCRIPT = os.path.join(BASE_DIR, "python_server.py")
DASHBOARD_SCRIPT = os.path.join(BASE_DIR, "web_dashboard.py")

processes = []

def signal_handler(sig, frame):
    print("\n[LAUNCHER] Shutdown Signal Received. Terminating processes...")
    cleanup()
    sys.exit(0)

def cleanup():
    for p in processes:
        if p.poll() is None: # If running
            print(f"[LAUNCHER] Terminating PID {p.pid}")
            if sys.platform == "win32":
                # Robust tree kill for Windows to prevent orphans
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            else:
                p.terminate()
            
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                if sys.platform != "win32": # taskkill already forced it
                    p.kill()

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("="*60)
    print(" NSCK AGI SYSTEM LAUNCHER")
    print("="*60)
    
    # 1. Start Brain (Python Server)
    print(f"[LAUNCHER] Starting Brain Core: {SERVER_SCRIPT}")
    server_proc = subprocess.Popen([PYTHON_EXE, SERVER_SCRIPT], cwd=BASE_DIR)
    processes.append(server_proc)
    time.sleep(2) # Give it time to bind ZMQ ports
    
    if server_proc.poll() is not None:
        print("[LAUNCHER] CRITICAL: Brain failed to start. Exiting.")
        return

    # 2. Start Interface (Web Dashboard)
    print(f"[LAUNCHER] Starting Mission Control: {DASHBOARD_SCRIPT}")
    dash_proc = subprocess.Popen([PYTHON_EXE, DASHBOARD_SCRIPT], cwd=BASE_DIR)
    processes.append(dash_proc)
    
    print("\n" + "-"*60)
    print(" SYSTEM ACTIVE ")
    print(" Access Mission Control at: http://localhost:5000")
    print(" Logs written to:         nsck_session.txt")
    print(" Press Ctrl+C to Shutdown")
    print("-"*60 + "\n")
    
    # Monitor Loop
    while True:
        time.sleep(1)
        # Check if server is alive
        if server_proc.poll() is not None:
             print("\n[LAUNCHER] CRITICAL: Brain process died! Shutting down.")
             cleanup()
             break
        
        # Check if dashboard is alive
        if dash_proc.poll() is not None:
             print("\n[LAUNCHER] NOTICE: Dashboard process ended. Restarting...")
             dash_proc = subprocess.Popen([PYTHON_EXE, DASHBOARD_SCRIPT], cwd=BASE_DIR)
             processes[-1] = dash_proc # Replace in list

if __name__ == "__main__":
    main()
