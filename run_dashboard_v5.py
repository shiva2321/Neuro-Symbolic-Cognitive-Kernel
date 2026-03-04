import subprocess
import time
import webbrowser
import os
import sys

def run():
    # Start the API
    print("[NSCK] Starting Societal World API V5 on port 8080...")
    api_path = os.path.abspath("d:/Node_network/nsck/api/api_v5.py")
    api_process = subprocess.Popen([sys.executable, api_path])
    
    # Wait for server to warm up
    time.sleep(2)
    
    # Open the Dashboard
    dashboard_path = os.path.abspath("d:/Node_network/nsck/api/web/v5/index.html")
    print(f"[NSCK] Opening Dashboard: {dashboard_path}")
    webbrowser.open(f"file://{dashboard_path}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[NSCK] Shutting down...")
        api_process.terminate()

if __name__ == "__main__":
    run()
