import zmq
import json
import time
import os

SERVER_ADDR = "tcp://127.0.0.1:5565"
OUTPUT_PATH = "d:\\NSCK_v1\\causal_dump.json"

def main():
    print(f">> Connecting to NSCK Server at {SERVER_ADDR}...")
    context = zmq.Context()
    sock = context.socket(zmq.PUSH)
    sock.connect(SERVER_ADDR)

    # Clean up previous dump
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    print(">> Sending 'dump_causal_graph' command...")
    payload = {
        "type": "admin",
        "cmd": "dump_causal_graph",
        "path": OUTPUT_PATH
    }
    sock.send_json(payload)

    print(">> Waiting for server response (file creation)...")
    for _ in range(10):
        if os.path.exists(OUTPUT_PATH):
            break
        time.sleep(0.5)
        print(".", end="", flush=True)
    print()

    if not os.path.exists(OUTPUT_PATH):
        print(f"\n[FAIL] Server did not create {OUTPUT_PATH}. Is it running the latest code?")
        return

    try:
        with open(OUTPUT_PATH, "r") as f:
            data = json.load(f)
        
        print(f"\n[SUCCESS] Retrieved Causal Graph Data!")
        
        total_links = sum(len(links) for links in data.values())
        print(f"Total Contexts: {len(data)} | Total Links: {total_links}")
        
        for ctx, links in data.items():
            print(f"\n--- Context: {ctx.upper()} ({len(links)} rules) ---")
            sorted_links = sorted(links, key=lambda x: x['strength'], reverse=True)
            for i, link in enumerate(sorted_links[:10]): # Top 10
                cause = link.get('cause', '?')
                effect = link.get('effect', '?')
                strength = link.get('strength', 0.0)
                ltype = link.get('type', 'causes')
                print(f"  {i+1}. [{ltype.upper()}] {cause} -> {effect} (conf={strength:.2f})")
                
    except Exception as e:
        print(f"\n[ERROR] Failed to parse graph data: {e}")

if __name__ == "__main__":
    main()
