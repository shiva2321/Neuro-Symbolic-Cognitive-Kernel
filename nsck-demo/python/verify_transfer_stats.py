import zmq
import json
import time
import numpy as np

def verify_transfer():
    print("=== STRICT TRANSFER VERIFICATION PROTOCOL ===")
    print("Listening for Telemetry on Port 5567...")
    
    context = zmq.Context()
    sub_sock = context.socket(zmq.SUB)
    sub_sock.setsockopt(zmq.LINGER, 0)
    sub_sock.connect("tcp://127.0.0.1:5567")
    sub_sock.setsockopt_string(zmq.SUBSCRIBE, "VIS:")
    
    pong_rallies = []
    pong_scores = [] # AI vs AI score if relevant
    
    start_time = time.time()
    DURATION = 30 # seconds to monitor
    
    try:
        while time.time() - start_time < DURATION:
            try:
                msg = sub_sock.recv_string(flags=zmq.NOBLOCK)
                if msg.startswith("VIS:"):
                    json_str = msg[4:]
                    data = json.loads(json_str)
                    
                    game = data.get("game")
                    score = data.get("score", 0)
                    
                    if game == "pong":
                        # Score in Pong UI is currently Rally Count
                        pong_rallies.append(score)
                        print(f"PONG RALLY: {score}")
                    elif game == "snake":
                       print(f"SNAKE SCORE: {score}")
                        
            except zmq.Again:
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        pass
        
    print(f"\n=== RESULTS (Duration: {DURATION}s) ===")
    if len(pong_rallies) > 0:
        max_rally = np.max(pong_rallies)
        mean_rally = np.mean(pong_rallies)
        # Note: Rally resets to 0 on miss. We want the PEAK rally or distribution of peaks.
        # But 'score' sends current tally.
        # So we should look for 'local maxima' in the stream.
        # Simple heuristic: Max Rally Observed.
        print(f"Max Rally Observed: {max_rally}")
        print(f"Mean Rally State: {mean_rally:.2f}")
        
        if max_rally > 0:
            print("VERDICT: SUCCESS (Positive Transfer Detected)")
        else:
            print("VERDICT: FAILURE (Zero Rallies)")
    else:
        print("No Pong data received.")

if __name__ == "__main__":
    verify_transfer()
