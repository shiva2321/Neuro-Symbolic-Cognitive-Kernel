import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import zmq
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from collections import deque
import sys
import os
import queue

# --- STYLE CONFIG ---
BG_COLOR = "#2e2e2e"
FG_COLOR = "#ffffff"
ACCENT_COLOR = "#007acc"
FONT_MAIN = ("Consolas", 10)
FONT_HEADER = ("Consolas", 12, "bold")

class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NSCK Mission Control: Brain Inspector")
        self.root.geometry("1600x900")
        self.root.configure(bg=BG_COLOR)
        
        # Internal State
        self.procs = {"Server": None, "Snake": None, "Pong": None}
        self.running = True
        
        # QUEUES (Thread-Safe Communication)
        self.log_queue = queue.Queue()
        self.zmq_queue = queue.Queue()
        
        # LOGGING BUFFER
        self.event_log = []
        
        # ZMQ
        self.context = zmq.Context()
        self.push_sock = self.context.socket(zmq.PUSH)
        self.push_sock.connect("tcp://127.0.0.1:5565") 
        
        self.sub_sock = self.context.socket(zmq.SUB)
        self.sub_sock.connect("tcp://127.0.0.1:5567") 
        self.sub_sock.setsockopt_string(zmq.SUBSCRIBE, "") 
        
        # Data Buffers (Only accessed by Main Thread)
        self.max_history = 100
        self.data = {
            "snake": {"agree": deque(maxlen=self.max_history), "loss": deque(maxlen=self.max_history), "score": deque(maxlen=self.max_history)},
            "pong":  {"agree": deque(maxlen=self.max_history), "loss": deque(maxlen=self.max_history), "score": deque(maxlen=self.max_history)}
        }
        self.current_vis = None 
        
        # UI Setup
        self._setup_ui()
        
        # Start ZMQ Thread
        threading.Thread(target=self._zmq_listener, daemon=True).start()
        
        # Start Main Thread Update Loop
        self._main_update_loop()

    def _setup_ui(self):
        # 1. CONTROLS
        control_frame = tk.Frame(self.root, bg=BG_COLOR, pady=10)
        control_frame.pack(fill="x", padx=10)
        
        # PROCESS CONTROLS
        tk.Label(control_frame, text="MANUAL:", bg=BG_COLOR, fg=FG_COLOR, font=FONT_HEADER).pack(side="left", padx=10)
        self.btn_server = self._make_proc_btn(control_frame, "Server", "python_server.py")
        self.btn_snake = self._make_proc_btn(control_frame, "Snake", "snake_ui.py")
        self.btn_pong = self._make_proc_btn(control_frame, "Pong", "pong_ui.py")
        
        # EXPERIMENTS
        tk.Label(control_frame, text="| EXPERIMENTS:", bg=BG_COLOR, fg=FG_COLOR, font=FONT_HEADER).pack(side="left", padx=20)
        tk.Button(control_frame, text="STRICT TRANSFER", command=self._run_transfer_exp, bg="#9900cc", fg="white", font=FONT_HEADER).pack(side="left", padx=5)
        
        tk.Label(control_frame, text="| ADMIN:", bg=BG_COLOR, fg=FG_COLOR, font=FONT_HEADER).pack(side="left", padx=20)
        tk.Button(control_frame, text="RESET", command=self._cmd_reset, bg="#cc0000", fg="white", font=FONT_HEADER).pack(side="left", padx=5)
        
        self.lbl_status = tk.Label(control_frame, text="READY", bg="black", fg="#00ff00", font=FONT_MAIN, width=30)
        self.lbl_status.pack(side="right", padx=10)

        # 2. MAIN LAYOUT
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=BG_COLOR)
        main_pane.pack(fill="both", expand=True, padx=10, pady=10)
        
        # LEFT: PERFORMANCE
        left_frame = tk.Frame(main_pane, bg=BG_COLOR)
        main_pane.add(left_frame, width=800)
        
        self.fig_perf, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(6, 8), facecolor=BG_COLOR)
        self.fig_perf.tight_layout(pad=3.0)
        self._setup_perf_plots()
        
        canvas_perf = FigureCanvasTkAgg(self.fig_perf, master=left_frame)
        canvas_perf.draw()
        canvas_perf.get_tk_widget().pack(fill="both", expand=True)
        
        # RIGHT: BRAIN INSPECTOR
        right_frame = tk.Frame(main_pane, bg=BG_COLOR)
        main_pane.add(right_frame)
        
        tk.Label(right_frame, text="BRAIN INSPECTOR (Visual & Motor Cortex)", bg=BG_COLOR, fg=FG_COLOR, font=FONT_HEADER).pack(pady=5)
        
        self.fig_brain, (self.ax_vis, self.ax_motor) = plt.subplots(2, 1, figsize=(5, 6), facecolor=BG_COLOR)
        self.fig_brain.tight_layout(pad=3.0)
        
        # Visual Grid
        self.ax_vis.set_title("Visual Input (10x10)", color=FG_COLOR)
        self.ax_vis.axis('off')
        self.im_vis = self.ax_vis.imshow([[0]*10]*10, cmap='gray', vmin=0, vmax=1)
        
        # Motor Probabilities
        self.ax_motor.set_title("Motor Probabilities", color=FG_COLOR)
        self.ax_motor.set_facecolor(BG_COLOR)
        self.ax_motor.tick_params(colors=FG_COLOR)
        self.ax_motor.set_ylim(0, 1)
        self.bar_motor = self.ax_motor.bar(["UP", "DN", "LF", "RT"], [0,0,0,0], color=ACCENT_COLOR)
        
        # New: System 2 Indicator
        self.ax_sys2 = self.fig_brain.add_axes([0.15, 0.92, 0.7, 0.05]) # Top bar overlay
        self.ax_sys2.axis('off')
        self.ind_sys2 = self.ax_sys2.text(0.5, 0.5, "SYSTEM 2: IDLE", ha='center', va='center', 
                                         color='gray', weight='bold', fontsize=10, 
                                         bbox=dict(facecolor='black', alpha=0.5))
        
        # New: Entropy Text Overlay
        self.txt_entropy = self.fig_brain.text(0.5, 0.02, "Entropy: 0.00", ha='center', color='white', fontsize=10)
        
        canvas_brain = FigureCanvasTkAgg(self.fig_brain, master=right_frame)
        canvas_brain.draw()
        canvas_brain.get_tk_widget().pack(fill="both", expand=True)

        # 3. BOTTOM: LOGS
        log_frame = tk.Frame(self.root, bg=BG_COLOR, height=150)
        log_frame.pack(fill="x", padx=10, pady=5)
        
        self.log_area = scrolledtext.ScrolledText(log_frame, bg="black", fg="#00ff00", font=("Consolas", 8), height=10)
        self.log_area.pack(fill="both", expand=True)

    def _setup_perf_plots(self):
        self.ax1.set_title("Agreement % (Competence)", color=FG_COLOR)
        self.ax1.set_facecolor(BG_COLOR)
        self.ax1.tick_params(colors=FG_COLOR)
        self.ax1.set_ylim(0, 100)
        self.line_snake_agree, = self.ax1.plot([], [], label="Snake", color="#00ff00")
        self.line_pong_agree, = self.ax1.plot([], [], label="Pong", color="#00ccff")
        self.ax1.legend(facecolor=BG_COLOR, labelcolor=FG_COLOR)
        
        self.ax2.set_title("Training Loss", color=FG_COLOR)
        self.ax2.set_facecolor(BG_COLOR)
        self.ax2.tick_params(colors=FG_COLOR)
        self.ax2.set_ylim(0, 3.0)
        self.line_snake_loss, = self.ax2.plot([], [], label="Snake", color="#00ff00", linestyle="--")
        self.line_pong_loss, = self.ax2.plot([], [], label="Pong", color="#00ccff", linestyle="--")
        self.ax2.legend(facecolor=BG_COLOR, labelcolor=FG_COLOR)
        
        # Fix Duplicate Legend
        # self.ax2.legend(facecolor=BG_COLOR, labelcolor=FG_COLOR) 
        
        self.ax3.set_title("Task Score (Survival/Rally)", color=FG_COLOR)
        self.ax3.set_facecolor(BG_COLOR)
        self.ax3.tick_params(colors=FG_COLOR)
        self.line_snake_score, = self.ax3.plot([], [], label="Snake", color="#00ff00")
        self.line_pong_score, = self.ax3.plot([], [], label="Pong", color="#00ccff")
        self.ax3.legend(facecolor=BG_COLOR, labelcolor=FG_COLOR)

    def _make_proc_btn(self, parent, name, script):
        btn = tk.Button(parent, text=f"START {name}", width=15, command=lambda: self._toggle_proc(name, script))
        btn.config(bg="green", fg="white", font=FONT_MAIN)
        btn.pack(side="left", padx=5)
        return btn

    def _toggle_proc(self, name, script, args=[]):
        if self.procs[name] is None:
            try:
                # Capture Stdout/Stderr
                base_dir = os.path.dirname(os.path.abspath(__file__))
                script_path = os.path.join(base_dir, script)
                
                cmd = [sys.executable, "-u", script_path] + args
                
                # Check for active experiment log dir
                log_file = None
                if hasattr(self, 'exp_dir') and self.exp_dir:
                     log_path = os.path.join(self.exp_dir, f"{name.lower()}.log")
                     log_file = open(log_path, "w", buffering=1)
                
                p = subprocess.Popen(
                    cmd, 
                    cwd=base_dir,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1 
                )
                self.procs[name] = p
                
                # Start Thread to read output
                threading.Thread(target=self._read_stream, args=(p.stdout, f"[{name}]", log_file), daemon=True).start()
                threading.Thread(target=self._read_stream, args=(p.stderr, f"[{name} ERR]", log_file), daemon=True).start()
                
                self.log_queue.put(f"Started {name} (PID: {p.pid})")
                
                if hasattr(self, f"btn_{name.lower()}"):
                    btn = getattr(self, f"btn_{name.lower()}")
                    btn.config(text=f"STOP {name}", bg="red")
            except Exception as e:
                self.log_queue.put(f"Error starting {name}: {e}")
        else:
            p = self.procs[name]
            p.terminate()
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
            self.procs[name] = None
            self.log_queue.put(f"Stopped {name}")
            if hasattr(self, f"btn_{name.lower()}"):
                btn = getattr(self, f"btn_{name.lower()}")
                btn.config(text=f"START {name}", bg="green")

    def _read_stream(self, stream, prefix, log_file=None):
        try:
            for line in iter(stream.readline, ''):
                self.log_queue.put(f"{prefix} {line.strip()}")
                if log_file:
                    log_file.write(f"[{time.strftime('%H:%M:%S')}] {line}")
        except Exception:
            pass
        finally:
            stream.close()
            if log_file: log_file.close()

    # --- EXPERIMENT LOGIC ---
    def _run_transfer_exp(self):
        # 1. Setup Environment
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", f"transfer_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.lbl_status.config(text=f"EXP RUNNING: {timestamp}", bg="yellow", fg="black")
        self.log_queue.put(f"=== STARTING STRICT TRANSFER EXPERIMENT ({timestamp}) ===")
        self.log_queue.put(f"Logs: {self.exp_dir}")

        # 2. Start Server (Strict Mode)
        self._toggle_proc("Server", "python_server.py", ["--no-teacher", "--freeze-pong"])
        
        # 3. Start Snake (Phase 1)
        self.root.after(2000, lambda: self._toggle_proc("Snake", "snake_ui.py"))
        
        # 4. Schedule Switch to Pong (Phase 2)
        self.root.after(15000, self._exp_switch_to_pong)
        
    def _exp_switch_to_pong(self):
        self.log_queue.put("=== EXPERIMENT PHASE 2: SWITCHING TO PONG ===")
        # Kill Snake
        if self.procs["Snake"]:
            self._toggle_proc("Snake", "snake_ui.py")
        
        # Start Pong
        self.root.after(2000, lambda: self._toggle_proc("Pong", "pong_ui.py"))
        
        # Schedule End
        self.root.after(30000, self._exp_end)

    def _exp_end(self):
        self.log_queue.put("=== EXPERIMENT COMPLETE ===")
        # Kill Pong
        if self.procs["Pong"]:
            self._toggle_proc("Pong", "pong_ui.py")
        # Kill Server
        if self.procs["Server"]:
            self._toggle_proc("Server", "python_server.py")
        
        self.lbl_status.config(text="EXP COMPLETE", bg="green", fg="white")
        self.exp_dir = None # Reset logging

    def _cmd_sleep(self):
        self.push_sock.send_json({"type": "admin", "cmd": "force_sleep"})
        self.log_queue.put("Sent FORCE SLEEP command")

    def _cmd_reset(self):
        self.push_sock.send_json({"type": "admin", "cmd": "reset_memory"})
        self.log_queue.put("Sent RESET MEMORY command")
        
    def _cmd_export_log(self):
        filename = f"run_log_{int(time.time())}.txt"
        try:
            # Section 1: Console
            console_content = self.log_area.get("1.0", "end")
            
            # Section 2: Brain Events
            events_content = "\n".join(self.event_log)
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write("=== SECTION 1: SYSTEM CONSOLE ===\n")
                f.write(console_content)
                f.write("\n\n=== SECTION 2: BRAIN EVENT STREAM (Decisions & Interventions) ===\n")
                f.write(events_content)
                
            messagebox.showinfo("Export Log", f"Log saved to {filename}\n(Includes {len(self.event_log)} brain events)")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _zmq_listener(self):
        while self.running:
            try:
                if self.sub_sock.poll(100):
                    msg = self.sub_sock.recv_string()
                    self.zmq_queue.put(msg) # Just pass to Main Thread
            except Exception as e:
                print(e)
                
    def _main_update_loop(self):
        # 1. Consume LOGS (Safe limit per tick)
        logs_processed = 0
        while not self.log_queue.empty() and logs_processed < 50:
            msg = self.log_queue.get_nowait()
            self.log_area.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
            logs_processed += 1
        if logs_processed > 0:
            self.log_area.see("end")
            
        # 2. Consume DATA (Safe limit)
        msgs_processed = 0
        vis_updated = False
        perf_updated = False
        
        while not self.zmq_queue.empty() and msgs_processed < 50:
            try:
                msg = self.zmq_queue.get_nowait()
                if msg.startswith("STATS:"):
                    payload = json.loads(msg[6:])
                    game = payload["game"]
                    if game in self.data:
                        self.data[game]["agree"].append(payload["agree_pct"])
                        self.data[game]["loss"].append(payload["loss"])
                        self.data[game]["score"].append(payload.get("score", 0)) # Assuming server sends score OR we track steps
                        perf_updated = True
                elif msg.startswith("VIS:"):
                    payload = json.loads(msg[4:])
                    self.current_vis = payload
                    vis_updated = True
                    
                    # LOG EVENT
                    ts = time.strftime('%H:%M:%S')
                    task = payload.get('task', 'UNK')
                    t_act = payload['teacher']
                    s_act = payload['student']
                    agree = payload['agreed']
                    probs = [f"{p:.2f}" for p in payload['probs']]
                    
                    actions = ["UP", "DN", "LF", "RT"]
                    t_str = actions[t_act] if 0 <= t_act < 4 else f"UNK({t_act})"
                    s_str = actions[s_act] if 0 <= s_act < 4 else f"UNK({s_act})"
                    
                    status = "AGREE" if agree else "INTERVENE"
                    extra = ""
                    
                    if payload.get("veto"):
                        status = "VETO"
                        extra = f" | {payload.get('veto_log')}"
                    
                    if payload.get("vsa_rescue"):
                        extra += " | [VSA RESCUE]"
                    
                    log_line = f"[{ts}] {task.upper()} | {status} | Teacher: {t_str} | Student: {s_str} {probs} (H={payload.get('entropy',0):.2f}){extra}"
                    self.event_log.append(log_line)
                    
                msgs_processed += 1
            except Exception:
                pass
                
        # 3. Update Plots (If needed)
        # Note: updating too fast freezes UI. Throttle updates.
        
        if perf_updated:
            self._update_perf_plots()
            
        if vis_updated and self.current_vis:
            self._update_vis_plot()
            
        # Schedule next tick (50ms = 20fps)
        self.root.after(50, self._main_update_loop)

    def _update_perf_plots(self):
        # Using list() to be thread-safe copy if needed, though we are now single threaded here.
        self.line_snake_agree.set_data(range(len(self.data["snake"]["agree"])), self.data["snake"]["agree"])
        self.line_pong_agree.set_data(range(len(self.data["pong"]["agree"])), self.data["pong"]["agree"])
        self.line_snake_loss.set_data(range(len(self.data["snake"]["loss"])), self.data["snake"]["loss"])
        self.line_pong_loss.set_data(range(len(self.data["pong"]["loss"])), self.data["pong"]["loss"])
        self.line_snake_score.set_data(range(len(self.data["snake"]["score"])), self.data["snake"]["score"])
        self.line_pong_score.set_data(range(len(self.data["pong"]["score"])), self.data["pong"]["score"])
        
        max_len = max(len(self.data["snake"]["agree"]), len(self.data["pong"]["agree"]), 1)
        self.ax1.set_xlim(0, max_len)
        self.ax2.set_xlim(0, max_len)
        self.ax3.set_xlim(0, max_len) # Autoscale limit?
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.fig_perf.canvas.draw_idle()

    def _update_vis_plot(self):
        # Heatmap
        self.im_vis.set_data(self.current_vis["grid"])
        # Bar Chart
        probs = self.current_vis["probs"]
        if len(probs) < 4: probs += [0] * (4-len(probs))
        for rect, h in zip(self.bar_motor.patches, probs):
            rect.set_height(h)
        
        agreed = self.current_vis.get("agreed", False)
        veto = self.current_vis.get("veto", False)
        
        # Visualize VSA Rescue (Gated)
        vsa_rescue = self.current_vis.get("vsa_rescue", False)
        entropy = self.current_vis.get("entropy", 0.0)

        # Update System 2 Indicator
        if veto:
            self.ind_sys2.set_text(f"SYSTEM 2 VETO: {self.current_vis.get('veto_log', 'ACTION')}")
            self.ind_sys2.set_bbox(dict(facecolor='red', alpha=0.8))
            self.ind_sys2.set_color('white')
            title = "VETO ACTIVE"
            color = "#ff0000"
        elif vsa_rescue:
            self.ind_sys2.set_text("SYSTEM 2: HELPING")
            self.ind_sys2.set_bbox(dict(facecolor='#00ccff', alpha=0.5))
            self.ind_sys2.set_color('white')
            title = "RESCUE ACTIVE"
            color = "#00ccff"
        elif agreed:
            self.ind_sys2.set_text("SYSTEM 2: IDLE")
            self.ind_sys2.set_bbox(dict(facecolor='black', alpha=0.5))
            self.ind_sys2.set_color('gray')
            title = "AGREEMENT"
            color = "#00ff00"
        else:
            self.ind_sys2.set_text("SYSTEM 2: IDLE")
            self.ind_sys2.set_bbox(dict(facecolor='black', alpha=0.5))
            self.ind_sys2.set_color('gray')
            title = "INTERVENTION"
            color = "#ffa500" # Orange for teacher intervention
            
        self.ax_motor.set_title(f"Motor Probs : {title}", color=color)
        

        
        if vsa_rescue:
             self.ax_motor.set_xlabel(f"*** VSA RESCUE (H={entropy:.2f}) ***", color='#00ccff', fontsize=10, weight='bold')
        elif entropy > 0.6:
             self.ax_motor.set_xlabel(f"High Uncertainty (H={entropy:.2f})", color='yellow', fontsize=8)
        else:
             self.ax_motor.set_xlabel(f"Confident (H={entropy:.2f})", color='gray', fontsize=8)
        
        # Update Overlay Text
        self.txt_entropy.set_text(f"Entropy: {entropy:.2f}")
        if entropy > 0.6:
            self.txt_entropy.set_color('yellow' if not vsa_rescue else '#00ccff')
        else:
            self.txt_entropy.set_color('white')
        
        self.fig_brain.canvas.draw_idle()

    def on_close(self):
        self.running = False
        for name, p in self.procs.items():
            if p: p.terminate()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
