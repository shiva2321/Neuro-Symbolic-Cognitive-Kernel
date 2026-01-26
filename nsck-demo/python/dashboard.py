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
        
        # ZMQ
        self.context = zmq.Context()
        self.push_sock = self.context.socket(zmq.PUSH)
        self.push_sock.connect("tcp://127.0.0.1:5555") 
        
        self.sub_sock = self.context.socket(zmq.SUB)
        self.sub_sock.connect("tcp://127.0.0.1:5557") 
        self.sub_sock.setsockopt_string(zmq.SUBSCRIBE, "") 
        
        # Data Buffers (Only accessed by Main Thread)
        self.max_history = 100
        self.data = {
            "snake": {"agree": deque(maxlen=self.max_history), "loss": deque(maxlen=self.max_history)},
            "pong":  {"agree": deque(maxlen=self.max_history), "loss": deque(maxlen=self.max_history)}
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
        
        tk.Label(control_frame, text="CONTROLS:", bg=BG_COLOR, fg=FG_COLOR, font=FONT_HEADER).pack(side="left", padx=10)
        
        self.btn_server = self._make_proc_btn(control_frame, "Server", "python_server.py")
        self.btn_snake = self._make_proc_btn(control_frame, "Snake", "snake_ui.py")
        self.btn_pong = self._make_proc_btn(control_frame, "Pong", "pong_ui.py")
        
        tk.Label(control_frame, text="| ADMIN:", bg=BG_COLOR, fg=FG_COLOR, font=FONT_HEADER).pack(side="left", padx=20)
        tk.Button(control_frame, text="FORCE SLEEP", command=self._cmd_sleep, bg="#ff9900", fg="black", font=FONT_HEADER).pack(side="left", padx=5)
        tk.Button(control_frame, text="RESET MEMORY", command=self._cmd_reset, bg="#cc0000", fg="white", font=FONT_HEADER).pack(side="left", padx=5)
        
        tk.Label(control_frame, text="| LOGS:", bg=BG_COLOR, fg=FG_COLOR, font=FONT_HEADER).pack(side="left", padx=20)
        tk.Button(control_frame, text="EXPORT LOG", command=self._cmd_export_log, bg="blue", fg="white", font=FONT_HEADER).pack(side="left", padx=5)

        # 2. MAIN LAYOUT
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=BG_COLOR)
        main_pane.pack(fill="both", expand=True, padx=10, pady=10)
        
        # LEFT: PERFORMANCE
        left_frame = tk.Frame(main_pane, bg=BG_COLOR)
        main_pane.add(left_frame, width=800)
        
        self.fig_perf, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 6), facecolor=BG_COLOR)
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

    def _make_proc_btn(self, parent, name, script):
        btn = tk.Button(parent, text=f"START {name}", width=15, command=lambda: self._toggle_proc(name, script))
        btn.config(bg="green", fg="white", font=FONT_MAIN)
        btn.pack(side="left", padx=5)
        return btn

    def _toggle_proc(self, name, script):
        if self.procs[name] is None:
            try:
                # Capture Stdout/Stderr
                # Resolve script path relative to this dashboard file
                base_dir = os.path.dirname(os.path.abspath(__file__))
                script_path = os.path.join(base_dir, script)
                
                cmd = [sys.executable, script_path]
                
                # Run in the script's directory so it finds its assets (weights, etc)
                p = subprocess.Popen(
                    cmd, 
                    cwd=base_dir,
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1 # Line buffered
                )
                self.procs[name] = p
                
                # Start Thread to read output
                threading.Thread(target=self._read_stream, args=(p.stdout, f"[{name}]"), daemon=True).start()
                threading.Thread(target=self._read_stream, args=(p.stderr, f"[{name} ERR]"), daemon=True).start()
                
                self.log_queue.put(f"Started {name} (PID: {p.pid})")
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
            btn = getattr(self, f"btn_{name.lower()}")
            btn.config(text=f"START {name}", bg="green")

    def _read_stream(self, stream, prefix):
        try:
            for line in iter(stream.readline, ''):
                self.log_queue.put(f"{prefix} {line.strip()}")
        except Exception:
            pass
        finally:
            stream.close()

    def _cmd_sleep(self):
        self.push_sock.send_json({"type": "admin", "cmd": "force_sleep"})
        self.log_queue.put("Sent FORCE SLEEP command")

    def _cmd_reset(self):
        self.push_sock.send_json({"type": "admin", "cmd": "reset_memory"})
        self.log_queue.put("Sent RESET MEMORY command")
        
    def _cmd_export_log(self):
        filename = f"run_log_{int(time.time())}.txt"
        try:
            content = self.log_area.get("1.0", "end")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Export Log", f"Log saved to {filename}")
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
                        perf_updated = True
                elif msg.startswith("VIS:"):
                    self.current_vis = json.loads(msg[4:])
                    vis_updated = True
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
        
        max_len = max(len(self.data["snake"]["agree"]), len(self.data["pong"]["agree"]), 1)
        self.ax1.set_xlim(0, max_len)
        self.ax2.set_xlim(0, max_len)
        self.fig_perf.canvas.draw_idle()

    def _update_vis_plot(self):
        # Heatmap
        self.im_vis.set_data(self.current_vis["grid"])
        # Bar Chart
        probs = self.current_vis["probs"]
        if len(probs) < 4: probs += [0] * (4-len(probs))
        for rect, h in zip(self.bar_motor.patches, probs):
            rect.set_height(h)
        
        agreed = self.current_vis["agreed"]
        title = "AGREEMENT" if agreed else "INTERVENTION!"
        color = "#00ff00" if agreed else "#ff0000"
        self.ax_motor.set_title(f"Motor Probs : {title}", color=color)
        
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
