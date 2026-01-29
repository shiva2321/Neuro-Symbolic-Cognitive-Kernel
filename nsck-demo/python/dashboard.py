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
import base64
import io
from PIL import Image, ImageDraw, ImageOps

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
        self.procs = {"Server": None, "Snake": None, "Pong": None, "Maze": None}
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
            "snake": {"agree": deque(maxlen=self.max_history), "loss": deque(maxlen=self.max_history), "score": deque(maxlen=self.max_history), "reward": deque(maxlen=self.max_history)},
            "pong":  {"agree": deque(maxlen=self.max_history), "loss": deque(maxlen=self.max_history), "score": deque(maxlen=self.max_history), "reward": deque(maxlen=self.max_history)},
            "maze":  {"agree": deque(maxlen=self.max_history), "loss": deque(maxlen=self.max_history), "score": deque(maxlen=self.max_history), "reward": deque(maxlen=self.max_history)},
            "char":  {"agree": deque(maxlen=self.max_history), "loss": deque(maxlen=self.max_history)} 
        }
        self.vis_data = {"snake": None, "pong": None, "maze": None, "char": None}
        
        # Cognitive state (for explanations)
        self.cognitive_state = {"explanation": "", "mode": "exploit", "confidence": 0.5}
        
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
        self.btn_maze = self._make_proc_btn(control_frame, "Maze", "maze_ui.py")
        
        # EXPERIMENTS
        tk.Label(control_frame, text="| EXPERIMENTS:", bg=BG_COLOR, fg=FG_COLOR, font=FONT_HEADER).pack(side="left", padx=20)
        tk.Button(control_frame, text="STRICT TRANSFER", command=self._run_transfer_exp, bg="#9900cc", fg="white", font=FONT_HEADER).pack(side="left", padx=5)
        tk.Button(control_frame, text="SNAKE→MAZE", command=self._run_maze_transfer, bg="#ff6600", fg="white", font=FONT_HEADER).pack(side="left", padx=5)
        
        
        tk.Label(control_frame, text="| ADMIN:", bg=BG_COLOR, fg=FG_COLOR, font=FONT_HEADER).pack(side="left", padx=20)
        self.btn_teacher = tk.Button(control_frame, text="TEACHER: ON", command=self._cmd_toggle_teacher, bg="#00aa00", fg="white", font=FONT_HEADER)
        self.btn_teacher.pack(side="left", padx=5)
        
        tk.Button(control_frame, text="EXPORT LOG", command=self._cmd_export_log, bg="#666666", fg="white", font=FONT_HEADER).pack(side="left", padx=5)
        tk.Button(control_frame, text="RESET MEM", command=self._cmd_reset, bg="#cc0000", fg="white", font=FONT_HEADER).pack(side="left", padx=5)
        tk.Button(control_frame, text="FULL RESET", command=self._cmd_full_reset, bg="#990000", fg="white", font=FONT_HEADER).pack(side="left", padx=5)
        
        # SLEEP CONTROLS
        tk.Label(control_frame, text="| SLEEP:", bg=BG_COLOR, fg=FG_COLOR, font=FONT_HEADER).pack(side="left", padx=10)
        tk.Button(control_frame, text="💤 SLEEP", command=self._cmd_universal_sleep, bg="#4400aa", fg="white", font=FONT_HEADER).pack(side="left", padx=5)
        
        self.lbl_status = tk.Label(control_frame, text="READY", bg="black", fg="#00ff00", font=FONT_MAIN, width=30)
        self.lbl_status.pack(side="right", padx=10)

        # 2. MAIN LAYOUT
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=BG_COLOR)
        main_pane.pack(fill="both", expand=True, padx=10, pady=10)
        
        # LEFT: NOTEBOOK (Performance + Letter Lab)
        left_book = ttk.Notebook(main_pane)
        main_pane.add(left_book, width=800)
        
        # TAB 1: PERF
        perf_frame = tk.Frame(left_book, bg=BG_COLOR)
        left_book.add(perf_frame, text="System Monitor")
        
        self.fig_perf, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(6, 8), facecolor=BG_COLOR)
        self.fig_perf.tight_layout(pad=3.0)
        self._setup_perf_plots()
        
        canvas_perf = FigureCanvasTkAgg(self.fig_perf, master=perf_frame)
        canvas_perf.draw()
        canvas_perf.get_tk_widget().pack(fill="both", expand=True)
        
        # TAB 2: COGNITIVE PANEL
        self._setup_cognitive_panel(left_book)
        
        # TAB 3: LETTER LAB
        self._setup_letter_lab(left_book)
        
        # RIGHT: BRAIN INSPECTOR (SPLIT VIEW)
        right_frame = tk.Frame(main_pane, bg=BG_COLOR)
        main_pane.add(right_frame)
        
        tk.Label(right_frame, text="BRAIN INSPECTOR (Parallel Attention)", bg=BG_COLOR, fg=FG_COLOR, font=FONT_HEADER).pack(pady=5)
        
        # 2x2 Grid: [Snake Vis, Pong Vis]
        #           [Snake Mot, Pong Mot]
        self.fig_brain, self.axs_brain = plt.subplots(2, 2, figsize=(8, 6), facecolor=BG_COLOR)
        self.fig_brain.tight_layout(pad=3.0)
        
        # Unpack Axes
        (self.ax_vis_snake, self.ax_vis_pong), (self.ax_motor_snake, self.ax_motor_pong) = self.axs_brain
        
        # --- SNAKE PANEL ---
        self.ax_vis_snake.set_title("SNAKE (Visual)", color=FG_COLOR)
        self.ax_vis_snake.axis('off')
        self.im_vis_snake = self.ax_vis_snake.imshow([[0]*10]*10, cmap='gray', vmin=0, vmax=1)
        
        self.ax_motor_snake.set_title("SNAKE (Motor)", color=FG_COLOR)
        self.ax_motor_snake.set_facecolor(BG_COLOR)
        self.ax_motor_snake.tick_params(colors=FG_COLOR)
        self.ax_motor_snake.set_ylim(0, 1)
        self.bar_motor_snake = self.ax_motor_snake.bar(["UP", "DN", "LF", "RT"], [0,0,0,0], color="#00ff00")

        # --- PONG PANEL ---
        self.ax_vis_pong.set_title("PONG (Visual)", color=FG_COLOR)
        self.ax_vis_pong.axis('off')
        self.im_vis_pong = self.ax_vis_pong.imshow([[0]*10]*10, cmap='gray', vmin=0, vmax=1)
        
        self.ax_motor_pong.set_title("PONG (Motor)", color=FG_COLOR)
        self.ax_motor_pong.set_facecolor(BG_COLOR)
        self.ax_motor_pong.tick_params(colors=FG_COLOR)
        self.ax_motor_pong.set_ylim(0, 1)
        self.bar_motor_pong = self.ax_motor_pong.bar(["UP", "DN", "LF", "RT"], [0,0,0,0], color="#00ccff")
        
        # --- OVERLAYS ---
        self.txt_entropy_snake = self.fig_brain.text(0.3, 0.02, "H: 0.00", ha='center', color='white', fontsize=10)
        self.txt_entropy_pong = self.fig_brain.text(0.7, 0.02, "H: 0.00", ha='center', color='white', fontsize=10)
        
        # System 2 Indicators (One per brain half)
        self.ind_sys2_snake = self.fig_brain.text(0.3, 0.52, "SYS2: IDLE", ha='center', va='center', 
                                         color='gray', weight='bold', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
        self.ind_sys2_pong = self.fig_brain.text(0.7, 0.52, "SYS2: IDLE", ha='center', va='center', 
                                         color='gray', weight='bold', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))

        canvas_brain = FigureCanvasTkAgg(self.fig_brain, master=right_frame)
        canvas_brain.draw()
        canvas_brain.get_tk_widget().pack(fill="both", expand=True)

        # 3. BOTTOM: LOGS (Enhanced & Organized)
        log_container = tk.Frame(self.root, bg=BG_COLOR)
        log_container.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Notebook for categorized logs
        self.log_notebook = ttk.Notebook(log_container)
        self.log_notebook.pack(fill="both", expand=True)
        
        # TAB 1: System Output
        system_frame = tk.Frame(self.log_notebook, bg=BG_COLOR)
        self.log_notebook.add(system_frame, text=" SYSTEM CONSOLE ")
        
        self.log_area = scrolledtext.ScrolledText(system_frame, bg="#1e1e1e", fg="#cccccc", 
                                                 font=("Consolas", 9), height=18)
        self.log_area.pack(fill="both", expand=True)
        
        # TAB 2: Brain Decision Stream
        brain_log_frame = tk.Frame(self.log_notebook, bg=BG_COLOR)
        self.log_notebook.add(brain_log_frame, text=" BRAIN EVENT STREAM ")
        
        self.brain_log_area = scrolledtext.ScrolledText(brain_log_frame, bg="#1e1e1e", fg="#00ff00", 
                                                       font=("Consolas", 9), height=18)
        self.brain_log_area.pack(fill="both", expand=True)
        
        # SETUP COLOR TAGS
        for area in [self.log_area, self.brain_log_area]:
            area.tag_config("SERVER", foreground="#00e5ff") # Cyan
            area.tag_config("SNAKE", foreground="#00ff00") # Green
            area.tag_config("PONG", foreground="#00ccff") # Light Blue
            area.tag_config("CHAR", foreground="#cc66ff") # Purple
            area.tag_config("ERROR", foreground="#ff3333", font=("Consolas", 9, "bold")) # Red
            area.tag_config("VETO", foreground="white", background="#cc0000") # White on Red
            area.tag_config("AGREE", foreground="#00cc00")
            area.tag_config("INTERVENE", foreground="#ff9900")
            area.tag_config("VSA", foreground="#33ccff", font=("Consolas", 9, "italic"))
            area.tag_config("TIMESTAMP", foreground="#666666")
            area.tag_config("TRANSFER", foreground="#ff6600")
            area.tag_config("EXPLORE", foreground="#ff00ff")

    def _setup_cognitive_panel(self, parent_book):
        """Cognitive reasoning panel - shows explanations, transfer, exploration."""
        frame = tk.Frame(parent_book, bg=BG_COLOR)
        parent_book.add(frame, text="Cognitive Panel")
        
        # Title
        tk.Label(frame, text="COGNITIVE STATE & EXPLANATIONS", 
                 bg=BG_COLOR, fg="#00ccff", font=FONT_HEADER).pack(pady=10)
        
        # Mode Indicator
        mode_frame = tk.Frame(frame, bg=BG_COLOR)
        mode_frame.pack(fill="x", padx=20)
        
        tk.Label(mode_frame, text="MODE:", bg=BG_COLOR, fg=FG_COLOR, font=FONT_MAIN).pack(side="left")
        self.lbl_cog_mode = tk.Label(mode_frame, text="EXPLOIT", bg="green", fg="white", 
                                      font=FONT_HEADER, width=12)
        self.lbl_cog_mode.pack(side="left", padx=10)
        
        tk.Label(mode_frame, text="CONFIDENCE:", bg=BG_COLOR, fg=FG_COLOR, font=FONT_MAIN).pack(side="left", padx=10)
        self.lbl_cog_conf = tk.Label(mode_frame, text="0.50", bg="black", fg="#00ff00", 
                                      font=FONT_HEADER, width=8)
        self.lbl_cog_conf.pack(side="left")
        
        # Transfer Status
        transfer_frame = tk.LabelFrame(frame, text="CROSS-TASK TRANSFER", 
                                        bg=BG_COLOR, fg="#ff6600", font=FONT_HEADER)
        transfer_frame.pack(fill="x", padx=20, pady=10)
        
        self.lbl_transfer_status = tk.Label(transfer_frame, 
                                            text="No transfer active", 
                                            bg=BG_COLOR, fg="gray", font=FONT_MAIN)
        self.lbl_transfer_status.pack(pady=5)
        
        self.txt_transfer_mappings = scrolledtext.ScrolledText(transfer_frame, 
                                                                bg="#1e1e1e", fg="#ff6600",
                                                                font=("Consolas", 8), height=4)
        self.txt_transfer_mappings.pack(fill="x", padx=10, pady=5)
        
        # Explanation Display
        explain_frame = tk.LabelFrame(frame, text="CURRENT EXPLANATION", 
                                       bg=BG_COLOR, fg="#00ccff", font=FONT_HEADER)
        explain_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.txt_explanation = scrolledtext.ScrolledText(explain_frame, 
                                                          bg="#1e1e1e", fg="#00ff00",
                                                          font=("Consolas", 9), height=8)
        self.txt_explanation.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Query Buttons
        query_frame = tk.Frame(explain_frame, bg=BG_COLOR)
        query_frame.pack(fill="x", pady=5)
        
        tk.Button(query_frame, text="WHY?", command=self._query_why, 
                  bg="#0066cc", fg="white", font=FONT_MAIN).pack(side="left", padx=5)
        tk.Button(query_frame, text="WHAT IF UP?", command=lambda: self._query_counterfactual("UP"),
                  bg="#006666", fg="white", font=FONT_MAIN).pack(side="left", padx=5)
        tk.Button(query_frame, text="WHAT IF DOWN?", command=lambda: self._query_counterfactual("DOWN"),
                  bg="#006666", fg="white", font=FONT_MAIN).pack(side="left", padx=5)
        tk.Button(query_frame, text="RULES", command=self._query_rules,
                  bg="#666600", fg="white", font=FONT_MAIN).pack(side="left", padx=5)

    def _setup_letter_lab(self, parent_book):
        frame = tk.Frame(parent_book, bg=BG_COLOR)
        parent_book.add(frame, text="Letter Lab (Handwriting)")
        
        # Grid Layout
        # Top: Controls
        # Center: Canvas
        
        ctrl_frame = tk.Frame(frame, bg=BG_COLOR)
        ctrl_frame.pack(pady=10)
        
        tk.Button(ctrl_frame, text="CLEAR", command=self._clear_canvas, bg="red", fg="white").pack(side="left", padx=5)
        tk.Button(ctrl_frame, text="PREDICT", command=self._predict_char, bg="blue", fg="white", font=FONT_HEADER).pack(side="left", padx=5)
        tk.Button(ctrl_frame, text="TRAIN (HANDWRITING)", command=self._train_handwritten, bg="green", fg="white").pack(side="left", padx=5)
        
        # GPU Toggle
        self.gpu_var = tk.BooleanVar(value=False)
        self.btn_gpu = tk.Checkbutton(ctrl_frame, text="GPU ACCEL", variable=self.gpu_var, 
                                      command=self._cmd_toggle_gpu, 
                                      bg=BG_COLOR, fg="yellow", selectcolor="black", activebackground=BG_COLOR)
        self.btn_gpu.pack(side="left", padx=10)
        
        # Sessions Input
        tk.Label(ctrl_frame, text="SESSIONS:", bg=BG_COLOR, fg=FG_COLOR).pack(side="left", padx=5)
        self.ent_sessions = tk.Entry(ctrl_frame, width=5, bg="black", fg="white", insertbackground="white")
        self.ent_sessions.insert(0, "1")
        self.ent_sessions.pack(side="left", padx=5)
        
        # Canvas
        self.cv_size = 280
        self.cv = tk.Canvas(frame, width=self.cv_size, height=self.cv_size, bg="black", cursor="cross")
        self.cv.pack(pady=10)
        self.cv.bind("<B1-Motion>", self._draw_kv)
        
        # Pillow Image for export (Black bg, white ink)
        self.image1 = Image.new("L", (self.cv_size, self.cv_size), 0)
        self.draw = ImageDraw.Draw(self.image1)
        
        tk.Label(frame, text="Draw a digit (0-9)", bg=BG_COLOR, fg="gray").pack()
        
        self.lbl_pred = tk.Label(frame, text="PREDICTION: ?", bg=BG_COLOR, fg="#00ccff", font=("Consolas", 24, "bold"))
        self.lbl_pred.pack(pady=10)

        # --- NEW SECTION: TYPED TEXT TRAINING ---
        typed_frame = tk.LabelFrame(frame, text="TYPED TEXT TRAINING (Generate Machine Data)", bg=BG_COLOR, fg="yellow", font=("Consolas", 10, "bold"), pady=10)
        typed_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Label(typed_frame, text="Enter text/words to teach the Brain:", bg=BG_COLOR, fg=FG_COLOR).pack(pady=5)
        self.ent_typed_text = tk.Entry(typed_frame, width=50, bg="black", fg="white", insertbackground="white")
        self.ent_typed_text.insert(0, "The quick brown fox jumps over the lazy dog 1234567890")
        self.ent_typed_text.pack(pady=5)
        
        tk.Button(typed_frame, text="TRAIN BRAIN ON THIS TEXT", command=self._train_typed_text, bg="#cc6600", fg="white", font=FONT_HEADER).pack(pady=10)
        
        tk.Label(typed_frame, text="(Generates perfect 10x10 typed characters for fast learning)", bg=BG_COLOR, fg="gray", font=("Consolas", 8)).pack()

    def _draw_kv(self, event):
        x, y = event.x, event.y
        r = 10
        self.cv.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255, outline=255)
        
    def _clear_canvas(self):
        self.cv.delete("all")
        self.image1 = Image.new("L", (self.cv_size, self.cv_size), 0)
        self.draw = ImageDraw.Draw(self.image1)
        self.lbl_pred.config(text="PREDICTION: ?")
        
    def _predict_char(self):
        # 1. Resize to 28x28 (Standard MNIST) or 10x10?
        # Server expects base64 image. Server resizes.
        # We send the 280x280 canvas image.
        
        buf = io.BytesIO()
        self.image1.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        
        payload = {"type": "predict_char", "image": img_b64}
        self.push_sock.send_json(payload)
        self.log_queue.put("[CHAR] Sending Prediction Request...")
        
    def _train_handwritten(self):
        try:
            epochs = int(self.ent_sessions.get())
        except ValueError:
            epochs = 1
            
        payload = {"type": "admin", "cmd": "train_char", "mode": "handwritten", "epochs": epochs}
        self.push_sock.send_json(payload)
        self.log_queue.put(f"[CHAR] Sent HANDWRITTEN TRAINING (EMNIST) (Sessions: {epochs})")

    def _train_typed_text(self):
        try:
            epochs = int(self.ent_sessions.get())
        except ValueError:
            epochs = 1
        
        text = self.ent_typed_text.get()
        if not text:
            messagebox.showwarning("Input Needed", "Please enter some text to train on.")
            return
            
        payload = {"type": "admin", "cmd": "train_char", "mode": "typed", "text": text, "epochs": epochs}
        self.push_sock.send_json(payload)
        self.log_queue.put(f"[CHAR] Sent TYPED TRAINING for: '{text}' (Sessions: {epochs})")

    def _cmd_toggle_gpu(self):
        device = "gpu" if self.gpu_var.get() else "cpu"
        self.push_sock.send_json({"type": "admin", "cmd": "set_device", "device": device})
        self.log_queue.put(f"DEVICE SWITCH REQUESTED: {device.upper()}")

    def _setup_perf_plots(self):
        self.ax1.set_title("Agreement % (Competence)", color=FG_COLOR)
        self.ax1.set_facecolor(BG_COLOR)
        self.ax1.tick_params(colors=FG_COLOR)
        self.ax1.set_ylim(0, 100)
        self.line_snake_agree, = self.ax1.plot([], [], label="Snake", color="#00ff00")
        self.line_pong_agree, = self.ax1.plot([], [], label="Pong", color="#00ccff")
        self.ax1.legend(facecolor=BG_COLOR, labelcolor=FG_COLOR)
        
        self.ax2.set_title("Training Loss (Left) | RL Reward (Right)", color=FG_COLOR)
        self.ax2.set_facecolor(BG_COLOR)
        self.ax2.tick_params(colors=FG_COLOR, axis='y', labelcolor='white')
        self.ax2.set_ylim(0, 3.0)
        
        # Loss Lines
        self.line_snake_loss, = self.ax2.plot([], [], label="Snake Loss", color="#00ff00", linestyle="--")
        self.line_pong_loss, = self.ax2.plot([], [], label="Pong Loss", color="#00ccff", linestyle="--")
        
        # RL Reward Twin Axis
        self.ax2_twin = self.ax2.twinx()
        self.ax2_twin.tick_params(colors=FG_COLOR, axis='y', labelcolor='yellow')
        self.ax2_twin.set_ylim(-12, 12)
        
        # Reward Lines (Solid)
        self.line_snake_reward, = self.ax2_twin.plot([], [], label="Snake R", color="#AAFFAA", linewidth=1, alpha=0.5)
        self.line_pong_reward, = self.ax2_twin.plot([], [], label="Pong R", color="#AACCEE", linewidth=1, alpha=0.5)
        
        # Combined Legend
        lines = [self.line_snake_loss, self.line_pong_loss, self.line_snake_reward, self.line_pong_reward]
        self.ax2.legend(lines, [l.get_label() for l in lines], facecolor=BG_COLOR, labelcolor=FG_COLOR, loc='upper right', fontsize=8)
        
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
                
                # FORCE PYTHON 3.11 (Contains Rust VSA + Torch + ZMQ)
                # The user's default 'sys.executable' maps to 3.14 which is incompatible with PyO3 0.20
                PYTHON_EXE = r"C:\Users\Asta\AppData\Local\Programs\Python\Python311\python.exe"
                cmd = [PYTHON_EXE, "-u", script_path] + args
                
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
    # --- EXPERIMENT LOGIC ---
    def _run_transfer_exp(self):
        # 1. Setup Environment
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", f"transfer_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.lbl_status.config(text=f"EXP CONFIG SET: {timestamp}", bg="yellow", fg="black")
        self.log_queue.put(f"=== STRICT TRANSFER CONFIG APPLIED ({timestamp}) ===")
        self.log_queue.put(f"Logs: {self.exp_dir}")
        self.log_queue.put("NOTE: AUTO-SEQUENCE DISABLED. PLEASE START GAMES MANUALLY.")

        # 2. Configure Server (Strict Mode) only. Do NOT force restart if already running.
        self.push_sock.send_json({"type": "admin", "cmd": "strict_transfer_cfg"})
        # Note: If server is not running, user must start it. 
        # If it is running, this command sets the flags.
    
    def _run_maze_transfer(self):
        """Run Snake→Maze zero-shot transfer experiment."""
        self.log_queue.put("=== SNAKE → MAZE TRANSFER EXPERIMENT ===")
        self.log_queue.put("Applying learned Snake rules to Maze domain...")
        
        # Update transfer status display
        if hasattr(self, 'lbl_transfer_status'):
            self.lbl_transfer_status.config(text="ACTIVE: Snake → Maze", fg="#00ff00")
        
        if hasattr(self, 'txt_transfer_mappings'):
            self.txt_transfer_mappings.delete("1.0", "end")
            mappings = [
                "SNAKE_HEAD ≈ MAZE_PLAYER via AGENT",
                "SNAKE_FOOD ≈ MAZE_EXIT via TARGET",
                "REL_ABOVE ≈ EXIT_ABOVE via TARGET_ABOVE",
                "REL_BELOW ≈ EXIT_BELOW via TARGET_BELOW",
                "ACTION_UP ≈ ACTION_UP via MOVE_UP",
                "ACTION_DOWN ≈ ACTION_DOWN via MOVE_DOWN",
            ]
            self.txt_transfer_mappings.insert("end", "\n".join(mappings))
        
        # Send transfer command to server
        self.push_sock.send_json({
            "type": "admin", 
            "cmd": "transfer_knowledge",
            "source": "snake",
            "target": "maze"
        })
        
        self._append_brain_event("[TRANSFER] Snake → Maze transfer activated")
    
    def _query_why(self):
        """Query the brain for explanation of current action."""
        self.push_sock.send_json({"type": "admin", "cmd": "explain_action"})
        self.log_queue.put("Sent EXPLAIN request...")
        
        # Show pending in explanation box
        if hasattr(self, 'txt_explanation'):
            self.txt_explanation.delete("1.0", "end")
            self.txt_explanation.insert("end", "Querying brain for explanation...")
    
    def _query_counterfactual(self, action):
        """Query 'what if' for alternative action."""
        self.push_sock.send_json({
            "type": "admin", 
            "cmd": "counterfactual",
            "action": action
        })
        self.log_queue.put(f"Sent COUNTERFACTUAL query: What if {action}?")
        
        if hasattr(self, 'txt_explanation'):
            self.txt_explanation.delete("1.0", "end")
            self.txt_explanation.insert("end", f"Simulating: What if ACTION_{action}?...")
    
    def _query_rules(self):
        """Query current learned rules."""
        self.push_sock.send_json({"type": "admin", "cmd": "list_rules"})
        self.log_queue.put("Sent RULES query...")
        
        if hasattr(self, 'txt_explanation'):
            self.txt_explanation.delete("1.0", "end")
            self.txt_explanation.insert("end", "Retrieving learned rules...")
        
    # REMOVED AUTO SEQUENCER logic (_exp_switch_to_pong, _exp_end) as per user request to be manual.

    def _cmd_universal_sleep(self):
        """Universal Sleep: Consolidate ALL memories (Snake, Pong, Maze, Char)."""
        self.push_sock.send_json({"type": "admin", "cmd": "force_sleep"})
        self.log_queue.put("💤 UNIVERSAL SLEEP: Consolidating all memories...")

    def _cmd_reset(self):
        self.push_sock.send_json({"type": "admin", "cmd": "reset_memory"})
        self.log_queue.put("Sent RESET MEMORY command")
    
    def _cmd_full_reset(self):
        """Full brain reset: reinit weights, clear buffer, delete saved model."""
        if messagebox.askyesno("FULL RESET", "This will DELETE all learned knowledge!\n\nThe brain will start from scratch.\n\nAre you sure?"):
            self.push_sock.send_json({"type": "admin", "cmd": "full_reset"})
            self.log_queue.put(">>> FULL BRAIN RESET INITIATED <<<")
        
    def _cmd_toggle_teacher(self):
        self.push_sock.send_json({"type": "admin", "cmd": "toggle_teacher"})
        # Update UI assumption (Optimistic)
        curs = self.btn_teacher.cget("text")
        if "ON" in curs:
            self.btn_teacher.config(text="TEACHER: OFF", bg="#660000")
            self.log_queue.put("TEACHER DISABLED")
        else:
            self.btn_teacher.config(text="TEACHER: ON", bg="#00aa00")
            self.log_queue.put("TEACHER ENABLED")
        
    def _cmd_export_log(self):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"NSCK_Full_Log_{timestamp}.txt"
        try:
            # 1. Gather Content from Widgets
            console_content = self.log_area.get("1.0", "end").strip()
            brain_content = self.brain_log_area.get("1.0", "end").strip()
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write("="*60 + "\n")
                f.write(f" NSCK SYSTEM LOG EXPORT - {time.ctime()}\n")
                f.write("="*60 + "\n\n")
                
                f.write("🛰️ SECTION 1: SYSTEM CONSOLE (Process Outputs)\n")
                f.write("-" * 50 + "\n")
                f.write(console_content if console_content else "(Empty)")
                f.write("\n\n" + "="*60 + "\n\n")
                
                f.write("🧠 SECTION 2: BRAIN EVENT STREAM (Decisions & Internal State)\n")
                f.write("-" * 50 + "\n")
                f.write(brain_content if brain_content else "(Empty)")
                f.write("\n\n" + "="*60 + "\n")
                f.write(" END OF LOG\n")
                f.write("="*60 + "\n")
                
            messagebox.showinfo("Export Success", f"All logs (Console + Brain) saved to:\n{filename}")
            self.log_queue.put(f"LOG EXPORTED: {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save log: {e}")

    def _zmq_listener(self):
        while self.running:
            try:
                if self.sub_sock.poll(100):
                    msg = self.sub_sock.recv_string()
                    self.zmq_queue.put(msg) # Just pass to Main Thread
            except Exception as e:
                print(e)
                
    def _append_log(self, text):
        """Routes system logs to the console tab with color-coding."""
        ts = time.strftime('%H:%M:%S')
        tag = None
        
        # Identify source for coloring
        clean_text = text
        if "[Server]" in text: tag = "SERVER"
        elif "[Snake]" in text: tag = "SNAKE"
        elif "[Pong]" in text: tag = "PONG"
        elif "[CHAR]" in text: tag = "CHAR"
        elif "ERR" in text: tag = "ERROR"
        
        self.log_area.insert("end", f"[{ts}] ", "TIMESTAMP")
        if tag:
            self.log_area.insert("end", text + "\n", tag)
        else:
            self.log_area.insert("end", text + "\n")
            
        self.log_area.see("end")

    def _append_brain_event(self, text):
        """Routes decision events to the Brain tab with semantic coloring."""
        ts = time.strftime('%H:%M:%S')
        tag = None
        
        if "VETO" in text: tag = "VETO"
        elif "AGREE" in text: tag = "AGREE"
        elif "INTERVENE" in text: tag = "INTERVENE"
        elif "VSA" in text: tag = "VSA"
        
        self.brain_log_area.insert("end", f"[{ts}] ", "TIMESTAMP")
        if tag:
            self.brain_log_area.insert("end", text + "\n", tag)
        else:
            self.brain_log_area.insert("end", text + "\n")
            
        self.brain_log_area.see("end")

    def _main_update_loop(self):
        # 1. Consume LOGS (Safe limit per tick)
        logs_processed = 0
        while not self.log_queue.empty() and logs_processed < 50:
            msg = self.log_queue.get_nowait()
            self._append_log(msg)
            logs_processed += 1
            
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
                        self.data[game]["score"].append(payload.get("score", 0)) 
                        self.data[game]["reward"].append(payload.get("reward", 0.0)) # RL Reward
                        perf_updated = True
                elif msg.startswith("VIS:"):
                    payload = json.loads(msg[4:])
                    
                    # ROUTER: Determine which Brain Region to update
                    task = payload.get('task', 'snake') 
                    # Normalize task name
                    if task == 'char_recognition': task = 'char'
                    
                    if task not in self.vis_data: task = 'snake_default'
                    
                    # Store if valid
                    if task in self.vis_data:
                        self.vis_data[task] = payload
                        vis_updated = True
                    
                    # LOG EVENT
                    ts = time.strftime('%H:%M:%S')
                    task = payload.get('task', 'UNK')
                    sid = payload.get('session_id', 'UNK')
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
                    
                    log_line = f"{task.upper()} ({sid}) | {status} | Teacher: {t_str} | Student: {s_str} {probs} (H={payload.get('entropy',0):.2f}){extra}"
                    self.event_log.append(f"[{ts}] {log_line}")
                    self._append_brain_event(log_line)
                    
                msgs_processed += 1
            except Exception:
                pass
                
        # 3. Update Plots (If needed)
        # Note: updating too fast freezes UI. Throttle updates.
        
        if perf_updated:
            self._update_perf_plots()
            
        if vis_updated:
            self._update_vis_plot()
            
        # Schedule next tick (50ms = 20fps)
        self.root.after(50, self._main_update_loop)

    def _update_perf_plots(self):
        # Using list() to be thread-safe copy if needed, though we are now single threaded here.
        x_snake = range(len(self.data["snake"]["agree"]))
        self.line_snake_agree.set_data(x_snake, self.data["snake"]["agree"])
        
        x_pong = range(len(self.data["pong"]["agree"]))
        self.line_pong_agree.set_data(x_pong, self.data["pong"]["agree"])
        
        # Loss Lines (Ax2 Left)
        self.line_snake_loss.set_data(range(len(self.data["snake"]["loss"])), self.data["snake"]["loss"])
        self.line_pong_loss.set_data(range(len(self.data["pong"]["loss"])), self.data["pong"]["loss"])
        
        # Reward Lines (Ax2 Right)
        # Note: reward list might be shorter than others if just added, ensure alignment? 
        # Actually append/append/append usually keeps them synced.
        self.line_snake_reward.set_data(range(len(self.data["snake"]["reward"])), self.data["snake"]["reward"])
        self.line_pong_reward.set_data(range(len(self.data["pong"]["reward"])), self.data["pong"]["reward"])
        
        if len(self.data.get("char", {}).get("loss", [])) > 0:
             # Just debug char loss on title
             self.ax2.set_title(f"Loss | CharLoss: {self.data['char']['loss'][-1]:.4f}", color=FG_COLOR)
        
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
        # Update Snake
        if self.vis_data["snake"]:
            self._update_single_brain(
                self.vis_data["snake"], 
                self.im_vis_snake, self.bar_motor_snake, self.ax_motor_snake, 
                self.ind_sys2_snake, self.txt_entropy_snake, "#00ff00"
            )
            
        # Update Pong
        if self.vis_data["pong"]:
            self._update_single_brain(
                self.vis_data["pong"], 
                self.im_vis_pong, self.bar_motor_pong, self.ax_motor_pong, 
                self.ind_sys2_pong, self.txt_entropy_pong, "#00ccff"
            )
            
        # Update Char (Hijack Snake or Pong? Or just rely on separate view? 
        # Plan said reuse Brain Inspector.
        # Let's hijack Snake temporarily if task is char?
        # Or just show in log.
        if self.vis_data.get("char"):
             d = self.vis_data["char"]
             # Show in Prediction Label
             if hasattr(self, 'lbl_pred'):
                 self.lbl_pred.config(text=f"PREDICTION: {d['prediction']} (H={d['entropy']:.2f})")
             
             # Also visualize on Snake Panel for "What Brain Sees" (10x10)
             self._update_single_brain(
                 d,
                 self.im_vis_snake, self.bar_motor_snake, self.ax_motor_snake,
                 self.ind_sys2_snake, self.txt_entropy_snake, "#ff00ff"
             )
             self.ax_vis_snake.set_title("CHARACTER IN (10x10)", color="#ff00ff")
             self.ax_motor_snake.set_title(f"PREDICTION: {d['prediction']}", color="#ff00ff")
             
             # Hack: Reset bar colors/labels for 10 classes? 
             # Existing bars are 4. Character output is 10.
             # We can't easily show 10 bars on 4-bar plot.
             # Just show top 4?

        self.fig_brain.canvas.draw_idle()

    def _update_single_brain(self, data, im, bars, ax_motor, ind_sys2, txt_entropy, base_color):
        # Heatmap
        im.set_data(data["grid"])
        
        # Bar Chart
        probs = data["probs"]
        if len(probs) < 4: probs += [0] * (4-len(probs))
        for rect, h in zip(bars.patches, probs):
            rect.set_height(h)
        
        agreed = data.get("agreed", False)
        veto = data.get("veto", False)
        vsa_rescue = data.get("vsa_rescue", False)
        entropy = data.get("entropy", 0.0)
        sid = data.get("session_id", "UNK")

        # Update System 2 Indicator
        if veto:
            ind_sys2.set_text(f"VETO: {data.get('veto_log', 'ACTION')}")
            ind_sys2.set_bbox(dict(facecolor='red', alpha=0.8))
            ind_sys2.set_color('white')
            title = f"VETO | {sid}"
            color = "#ff0000"
        elif vsa_rescue:
            ind_sys2.set_text("SYS2: HELPING")
            ind_sys2.set_bbox(dict(facecolor='#00ccff', alpha=0.5))
            ind_sys2.set_color('white')
            title = f"RESCUE | {sid}"
            color = "#00ccff"
        elif agreed:
            ind_sys2.set_text("SYS2: IDLE")
            ind_sys2.set_bbox(dict(facecolor='black', alpha=0.5))
            ind_sys2.set_color('gray')
            title = f"AGREEMENT | {sid}"
            color = base_color
        else:
            ind_sys2.set_text("SYS2: MONITOR")
            ind_sys2.set_bbox(dict(facecolor='black', alpha=0.5))
            ind_sys2.set_color('gray')
            title = f"Teacher Intervention | {sid}"
            color = "#ffa500" 
            
        # USER REQUEST: Explicitly state WHO made the decision
        # If Teacher Active is FALSE -> Brain is always in charge (even if 'agreed' is false)
        teacher_active = data.get("teacher_active", True)
        
        if not teacher_active:
             decision_source = "DECISION: BRAIN (ALONE)"
        elif agreed:
             decision_source = "DECISION: BRAIN (SNN)"
        elif veto:
             decision_source = "DECISION: BRAIN (SYS2)"
        else:
             decision_source = "DECISION: TEACHER (ASSIST)"
             
        # Add source to Title
        ax_motor.set_title(f"{title}\n{decision_source}", color=color, fontsize=8)
        
        if vsa_rescue:
             ax_motor.set_xlabel(f"*** VSA (H={entropy:.2f}) ***", color='#00ccff', fontsize=8, weight='bold')
        elif entropy > 0.6:
             ax_motor.set_xlabel(f"High Uncertainty (H={entropy:.2f})", color='yellow', fontsize=8)
        else:
             ax_motor.set_xlabel(f"Confident (H={entropy:.2f})", color='gray', fontsize=8)
        
        # Update Overlay Text
        txt_entropy.set_text(f"H: {entropy:.2f}")
        if entropy > 0.6:
            txt_entropy.set_color('yellow' if not vsa_rescue else '#00ccff')
        else:
            txt_entropy.set_color('white')

    def on_close(self):
        print("[Dashboard] Closing... Killing subprocesses.")
        self.running = False
        self.log_queue.put("DASHBOARD EXITING...")
        
        for name, p in self.procs.items():
            if p:
                print(f"[Dashboard] Killing {name} (PID: {p.pid})...")
                try:
                    p.kill() # Force kill (SIGKILL/TerminateProcess)
                    p.wait(timeout=1)
                except Exception as e:
                    print(f"Error killing {name}: {e}")
        
        try:
            self.root.destroy()
        except:
            pass
        print("[Dashboard] Bye.")
        sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
