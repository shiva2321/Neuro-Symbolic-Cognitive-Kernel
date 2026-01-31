"""
NSCK Maze Game UI
Simple tkinter UI for the Maze game environment.
"""
import tkinter as tk
from tkinter import ttk
import zmq
import json
import time
import uuid
import random
import sys
import os
import base64

import numpy as np
import cv2

# Import maze game from parent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maze_game import MazeGame, Cell

# --- CONFIG ---
CELL_SIZE = 30
BG_COLOR = "#2e2e2e"
WALL_COLOR = "#444444"
EMPTY_COLOR = "#1a1a1a"
PLAYER_COLOR = "#00ff00"
EXIT_COLOR = "#ff0000"
VISITED_COLOR = "#0a3a0a"
FG_COLOR = "#ffffff"

class MazeUI:
    """UI for Maze game with ZMQ communication to brain server."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("NSCK Maze Game")
        self.root.configure(bg=BG_COLOR)
        
        # Game instance
        self.game = MazeGame(width=15, height=15, wall_density=0.2)
        
        # Session tracking
        self.session_id = str(uuid.uuid4())[:8]
        self.step_count = 0
        self.score = 0
        
        # ZMQ Setup
        self.context = zmq.Context()
        self.push_sock = self.context.socket(zmq.PUSH)
        self.push_sock.connect("tcp://127.0.0.1:5565")
        
        self.sub_sock = self.context.socket(zmq.SUB)
        self.sub_sock.connect("tcp://127.0.0.1:5566")
        self.sub_sock.setsockopt_string(zmq.SUBSCRIBE, "MAZE:")
        
        # Setup UI
        self._setup_ui()
        
        # Start game loop
        self._game_loop()
    
    def _setup_ui(self):
        """Create the UI elements."""
        # Top frame - info
        top_frame = tk.Frame(self.root, bg=BG_COLOR)
        top_frame.pack(fill="x", padx=10, pady=5)
        
        self.lbl_score = tk.Label(top_frame, text="Score: 0", 
                                   bg=BG_COLOR, fg="#00ff00", 
                                   font=("Consolas", 14, "bold"))
        self.lbl_score.pack(side="left")
        
        self.lbl_steps = tk.Label(top_frame, text="Steps: 0",
                                   bg=BG_COLOR, fg="#cccccc",
                                   font=("Consolas", 12))
        self.lbl_steps.pack(side="left", padx=20)
        
        self.lbl_session = tk.Label(top_frame, text=f"Session: {self.session_id}",
                                     bg=BG_COLOR, fg="#666666",
                                     font=("Consolas", 10))
        self.lbl_session.pack(side="right")
        
        # Canvas for maze
        canvas_width = self.game.width * CELL_SIZE
        canvas_height = self.game.height * CELL_SIZE
        
        self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height,
                                 bg=EMPTY_COLOR, highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)
        
        # Bottom frame - controls
        bottom_frame = tk.Frame(self.root, bg=BG_COLOR)
        bottom_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Button(bottom_frame, text="NEW MAZE", command=self._reset_game,
                  bg="#0066cc", fg="white", font=("Consolas", 10)).pack(side="left", padx=5)
        
        self.lbl_status = tk.Label(bottom_frame, text="Use Arrow Keys or let Brain play",
                                    bg=BG_COLOR, fg="#888888", font=("Consolas", 9))
        self.lbl_status.pack(side="right")
        
        # Bind keys
        self.root.bind("<Up>", lambda e: self._human_move("UP"))
        self.root.bind("<Down>", lambda e: self._human_move("DOWN"))
        self.root.bind("<Left>", lambda e: self._human_move("LEFT"))
        self.root.bind("<Right>", lambda e: self._human_move("RIGHT"))
        
        # Initial render
        self._render()
    
    def _render(self):
        """Render the maze on canvas."""
        self.canvas.delete("all")
        
        if self.game.state is None:
            return
        
        state = self.game.state
        
        for y in range(state.height):
            for x in range(state.width):
                x0 = x * CELL_SIZE
                y0 = y * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                
                # Determine cell color
                if (x, y) == state.player_pos:
                    color = PLAYER_COLOR
                elif (x, y) == state.exit_pos:
                    color = EXIT_COLOR
                elif state.maze[y][x] == Cell.WALL.value:
                    color = WALL_COLOR
                elif (x, y) in state.visited:
                    color = VISITED_COLOR
                else:
                    color = EMPTY_COLOR
                
                self.canvas.create_rectangle(x0, y0, x1, y1, 
                                              fill=color, outline="#333333")
        
        # Draw player symbol
        px, py = state.player_pos
        cx = px * CELL_SIZE + CELL_SIZE // 2
        cy = py * CELL_SIZE + CELL_SIZE // 2
        r = CELL_SIZE // 3
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, fill=PLAYER_COLOR, outline="white")
        
        # Draw exit symbol
        ex, ey = state.exit_pos
        cx = ex * CELL_SIZE + CELL_SIZE // 2
        cy = ey * CELL_SIZE + CELL_SIZE // 2
        self.canvas.create_text(cx, cy, text="EXIT", fill="white", 
                                 font=("Consolas", 8, "bold"))
    
    def _render_for_brain(self):
        """
        Render the maze as a 10x10 grayscale image for the SNN brain.
        
        ALIGNED WITH SNAKE ENCODING for transfer learning:
        - Background/Empty: 0 (black) - same as Snake background
        - Target (Exit): 128 (gray) - same as Snake food!
        - Danger (Walls): 255 (white) - same as Snake body (avoid!)
        - Player: Not shown (brain controls player, doesn't need to see itself)
        
        This mapping enables the Snake-trained visual cortex to interpret:
        - Walls as "dangerous" (like snake body) -> avoid
        - Exit as "target" (like food) -> move toward
        
        Returns:
            Base64 encoded PNG image string, or empty string if no state.
        """
        if self.game.state is None:
            return ""
        
        state = self.game.state
        
        # Create full-resolution image first
        full_img = np.zeros((state.height, state.width), dtype=np.uint8)
        
        for y in range(state.height):
            for x in range(state.width):
                if state.maze[y][x] == Cell.WALL.value:
                    full_img[y, x] = 255  # Wall = DANGER (like snake body)
                # Empty and visited cells stay 0 (background)
        
        # Mark exit as TARGET (like snake food)
        ex, ey = state.exit_pos
        full_img[ey, ex] = 128
        
        # Resize to 10x10 for SNN
        brain_img = cv2.resize(full_img, (10, 10), interpolation=cv2.INTER_AREA)
        
        # Encode as PNG and then base64
        _, buffer = cv2.imencode('.png', brain_img)
        img_bytes = buffer.tobytes()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def _human_move(self, action):
        """Handle human input."""
        self._execute_move(f"ACTION_{action}", is_human=True)
    
    def _execute_move(self, action, is_human=False):
        """Execute a move and update state."""
        if self.game.state is None or self.game.state.done:
            return
        
        _, reward, done = self.game.step(action)
        self.step_count += 1
        
        if done:
            self.score = self.game.state.score
            self.lbl_status.config(text=f"Goal reached! Score: {self.score} (Auto-restart in 1s...)", fg="#00ff00")
            # AUTO-RESTART: New maze after 1 second delay
            self.root.after(1000, self._reset_game)
        
        # Update labels
        self.lbl_score.config(text=f"Score: {self.game.state.score}")
        self.lbl_steps.config(text=f"Steps: {self.step_count}")
        
        self._render()
    
    def _reset_game(self):
        """Reset to new maze."""
        self.game.reset()
        self.session_id = str(uuid.uuid4())[:8]
        self.step_count = 0
        self.score = 0
        
        self.lbl_session.config(text=f"Session: {self.session_id}")
        self.lbl_score.config(text="Score: 0")
        self.lbl_steps.config(text="Steps: 0")
        self.lbl_status.config(text="New maze! Use Arrow Keys or let Brain play", fg="#888888")
        
        self._render()
    
    def _game_loop(self):
        """Main game loop - send state to brain and receive commands."""
        if self.game.state and not self.game.state.done:
            # Get state for brain
            state = self.game.get_state_dict()
            state["game"] = "maze"
            state["session_id"] = self.session_id
            state["step"] = self.step_count
            
            # Add visual input for brain (10x10 grayscale image)
            state["image"] = self._render_for_brain()
            
            # Send to server
            self.push_sock.send_json(state)
            
            # Check for brain response
            try:
                if self.sub_sock.poll(50):
                    msg = self.sub_sock.recv_string()
                    if msg.startswith("MAZE:"):
                        cmd = msg[5:]
                        self._execute_move(f"ACTION_{cmd}", is_human=False)
                        self.lbl_status.config(text=f"Brain: {cmd}", fg="#00ccff")
            except zmq.ZMQError:
                pass
        
        # Schedule next loop
        self.root.after(100, self._game_loop)


def main():
    root = tk.Tk()
    app = MazeUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
