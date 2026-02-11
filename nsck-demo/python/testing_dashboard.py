"""
NSCK Comprehensive Testing Dashboard
=====================================
A complete web-based dashboard for testing all capabilities of the NSCK
cognitive system.  Provides:

* **Chat** – Submit text samples and chat interactively with the system.
* **Game Simulations** – Attach Snake / Pong / Maze simulations
  (sequentially or simultaneously) and watch the system play & learn.
* **System Monitor** – Observe emotions, reasoning traces, self-model
  metrics, metacognition, curiosity, and consciousness in real time.
* **Log Export** – Export every log, metric, reasoning trace, and system
  state to a downloadable text file for examination.

The dashboard is a self-contained Flask application with an embedded
single-page HTML/JS front-end.
"""

import os
import io
import json
import time
import threading
import logging
from datetime import datetime, timezone
from collections import deque
from typing import Dict, Any, List, Optional

from flask import Flask, request, jsonify, send_file, render_template_string

import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from knowledge_integration import KnowledgeIntegration, CognitiveResponse
from multimodal_processor import MultimodalInput
from context_engine import ContextFrame
from simulation import sim_snake, sim_pong
from emotion_system import EmotionSystem
from self_model import SelfModel
from language_module import LanguageModule
from dialogue_manager import DialogueManager
from maze_game import MazeGame
import heapq
from collections import defaultdict

class LearnedPolicy:
    """
    Simple lookup-table learner for dashboard simulations.
    Maps state_hash -> {action: count}.
    """
    def __init__(self):
        self.policy = defaultdict(lambda: defaultdict(int))

    def train(self, state_hash: str, action: str):
        self.policy[state_hash][action] += 1

    def predict(self, state_hash: str) -> Optional[str]:
        if state_hash not in self.policy:
            return None
        # Return action with highest count
        counts = self.policy[state_hash]
        return max(counts, key=counts.get)

    def reset(self):
        """Clear all learned state-action mappings."""
        self.policy.clear()
        print("[POLICY] Dashboard learned policy reset.")

# Global Policy Instance
_dashboard_policy = LearnedPolicy()

# Optional imports – gracefully degrade if not available
try:
    from cognitive_engine import CognitiveEngine
except ImportError:
    CognitiveEngine = None

try:
    from curiosity import CuriosityModule
except Exception:
    CuriosityModule = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nsck_testing_dashboard")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_system: Optional[KnowledgeIntegration] = None
_emotion: Optional[EmotionSystem] = None
_self_model: Optional[SelfModel] = None
_language: Optional[LanguageModule] = None
_dialogue: Optional[DialogueManager] = None

_activity_log: deque = deque(maxlen=2000)
_chat_history: List[Dict[str, str]] = []

# Game simulation state
_game_sessions: Dict[str, Dict[str, Any]] = {}
_game_lock = threading.Lock()
_game_threads: Dict[str, threading.Event] = {}

# Game simulation constants
EXPLORATION_PROBABILITY = 0.1   # Chance of random action in snake
MAZE_EXPLORATION_PROBABILITY = 0.2  # Chance of random action in maze
MIN_POLL_INTERVAL_MS = 200  # Minimum polling interval for game state updates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(category: str, message: str, data: Dict[str, Any] = None):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "category": category,
        "message": message,
    }
    if data:
        entry["data"] = data
    _activity_log.append(entry)
    logger.info("[%s] %s", category, message)


def _get_system() -> KnowledgeIntegration:
    global _system
    if _system is None:
        _system = KnowledgeIntegration()
        _log("system", "KnowledgeIntegration initialised")
    return _system


def _get_emotion() -> EmotionSystem:
    global _emotion
    if _emotion is None:
        _emotion = EmotionSystem()
        _log("system", "EmotionSystem initialised")
    return _emotion


def _get_self_model() -> SelfModel:
    global _self_model
    if _self_model is None:
        _self_model = SelfModel()
        _log("system", "SelfModel initialised")
    return _self_model


def _get_language() -> LanguageModule:
    global _language
    if _language is None:
        _language = LanguageModule()
        _log("system", f"LanguageModule initialised (mock={_language.mock_mode})")
    return _language


def _get_dialogue() -> DialogueManager:
    global _dialogue
    if _dialogue is None:
        _dialogue = DialogueManager(None, _get_language())
        _log("system", "DialogueManager initialised")
    return _dialogue


# ---------------------------------------------------------------------------
# Game simulation helpers
# ---------------------------------------------------------------------------

def _init_snake() -> Dict[str, Any]:
    return {
        "type": "snake",
        "head": (5, 5),
        "body": [(5, 5)],
        "food": (np.random.randint(0, 10), np.random.randint(0, 10)),
        "score": 0,
        "steps": 0,
        "done": False,
        "history": [],
    }


def _init_pong() -> Dict[str, Any]:
    return {
        "type": "pong",
        "p1_y": 12,
        "ball_x": 15,
        "ball_y": 15,
        "ball_dx": -1,
        "ball_dy": 1,
        "score": 0,
        "steps": 0,
        "done": False,
        "history": [],
    }


def _init_maze() -> Dict[str, Any]:
    # Use actual MazeGame for random generation
    game = MazeGame(width=10, height=10)
    # game.reset() is called in __init__
    
    state_dict = game.state.to_dict()
    # Convert list of lists to set of tuples for dashboard logic
    walls = set()
    for w in state_dict["walls"]:
        walls.add(tuple(w))
        
    return {
        "type": "maze",
        "player": state_dict["player_pos"],
        "exit": state_dict["exit_pos"],
        "walls": walls,
        "size": 10,
        "score": 0,
        "steps": 0,
        "done": False,
        "history": [],
    }


def solve_maze_astar(start, goal, walls, width=10, height=10):
    """A* Solver for Maze."""
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            if not path: return "UP" # Should not happen if start!=goal
            
            # Determine first move
            next_step = path[0]
            dx, dy = next_step[0] - start[0], next_step[1] - start[1]
            if dx == 1: return "RIGHT"
            if dx == -1: return "LEFT"
            if dy == 1: return "DOWN"
            if dy == -1: return "UP"
            return "UP"

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                if neighbor in walls: continue
                
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None # No path

def solve_snake_bfs(head, food, body, width=10, height=10):
    """BFS Solver for Snake."""
    queue = [(head, [])]
    visited = {head}
    body_set = set(body) # Includes tail, which might move, but treating as obstacle is safer

    while queue:
        current, path = queue.pop(0)
        
        if current == food:
            if not path: return None
            # Extract first move direction
            next_step = path[0]
            dx, dy = next_step[0] - head[0], next_step[1] - head[1]
            if dx == 1: return "RIGHT"
            if dx == -1: return "LEFT"
            if dy == 1: return "DOWN"
            if dy == -1: return "UP"
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            # Snake wraps? The sim_snake uses modulo, so we should too?
            # Wait, existing sim_snake has toriodal wrap.
            nx, ny = (current[0] + dx) % width, (current[1] + dy) % height
            
            if (nx, ny) not in visited and (nx, ny) not in body_set:
                visited.add((nx, ny))
                new_path = path + [(nx, ny)] if not path else path # optimize? no need path is short
                # Actually we just need the first move.
                # Let's store 'first_move' in queue instead of full path for memory?
                # But here map is small (10x10). Full path is fine.
                if not path:
                    queue.append(((nx, ny), [(nx, ny)]))
                else:
                    queue.append(((nx, ny), path)) # Propagate first move
                    
    return None # No path found

def _choose_action_for_game(game_state: Dict) -> str:
    """Use Teacher (Solver) or Student (Learned Policy) to pick action."""
    game_type = game_state["type"]
    teacher_active = game_state.get("teacher_active", True) # Default to ON
    
    # helper to hash state
    def get_state_hash(gs):
        if gs["type"] == "snake":
            # Relative food position + immediate danger? 
            # Or just head/food relative?
            # For robust learning we need local view. 
            # Simple absolute state:
            return f"snake:{gs['head']}:{gs['food']}"
        elif gs["type"] == "maze":
            return f"maze:{gs['player']}"
        elif gs["type"] == "pong":
            return f"pong:{gs['ball_x']}:{gs['ball_y']}:{gs['p1_y']}"
        return "unknown"

    state_hash = get_state_hash(game_state)
    suggested_action = None

    if teacher_active:
        # --- TEACHER MODE: SOLVE & TRAIN ---
        if game_type == "snake":
            suggested_action = solve_snake_bfs(game_state["head"], game_state["food"], game_state["body"])
            if not suggested_action: # Fallback if no path
                 # naive
                 hx, hy = game_state["head"]
                 fx, fy = game_state["food"]
                 dx, dy = fx - hx, fy - hy
                 if abs(dx) > abs(dy): suggested_action = "RIGHT" if dx > 0 else "LEFT"
                 else: suggested_action = "DOWN" if dy > 0 else "UP"

        elif game_type == "maze":
            # Convert set of walls to list for checking
            walls = game_state["walls"]
            suggested_action = solve_maze_astar(game_state["player"], game_state["exit"], walls)
            if not suggested_action: suggested_action = "UP" # Stuck?

        elif game_type == "pong":
            # Perfect tracking
            ball_y = game_state["ball_y"]
            p1_y = game_state["p1_y"]
            if ball_y < p1_y: suggested_action = "UP"
            elif ball_y > p1_y + 6: suggested_action = "DOWN"
            else: suggested_action = "UP" if np.random.random() < 0.5 else "DOWN"

        # Train Policy
        if suggested_action:
            _dashboard_policy.train(state_hash, suggested_action)
            
        action = suggested_action if suggested_action else "UP"

    else:
        # --- STUDENT MODE: RECALL or FAIL ---
        learned_action = _dashboard_policy.predict(state_hash)
        
        if learned_action:
            action = learned_action
        else:
            # Cold Start / Unknown State -> Random Exploration
            action = np.random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
            
    return action


def _step_game(session_id: str) -> Dict[str, Any]:
    """Advance one game step, returning a summary."""
    with _game_lock:
        gs = _game_sessions.get(session_id)
        if gs is None or gs["done"]:
            return {"error": "session not found or game over"}

        # manual intervention?
        if gs.get("manual_action"):
            action = gs.pop("manual_action")
            # Clear it so we don't repeat
        else:
            action = _choose_action_for_game(gs)
            
        game_type = gs["type"]
        reward = 0.0
        done = False

        if game_type == "snake":
            state_in = {"head": gs["head"], "body": list(gs["body"]),
                        "food": gs["food"]}
            next_state, collision = sim_snake(state_in, action)
            gs["head"] = next_state["head"]
            if gs["head"] == gs["food"]:
                gs["score"] += 1
                gs["food"] = (np.random.randint(0, 10), np.random.randint(0, 10))
                gs["body"].append(gs["head"])
                reward = 1.0
            else:
                if len(gs["body"]) > 1:
                    gs["body"].pop(0)
                gs["body"].append(gs["head"])
            if collision:
                done = True
                reward = -1.0
            gs["steps"] += 1
            if gs["steps"] >= 200:
                done = True

        elif game_type == "pong":
            state_in = {
                "p1_y": gs["p1_y"],
                "ball_x": gs["ball_x"],
                "ball_y": gs["ball_y"],
                "ball_dx": gs["ball_dx"],
                "ball_dy": gs["ball_dy"],
            }
            next_state, miss = sim_pong(state_in, action)
            gs["p1_y"] = next_state["p1_y"]
            gs["ball_x"] = next_state["ball_x"]
            gs["ball_y"] = next_state["ball_y"]
            gs["ball_dx"] = next_state["ball_dx"]
            gs["ball_dy"] = next_state["ball_dy"]
            if gs["ball_y"] <= 0 or gs["ball_y"] >= 29:
                gs["ball_dy"] = -gs["ball_dy"]
            if gs["ball_x"] >= 28:
                gs["ball_dx"] = -gs["ball_dx"]
            if miss:
                reward = -1.0
                gs["ball_x"] = 15
                gs["ball_y"] = 15
                gs["ball_dx"] = -1
                gs["ball_dy"] = np.random.choice([-1, 1])
            else:
                if gs["ball_dx"] < 0 and gs["ball_x"] <= 2:
                    gs["score"] += 1
                    reward = 1.0
                    gs["ball_dx"] = -gs["ball_dx"]
            gs["steps"] += 1
            if gs["steps"] >= 300:
                done = True

        elif game_type == "maze":
            px, py = gs["player"]
            nx, ny = px, py
            if action == "UP":
                ny -= 1
            elif action == "DOWN":
                ny += 1
            elif action == "LEFT":
                nx -= 1
            elif action == "RIGHT":
                nx += 1
            if (nx, ny) not in gs["walls"] and 0 <= nx < gs["size"] and 0 <= ny < gs["size"]:
                gs["player"] = (nx, ny)
            if gs["player"] == gs["exit"]:
                gs["score"] += 1
                done = True
                reward = 1.0
            gs["steps"] += 1
            if gs["steps"] >= 200:
                done = True

        gs["done"] = done

        # Update self-model with result
        sm = _get_self_model()
        success = reward > 0
        sm.update(game_type, 0.5, success, action, reward)

        step_info = {
            "session_id": session_id,
            "game_type": game_type,
            "step": gs["steps"],
            "action": action,
            "reward": reward,
            "score": gs["score"],
            "done": done,
            "reasoning": f"Chose {action} based on target-seeking heuristic + exploration",
        }
        gs["history"].append(step_info)
        _log("game", f"[{game_type}] step={gs['steps']} action={action} reward={reward} score={gs['score']}", step_info)
        return step_info


def _run_game_loop(session_id: str, stop_event: threading.Event, speed: float = 0.3):
    """Background thread that auto-steps a game."""
    while not stop_event.is_set():
        result = _step_game(session_id)
        if result.get("error"):
            break
            
        if result.get("done"):
            # Auto-restart logic for dashboard simulation
            time.sleep(1.0) # Pause for effect
            with _game_lock:
                gs = _game_sessions.get(session_id)
                if gs:
                    # Reset generic state
                    gs["score"] = 0
                    gs["steps"] = 0
                    gs["done"] = False
                    gs["history"] = []
                    
                    # Reset specific game state
                    if gs["type"] == "snake":
                        gs["head"] = (5, 5)
                        gs["body"] = [(5, 5)]
                        gs["food"] = (np.random.randint(0, 10), np.random.randint(0, 10))
                    elif gs["type"] == "pong":
                        gs["score"] = 0 # distinct from generic score?
                        gs["ball_x"] = 15
                        gs["ball_y"] = 15
                    elif gs["type"] == "maze":
                        # Regenerate maze walls/exit
                        new_maze = _init_maze()
                        gs["player"] = new_maze["player"]
                        gs["exit"] = new_maze["exit"]
                        gs["walls"] = new_maze["walls"]
            _log("game", f"Auto-restarting session {session_id}")
            continue
        stop_event.wait(speed)


# ---------------------------------------------------------------------------
# Routes – Dashboard page
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


# ---------------------------------------------------------------------------
# Routes – Chat
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True, silent=True) or {}
    user_msg = data.get("message", "").strip()
    sample_text = data.get("sample_text", "").strip()
    if not user_msg:
        return jsonify({"error": "empty message"}), 400

    _log("chat", f"User: {user_msg}")

    system = _get_system()
    emo = _get_emotion()

    # If there is a sample text, process it through the cognitive pipeline first
    reasoning_trace = []
    disambiguations = []
    confidence = 0.0
    answer = ""

    if sample_text:
        inp = MultimodalInput(text=sample_text)
        resp = system.process_input(inp, task_tag="chat")
        reasoning_trace.extend(resp.reasoning_trace)
        disambiguations = [
            {"concept": d.concept, "meaning": d.meaning,
             "context_domain": d.context_domain,
             "confidence": d.confidence, "explanation": d.explanation}
            for d in resp.disambiguations
        ]

    # Now process the user message itself
    inp2 = MultimodalInput(text=user_msg)
    resp2 = system.process_input(inp2, task_tag="chat")
    answer = resp2.answer
    confidence = resp2.confidence
    reasoning_trace.extend(resp2.reasoning_trace)

    # Also try dialogue manager for a conversational response
    dm = _get_dialogue()
    dialogue_response = dm.process_turn(user_msg)

    # Combine answers
    if dialogue_response and dialogue_response != "I'm not sure what you mean. Can you rephrase?":
        combined = f"{answer}\n\n[Dialogue]: {dialogue_response}"
    else:
        combined = answer

    # Emotion info
    emotion_info = {
        "emotion": emo.current_emotion,
        "valence": round(emo.valence, 3),
        "arousal": round(emo.arousal, 3),
        "blend": emo.get_emotion_blend(),
        "mood": emo.get_mood(),
    }

    entry = {
        "role": "user",
        "content": user_msg,
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
    }
    _chat_history.append(entry)
    assistant_entry = {
        "role": "assistant",
        "content": combined,
        "confidence": confidence,
        "reasoning_trace": reasoning_trace,
        "emotion": emotion_info,
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
    }
    _chat_history.append(assistant_entry)

    _log("chat", f"Assistant: {combined[:120]}...", {
        "confidence": confidence,
        "trace_len": len(reasoning_trace),
    })

    return jsonify({
        "answer": combined,
        "confidence": confidence,
        "reasoning_trace": reasoning_trace,
        "disambiguations": disambiguations,
        "emotion": emotion_info,
        "stats": system.get_statistics(),
    })


@app.route("/api/chat/history")
def api_chat_history():
    return jsonify({"history": _chat_history})


# ---------------------------------------------------------------------------
# Routes – Process (cognitive pipeline)
# ---------------------------------------------------------------------------

@app.route("/api/process", methods=["POST"])
def api_process():
    try:
        system = _get_system()
        text = request.form.get("text", None)
        task_tag = request.form.get("task_tag", "general")
        structured_raw = request.form.get("structured", None)

        structured = None
        if structured_raw:
            try:
                structured = json.loads(structured_raw)
            except json.JSONDecodeError:
                structured = {"raw": structured_raw}

        inp = MultimodalInput(text=text or None, structured=structured)
        _log("input", f"Processing: text={bool(text)}, structured={bool(structured)}, task={task_tag}")
        response = system.process_input(inp, task_tag=task_tag)
        _log("response", f"Generated response (conf={response.confidence:.2f})")

        disamb_data = []
        for d in response.disambiguations:
            disamb_data.append({
                "concept": d.concept,
                "meaning": d.meaning,
                "context_domain": d.context_domain,
                "confidence": d.confidence,
                "explanation": d.explanation,
                "alternatives": d.alternative_meanings,
            })

        return jsonify({
            "answer": response.answer,
            "confidence": response.confidence,
            "reasoning_trace": response.reasoning_trace,
            "recalled_episodes": response.recalled_episodes,
            "disambiguations": disamb_data,
            "learned_facts": response.learned_facts,
            "emotional_context": response.emotional_context,
            "stats": system.get_statistics(),
        })
    except Exception as e:
        logger.exception("Error processing input")
        _log("error", str(e))
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Routes – Game simulations
# ---------------------------------------------------------------------------

@app.route("/api/game/start", methods=["POST"])
def api_game_start():
    data = request.get_json(force=True, silent=True) or {}
    game_type = data.get("game", "snake")
    teacher_active = data.get("teacher_active", True)
    auto_play = data.get("auto_play", False)
    speed = float(data.get("speed", 0.3))

    if game_type == "snake":
        gs = _init_snake()
    elif game_type == "pong":
        gs = _init_pong()
    elif game_type == "maze":
        gs = _init_maze()
    else:
        return jsonify({"error": "unknown game"}), 400

    session_id = f"{game_type}_{int(time.time()*1000)}"
    gs["session_id"] = session_id
    gs["type"] = game_type
    gs["teacher_active"] = teacher_active
    
    with _game_lock:
        _game_sessions[session_id] = gs

    _log("game", f"Started {game_type} session {session_id}")

    if auto_play:
        stop_event = threading.Event()
        _game_threads[session_id] = stop_event
        t = threading.Thread(target=_run_game_loop, args=(session_id, stop_event, speed), daemon=True)
        t.start()
        _log("game", f"Auto-play started for {session_id}")

    return jsonify({"session_id": session_id, "game_type": game_type,
                    "auto_play": auto_play, "state": _serialize_game(gs)})


@app.route("/api/game/step", methods=["POST"])
def api_game_step():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id", "")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    result = _step_game(session_id)
    with _game_lock:
        gs = _game_sessions.get(session_id, {})
    return jsonify({**result, "state": _serialize_game(gs)})


@app.route("/api/game/action", methods=["POST"])
def api_game_action():
    """Handle manual intervention from client."""
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id", "")
    action = data.get("action", "")
    
    if not session_id or not action:
        return jsonify({"error": "session_id and action required"}), 400
        
    with _game_lock:
        gs = _game_sessions.get(session_id)
        if gs:
            gs["manual_action"] = action # Queue for next step
            
    return jsonify({"status": "queued", "action": action})


@app.route("/api/game/stop", methods=["POST"])
def api_game_stop():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id", "")
    if session_id in _game_threads:
        _game_threads[session_id].set()
        del _game_threads[session_id]
        _log("game", f"Stopped auto-play for {session_id}")
    return jsonify({"status": "stopped", "session_id": session_id})


@app.route("/api/game/state")
def api_game_state():
    session_id = request.args.get("session_id", "")
    with _game_lock:
        gs = _game_sessions.get(session_id)
    if gs is None:
        return jsonify({"error": "session not found"}), 404
    return jsonify({"state": _serialize_game(gs),
                    "history": gs.get("history", [])[-20:]})


@app.route("/api/game/sessions")
def api_game_sessions():
    with _game_lock:
        sessions = []
        for sid, gs in _game_sessions.items():
            sessions.append({
                "session_id": sid,
                "game_type": gs["type"],
                "score": gs["score"],
                "steps": gs["steps"],
                "done": gs["done"],
                "auto_play": sid in _game_threads,
            })
    return jsonify({"sessions": sessions})


def _serialize_game(gs: Dict) -> Dict:
    """Create JSON-safe representation of game state."""
    if gs is None:
        return {}
    result = {}
    for k, v in gs.items():
        if k == "walls":
            result[k] = [list(w) for w in v]
        elif k == "history":
            continue  # skip history in state serialization
        elif isinstance(v, (list, tuple)):
            result[k] = list(v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Routes – System monitoring
# ---------------------------------------------------------------------------

@app.route("/api/monitor")
def api_monitor():
    """Full system status snapshot."""
    emo = _get_emotion()
    sm = _get_self_model()

    task_perf = {}
    for task_tag in list(sm.task_stats.keys()):
        stats = sm.get_stats(task_tag)
        task_perf[task_tag] = {
            "attempts": stats["attempts"],
            "successes": stats["successes"],
            "success_rate": stats["successes"] / stats["attempts"] if stats["attempts"] > 0 else 0,
            "trend": sm.get_improvement_trend(task_tag),
            "context_breakdown": sm.get_context_performance(task_tag),
        }

    return jsonify({
        "emotion": {
            "current": emo.current_emotion,
            "valence": round(emo.valence, 4),
            "arousal": round(emo.arousal, 4),
            "intensity": round(emo.emotion_intensity, 4),
            "blend": emo.get_emotion_blend(),
            "mood": emo.get_mood(),
            "history": emo.emotion_history[-30:],
        },
        "self_model": task_perf,
        "system_stats": _get_system().get_statistics(),
    })


@app.route("/api/stats")
def api_stats():
    system = _get_system()
    return jsonify(system.get_statistics())


@app.route("/api/logs")
def api_logs():
    return jsonify({"logs": list(_activity_log)})


@app.route("/api/knowledge")
def api_knowledge():
    system = _get_system()
    entries = []
    for k in system.knowledge:
        entries.append({
            "concept": k.concept,
            "relation": k.relation,
            "target": k.target,
            "confidence": k.confidence,
            "source": k.source,
            "timestamp": k.timestamp,
        })
    return jsonify({"knowledge": entries})


@app.route("/api/context/stats")
def api_context_stats():
    system = _get_system()
    return jsonify(system.context.get_statistics())


# ---------------------------------------------------------------------------
# Routes – Export
# ---------------------------------------------------------------------------

@app.route("/api/export/text")
def export_text():
    """Export everything to a single text file for examination."""
    system = _get_system()
    emo = _get_emotion()
    sm = _get_self_model()
    lines = []

    lines.append("=" * 80)
    lines.append("NSCK COGNITIVE SYSTEM — FULL EXPORT")
    lines.append(f"Exported at: {datetime.now(timezone.utc).isoformat()}Z")
    lines.append("=" * 80)
    lines.append("")

    # System stats
    lines.append("-" * 60)
    lines.append("SYSTEM STATISTICS")
    lines.append("-" * 60)
    stats = system.get_statistics()
    for k, v in stats.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # Emotion state
    lines.append("-" * 60)
    lines.append("EMOTIONAL STATE")
    lines.append("-" * 60)
    lines.append(f"  Current emotion: {emo.current_emotion}")
    lines.append(f"  Valence: {emo.valence:.4f}")
    lines.append(f"  Arousal: {emo.arousal:.4f}")
    lines.append(f"  Intensity: {emo.emotion_intensity:.4f}")
    lines.append(f"  Blend: {json.dumps(emo.get_emotion_blend(), default=str)}")
    lines.append(f"  Mood: {json.dumps(emo.get_mood(), default=str)}")
    lines.append("")

    # Emotion history
    lines.append("-" * 60)
    lines.append("EMOTION HISTORY")
    lines.append("-" * 60)
    for eh in emo.emotion_history:
        lines.append(f"  step={eh['step']} emotion={eh['emotion']} "
                     f"v={eh['valence']} a={eh['arousal']}")
    lines.append("")

    # Self-model performance
    lines.append("-" * 60)
    lines.append("SELF-MODEL PERFORMANCE")
    lines.append("-" * 60)
    for task_tag in list(sm.task_stats.keys()):
        stats = sm.get_stats(task_tag)
        lines.append(f"  Task: {task_tag}")
        lines.append(f"    Attempts: {stats['attempts']}")
        lines.append(f"    Successes: {stats['successes']}")
        sr = stats['successes'] / stats['attempts'] if stats['attempts'] > 0 else 0
        lines.append(f"    Success rate: {sr:.2%}")
        lines.append(f"    Trend: {sm.get_improvement_trend(task_tag)}")
        ctx = sm.get_context_performance(task_tag)
        if ctx:
            lines.append(f"    Context breakdown: {json.dumps(ctx, default=str)}")
    lines.append("")

    # Knowledge base
    lines.append("-" * 60)
    lines.append("KNOWLEDGE BASE")
    lines.append("-" * 60)
    for k in system.knowledge:
        lines.append(f"  [{k.source}] {k.concept} --{k.relation}--> {k.target}  "
                     f"(conf={k.confidence:.2f})")
    lines.append("")

    # Chat history
    lines.append("-" * 60)
    lines.append("CHAT HISTORY")
    lines.append("-" * 60)
    for ch in _chat_history:
        ts = ch.get("timestamp", "")
        role = ch.get("role", "?")
        content = ch.get("content", "")
        lines.append(f"  [{ts}] {role}: {content}")
        if ch.get("reasoning_trace"):
            for t in ch["reasoning_trace"]:
                lines.append(f"    → {t}")
        if ch.get("emotion"):
            lines.append(f"    emotion: {json.dumps(ch['emotion'], default=str)}")
    lines.append("")

    # Game sessions
    lines.append("-" * 60)
    lines.append("GAME SESSIONS")
    lines.append("-" * 60)
    with _game_lock:
        for sid, gs in _game_sessions.items():
            lines.append(f"  Session: {sid} ({gs['type']})")
            lines.append(f"    Score: {gs['score']}, Steps: {gs['steps']}, Done: {gs['done']}")
            for h in gs.get("history", []):
                lines.append(f"    step={h['step']} action={h['action']} "
                             f"reward={h['reward']} score={h['score']} "
                             f"reasoning={h.get('reasoning', '')}")
    lines.append("")

    # Activity log
    lines.append("-" * 60)
    lines.append("ACTIVITY LOG")
    lines.append("-" * 60)
    for entry in _activity_log:
        ts = entry.get("timestamp", "")
        cat = entry.get("category", "")
        msg = entry.get("message", "")
        lines.append(f"  [{ts}] [{cat}] {msg}")
        if entry.get("data"):
            lines.append(f"    data: {json.dumps(entry['data'], default=str)}")
    lines.append("")

    # Context engine
    lines.append("-" * 60)
    lines.append("CONTEXT ENGINE")
    lines.append("-" * 60)
    ctx_stats = system.context.get_statistics()
    for k, v in ctx_stats.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    lines.append("=" * 80)
    lines.append("END OF EXPORT")
    lines.append("=" * 80)

    text = "\n".join(lines)
    buf = io.BytesIO(text.encode("utf-8"))
    _log("export", "Exported full system data as text")
    return send_file(
        buf,
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"nsck_full_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt",
    )


@app.route("/api/export/json")
def export_json():
    system = _get_system()
    emo = _get_emotion()
    sm = _get_self_model()

    export_data = {
        "exported_at": datetime.now(timezone.utc).isoformat() + "Z",
        "system_stats": system.get_statistics(),
        "knowledge_base": [
            {"concept": k.concept, "relation": k.relation, "target": k.target,
             "confidence": k.confidence, "source": k.source, "timestamp": k.timestamp}
            for k in system.knowledge
        ],
        "context_engine_stats": system.context.get_statistics(),
        "emotion": {
            "current": emo.current_emotion,
            "valence": emo.valence,
            "arousal": emo.arousal,
            "blend": emo.get_emotion_blend(),
            "mood": emo.get_mood(),
            "history": emo.emotion_history,
        },
        "self_model": {
            task: sm.get_stats(task)
            for task in sm.task_stats
        },
        "chat_history": _chat_history,
        "game_sessions": {
            sid: _serialize_game(gs)
            for sid, gs in _game_sessions.items()
        },
        "activity_log": list(_activity_log),
    }

    buf = io.BytesIO()
    buf.write(json.dumps(export_data, indent=2, default=str).encode("utf-8"))
    buf.seek(0)
    _log("export", "Exported system data as JSON")
    return send_file(
        buf,
        mimetype="application/json",
        as_attachment=True,
        download_name=f"nsck_export_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
    )


# ---------------------------------------------------------------------------
# Routes – Feedback & Correction
# ---------------------------------------------------------------------------

@app.route("/api/correct", methods=["POST"])
def api_correct():
    system = _get_system()
    data = request.get_json()
    concept = data.get("concept", "")
    relation = data.get("relation", "")
    wrong_target = data.get("wrong_target", "")
    correct_target = data.get("correct_target", "")
    if not all([concept, relation, correct_target]):
        return jsonify({"error": "Missing required fields"}), 400
    system.correct_knowledge(concept, relation, wrong_target, correct_target)
    _log("correction", f"Corrected: {concept} {relation} {wrong_target} → {correct_target}")
    return jsonify({"status": "ok", "stats": system.get_statistics()})


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    system = _get_system()
    data = request.get_json()
    task_tag = data.get("task_tag", "general")
    reward = float(data.get("reward", 0.0))
    concepts = data.get("concepts", [])
    system.learn_from_feedback(task_tag, reward, concepts)
    _log("feedback", f"Feedback: task={task_tag}, reward={reward}")
    return jsonify({"status": "ok"})




@app.route("/api/system/reset", methods=["POST"])
def api_system_reset():
    """Perform a full system reset: clear memories, files, and restart games."""
    global _chat_history
    
    _log("system", ">>> INITIATING FULL BRAIN RESET <<<")
    
    # 1. Stop all game simulations (Threads and Sessions)
    _log("system", "Stopping all game threads...")
    with _game_lock:
        # Signal all threads to stop
        for sid, stop_event in _game_threads.items():
            stop_event.set()
        _game_threads.clear()
        
        # Mark all sessions as done
        for sid, gs in _game_sessions.items():
            try:
                gs["env"].done = True
                _log("game", f"Terminated session {gs['game_type']} ({sid})")
            except:
                pass
    
    # Give threads a moment to catch the signal
    time.sleep(0.5)

    # 2. Reset Cognitive Modules and Transient States
    system = _get_system()
    emo = _get_emotion()
    sm = _get_self_model()
    
    system.reset()
    emo.reset()
    sm.reset()
    _dashboard_policy.reset()
    
    # Reset dialogue if it exists
    global _dialogue
    if _dialogue:
        _dialogue.reset()
    
    # 3. Clear transient dashboard state
    _chat_history.clear()
    _activity_log.clear() # Optional: decide if log should persist. Implementation plan says clear.
    _log("system", "Dashboard transients cleared.")
    
    # 4. Clear Persistent Files
    # Identify database path from BrainStore if possible, or use default path.
    # Looking at persistence.py learnings: nsck_brain.db is the target.
    db_path = "nsck_brain.db"
    if os.path.exists(db_path):
        try:
            # We need to Ensure the sqlite connection is closed first.
            # KnowledgeIntegration.reset() calls self.episodic.reset(),
            # which might need to close the store.
            if system.episodic.store:
                system.episodic.store.close()
            
            os.remove(db_path)
            _log("system", f"Deleted persistent database: {db_path}")
        except Exception as e:
            _log("error", f"Failed to delete {db_path}: {e}")

    log_csv = "training_log.csv"
    if os.path.exists(log_csv):
        try:
            os.remove(log_csv)
            _log("system", f"Deleted training log: {log_csv}")
        except Exception as e:
            _log("error", f"Failed to delete {log_csv}: {e}")

    _log("system", ">>> FULL BRAIN RESET COMPLETE <<<")
    
    return jsonify({
        "status": "ok", 
        "message": "Brain cleared and system re-initialized.",
        "stats": system.get_statistics()
    })


# ---------------------------------------------------------------------------
# HTML Dashboard Template
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NSCK Testing Dashboard</title>
<style>
:root {
  --bg: #0d1117; --panel: #161b22; --border: #30363d;
  --text: #c9d1d9; --accent: #58a6ff; --green: #3fb950;
  --red: #f85149; --yellow: #d29922; --purple: #bc8cff;
  --orange: #f0883e; --cyan: #39d2c0;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); font-size: 14px; }

/* ---- HEADER ---- */
.header {
  background: linear-gradient(135deg, #161b22, #1c2333);
  border-bottom: 2px solid var(--accent);
  padding: 12px 24px; display: flex; align-items: center; gap: 20px;
  position: sticky; top: 0; z-index: 100;
}
.header h1 { font-size: 20px; color: var(--accent); white-space: nowrap; }
.header .status { font-size: 13px; color: var(--green); }
.header-actions { display: flex; gap: 8px; margin-left: auto; }

/* ---- TABS ---- */
.tab-bar {
  display: flex; gap: 2px; padding: 8px 24px 0; background: var(--panel);
  border-bottom: 1px solid var(--border);
}
.tab-btn {
  padding: 10px 20px; background: transparent; color: var(--text);
  border: 1px solid transparent; border-bottom: none; border-radius: 6px 6px 0 0;
  cursor: pointer; font-size: 13px; font-weight: 600; transition: all 0.2s;
}
.tab-btn:hover { background: rgba(88,166,255,0.1); }
.tab-btn.active { background: var(--bg); color: var(--accent); border-color: var(--border); }

.tab-content { display: none; padding: 16px 24px; }
.tab-content.active { display: block; }

/* ---- PANELS ---- */
.panel {
  background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; margin-bottom: 16px;
}
.panel h2 {
  font-size: 13px; color: var(--accent); margin-bottom: 12px;
  text-transform: uppercase; letter-spacing: 1px;
  border-bottom: 1px solid var(--border); padding-bottom: 8px;
}

/* ---- GRID ---- */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.full-width { grid-column: 1 / -1; }

/* ---- FORMS ---- */
textarea, input[type="text"], select {
  width: 100%; background: var(--bg); color: var(--text);
  border: 1px solid var(--border); border-radius: 4px; padding: 8px;
  font-family: inherit; font-size: 13px; resize: vertical;
}
textarea:focus, input:focus { border-color: var(--accent); outline: none; }
button {
  background: var(--accent); color: #000; border: none; padding: 8px 16px;
  border-radius: 4px; cursor: pointer; font-weight: 600; font-size: 13px;
  transition: all 0.2s;
}
button:hover { opacity: 0.85; transform: translateY(-1px); }
button.secondary { background: var(--border); color: var(--text); }
button.danger { background: var(--red); color: #fff; }
button.success { background: var(--green); color: #000; }
button.warn { background: var(--yellow); color: #000; }
button:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

/* ---- CHAT ---- */
.chat-area {
  background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
  height: 400px; overflow-y: auto; padding: 12px; margin-bottom: 8px;
}
.chat-msg {
  margin-bottom: 12px; padding: 10px 14px; border-radius: 8px;
  max-width: 85%; font-size: 13px; line-height: 1.5;
}
.chat-msg.user {
  background: #1c3a5e; margin-left: auto; border-bottom-right-radius: 2px;
}
.chat-msg.assistant {
  background: #1c2d1c; border-bottom-left-radius: 2px;
}
.chat-msg .meta { font-size: 11px; color: #8b949e; margin-top: 4px; }
.chat-input-row { display: flex; gap: 8px; }
.chat-input-row textarea { height: 40px; flex: 1; }

/* ---- GAME ---- */
.game-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px;
}
.game-card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px;
}
.game-card h3 { color: var(--accent); font-size: 14px; margin-bottom: 8px; }
.game-canvas {
  background: #000; border: 1px solid var(--border); border-radius: 4px;
  width: 100%; aspect-ratio: 1; margin: 8px 0; position: relative;
}
.game-stats { display: flex; gap: 16px; font-size: 12px; margin: 8px 0; }
.game-stats .stat-item { background: var(--bg); padding: 4px 10px; border-radius: 4px; }
.game-stats .stat-item strong { color: var(--accent); }
.game-controls { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
.game-log {
  background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
  height: 120px; overflow-y: auto; padding: 6px; font-size: 11px;
  font-family: 'Cascadia Code', 'Fira Code', monospace; margin-top: 8px;
}

/* ---- MONITOR ---- */
.stat-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 8px; }
.stat-card {
  background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
  padding: 12px; text-align: center;
}
.stat-card .val { font-size: 22px; font-weight: 700; color: var(--accent); }
.stat-card .lbl { font-size: 10px; color: #8b949e; text-transform: uppercase; margin-top: 2px; }

.emotion-bar {
  display: flex; gap: 4px; height: 24px; border-radius: 4px; overflow: hidden;
  margin: 8px 0;
}
.emotion-segment {
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 600; color: #000; min-width: 30px;
  transition: width 0.3s ease;
}

/* ---- LOG VIEWER ---- */
.log-viewer {
  background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
  height: 400px; overflow-y: auto; padding: 8px;
  font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 12px;
}
.log-entry { padding: 3px 0; border-bottom: 1px solid rgba(48,54,61,0.5); }
.log-entry .ts { color: var(--yellow); margin-right: 6px; }
.log-entry .cat { color: var(--purple); margin-right: 6px; font-weight: 600; }

/* ---- TRACE PANEL ---- */
.trace-item {
  padding: 6px 10px; margin: 3px 0; background: var(--bg);
  border-left: 3px solid var(--cyan); border-radius: 0 4px 4px 0;
  font-size: 12px;
}
.reasoning-box {
  background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
  max-height: 300px; overflow-y: auto; padding: 8px;
}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* ---- LOADING ---- */
.spinner {
  display: inline-block; width: 14px; height: 14px;
  border: 2px solid var(--border); border-top-color: var(--accent);
  border-radius: 50%; animation: spin 0.6s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.badge {
  display: inline-block; padding: 2px 8px; border-radius: 10px;
  font-size: 11px; font-weight: 600;
}
.badge-green { background: var(--green); color: #000; }
.badge-red { background: var(--red); color: #fff; }
.badge-yellow { background: var(--yellow); color: #000; }

@media (max-width: 900px) {
  .grid-2, .grid-3 { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<!-- ========== HEADER ========== -->
<div class="header">
  <h1>🧠 NSCK Testing Dashboard</h1>
  <div class="status" id="sys-status">● System Ready</div>
  <div class="header-actions">
    <button class="secondary" onclick="refreshAll()">↻ Refresh All</button>
    <button class="warn" onclick="exportText()">📄 Export TXT</button>
    <button class="secondary" onclick="exportJSON()">📥 Export JSON</button>
    <button class="danger" onclick="fullReset()">⚠ RESET BRAIN</button>
  </div>
</div>

<!-- ========== TAB BAR ========== -->
<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('chat')">💬 Chat & Test</button>
  <button class="tab-btn" onclick="switchTab('games')">🎮 Game Simulations</button>
  <button class="tab-btn" onclick="switchTab('monitor')">📊 System Monitor</button>
  <button class="tab-btn" onclick="switchTab('logs')">📋 Logs & Export</button>
</div>

<!-- ================================================================ -->
<!-- TAB 1: CHAT & TEST -->
<!-- ================================================================ -->
<div class="tab-content active" id="tab-chat">
  <div class="grid-2">
    <!-- Left: Sample Text + Chat -->
    <div>
      <div class="panel">
        <h2>📝 Text Sample</h2>
        <textarea id="sample-text" rows="4"
          placeholder="Paste or type a text sample here. The system will process and understand it, then you can chat about it..."></textarea>
        <div style="margin-top:8px">
          <button onclick="loadSample('nature')">🌿 Nature</button>
          <button onclick="loadSample('science')">🔬 Science</button>
          <button onclick="loadSample('story')">📖 Story</button>
          <button class="secondary" onclick="document.getElementById('sample-text').value=''">Clear</button>
        </div>
      </div>

      <div class="panel">
        <h2>💬 Conversation</h2>
        <div class="chat-area" id="chat-area"></div>
        <div class="chat-input-row">
          <textarea id="chat-input" placeholder="Type your message..." onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendChat();}"></textarea>
          <button onclick="sendChat()" id="chat-send-btn">Send</button>
        </div>
      </div>
    </div>

    <!-- Right: Reasoning & Response Info -->
    <div>
      <div class="panel">
        <h2>🔍 Reasoning Trace</h2>
        <div class="reasoning-box" id="reasoning-trace">
          <div style="color:#8b949e">Send a message to see the system's reasoning process...</div>
        </div>
      </div>

      <div class="panel">
        <h2>🎯 Response Details</h2>
        <div style="margin-bottom:8px">
          <strong>Confidence:</strong> <span id="chat-confidence" style="color:var(--green)">—</span>
        </div>
        <div style="margin-bottom:8px">
          <strong>Emotion:</strong> <span id="chat-emotion">—</span>
        </div>
        <div style="margin-bottom:8px">
          <strong>Disambiguations:</strong>
          <div id="chat-disambiguations" style="font-size:12px;color:#8b949e">None yet</div>
        </div>
      </div>

      <div class="panel">
        <h2>📊 Quick Stats</h2>
        <div class="stat-cards" id="chat-stats">
          <div class="stat-card"><div class="val" id="qs-queries">0</div><div class="lbl">Queries</div></div>
          <div class="stat-card"><div class="val" id="qs-knowledge">0</div><div class="lbl">Knowledge</div></div>
          <div class="stat-card"><div class="val" id="qs-episodes">0</div><div class="lbl">Episodes</div></div>
          <div class="stat-card"><div class="val" id="qs-facts">0</div><div class="lbl">Facts</div></div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ================================================================ -->
<!-- TAB 2: GAME SIMULATIONS -->
<!-- ================================================================ -->
<div class="tab-content" id="tab-games">
  <div class="panel">
    <h2>🎮 Launch Game Simulations</h2>
    <p style="font-size:12px;color:#8b949e;margin-bottom:12px">
      Start game simulations for the cognitive system to play.
      The system uses heuristic + self-model reasoning to choose actions.
      You can run multiple games simultaneously or sequentially.
    </p>
    <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
      <select id="game-type" style="width:140px">
        <option value="snake">🐍 Snake</option>
        <option value="pong">🏓 Pong</option>
        <option value="maze">🏰 Maze</option>
      </select>
      <label style="font-size:12px;display:flex;align-items:center;gap:4px">
        <input type="checkbox" id="teacherMode" checked> Teacher Mode (Learn)
      </label>
      <label style="font-size:12px;display:flex;align-items:center;gap:4px">
        <input type="checkbox" id="auto-play" checked> Auto-play
      </label>
      <label style="font-size:12px;display:flex;align-items:center;gap:4px">
        Speed: <input type="range" id="game-speed" min="50" max="1000" value="300" style="width:100px">
        <span id="speed-label">300ms</span>
      </label>
      <button class="success" onclick="startGame()">▶ Start Game</button>
      <button class="secondary" onclick="startAllGames()">▶▶ Start All Three</button>
      <button class="danger" onclick="stopAllGames()">■ Stop All</button>
    </div>
  </div>

  <div class="game-grid" id="game-grid">
    <div class="panel" style="grid-column:1/-1;text-align:center;color:#8b949e;padding:40px">
      No games running. Click "Start Game" to launch a simulation.
    </div>
  </div>
</div>

<!-- ================================================================ -->
<!-- TAB 3: SYSTEM MONITOR -->
<!-- ================================================================ -->
<div class="tab-content" id="tab-monitor">
  <div class="grid-2">
    <!-- Emotion Monitor -->
    <div class="panel">
      <h2>😊 Emotional State</h2>
      <div style="display:flex;gap:20px;margin-bottom:12px">
        <div>
          <div style="font-size:11px;color:#8b949e">Current Emotion</div>
          <div style="font-size:24px;font-weight:700" id="mon-emotion">neutral</div>
        </div>
        <div>
          <div style="font-size:11px;color:#8b949e">Valence</div>
          <div style="font-size:20px;font-weight:600" id="mon-valence">0.00</div>
        </div>
        <div>
          <div style="font-size:11px;color:#8b949e">Arousal</div>
          <div style="font-size:20px;font-weight:600" id="mon-arousal">0.00</div>
        </div>
        <div>
          <div style="font-size:11px;color:#8b949e">Intensity</div>
          <div style="font-size:20px;font-weight:600" id="mon-intensity">0.00</div>
        </div>
      </div>
      <div style="font-size:12px;margin-bottom:4px;color:#8b949e">Emotion Blend:</div>
      <div class="emotion-bar" id="emotion-bar"></div>
      <div style="font-size:12px;margin-top:8px">
        <strong>Mood:</strong> <span id="mon-mood">—</span>
      </div>
    </div>

    <!-- Self-Model Performance -->
    <div class="panel">
      <h2>🎯 Self-Model Performance</h2>
      <div id="self-model-perf" style="font-size:13px">
        <div style="color:#8b949e">No task performance data yet. Run some games or chat to generate data.</div>
      </div>
    </div>

    <!-- System Stats -->
    <div class="panel">
      <h2>📈 System Statistics</h2>
      <div class="stat-cards" id="monitor-stats">
        <div class="stat-card"><div class="val" id="ms-queries">0</div><div class="lbl">Queries</div></div>
        <div class="stat-card"><div class="val" id="ms-knowledge">0</div><div class="lbl">Knowledge</div></div>
        <div class="stat-card"><div class="val" id="ms-episodes">0</div><div class="lbl">Episodes</div></div>
        <div class="stat-card"><div class="val" id="ms-facts">0</div><div class="lbl">Facts</div></div>
        <div class="stat-card"><div class="val" id="ms-corrections">0</div><div class="lbl">Corrections</div></div>
        <div class="stat-card"><div class="val" id="ms-concepts">0</div><div class="lbl">Concepts</div></div>
      </div>
    </div>

    <!-- Knowledge Browser -->
    <div class="panel">
      <h2>📚 Knowledge Base</h2>
      <div style="max-height:300px;overflow-y:auto" id="knowledge-list">
        <div style="color:#8b949e;font-size:12px">Loading knowledge base...</div>
      </div>
    </div>
  </div>

  <div class="panel">
    <h2>📉 Emotion History</h2>
    <div style="max-height:200px;overflow-y:auto;font-size:12px;font-family:monospace" id="emotion-history">
      <div style="color:#8b949e">No emotion history yet</div>
    </div>
  </div>
</div>

<!-- ================================================================ -->
<!-- TAB 4: LOGS & EXPORT -->
<!-- ================================================================ -->
<div class="tab-content" id="tab-logs">
  <div class="panel">
    <h2>📋 Activity Log</h2>
    <div style="display:flex;gap:8px;margin-bottom:8px">
      <button class="secondary" onclick="refreshLogs()">↻ Refresh</button>
      <button class="warn" onclick="exportText()">📄 Export Full TXT</button>
      <button class="secondary" onclick="exportJSON()">📥 Export JSON</button>
      <select id="log-filter" onchange="filterLogs()" style="width:140px">
        <option value="all">All Categories</option>
        <option value="chat">Chat</option>
        <option value="game">Game</option>
        <option value="system">System</option>
        <option value="input">Input</option>
        <option value="response">Response</option>
        <option value="error">Errors</option>
        <option value="export">Export</option>
      </select>
    </div>
    <div class="log-viewer" id="log-viewer">
      <div style="color:#8b949e">Loading logs...</div>
    </div>
  </div>

  <div class="grid-2">
    <div class="panel">
      <h2>💬 Chat History Export</h2>
      <div style="max-height:300px;overflow-y:auto;font-size:12px" id="chat-history-export">
        <div style="color:#8b949e">No chat history yet</div>
      </div>
    </div>
    <div class="panel">
      <h2>🎮 Game History Export</h2>
      <div style="max-height:300px;overflow-y:auto;font-size:12px" id="game-history-export">
        <div style="color:#8b949e">No game sessions yet</div>
      </div>
    </div>
  </div>
</div>

<!-- ================================================================ -->
<!-- JAVASCRIPT -->
<!-- ================================================================ -->
<script>
const API = '';
const MIN_POLL_INTERVAL_MS = 200;
let allLogs = [];
let gamePollers = {};

// ---- Sample texts ----
const SAMPLES = {
  nature: "The red rose blooms in the garden, its petals glistening with morning dew. A butterfly lands gently on its soft surface, drawn by the sweet fragrance that fills the warm summer air.",
  science: "Photosynthesis is the process by which green plants convert sunlight into chemical energy. Chlorophyll in the leaves absorbs light, which drives the reaction that turns carbon dioxide and water into glucose and oxygen.",
  story: "The old wizard climbed the tower stairs, his staff tapping each step. At the top, he found the ancient book of spells, its pages glowing with a faint blue light. He knew this was the key to saving the kingdom."
};

function loadSample(key) {
  document.getElementById('sample-text').value = SAMPLES[key] || '';
}

// ---- Tabs ----
function switchTab(tabId) {
  document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(tb => tb.classList.remove('active'));
  document.getElementById('tab-' + tabId).classList.add('active');
  event.target.classList.add('active');
  if (tabId === 'monitor') refreshMonitor();
  if (tabId === 'logs') refreshLogs();
}

// ---- Chat ----
async function sendChat() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;

  const sampleText = document.getElementById('sample-text').value.trim();
  const btn = document.getElementById('chat-send-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';

  appendChat('user', msg);
  input.value = '';

  try {
    const resp = await fetch(API + '/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: msg, sample_text: sampleText})
    });
    const data = await resp.json();
    if (data.error) {
      appendChat('assistant', '❌ Error: ' + data.error);
    } else {
      appendChat('assistant', data.answer, data);
      updateReasoningTrace(data.reasoning_trace || []);
      updateChatDetails(data);
    }
  } catch(e) {
    appendChat('assistant', '❌ Network error: ' + e.message);
  }
  btn.disabled = false;
  btn.textContent = 'Send';
}

function appendChat(role, content, data) {
  const area = document.getElementById('chat-area');
  const div = document.createElement('div');
  div.className = 'chat-msg ' + role;
  let html = content.replace(/\n/g, '<br>');
  if (data && data.confidence !== undefined) {
    html += `<div class="meta">Confidence: ${(data.confidence*100).toFixed(1)}% | Emotion: ${data.emotion?.emotion || '—'}</div>`;
  }
  div.innerHTML = html;
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
}

function updateReasoningTrace(trace) {
  const box = document.getElementById('reasoning-trace');
  if (!trace || trace.length === 0) {
    box.innerHTML = '<div style="color:#8b949e">No reasoning trace available</div>';
    return;
  }
  box.innerHTML = trace.map(t => `<div class="trace-item">${t}</div>`).join('');
  box.scrollTop = box.scrollHeight;
}

function updateChatDetails(data) {
  document.getElementById('chat-confidence').textContent =
    data.confidence !== undefined ? (data.confidence*100).toFixed(1) + '%' : '—';

  const emo = data.emotion || {};
  const emoStr = `${emo.emotion || '—'} (v=${emo.valence?.toFixed(2) || 0}, a=${emo.arousal?.toFixed(2) || 0})`;
  document.getElementById('chat-emotion').textContent = emoStr;

  const dBox = document.getElementById('chat-disambiguations');
  const disambs = data.disambiguations || [];
  if (disambs.length === 0) {
    dBox.innerHTML = '<span style="color:#8b949e">None</span>';
  } else {
    dBox.innerHTML = disambs.map(d =>
      `<div style="margin:3px 0;padding:4px;background:var(--bg);border-radius:3px">
        <strong>'${d.concept}'</strong> → ${d.meaning}
        <span style="color:#8b949e">(${(d.confidence*100).toFixed(0)}%)</span>
      </div>`
    ).join('');
  }

  if (data.stats) {
    document.getElementById('qs-queries').textContent = data.stats.queries_processed || 0;
    document.getElementById('qs-knowledge').textContent = data.stats.knowledge_entries || 0;
    document.getElementById('qs-episodes').textContent = data.stats.episodes_recorded || 0;
    document.getElementById('qs-facts').textContent = data.stats.facts_learned || 0;
  }
}

// ---- Games ----
async function startGame() {
  const gameType = document.getElementById('game-type').value;
  const teacherActive = document.getElementById('teacherMode').checked;
  const autoPlay = document.getElementById('auto-play').checked;
  const speed = parseInt(document.getElementById('game-speed').value);

  try {
    const resp = await fetch('/api/game/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
          game: gameType, 
          teacher_active: teacherActive, 
          auto_play: autoPlay, 
          speed: speed/1000 // Convert ms to seconds
      })
    });
    const data = await resp.json();
    if (data.error) { alert('Error: ' + data.error); return; }
    addGameCard(data.session_id, data.game_type, data.state);
    if (autoPlay) startGamePoller(data.session_id, speed);
  } catch(e) { alert('Error: ' + e.message); }
}

async function startAllGames() {
  const speed = parseInt(document.getElementById('game-speed').value);
  for (const gt of ['snake', 'pong', 'maze']) {
    try {
      const resp = await fetch(API + '/api/game/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({game_type: gt, auto_play: true, speed: speed/1000})
      });
      const data = await resp.json();
      if (!data.error) {
        addGameCard(data.session_id, data.game_type, data.state);
        startGamePoller(data.session_id, speed);
      }
    } catch(e) {}
  }
}

function addGameCard(sessionId, gameType, state) {
  const grid = document.getElementById('game-grid');
  // Remove placeholder if present
  if (grid.querySelector('.panel[style]')) {
    grid.innerHTML = '';
  }

  const icons = {snake: '🐍', pong: '🏓', maze: '🏰'};
  const card = document.createElement('div');
  card.className = 'game-card';
  card.id = 'game-' + sessionId;
  card.innerHTML = `
    <h3>${icons[gameType]||'🎮'} ${gameType.toUpperCase()} <span class="badge badge-green" id="badge-${sessionId}">RUNNING</span></h3>
    <canvas class="game-canvas" id="canvas-${sessionId}" width="200" height="200"></canvas>
    <div class="game-stats">
      <div class="stat-item">Score: <strong id="score-${sessionId}">0</strong></div>
      <div class="stat-item">Steps: <strong id="steps-${sessionId}">0</strong></div>
      <div class="stat-item">Action: <strong id="action-${sessionId}">—</strong></div>
    </div>
    <div class="game-controls">
      <button onclick="stepGame('${sessionId}')" class="secondary">Step</button>
      <button onclick="stopGame('${sessionId}')" class="danger">Stop</button>
    </div>
    <div class="game-log" id="log-${sessionId}"></div>
  `;
  // Add Selection Logic
  card.onclick = function() {
      selectGameSession(sessionId);
  };
  grid.appendChild(card);
  
  // Auto-select if first
  if (!selectedSessionId) selectGameSession(sessionId);
  
  renderGame(sessionId, state, gameType);
}

// Global Selection State
let selectedSessionId = null;

function selectGameSession(sid) {
    if (selectedSessionId) {
        const old = document.getElementById('game-' + selectedSessionId);
        if (old) old.style.borderColor = '#30363d'; // Default border
    }
    selectedSessionId = sid;
    const curr = document.getElementById('game-' + sid);
    if (curr) curr.style.borderColor = '#58a6ff'; // Highlight blue
}

// Global Keyboard Listener
document.addEventListener('keydown', async (e) => {
    // Only capture if game is selected
    if (!selectedSessionId) return;
    
    // Map keys
    let action = null;
    if (e.key === 'ArrowUp') action = 'UP';
    else if (e.key === 'ArrowDown') action = 'DOWN';
    else if (e.key === 'ArrowLeft') action = 'LEFT';
    else if (e.key === 'ArrowRight') action = 'RIGHT';
    
    if (action) {
        e.preventDefault(); // Stop scrolling
        // Send manual action
        try {
            await fetch(API + '/api/game/action', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({session_id: selectedSessionId, action: action})
            });
            // Optional: Optimistic UI update? No, let poll handle it.
            console.log(`Sent manual action ${action} to ${selectedSessionId}`);
        } catch(err) {
            console.error(err);
        }
    }
});

function startGamePoller(sessionId, intervalMs) {
  if (gamePollers[sessionId]) clearInterval(gamePollers[sessionId]);
  gamePollers[sessionId] = setInterval(async () => {
    try {
      const resp = await fetch(API + '/api/game/state?session_id=' + sessionId);
      const data = await resp.json();
      if (data.error) { clearInterval(gamePollers[sessionId]); return; }
      const state = data.state;
      updateGameCard(sessionId, state, data.history || []);
      if (state.done) {
        // Do NOT stop polling for auto-play games!
        // The backend will auto-restart.
        // We just update the badge briefly?
        const badge = document.getElementById('badge-' + sessionId);
        if (badge) { badge.className = 'badge badge-yellow'; badge.textContent = 'RESTARTING'; }
      } else {
         const badge = document.getElementById('badge-' + sessionId);
         if (badge && badge.textContent !== 'STOPPED') { 
             badge.className = 'badge badge-green'; 
             badge.textContent = 'RUNNING'; 
         }
      }
    } catch(e) {}
  }, Math.max(intervalMs, MIN_POLL_INTERVAL_MS));
}

function updateGameCard(sessionId, state, history) {
  const scoreEl = document.getElementById('score-' + sessionId);
  const stepsEl = document.getElementById('steps-' + sessionId);
  const actionEl = document.getElementById('action-' + sessionId);
  if (scoreEl) scoreEl.textContent = state.score || 0;
  if (stepsEl) stepsEl.textContent = state.steps || 0;
  if (history.length > 0) {
    const last = history[history.length - 1];
    if (actionEl) actionEl.textContent = last.action || '—';
  }
  renderGame(sessionId, state, state.type);
  // Update game log
  const logEl = document.getElementById('log-' + sessionId);
  if (logEl && history.length > 0) {
    const recent = history.slice(-8);
    logEl.innerHTML = recent.map(h =>
      `<div>step ${h.step}: ${h.action} → r=${h.reward} | ${h.reasoning || ''}</div>`
    ).join('');
    logEl.scrollTop = logEl.scrollHeight;
  }
}

function renderGame(sessionId, state, gameType) {
  const canvas = document.getElementById('canvas-' + sessionId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, w, h);

  if (gameType === 'snake') {
    const cellW = w / 10, cellH = h / 10;
    // Draw body
    ctx.fillStyle = '#3fb950';
    (state.body || []).forEach(([x,y]) => {
      ctx.fillRect(x*cellW+1, y*cellH+1, cellW-2, cellH-2);
    });
    // Draw head
    if (state.head) {
      ctx.fillStyle = '#58a6ff';
      ctx.fillRect(state.head[0]*cellW+1, state.head[1]*cellH+1, cellW-2, cellH-2);
    }
    // Draw food
    if (state.food) {
      ctx.fillStyle = '#f85149';
      ctx.beginPath();
      ctx.arc(state.food[0]*cellW+cellW/2, state.food[1]*cellH+cellH/2, cellW/3, 0, Math.PI*2);
      ctx.fill();
    }
  } else if (gameType === 'pong') {
    const scaleX = w / 30, scaleY = h / 30;
    // Paddle
    ctx.fillStyle = '#58a6ff';
    ctx.fillRect(1*scaleX, (state.p1_y||0)*scaleY, scaleX, 6*scaleY);
    // Ball
    ctx.fillStyle = '#f85149';
    ctx.beginPath();
    ctx.arc((state.ball_x||15)*scaleX, (state.ball_y||15)*scaleY, scaleX/2, 0, Math.PI*2);
    ctx.fill();
    // Right wall
    ctx.fillStyle = '#30363d';
    ctx.fillRect(28*scaleX, 0, 2*scaleX, h);
  } else if (gameType === 'maze') {
    const size = state.size || 10;
    const cellW = w / size, cellH = h / size;
    // Walls
    ctx.fillStyle = '#30363d';
    (state.walls || []).forEach(([x,y]) => {
      ctx.fillRect(x*cellW, y*cellH, cellW, cellH);
    });
    // Exit
    if (state.exit) {
      ctx.fillStyle = '#3fb950';
      ctx.fillRect(state.exit[0]*cellW+2, state.exit[1]*cellH+2, cellW-4, cellH-4);
    }
    // Player
    if (state.player) {
      ctx.fillStyle = '#58a6ff';
      ctx.beginPath();
      ctx.arc(state.player[0]*cellW+cellW/2, state.player[1]*cellH+cellH/2, cellW/3, 0, Math.PI*2);
      ctx.fill();
    }
  }
}

async function stepGame(sessionId) {
  try {
    const resp = await fetch(API + '/api/game/step', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId})
    });
    const data = await resp.json();
    if (data.state) updateGameCard(sessionId, data.state, [data]);
  } catch(e) {}
}

async function stopGame(sessionId) {
  if (gamePollers[sessionId]) {
    clearInterval(gamePollers[sessionId]);
    delete gamePollers[sessionId];
  }
  try {
    await fetch(API + '/api/game/stop', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId})
    });
  } catch(e) {}
  const badge = document.getElementById('badge-' + sessionId);
  if (badge) { badge.className = 'badge badge-yellow'; badge.textContent = 'STOPPED'; }
}

async function fullReset() {
  if (!confirm("⚠️ CAUTION: Are you sure you want to clear the entire brain? This will erase all learned knowledge, experiences, and model weights. This cannot be undone.")) {
    return;
  }
  
  try {
    const resp = await fetch(API + '/api/system/reset', { method: 'POST' });
    const data = await resp.json();
    if (data.status === 'ok') {
      alert("✅ Brain cleared and system re-initialized.");
      // Clear chat
      document.getElementById('chat-area').innerHTML = '';
      document.getElementById('reasoning-trace').innerHTML = '<div style="color:#8b949e">Send a message to see the system\'s reasoning process...</div>';
      // Stop all pollers
      stopAllGames();
      // Refresh
      refreshAll();
    } else {
      alert("❌ Error: " + data.error);
    }
  } catch(e) {
    alert("❌ Network error during reset: " + e.message);
  }
}

async function stopAllGames() {
  for (const sid of Object.keys(gamePollers)) {
    await stopGame(sid);
  }
  try {
    const resp = await fetch(API + '/api/game/sessions');
    const data = await resp.json();
    for (const s of (data.sessions || [])) {
      if (!s.done) {
        await fetch(API + '/api/game/stop', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({session_id: s.session_id})
        });
      }
    }
  } catch(e) {}
}

document.getElementById('game-speed').addEventListener('input', function() {
  document.getElementById('speed-label').textContent = this.value + 'ms';
});

// ---- Monitor ----
const EMOTION_COLORS = {
  joy: '#FFD700', trust: '#90EE90', fear: '#9370DB', surprise: '#FF69B4',
  sadness: '#4682B4', disgust: '#556B2F', anger: '#DC143C',
  anticipation: '#FF8C00', neutral: '#808080'
};

async function refreshMonitor() {
  try {
    const resp = await fetch(API + '/api/monitor');
    const data = await resp.json();

    // Emotion
    const emo = data.emotion || {};
    document.getElementById('mon-emotion').textContent = emo.current || 'neutral';
    document.getElementById('mon-valence').textContent = (emo.valence || 0).toFixed(3);
    document.getElementById('mon-arousal').textContent = (emo.arousal || 0).toFixed(3);
    document.getElementById('mon-intensity').textContent = (emo.intensity || 0).toFixed(3);

    // Emotion blend bar
    const blendBar = document.getElementById('emotion-bar');
    const blend = emo.blend || {};
    blendBar.innerHTML = Object.entries(blend)
      .sort((a,b) => b[1]-a[1])
      .map(([e,w]) => {
        const pct = (w*100).toFixed(0);
        return `<div class="emotion-segment" style="width:${pct}%;background:${EMOTION_COLORS[e]||'#666'}"
          title="${e}: ${pct}%">${pct > 8 ? e.slice(0,3) : ''}</div>`;
      }).join('');

    // Mood
    const mood = emo.mood || {};
    document.getElementById('mon-mood').textContent =
      `${mood.dominant_emotion || 'neutral'} (v=${(mood.avg_valence||0).toFixed(2)}, a=${(mood.avg_arousal||0).toFixed(2)}, stability=${(mood.stability||0).toFixed(2)})`;

    // Emotion history
    const histEl = document.getElementById('emotion-history');
    const hist = emo.history || [];
    if (hist.length > 0) {
      histEl.innerHTML = hist.map(h =>
        `<span style="color:var(--yellow)">step ${h.step}</span> ` +
        `<span style="color:${EMOTION_COLORS[h.emotion]||'#ccc'}">${h.emotion}</span> ` +
        `v=${h.valence} a=${h.arousal}`
      ).join('<br>');
    }

    // Self-model
    const smEl = document.getElementById('self-model-perf');
    const sm = data.self_model || {};
    if (Object.keys(sm).length === 0) {
      smEl.innerHTML = '<div style="color:#8b949e">No task data yet</div>';
    } else {
      smEl.innerHTML = Object.entries(sm).map(([task, perf]) => {
        const sr = (perf.success_rate * 100).toFixed(1);
        const trendIcon = perf.trend === 'improving' ? '📈' : (perf.trend === 'declining' ? '📉' : '➡️');
        return `<div style="margin:6px 0;padding:8px;background:var(--bg);border-radius:4px">
          <strong>${task}</strong> ${trendIcon}
          <span style="color:var(--green)">${sr}%</span> success
          (${perf.attempts} attempts, ${perf.successes} wins)
        </div>`;
      }).join('');
    }

    // System stats
    const ss = data.system_stats || {};
    document.getElementById('ms-queries').textContent = ss.queries_processed || 0;
    document.getElementById('ms-knowledge').textContent = ss.knowledge_entries || 0;
    document.getElementById('ms-episodes').textContent = ss.episodes_recorded || 0;
    document.getElementById('ms-facts').textContent = ss.facts_learned || 0;
    document.getElementById('ms-corrections').textContent = ss.corrections_made || 0;
    document.getElementById('ms-concepts').textContent = ss.semantic_concepts || 0;

    // Knowledge base
    const kResp = await fetch(API + '/api/knowledge');
    const kData = await kResp.json();
    const kList = document.getElementById('knowledge-list');
    const entries = kData.knowledge || [];
    if (entries.length === 0) {
      kList.innerHTML = '<div style="color:#8b949e">No knowledge entries</div>';
    } else {
      kList.innerHTML = entries.slice(0, 50).map(k =>
        `<div style="padding:3px 0;border-bottom:1px solid var(--border);font-size:12px">
          <span style="color:var(--accent)">${k.concept}</span>
          <span style="color:#8b949e">—${k.relation}→</span>
          <span style="color:var(--green)">${k.target}</span>
          <span style="color:#8b949e">(${(k.confidence*100).toFixed(0)}%, ${k.source})</span>
        </div>`
      ).join('');
      if (entries.length > 50) {
        kList.innerHTML += `<div style="color:#8b949e;text-align:center;padding:4px">... and ${entries.length-50} more</div>`;
      }
    }
  } catch(e) { console.error('Monitor refresh error:', e); }
}

// ---- Logs ----
async function refreshLogs() {
  try {
    const resp = await fetch(API + '/api/logs');
    const data = await resp.json();
    allLogs = data.logs || [];
    renderLogs(allLogs);

    // Chat history
    const chatResp = await fetch(API + '/api/chat/history');
    const chatData = await chatResp.json();
    const chatEl = document.getElementById('chat-history-export');
    const hist = chatData.history || [];
    if (hist.length === 0) {
      chatEl.innerHTML = '<div style="color:#8b949e">No chat history</div>';
    } else {
      chatEl.innerHTML = hist.map(h => {
        const prefix = h.role === 'user' ? '👤' : '🤖';
        return `<div style="margin:4px 0;padding:4px;background:var(--bg);border-radius:3px">
          ${prefix} <strong>${h.role}</strong>: ${(h.content||'').substring(0,200)}
          ${h.confidence !== undefined ? `<br><span style="color:#8b949e">conf=${(h.confidence*100).toFixed(1)}%</span>` : ''}
        </div>`;
      }).join('');
    }

    // Game history
    const gameResp = await fetch(API + '/api/game/sessions');
    const gameData = await gameResp.json();
    const gameEl = document.getElementById('game-history-export');
    const sessions = gameData.sessions || [];
    if (sessions.length === 0) {
      gameEl.innerHTML = '<div style="color:#8b949e">No game sessions</div>';
    } else {
      gameEl.innerHTML = sessions.map(s =>
        `<div style="margin:4px 0;padding:4px;background:var(--bg);border-radius:3px">
          ${s.game_type.toUpperCase()} (${s.session_id})
          Score: ${s.score} | Steps: ${s.steps} |
          <span class="badge ${s.done ? 'badge-red' : 'badge-green'}">${s.done ? 'DONE' : 'RUNNING'}</span>
        </div>`
      ).join('');
    }
  } catch(e) { console.error('Log refresh error:', e); }
}

function renderLogs(logs) {
  const viewer = document.getElementById('log-viewer');
  if (logs.length === 0) {
    viewer.innerHTML = '<div style="color:#8b949e">No log entries</div>';
    return;
  }
  viewer.innerHTML = logs.slice().reverse().map(e =>
    `<div class="log-entry">
      <span class="ts">${(e.timestamp||'').slice(11,19)}</span>
      <span class="cat">[${e.category}]</span>
      ${e.message}
    </div>`
  ).join('');
}

function filterLogs() {
  const filter = document.getElementById('log-filter').value;
  if (filter === 'all') {
    renderLogs(allLogs);
  } else {
    renderLogs(allLogs.filter(l => l.category === filter));
  }
}

// ---- Exports ----
function exportText() { window.location.href = API + '/api/export/text'; }
function exportJSON() { window.location.href = API + '/api/export/json'; }

// ---- Global refresh ----
async function refreshAll() {
  refreshMonitor();
  refreshLogs();
  try {
    const resp = await fetch(API + '/api/stats');
    const data = await resp.json();
    document.getElementById('qs-queries').textContent = data.queries_processed || 0;
    document.getElementById('qs-knowledge').textContent = data.knowledge_entries || 0;
    document.getElementById('qs-episodes').textContent = data.episodes_recorded || 0;
    document.getElementById('qs-facts').textContent = data.facts_learned || 0;
  } catch(e) {}
}

// ---- Auto refresh monitor every 3s ----
setInterval(() => {
  if (document.getElementById('tab-monitor').classList.contains('active')) {
    refreshMonitor();
  }
}, 3000);

// Initial load
refreshAll();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the testing dashboard server."""
    port = int(os.environ.get("NSCK_TESTING_PORT", 5051))
    print(f"\n{'='*60}")
    print(f"  NSCK Testing Dashboard")
    print(f"  Open http://localhost:{port} in your browser")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
