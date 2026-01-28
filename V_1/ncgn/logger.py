"""
NCGN v7 Activity Logger

Detailed logging of all brain activities for examination and debugging.
Logs ticks, learning events, inputs, and game actions.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import threading


@dataclass
class LogEntry:
    """Single log entry."""
    timestamp: str
    event_type: str
    data: Dict[str, Any]
    
    def to_dict(self) -> dict:
        return asdict(self)


class BrainLogger:
    """
    Detailed activity logger for Brain operations.
    
    Logs:
    - Tick events: activations, firing, energy flow
    - Learning events: weight changes, LTP/LTD
    - Input events: injections, file uploads, queries
    - Game events: actions, rewards, episodes
    """
    
    def __init__(self, enabled: bool = False, max_entries: int = 10000):
        self.enabled = enabled
        self.max_entries = max_entries
        self._entries: List[LogEntry] = []
        self._lock = threading.Lock()
        self._session_start = datetime.now().isoformat()
    
    def enable(self) -> None:
        """Enable logging."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable logging."""
        self.enabled = False
    
    def clear(self) -> None:
        """Clear all log entries."""
        with self._lock:
            self._entries.clear()
            self._session_start = datetime.now().isoformat()
    
    def _log(self, event_type: str, data: Dict[str, Any]) -> None:
        """Add a log entry."""
        if not self.enabled:
            return
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            data=data
        )
        
        with self._lock:
            self._entries.append(entry)
            
            # Trim if too many entries
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries:]
    
    # === Tick Events ===
    
    def log_tick(
        self,
        tick_number: int,
        total_energy: float,
        active_count: int,
        firing_count: int,
        top_active: Dict[str, float]
    ) -> None:
        """Log a propagation tick."""
        self._log("tick", {
            "tick": tick_number,
            "total_energy": round(total_energy, 4),
            "active_count": active_count,
            "firing_count": firing_count,
            "top_active": {k: round(v, 4) for k, v in list(top_active.items())[:5]}
        })
    
    def log_propagation(
        self,
        source: str,
        target: str,
        energy_transferred: float
    ) -> None:
        """Log energy propagation between nodes."""
        self._log("propagation", {
            "source": source,
            "target": target,
            "energy": round(energy_transferred, 4)
        })
    
    # === Learning Events ===
    
    def log_learning(
        self,
        source: str,
        target: str,
        old_weight: float,
        new_weight: float,
        reward: float,
        learning_type: str  # "ltp" or "ltd"
    ) -> None:
        """Log a weight update."""
        self._log("learning", {
            "source": source,
            "target": target,
            "old_weight": round(old_weight, 4),
            "new_weight": round(new_weight, 4),
            "delta": round(new_weight - old_weight, 4),
            "reward": round(reward, 4),
            "type": learning_type
        })
    
    def log_reward(self, reward: float, rpe: float) -> None:
        """Log reward signal."""
        self._log("reward", {
            "reward": round(reward, 4),
            "rpe": round(rpe, 4)
        })
    
    # === Input Events ===
    
    def log_injection(self, label: str, energy: float) -> None:
        """Log energy injection."""
        self._log("injection", {
            "label": label,
            "energy": round(energy, 4)
        })
    
    def log_concept_added(self, label: str, idx: int) -> None:
        """Log new concept creation."""
        self._log("concept_added", {
            "label": label,
            "index": idx
        })
    
    def log_connection_added(
        self,
        source: str,
        target: str,
        weight: float
    ) -> None:
        """Log new connection creation."""
        self._log("connection_added", {
            "source": source,
            "target": target,
            "weight": round(weight, 4)
        })
    
    def log_file_upload(
        self,
        filename: str,
        concepts_extracted: int,
        connections_created: int
    ) -> None:
        """Log file upload and processing."""
        self._log("file_upload", {
            "filename": filename,
            "concepts": concepts_extracted,
            "connections": connections_created
        })
    
    def log_query(self, question: str, answer: str) -> None:
        """Log a query and response."""
        self._log("query", {
            "question": question,
            "answer": answer[:200]  # Truncate long answers
        })
    
    # === Game Events ===
    
    def log_game_action(
        self,
        game: str,
        action: str,
        reward: float,
        done: bool
    ) -> None:
        """Log a game action."""
        self._log("game_action", {
            "game": game,
            "action": action,
            "reward": round(reward, 4),
            "done": done
        })
    
    def log_game_episode(
        self,
        game: str,
        episode: int,
        total_reward: float,
        steps: int
    ) -> None:
        """Log game episode completion."""
        self._log("game_episode", {
            "game": game,
            "episode": episode,
            "total_reward": round(total_reward, 4),
            "steps": steps
        })
    
    # === Retrieval ===
    
    def get_entries(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Get log entries with optional filtering."""
        with self._lock:
            entries = self._entries
            
            if event_type:
                entries = [e for e in entries if e.event_type == event_type]
            
            # Return most recent first
            entries = list(reversed(entries))
            
            return [e.to_dict() for e in entries[offset:offset + limit]]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of logged events."""
        with self._lock:
            counts = {}
            for entry in self._entries:
                counts[entry.event_type] = counts.get(entry.event_type, 0) + 1
            
            return {
                "session_start": self._session_start,
                "total_entries": len(self._entries),
                "event_counts": counts,
                "logging_enabled": self.enabled
            }
    
    def export_session(self, path: str) -> str:
        """Export all log entries to JSON file."""
        export_path = Path(path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            data = {
                "session_start": self._session_start,
                "export_time": datetime.now().isoformat(),
                "entries": [e.to_dict() for e in self._entries]
            }
        
        with open(export_path, "w") as f:
            json.dump(data, f, indent=2)
        
        return str(export_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics."""
        return self.get_summary()
