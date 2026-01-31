"""
NSCK Persistence Module
SQLite-based storage for concepts, rules, and episodes.
"""
import sqlite3
import pickle
import time
import threading
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Rule:
    """Stored rule with metadata."""
    id: Optional[int]
    condition: frozenset  # Set of predicate names
    consequence: str  # Action name
    priority: int = 1
    source: str = "bootstrap"  # "bootstrap", "learned", "instructed"
    task_tag: str = "global"
    scope: str = "task_local"  # "task_local", "candidate_global", "global"
    support_count: int = 0
    success_rate: float = 0.0
    created_at: float = 0.0


@dataclass
class Concept:
    """Stored concept with metadata."""
    id: Optional[int]
    name: str
    hv_bytes: bytes  # Serialized HyperVector
    concept_type: str  # "ACTION", "RELATION", "OBJECT", etc.
    task_tag: str = "global"
    created_at: float = 0.0
    access_count: int = 0


@dataclass
class Episode:
    """Stored episode for episodic memory."""
    id: Optional[int]
    timestamp: float
    task_tag: str
    situation_hv_bytes: bytes  # Serialized HyperVector
    state_sketch: Dict[str, Any]  # Compressed state (not full state)
    action: str
    outcome: str
    reward: float
    impact_score: float = 0.0


class BrainStore:
    """
    SQLite-based persistence for NSCK brain state.
    
    Uses WAL mode for better concurrency and write-behind
    buffering for performance.
    """
    
    def __init__(self, db_path: str = "nsck_brain.db"):
        """
        Initialize brain store.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._buffer_lock = threading.Lock()
        self._episode_buffer: List[Episode] = []
        self._buffer_limit = 100  # Flush every N episodes
        self._last_flush = time.time()
        self._flush_interval = 30.0  # Flush every N seconds
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        
        # Rules table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                condition BLOB,
                consequence TEXT,
                priority INTEGER DEFAULT 1,
                source TEXT DEFAULT 'bootstrap',
                task_tag TEXT DEFAULT 'global',
                scope TEXT DEFAULT 'task_local',
                support_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                created_at REAL
            )
        """)
        
        # Concepts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                hv_bytes BLOB,
                concept_type TEXT,
                task_tag TEXT DEFAULT 'global',
                created_at REAL,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        # Episodes table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                task_tag TEXT,
                situation_hv_bytes BLOB,
                state_sketch BLOB,
                action TEXT,
                outcome TEXT,
                reward REAL,
                impact_score REAL DEFAULT 0.0
            )
        """)
        
        # Indices for common queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rules_task ON rules(task_tag)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_concepts_task ON concepts(task_tag)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task_tag)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_impact ON episodes(impact_score)")
        
        conn.commit()
        conn.close()
    
    # === RULES ===
    
    def save_rule(self, rule: Rule) -> int:
        """Save or update a rule. Returns rule ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if rule.id is None:
            cursor.execute("""
                INSERT INTO rules (condition, consequence, priority, source, 
                                   task_tag, scope, support_count, success_rate, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pickle.dumps(rule.condition),
                rule.consequence,
                rule.priority,
                rule.source,
                rule.task_tag,
                rule.scope,
                rule.support_count,
                rule.success_rate,
                time.time()
            ))
            rule_id = cursor.lastrowid
        else:
            cursor.execute("""
                UPDATE rules SET condition=?, consequence=?, priority=?, source=?,
                                 task_tag=?, scope=?, support_count=?, success_rate=?
                WHERE id=?
            """, (
                pickle.dumps(rule.condition),
                rule.consequence,
                rule.priority,
                rule.source,
                rule.task_tag,
                rule.scope,
                rule.support_count,
                rule.success_rate,
                rule.id
            ))
            rule_id = rule.id
        
        conn.commit()
        conn.close()
        return rule_id
    
    def load_rules(self, task_tag: Optional[str] = None, scope: Optional[str] = None) -> List[Rule]:
        """Load rules, optionally filtered by task and scope."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM rules WHERE 1=1"
        params = []
        
        if task_tag:
            query += " AND (task_tag=? OR task_tag='global')"
            params.append(task_tag)
        
        if scope:
            query += " AND scope=?"
            params.append(scope)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        rules = []
        for row in rows:
            rules.append(Rule(
                id=row[0],
                condition=pickle.loads(row[1]),
                consequence=row[2],
                priority=row[3],
                source=row[4],
                task_tag=row[5],
                scope=row[6],
                support_count=row[7],
                success_rate=row[8],
                created_at=row[9]
            ))
        
        return rules
    
    def delete_rule(self, rule_id: int):
        """Delete a rule by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM rules WHERE id=?", (rule_id,))
        conn.commit()
        conn.close()
    
    # === CONCEPTS ===
    
    def save_concept(self, concept: Concept) -> int:
        """Save or update a concept. Returns concept ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO concepts 
            (name, hv_bytes, concept_type, task_tag, created_at, access_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            concept.name,
            concept.hv_bytes,
            concept.concept_type,
            concept.task_tag,
            time.time(),
            concept.access_count
        ))
        
        concept_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return concept_id
    
    def load_concept(self, name: str) -> Optional[Concept]:
        """Load a concept by name."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM concepts WHERE name=?", (name,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return Concept(
            id=row[0],
            name=row[1],
            hv_bytes=row[2],
            concept_type=row[3],
            task_tag=row[4],
            created_at=row[5],
            access_count=row[6]
        )
    
    def load_concepts(self, task_tag: Optional[str] = None) -> List[Concept]:
        """Load all concepts, optionally filtered by task."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if task_tag:
            cursor.execute(
                "SELECT * FROM concepts WHERE task_tag=? OR task_tag='global'",
                (task_tag,)
            )
        else:
            cursor.execute("SELECT * FROM concepts")
        
        rows = cursor.fetchall()
        conn.close()
        
        return [Concept(
            id=row[0], name=row[1], hv_bytes=row[2],
            concept_type=row[3], task_tag=row[4],
            created_at=row[5], access_count=row[6]
        ) for row in rows]
    
    # === EPISODES ===
    
    def record_episode(self, episode: Episode):
        """Buffer an episode for later persistence."""
        with self._buffer_lock:
            self._episode_buffer.append(episode)
            
            # Check if we should flush
            should_flush = (
                len(self._episode_buffer) >= self._buffer_limit or
                time.time() - self._last_flush > self._flush_interval
            )
        
        if should_flush:
            self.flush_episodes()
    
    def flush_episodes(self):
        """Flush buffered episodes to database."""
        with self._buffer_lock:
            if not self._episode_buffer:
                return
            
            episodes = self._episode_buffer.copy()
            self._episode_buffer.clear()
            self._last_flush = time.time()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executemany("""
            INSERT INTO episodes 
            (timestamp, task_tag, situation_hv_bytes, state_sketch, action, outcome, reward, impact_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (ep.timestamp, ep.task_tag, ep.situation_hv_bytes,
             pickle.dumps(ep.state_sketch), ep.action, ep.outcome, ep.reward, ep.impact_score)
            for ep in episodes
        ])
        
        conn.commit()
        conn.close()
    
    def load_recent_episodes(self, task_tag: str, limit: int = 100) -> List[Episode]:
        """Load most recent episodes for a task."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM episodes 
            WHERE task_tag=? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (task_tag, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [Episode(
            id=row[0], timestamp=row[1], task_tag=row[2],
            situation_hv_bytes=row[3], state_sketch=pickle.loads(row[4]),
            action=row[5], outcome=row[6], reward=row[7],
            impact_score=row[8] if len(row) > 8 else 0.0
        ) for row in rows]
    
    def count_episodes(self, task_tag: Optional[str] = None) -> int:
        """Count episodes, optionally filtered by task."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if task_tag:
            cursor.execute("SELECT COUNT(*) FROM episodes WHERE task_tag=?", (task_tag,))
        else:
            cursor.execute("SELECT COUNT(*) FROM episodes")
        
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def sample_random_episodes(self, limit: int = 32) -> List[Episode]:
        """
        Sample random episodes from the database (for Deep Dreaming).
        Uses ORDER BY RANDOM() which is sufficient for SQLite at this scale.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM episodes 
            ORDER BY RANDOM() 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [Episode(
            id=row[0], timestamp=row[1], task_tag=row[2],
            situation_hv_bytes=row[3], state_sketch=pickle.loads(row[4]),
            action=row[5], outcome=row[6], reward=row[7],
            impact_score=row[8] if len(row) > 8 else 0.0
        ) for row in rows]
    
    def prune_old_episodes(self, task_tag: str, keep_count: int = 10000):
        """Remove oldest episodes beyond capacity."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get ID threshold
        cursor.execute("""
            SELECT id FROM episodes 
            WHERE task_tag=? 
            ORDER BY timestamp DESC 
            LIMIT 1 OFFSET ?
        """, (task_tag, keep_count))
        
        row = cursor.fetchone()
        if row:
            threshold_id = row[0]
            cursor.execute(
                "DELETE FROM episodes WHERE task_tag=? AND id < ?",
                (task_tag, threshold_id)
            )
            conn.commit()
        
        conn.close()
    
    def close(self):
        """Flush and close database."""
        self.flush_episodes()
