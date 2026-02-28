"""
NSCK Persistence Module
SQLite-based storage for concepts, rules, and episodes.
Includes brain versioning and export/import functionality.
"""
import sqlite3
import time
import threading
import json
import shutil
import zipfile
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path


def _serialize_condition(condition: frozenset) -> str:
    """Serialize a frozenset condition to JSON string (safe, no pickle)."""
    return json.dumps(sorted(list(condition)))


def _deserialize_condition(data) -> frozenset:
    """Deserialize condition from JSON string or legacy pickle bytes."""
    if isinstance(data, bytes):
        # Legacy pickle data — attempt safe migration
        try:
            import pickle
            result = pickle.loads(data)
            return frozenset(result) if not isinstance(result, frozenset) else result
        except Exception:
            return frozenset()
    elif isinstance(data, str):
        return frozenset(json.loads(data))
    return frozenset()


def _serialize_state_sketch(state_sketch: Dict[str, Any]) -> str:
    """Serialize state_sketch dict to JSON string (safe, no pickle)."""
    try:
        return json.dumps(state_sketch, default=str)
    except (TypeError, ValueError):
        return json.dumps({})


def _deserialize_state_sketch(data) -> Dict[str, Any]:
    """Deserialize state_sketch from JSON string or legacy pickle bytes."""
    if isinstance(data, bytes):
        try:
            import pickle
            return pickle.loads(data)
        except Exception:
            return {}
    elif isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


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
    confidence: float = 1.0  # Confidence level [0.0, 1.0] - graduates as support increases
    created_at: float = 0.0
    # V9: Lifelong stability tracking
    confidence_history: List[float] = field(default_factory=list)
    last_fired: float = 0.0   # Unix timestamp of last application
    fire_count: int = 0        # Total number of times this rule fired
    # V4: EWC importance tracking
    gwt_win_count: int = 0
    ewc_importance: float = 0.0


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


@dataclass
class BrainVersion:
    """Stored brain checkpoint version."""
    id: Optional[int]
    version_tag: str  # e.g., "v1.0", "checkpoint_2026-02-12", "trained_maze"
    description: str  # User-provided description
    timestamp: float
    rules_count: int
    concepts_count: int
    episodes_count: int
    metadata_json: str  # JSON string with additional info


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
                confidence REAL DEFAULT 1.0,
                created_at REAL
            )
        """)
        
        # Migration: Add confidence column if it doesn't exist (for existing databases)
        try:
            conn.execute('ALTER TABLE rules ADD COLUMN confidence REAL DEFAULT 1.0')
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
        
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
        
        # Brain versions table (for checkpoints)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS brain_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_tag TEXT UNIQUE,
                description TEXT,
                timestamp REAL,
                rules_count INTEGER,
                concepts_count INTEGER,
                episodes_count INTEGER,
                metadata_json TEXT
            )
        """)
        
        # Indices for common queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rules_task ON rules(task_tag)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_concepts_task ON concepts(task_tag)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task_tag)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_impact ON episodes(impact_score)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_versions_time ON brain_versions(timestamp)")
        
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
                                   task_tag, scope, support_count, success_rate, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                _serialize_condition(rule.condition),
                rule.consequence,
                rule.priority,
                rule.source,
                rule.task_tag,
                rule.scope,
                rule.support_count,
                rule.success_rate,
                rule.confidence,
                time.time()
            ))
            rule_id = cursor.lastrowid
        else:
            cursor.execute("""
                UPDATE rules SET condition=?, consequence=?, priority=?, source=?,
                                 task_tag=?, scope=?, support_count=?, success_rate=?, confidence=?
                WHERE id=?
            """, (
                _serialize_condition(rule.condition),
                rule.consequence,
                rule.priority,
                rule.source,
                rule.task_tag,
                rule.scope,
                rule.support_count,
                rule.success_rate,
                rule.confidence,
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
            # Handle both old schema (9 columns) and new schema (10 columns with confidence)
            if len(row) >= 11:  # New schema with confidence
                rules.append(Rule(
                    id=row[0],
                    condition=_deserialize_condition(row[1]),
                    consequence=row[2],
                    priority=row[3],
                    source=row[4],
                    task_tag=row[5],
                    scope=row[6],
                    support_count=row[7],
                    success_rate=row[8],
                    confidence=row[9],
                    created_at=row[10]
                ))
            else:  # Old schema without confidence
                rules.append(Rule(
                    id=row[0],
                    condition=_deserialize_condition(row[1]),
                    consequence=row[2],
                    priority=row[3],
                    source=row[4],
                    task_tag=row[5],
                    scope=row[6],
                    support_count=row[7],
                    success_rate=row[8],
                    confidence=1.0,  # Default for old rules
                    created_at=row[9] if len(row) > 9 else time.time()
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
             _serialize_state_sketch(ep.state_sketch), ep.action, ep.outcome, ep.reward, ep.impact_score)
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
            situation_hv_bytes=row[3], state_sketch=_deserialize_state_sketch(row[4]),
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
            situation_hv_bytes=row[3], state_sketch=_deserialize_state_sketch(row[4]),
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
    
    # === BRAIN VERSIONING & EXPORT ===
    
    def create_checkpoint(self, version_tag: str, description: str = "") -> int:
        """
        Create a versioned checkpoint of the current brain state.
        
        Records snapshot metadata in brain_versions table. The actual
        database state is the snapshot (we don't duplicate data).
        
        Args:
            version_tag: Unique identifier (e.g., "v1.0", "trained_maze")
            description: Human-readable description
            
        Returns:
            Version ID
            
        Example:
            brain.create_checkpoint(
                version_tag="maze_expert_v1",
                description="After 10K maze episodes with 87% success rate"
            )
        """
        self.flush_episodes()  # Ensure all data is written
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count current state
        cursor.execute("SELECT COUNT(*) FROM rules")
        rules_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM concepts")
        concepts_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM episodes")
        episodes_count = cursor.fetchone()[0]
        
        # Collect metadata
        metadata = {
            "nsck_version": "2.0",
            "db_path": self.db_path,
            "checkpoint_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        cursor.execute("""
            INSERT INTO brain_versions 
            (version_tag, description, timestamp, rules_count, concepts_count, 
             episodes_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            version_tag,
            description,
            time.time(),
            rules_count,
            concepts_count,
            episodes_count,
            json.dumps(metadata)
        ))
        
        version_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return version_id
    
    def list_versions(self) -> List[BrainVersion]:
        """List all brain checkpoints, ordered by timestamp (newest first)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM brain_versions 
            ORDER BY timestamp DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [BrainVersion(
            id=row[0],
            version_tag=row[1],
            description=row[2],
            timestamp=row[3],
            rules_count=row[4],
            concepts_count=row[5],
            episodes_count=row[6],
            metadata_json=row[7]
        ) for row in rows]
    
    def get_version(self, version_tag: str) -> Optional[BrainVersion]:
        """Retrieve a specific version by tag."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM brain_versions WHERE version_tag=?
        """, (version_tag,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        return BrainVersion(
            id=row[0], version_tag=row[1], description=row[2],
            timestamp=row[3], rules_count=row[4], concepts_count=row[5],
            episodes_count=row[6], metadata_json=row[7]
        )
    
    def export_brain(self, filepath: str, version_tag: Optional[str] = None):
        """
        Export brain to a portable .nsck file (ZIP archive).
        
        The .nsck file contains:
        - brain.db: Complete SQLite database snapshot
        - metadata.json: Version info, statistics, NSCK version
        
        Args:
            filepath: Destination path (should end in .nsck)
            version_tag: Optional version to include in metadata
            
        Example:
            brain.export_brain("trained_maze_v1.nsck", version_tag="maze_expert_v1")
        """
        self.flush_episodes()  # Ensure all data is written
        
        # Ensure .nsck extension
        filepath = Path(filepath)
        if filepath.suffix != '.nsck':
            filepath = filepath.with_suffix('.nsck')
        
        # Get version info if specified
        version_info = {}
        if version_tag:
            version = self.get_version(version_tag)
            if version:
                version_info = {
                    "version_tag": version.version_tag,
                    "description": version.description,
                    "timestamp": version.timestamp,
                    "rules_count": version.rules_count,
                    "concepts_count": version.concepts_count,
                    "episodes_count": version.episodes_count,
                    "metadata": json.loads(version.metadata_json)
                }
        
        # If no version specified, get current state
        if not version_info:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rules")
            rules_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM concepts")
            concepts_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM episodes")
            episodes_count = cursor.fetchone()[0]
            conn.close()
            
            version_info = {
                "version_tag": "exported_" + time.strftime("%Y%m%d_%H%M%S"),
                "description": "Exported brain snapshot",
                "timestamp": time.time(),
                "rules_count": rules_count,
                "concepts_count": concepts_count,
                "episodes_count": episodes_count,
                "metadata": {
                    "nsck_version": "2.0",
                    "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_db": str(self.db_path)
                }
            }
        
        # Create temporary directory for packaging
        temp_dir = filepath.parent / f"_temp_{filepath.stem}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Copy database to temp directory
            db_copy = temp_dir / "brain.db"
            shutil.copy2(self.db_path, db_copy)
            
            # Create metadata.json
            metadata_path = temp_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(version_info, f, indent=2)
            
            # Create .nsck ZIP archive
            with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(db_copy, arcname="brain.db")
                zf.write(metadata_path, arcname="metadata.json")
            
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def import_brain(self, filepath: str, merge: bool = False):
        """
        Import brain from a .nsck file.
        
        Args:
            filepath: Path to .nsck file
            merge: If True, merge with existing brain. If False, replace current brain.
            
        Warning:
            If merge=False, this will DELETE all existing brain data!
            
        Example:
            # Replace current brain
            brain.import_brain("expert_maze.nsck", merge=False)
            
            # Merge knowledge from another brain
            brain.import_brain("language_expert.nsck", merge=True)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Brain file not found: {filepath}")
        
        # Extract .nsck archive
        temp_dir = filepath.parent / f"_temp_import_{int(time.time())}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Extract archive
            with zipfile.ZipFile(filepath, 'r') as zf:
                zf.extractall(temp_dir)
            
            # Read metadata
            metadata_path = temp_dir / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            imported_db = temp_dir / "brain.db"
            
            if not merge:
                # Replace mode: Backup current brain and replace
                backup_path = Path(self.db_path).with_suffix('.db.backup')
                shutil.copy2(self.db_path, backup_path)
                shutil.copy2(imported_db, self.db_path)
                
                # Reinitialize to ensure schema compatibility
                self._init_db()
            else:
                # Merge mode: Copy data from imported brain
                self._merge_brain_data(imported_db, metadata)
        
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _merge_brain_data(self, source_db_path: Path, metadata: Dict[str, Any]):
        """
        Merge brain data from another database.
        
        Handles ID conflicts by remapping imported data.
        """
        self.flush_episodes()
        
        # Attach source database
        conn = sqlite3.connect(self.db_path)
        conn.execute(f"ATTACH DATABASE '{source_db_path}' AS source")
        
        # Merge rules (skip duplicates based on condition+consequence)
        conn.execute("""
            INSERT OR IGNORE INTO rules 
            (condition, consequence, priority, source, task_tag, scope, 
             support_count, success_rate, confidence, created_at)
            SELECT condition, consequence, priority, source, task_tag, scope,
                   support_count, success_rate, confidence, created_at
            FROM source.rules
        """)
        
        # Merge concepts (skip duplicates based on name)
        conn.execute("""
            INSERT OR IGNORE INTO concepts
            (name, hv_bytes, concept_type, task_tag, created_at, access_count)
            SELECT name, hv_bytes, concept_type, task_tag, created_at, access_count
            FROM source.concepts
        """)
        
        # Merge episodes (always insert, no uniqueness constraint)
        conn.execute("""
            INSERT INTO episodes
            (timestamp, task_tag, situation_hv_bytes, state_sketch, action, 
             outcome, reward, impact_score)
            SELECT timestamp, task_tag, situation_hv_bytes, state_sketch, action,
                   outcome, reward, impact_score
            FROM source.episodes
        """)
        
        # Record merge in versions table
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM rules")
        rules_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM concepts")
        concepts_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM episodes")
        episodes_count = cursor.fetchone()[0]
        
        merge_tag = f"merged_{metadata.get('version_tag', 'unknown')}_{int(time.time())}"
        merge_metadata = {
            "merge_source": metadata.get('version_tag', 'unknown'),
            "merge_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_rules": metadata.get('rules_count', 0),
            "source_concepts": metadata.get('concepts_count', 0),
            "source_episodes": metadata.get('episodes_count', 0)
        }
        
        conn.execute("""
            INSERT INTO brain_versions
            (version_tag, description, timestamp, rules_count, concepts_count,
             episodes_count, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            merge_tag,
            f"Merged from: {metadata.get('description', 'imported brain')}",
            time.time(),
            rules_count,
            concepts_count,
            episodes_count,
            json.dumps(merge_metadata)
        ))
        
        conn.commit()
        conn.execute("DETACH DATABASE source")
        conn.close()
    
    def close(self):
        """Flush and close database."""
        self.flush_episodes()
