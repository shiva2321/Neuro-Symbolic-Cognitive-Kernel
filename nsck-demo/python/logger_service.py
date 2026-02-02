
import sqlite3
import threading
import queue
import time
import json
import os
from typing import Dict, Any, Optional

class StructuredLogger:
    """
    Robust logging service for the NSCK AGI.
    
    Features:
    1. Dual-Write: Writes to both SQLite (structural) and TXT (human readable).
    2. Non-Blocking: Uses a background thread to handle IO.
    3. Fail-Safe: Flushes on crash or shutdown.
    4. Categorized: Supports different log levels and sources.
    """
    
    DB_NAME = "nsck_logs.db"
    TXT_NAME = "nsck_session.txt"
    
    def __init__(self, db_path=None, txt_path=None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = db_path or os.path.join(self.base_dir, self.DB_NAME)
        self.txt_path = txt_path or os.path.join(self.base_dir, self.TXT_NAME)
        
        self.log_queue = queue.Queue()
        self.running = True
        
        # Start the IO thread
        self.thread = threading.Thread(target=self._io_loop, daemon=True)
        self.thread.start()
        
        self.log("SYSTEM", "Logger initialized.")

    def log(self, source: str, message: str, level: str = "INFO", metadata: Optional[Dict[str, Any]] = None):
        """
        Public API to log an event.
        Returns immediately (Non-blocking).
        """
        event = {
            "timestamp": time.time(),
            "source": source,
            "message": message,
            "level": level,
            "metadata": json.dumps(metadata) if metadata else "{}"
        }
        self.log_queue.put(event)

    def telemetry(self, task: str, metric: str, value: float, step: int = 0):
        """
        Log numerical data for charting.
        """
        event = {
            "type": "telemetry",
            "timestamp": time.time(),
            "task": task,
            "metric": metric,
            "value": value,
            "step": step
        }
        self.log_queue.put(event)

    def close(self):
        """
        Graceful shutdown. Waits for queue to empty.
        """
        self.log("SYSTEM", "Logger shutting down...")
        self.running = False
        self.thread.join(timeout=2.0)

    def _io_loop(self):
        """
        Background thread that actually writes to disk.
        """
        # 1. Open DB Connection
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 2. Initialize Schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                source TEXT,
                level TEXT,
                message TEXT,
                metadata TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                task TEXT,
                metric TEXT,
                value REAL,
                step INTEGER
            )
        ''')
        conn.commit()
        
        # 3. Open Text File
        # Use 'a' (append) mode. 'utf-8' encoding.
        txt_file = open(self.txt_path, "a", encoding="utf-8")
        
        while self.running or not self.log_queue.empty():
            try:
                # Get event with timeout to allow checking self.running
                event = self.log_queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            # Handle Event
            try:
                if event.get("type") == "telemetry":
                    # DB Insert Only for Telemetry (Keep txt clean)
                    cursor.execute(
                        "INSERT INTO telemetry (timestamp, task, metric, value, step) VALUES (?, ?, ?, ?, ?)",
                        (event["timestamp"], event["task"], event["metric"], event["value"], event["step"])
                    )
                else:
                    # DB Insert
                    cursor.execute(
                        "INSERT INTO logs (timestamp, source, level, message, metadata) VALUES (?, ?, ?, ?, ?)",
                        (event["timestamp"], event["source"], event["level"], event["message"], event["metadata"])
                    )
                    
                    # TXT Append
                    time_str = time.strftime("%H:%M:%S", time.localtime(event["timestamp"]))
                    log_line = f"[{time_str}] [{event['level']}] [{event['source']}] {event['message']}\n"
                    txt_file.write(log_line)
                    txt_file.flush() # Flush text immediately for real-time tail
                    
                conn.commit() # Commit DB transaction
                
            except Exception as e:
                print(f"LOGGER ERROR: {e}")
                
        # Cleanup
        conn.close()
        txt_file.close()

# Singleton Instance (Lazy init)
_logger_instance = None

def get_logger():
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = StructuredLogger()
    return _logger_instance

if __name__ == "__main__":
    # Test
    log = StructuredLogger()
    log.log("TEST", "Hello World")
    log.telemetry("snake", "score", 100)
    time.sleep(1)
    log.close()
    print("Test Complete. Check nsck_logs.db and nsck_session.txt")
