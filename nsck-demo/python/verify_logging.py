
import unittest
import os
import time
import sqlite3
from logger_service import StructuredLogger

class TestLoggerService(unittest.TestCase):
    
    def setUp(self):
        self.test_db = "test_logs.db"
        self.test_txt = "test_session.txt"
        
        # Clean previous run
        if os.path.exists(self.test_db): os.remove(self.test_db)
        if os.path.exists(self.test_txt): os.remove(self.test_txt)
        
        self.logger = StructuredLogger(db_path=self.test_db, txt_path=self.test_txt)
        
    def tearDown(self):
        self.logger.close()
        # Wait a bit for file release
        time.sleep(0.5)
        # Cleanup is manual if you want to inspect files
        # if os.path.exists(self.test_db): os.remove(self.test_db)
        # if os.path.exists(self.test_txt): os.remove(self.test_txt)

    def test_dual_write(self):
        """Verify writes go to both DB and TXT."""
        msg = "Unit Test Message"
        self.logger.log("UNITTEST", msg)
        
        time.sleep(0.5) # Allow thread to write
        
        # Check DB
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT message FROM logs WHERE source='UNITTEST'")
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], msg)
        conn.close()
        
        # Check TXT
        with open(self.test_txt, "r") as f:
            content = f.read()
            self.assertIn(msg, content)
            
    def test_stress_logging(self):
        """Log 1000 messages quickly and verify count."""
        count = 1000
        start = time.time()
        for i in range(count):
            self.logger.log("STRESS", f"Message {i}")
            
        # Give it time to flush
        while self.logger.log_queue.qsize() > 0:
            time.sleep(0.1)
            
        duration = time.time() - start
        print(f"Logged {count} items in {duration:.4f}s (Main Thread)")
        
        # Verify
        time.sleep(1.0) # Ensure DB commit
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM logs WHERE source='STRESS'")
        rows = cursor.fetchone()[0]
        self.assertEqual(rows, count)
        conn.close()

if __name__ == "__main__":
    unittest.main()
