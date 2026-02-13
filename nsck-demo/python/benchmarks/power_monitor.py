import psutil
import time
import threading
import os
import sys

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../')))
from python.core.language.text_knowledge_learner import TextKnowledgeLearner
from python.core.language.language_module import LanguageModule

class PowerMonitor:
    def __init__(self, tdp_watts=65.0, idle_watts=10.0, poll_interval=0.1):
        self.tdp = tdp_watts
        self.idle = idle_watts
        self.interval = poll_interval
        self.running = False
        self.stats = []
        self.thread = None

    def start(self):
        self.running = True
        self.stats = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor_loop(self):
        process = psutil.Process(os.getpid())
        while self.running:
            cpu_pct = process.cpu_percent(interval=None) / psutil.cpu_count() # Normalized per core? Or total system?
            # psutil.cpu_percent() defaults to total system wide or process?
            # process.cpu_percent() is process specific.
            # But converting to system wattage needs system usage.
            
            sys_cpu = psutil.cpu_percent(interval=None)
            mem_mb = process.memory_info().rss / (1024 * 1024)
            
            # Simple Linear Power Model
            # Watts = Idle + (TDP * (System_CPU / 100))
            # (Assuming Single CPU system)
            watts = self.idle + (self.tdp * (sys_cpu / 100.0))
            
            self.stats.append({
                'time': time.time(),
                'cpu_sys': sys_cpu,
                'mem_mb': mem_mb,
                'watts': watts
            })
            time.sleep(self.interval)

    def get_summary(self):
        if not self.stats:
            return "No data collected."
        
        avg_watts = sum(s['watts'] for s in self.stats) / len(self.stats)
        peak_watts = max(s['watts'] for s in self.stats)
        avg_cpu = sum(s['cpu_sys'] for s in self.stats) / len(self.stats)
        peak_mem = max(s['mem_mb'] for s in self.stats)
        total_energy_joules = avg_watts * (self.stats[-1]['time'] - self.stats[0]['time'])
        
        return {
            "avg_watts": avg_watts,
            "peak_watts": peak_watts,
            "avg_cpu_percent": avg_cpu,
            "peak_memory_mb": peak_mem,
            "total_energy_joules": total_energy_joules,
            "duration_seconds": self.stats[-1]['time'] - self.stats[0]['time']
        }

def run_benchmark():
    print("=== Power Benchmark: Xylophone Planets Learning ===")
    
    # Setup
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "test_corpus")
    corpus_file = os.path.join(data_dir, "xylophone_planets.txt")
    
    monitor = PowerMonitor(poll_interval=0.01)
    
    # Init Learner
    print("[1] Initializing...")
    lang_mod = LanguageModule()
    learner = TextKnowledgeLearner(language_module=lang_mod)
    
    # Start Monitor
    print("[2] Starting Power Monitor...")
    # Warm up psutil before starting monitor
    psutil.cpu_percent()
    time.sleep(0.1)
    
    monitor.start()
    
    start_time = time.time()
    
    # Run Workload
    print(f"[3] Learning from {os.path.basename(corpus_file)}...")
    # Repeat learning to create load
    for _ in range(5):
        session = learner.learn_from_text_file(corpus_file)
    
    # Run Queries to simulate reasoning load
    print("[4] Running Reasoning Queries (x100)...")
    q_start = time.time()
    for _ in range(100):
        learner.query_learned_knowledge("What processes support life on Xylophone planets?")
    q_end = time.time()
    
    print(f"    Queries took {q_end - q_start:.4f}s")
    
    # Ensure at least 1 second of monitoring
    elapsed = time.time() - start_time
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    
    end_time = time.time()
    
    # Stop Monitor
    monitor.stop()
    print("[5] Benchmark Complete.")
    
    # Report
    summary = monitor.get_summary()
    print("\n=== Power Analysis ===")
    print(f"Duration:     {summary['duration_seconds']:.2f} s")
    print(f"Avg Power:    {summary['avg_watts']:.2f} W")
    print(f"Peak Power:   {summary['peak_watts']:.2f} W")
    print(f"Total Energy: {summary['total_energy_joules']:.2f} J")
    print(f"Avg CPU Util: {summary['avg_cpu_percent']:.1f} %")
    print(f"Peak Memory:  {summary['peak_memory_mb']:.1f} MB")
    print("======================")

if __name__ == "__main__":
    # Warm up psutil
    psutil.cpu_percent()
    run_benchmark()
