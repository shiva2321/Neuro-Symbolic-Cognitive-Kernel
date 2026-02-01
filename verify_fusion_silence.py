
import sys
import os
import multiprocessing

# Add the directory to sys.path
sys.path.append(r'd:\NSCK_v1\nsck-demo\python')

import symbol_grounding

def worker_task(i):
    # This should NOT print fusion logs because it doesn't call get_kernel_engine()
    print(f"Worker {i} starting...")
    import symbol_grounding
    print(f"Worker {i} finished.")

def test_lazy_init():
    print("Main process: Importing symbol_grounding...")
    # This should NOT print fusion logs yet
    import symbol_grounding
    
    print("\nMain process: Spawning workers (simulating DataLoader)...")
    processes = []
    for i in range(2):
        p = multiprocessing.Process(target=worker_task, args=(i,))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    print("\nMain process: Accessing ActionSemantics (should trigger fusion ONCE)...")
    from symbol_grounding import ActionSemantics
    ActionSemantics.get_goal_alignment("snake", {"head": [0,0], "food": [1,1]})
    
    print("\nTest complete.")

if __name__ == "__main__":
    # Correct multiprocessing start for Windows
    multiprocessing.freeze_support()
    test_lazy_init()
