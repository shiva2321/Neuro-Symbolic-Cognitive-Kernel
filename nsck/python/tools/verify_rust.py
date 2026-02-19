
import sys
import os

# Path hack
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    import python.core.vsa.hypervec_shim as hypervec_rs
    print(f"[Rust Check] Hypervec Shim loaded: {hypervec_rs}")
    
    # Check if Rust class is available
    if hypervec_rs.SemanticMemoryConcurrent is not None:
         print("[Rust Check] SUCCESS: SemanticMemoryConcurrent is available (Rust backend).")
         
         # Test instantiation
         mem = hypervec_rs.SemanticMemoryConcurrent()
         print("[Rust Check] Instantiated Rust Memory.")
    else:
         print("[Rust Check] FAILURE: SemanticMemoryConcurrent is None (Python Fallback).")
         print("[Rust Check] This means the Rust extension (.pyd/.so) was not found or failed to load.")

except ImportError as e:
    print(f"[Rust Check] ImportError: {e}")
except Exception as e:
    print(f"[Rust Check] Unexpected Error: {e}")
