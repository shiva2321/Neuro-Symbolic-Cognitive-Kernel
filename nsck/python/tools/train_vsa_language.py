
import sys
import os
import time
from typing import Dict

# Path setup: resolve project root relative to this script
# script is in: <project_root>/nsck/python/tools
# we need <project_root>/nsck in sys.path to import 'python' package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from python.core.reasoning.cognitive_engine import create_cognitive_engine
from python.tools.dataset_streamer import TextStreamer

def train_language_loop(limit: int = 100):
    print(f"\n[Training] Starting VSA Language Training (Limit: {limit} sentences)...")
    
    # 1. Initialize Engine (with VSA Language)
    engine = create_cognitive_engine()
    # Force VSA just in case
    if not engine.language.use_vsa:
        print("[Error] Engine failed to initialize in VSA mode.")
        return

    # 2. Initialize Streamer
    streamer = TextStreamer(dataset_name="wikitext", subset="wikitext-2-v1", split="train")
    
    # 3. Training Loop
    count = 0
    start_time = time.time()
    # Access memory through the VSA backend
    # LanguageModule -> VSALanguageModule -> SemanticMemory
    if hasattr(engine.language, "vsa_backend") and engine.language.vsa_backend:
        memory = engine.language.vsa_backend.memory
    else:
        print("[Error] No VSA backend found.")
        return

    vocab_start = len(memory.concept_graph)
    
    print(f"[Training] Initial Vocab Size: {vocab_start}")
    
    for sentence in streamer.stream_sentences(limit=limit):
        count += 1
        print(f"\n--- Sentence {count} ---")
        try:
             print(f"Input: {sentence[:100]}..." if len(sentence) > 100 else f"Input: {sentence}")
        except UnicodeEncodeError:
             print(f"Input: {sentence[:100].encode('utf-8', 'replace')}...")
        
        # Parse & Learn
        # understand() now does dynamic learning via NLTK tagging + SemanticMemory.add_concept
        result = engine.language.understand(sentence)
        
        # Analyze Result
        intent = result.get("intent")
        entities = result.get("entities")
        print(f"Parsed: Intent={intent}, Entities={entities}")
        
        # Check Resonator Confidence? (Not easily exposed yet, but result implies some structure)
        
    end_time = time.time()
    duration = end_time - start_time
    vocab_end = len(memory.concept_graph)
    
    print("\n[Training] Complete.")
    print(f"Processed: {count} sentences")
    print(f"Time: {duration:.2f}s ({count/duration:.1f} sent/s)")
    print(f"Vocabulary Growth: {vocab_start} -> {vocab_end} (+{vocab_end - vocab_start} new concepts)")
    
    # Verify a new word
    # e.g. check if a common word from WikiText exists now
    if "valkyria" in memory.concept_hvs: # WikiText-2 starts with Valkyria Chronicles usually
         print("[Verification] Learned 'valkyria'!")

    # SAVE MEMORY
    save_path = os.path.join(project_root, "data", "vsa_memory.pkl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # We need to call save on the Python class wrapper, usually engine.language.memory
    # But engine.language might not expose it directly if it's VSALanguageModule wrapped.
    # Check VSALanguageModule in vsa_language_module.py
    
    # We accessed 'memory' via engine.language.vsa_backend.memory earlier.
    # That object is the SemanticMemory instance.
    if hasattr(memory, "save"):
        memory.save(save_path)
    else:
        print("[Error] Memory object does not have save method.")

if __name__ == "__main__":
    # Run a small batch to prove it works
    train_language_loop(limit=50)
