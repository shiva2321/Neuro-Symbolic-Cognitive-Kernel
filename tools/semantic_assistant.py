import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ncgn.semantic import semantic_brain

def print_header():
    print("="*60)
    print("🧠 NCGN KNOWLEDGE ASSISTANT v1.0")
    print("   Architecture: Graph-Based RDF Triples")
    print("   Storage:      Binary Flash Memory (mmap)")
    print("   Logic:        Deterministic Pathfinding")
    print("="*60)

def load_file(ai, filepath):
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    print(f"\n📖 Reading {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            # Chunking for better processing visualization
            print(f"   Size: {len(text)} bytes. Ingesting...")
            start = time.time()
            ai.learn_rdf(text)
            end = time.time()
            print(f"✓ Learned in {end - start:.2f} seconds.")
    except Exception as e:
        print(f"❌ Error: {e}")

def main_loop():
    ai = semantic_brain.SemanticBrain()
    print_header()

    while True:
        print("\nCOMMANDS: /load [file] | /wipe | /exit")
        user_input = input("USER > ").strip()

        if not user_input: continue

        # COMMAND HANDLING
        if user_input.upper() == "/EXIT":
            print("🔌 Shutting down.")
            ai.brain.close()
            break

        elif user_input.upper() == "/WIPE":
            confirm = input("⚠️  Are you sure you want to delete all memory? (y/n): ")
            if confirm.lower() == 'y':
                ai.factory_reset()
                print("✨ Memory wiped clean.")
            else:
                print("   Cancelled.")

        elif user_input.upper().startswith("/LOAD "):
            filepath = user_input[6:].strip()
            # Remove quotes if user added them
            filepath = filepath.replace('"', '').replace("'", "")
            load_file(ai, filepath)

        # QUERY HANDLING
        else:
            # Assume it's a question
            if "?" not in user_input: user_input += "?"
            ai.query(user_input)

if __name__ == "__main__":
    main_loop()