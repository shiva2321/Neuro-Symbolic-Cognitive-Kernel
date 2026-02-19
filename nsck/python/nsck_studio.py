#!/usr/bin/env python3
"""
NSCK Studio: The Neuro-Symbolic Cognitive Kernel
================================================
The main entry point for the "Embodied" Agent.

Architecture:
- Brain: CognitiveEngine (Global Workspace, Planning, Safety)
- Eyes:  SNNPerceptionModule (Spikes -> Hypervectors)
- Voice: VSALanguageModule (Neuro-Symbolic Parsing)

Usage:
    python nsck_studio.py
"""

import sys
import os
import time
import numpy as np
from typing import Optional

# Ensure project root is in path so 'python.core...' works
# We need 'd:\Node_network\nsck' to be in sys.path
# Script is at: d:\Node_network\nsck\python\nsck_studio.py
# Dirname: d:\Node_network\nsck\python
# .. : d:\Node_network\nsck
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

# Also add the grand-parent just in case 'from nsck.python...' is used anywhere
grand_parent = os.path.abspath(os.path.join(root_path, ".."))
if grand_parent not in sys.path:
    sys.path.append(grand_parent)

from python.core.reasoning.cognitive_engine import create_cognitive_engine
from python.core.perception.grounding_verifier import GroundingVerifier

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def type_text(text: str, delay: float = 0.01):
    """Simulate typing effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

class AgentShell:
    def __init__(self):
        print(f"{BOLD}{CYAN}Initializing NSCK Neuro-Symbolic Kernel...{RESET}")
        
        # 1. Create Engine (Auto-loads SNN and Language if configured)
        self.engine = create_cognitive_engine()
        
        # 2. Register Default Task (The "Studio" Environment)
        self.engine.register_task("studio", verifier=GroundingVerifier())
        
        print(f"{GREEN}[OK] Brain Online (CognitiveEngine){RESET}")
        
        if self.engine.perception:
            print(f"{GREEN}[OK] Eyes Online (SNNPerceptionModule - 256 Neurons){RESET}")
        else:
            print(f"{YELLOW}[WARN] Eyes Offline (SNN module missing){RESET}")
            
        if self.engine.language.use_vsa:
             print(f"{GREEN}[OK] Voice Online (VSA Neuro-Symbolic Parser){RESET}")
        else:
             print(f"{YELLOW}[WARN] Voice Offline (Using LLM/Mock){RESET}")
             
        # 3. Load Memory if available
        # Check for data/vsa_memory.pkl
        memory_path = os.path.join(root_path, "data", "vsa_memory.pkl")
        if os.path.exists(memory_path):
             print(f"{CYAN}[Memory] Loading persisted knowledge from {memory_path}...{RESET}")
             try:
                 # Access memory via VSA backend if possible
                 if hasattr(self.engine.language, "vsa_backend"):
                     self.engine.language.vsa_backend.memory.load(memory_path)
             except Exception as e:
                 print(f"{RED}[Memory] Failed to load: {e}{RESET}")
        
        # 4. Register Thought Stream Listener
        self.engine.register_broadcaster(self._on_broadcast)
        
    def _on_broadcast(self, msg_type: str, content: str):
        """Real-time thought stream."""
        print(f"{YELLOW}   [Thought] {msg_type}: {content}{RESET}")

    def perceive_environment(self) -> np.ndarray:
        """Simulate sensory input (random noise for now)."""
        # In a real system, this comes from camera/mic.
        # Here we generate a random 64-dim pattern to stimulate the SNN.
        return np.random.rand(64).astype(np.float32)

    def run_loop(self):
        print(f"\n{BOLD}NSCK Agent Ready.{RESET}")
        print("Commands: 'perceive' (feed SNN), 'status', 'quit', or type a sentence.\n")
        
        while True:
            try:
                user_input = input(f"{CYAN}USER > {RESET}").strip()
                if not user_input: continue
                
                if user_input.lower() in ("quit", "exit"):
                    print("Shutting down.")
                    break
                    
                elif user_input.lower() == "status":
                    stats = self.engine.get_stats()
                    print(f"\n{BOLD}System Status:{RESET}")
                    for k, v in stats.items():
                        print(f"  {k}: {v}")
                    print()
                    
                elif user_input.lower() == "perceive":
                    print(f"{CYAN}[Visual Cortex] Scanning environment...{RESET}")
                    sensory_data = self.perceive_environment()
                    
                    # Run Full Cognitive Cycle (SNN -> GWT -> Action)
                    state = self.engine.perceive_and_decide(sensory_data, task_tag="studio")
                    
                    print(f"\n{GREEN}DECISION -> {state.chosen_action}{RESET}")
                    print(f"Confidence: {state.confidence:.2f}")
                    if state.explanation:
                        print(f"Reason: {state.explanation}")
                    print()
                    
                elif user_input.lower().startswith("train "):
                    # Syntax: train <dataset> <steps>
                    parts = user_input.split()
                    dataset = parts[1] if len(parts) > 1 else "wikitext"
                    try:
                        steps = int(parts[2]) if len(parts) > 2 else 100
                    except ValueError:
                        steps = 100
                        
                    print(f"{CYAN}[Training] Initializing VSA Trainer on '{dataset}' for {steps} steps...{RESET}")
                    
                    # Lazy Import
                    from python.core.training.vsa_trainer import VSATrainer
                    trainer = VSATrainer(memory_path=os.path.join(root_path, "data", "vsa_memory.pkl"))
                    trainer.train(dataset_name=dataset, steps=steps)
                    
                    # Reload memory into engine after training
                    if hasattr(self.engine.language, "vsa_backend"):
                        self.engine.language.vsa_backend.memory.load(trainer.memory_path)
                        print(f"{GREEN}[System] Memory reloaded. New concept count: {len(trainer.memory.concept_hvs)}{RESET}")

                elif user_input.lower() == "test cognition":
                    print(f"{CYAN}[Testing] Running Cognitive Turing Test...{RESET}")
                    from python.tests.cognitive_turing_test import CognitiveTuringTest
                    tester = CognitiveTuringTest(memory_path=os.path.join(root_path, "data", "vsa_memory.pkl"))
                    tester.run_all()
                    print(f"{GREEN}[Testing] Complete. Check COGNITIVE_REPORT_CARD.md{RESET}")
                    
                else:
                    # Treat as Natural Language
                    # 1. Understand (VSA Parser)
                    print(f"{CYAN}[Language] Listening...{RESET}")
                    understanding = self.engine.language.understand(user_input)
                    
                    intent = understanding["structured_output"]
                    intent_name = intent.get('intent')
                    entities = intent.get('entities', [])
                    
                    if intent_name and intent_name != "unknown":
                        print(f"  {GREEN}Understood:{RESET} {BOLD}{intent_name}{RESET} on {entities}")
                    elif entities:
                         print(f"  {YELLOW}I heard you mention {entities}, but I'm not sure what to do with them.{RESET}")
                    else:
                         print(f"  {YELLOW}I didn't understand that.{RESET}")

                    if intent.get("relation") == "vsa_parsed" and hasattr(self.engine.language, "vsa_backend"):
                        mem_count = len(self.engine.language.vsa_backend.memory.concept_hvs)
                        print(f"  (Parsed via VSA: {mem_count} concepts in memory)")
                        
                    # 2. Respond (Mock for now, or VSA generation later)
                    response = self.engine.process_dialogue(user_input)
                    type_text(f"{GREEN}AGENT: {response}{RESET}")
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except Exception as e:
                print(f"{RED}Error: {e}{RESET}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    agent = AgentShell()
    agent.run_loop()
