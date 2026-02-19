"""
Cognitive Control Loop (Basal Ganglia Emulation)
================================================

Orchestrates the flow of information between cortical modules.
Simulates the "Action Selection" loop of the Basal Ganglia.

Cycle:
1. Perception (Input -> Working Memory)
2. Utility Calculation (Striatum)
3. Action Selection (GPi/Thalamus)
4. Execution (Cortical Update)
"""

from typing import Dict, Any, List
import time
from python.core.language.parser import LeftCornerParser
import python.core.vsa.hypervec_shim as hv

class CognitiveController:
    def __init__(self):
        self.parser = LeftCornerParser()
        self.cycle_count = 0
        self.working_memory: Dict[str, Any] = {} # "Global Workspace"
    
    def perceive(self, input_signal: Any):
        """Phase 1: Perception"""
        # In a real system, this converts raw data to spikes/vectors.
        # Here we accept token lists for language.
        self.working_memory["input"] = input_signal

    def select_action(self) -> str:
        """Phase 2 & 3: Utility & Selection (Basal Ganglia)"""
        # Check state of buffers
        inp = self.working_memory.get("input")
        
        if isinstance(inp, list) and inp:
            # If we have tokens, we need to PARSE
            return "PARSE_SENTENCE"
        
        if self.parser.tree.similarity_robust(hv.HyperVector(seed=0)) < 0.99:
            # If tree is not empty (approx), maybe we can ACT?
            return "REPORT_RESULT"
            
        return "WAIT"

    def execute(self, action: str):
        """Phase 4: Execution (Thalamus/Cortex)"""
        if action == "PARSE_SENTENCE":
            tokens = self.working_memory["input"]
            print(f"[Control] Executing Parse on {tokens}...")
            # The parser itself runs a micro-loop of Shift/Reduce
            # In a full SNN, this would be step-by-step.
            self.parser.parse_sentence(tokens)
            self.working_memory["input"] = None # Consumed
            
        elif action == "REPORT_RESULT":
            print(f"[Control] Parse Complete. Tree State: [Vector]")
            # In a real system, we'd output to Motor Cortex
            self.report()
            # Clear tree after reporting?
            # self.parser.reset() 

    def report(self):
        # Analyze the tree (introspection)
        # We can try to extract the specific structure
        pass

    def step(self):
        """Run one cognitive cycle (approx 50ms)"""
        self.cycle_count += 1
        action = self.select_action()
        if action != "WAIT":
            self.execute(action)

    def run_pipeline(self, input_data):
        self.perceive(input_data)
        # Run until idle
        for _ in range(10): 
            self.step()
            if self.working_memory["input"] is None and self.parser.symbolic_stack == []:
                break
