"""
NSCK Language Module
====================
Role: The "Broca's Area" of the system.
Responsibility:
1. Translate Natural Language -> Grounded VSA Hypervectors (Understanding)
2. Translate VSA Hypervectors -> Natural Language (Generation)

CRITICAL ARCHITECTURAL CONSTRAINT:
----------------------------------
The LLM is a PERIPHERAL. It does NOT make decisions.
It simply translates intent/meaning to/from the core system's symbolic language.
"""

import os
import sys
from typing import Dict, Any, Optional, List
import json

# Try to import dependencies, handle gracefully if missing
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

# Import Core Semantics
from lingua_cortex import get_lingua_cortex, SemanticFingerprint

# Fallback config
DEFAULT_MODEL_PATH = "models/phi-3-mini-4k-instruct.Q4_K_M.gguf"

class LanguageModule:
    """
    Interfaces with a local LLM to provide natural language capabilities.
    Maintains strict separation between 'Language' (LLM) and 'Thought' (VSA).
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.cortex = get_lingua_cortex()
        self.llm = None
        self.mock_mode = False
        
        if not LLAMA_AVAILABLE:
            print("WARNING: llama-cpp-python not installed. LanguageModule running in MOCK mode.")
            self.mock_mode = True
            return

        if not os.path.exists(model_path):
            print(f"WARNING: Model file not found at {model_path}. LanguageModule running in MOCK mode.")
            self.mock_mode = True
            return

        # Initialize LLM only if available and file exists
        print(f"Loading LLM from {model_path}...")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=4,
                verbose=False
            )
            print("LLM loaded successfully.")
        except Exception as e:
            print(f"ERROR loading LLM: {e}. Reverting to MOCK mode.")
            self.mock_mode = True

    def understand(self, text: str) -> Dict[str, Any]:
        """
        Input: User text (e.g. "Go to the food")
        Output: Structured intent + Grounded Semantic Hypervector
        """
        if self.mock_mode:
            return self._mock_understand(text)
            
        # 1. Prompt LLM to extract structure
        system_prompt = (
            "You are a parser. Extract intent and entities from the user's command. "
            "Output JSON only. "
            "Schema: {'intent': str, 'entities': List[str], 'relation': str}"
        )
        
        prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{text}<|end|>\n<|assistant|>\n"
        
        response = self.llm(
            prompt, 
            max_tokens=128, 
            stop=["<|end|>"], 
            echo=False,
            temperature=0.1 # Deterministic for parsing
        )
        
        raw_output = response['choices'][0]['text'].strip()
        
        try:
            # Simple heuristic cleaning if LLM adds markdown
            if "```json" in raw_output:
                raw_output = raw_output.split("```json")[1].split("```")[0]
            elif "{" in raw_output:
                raw_output = "{" + raw_output.split("{", 1)[1]
                
            structured = json.loads(raw_output)
        except Exception as e:
            print(f"LLM Parsing failed: {e}. Output was: {raw_output}")
            structured = {"intent": "unknown", "entities": [], "relation": "none"}

        # 2. Ground to VSA (Link to Semantic Map)
        # This crosses the boundary from "Text" to "Meaning" (Vectors)
        grounded_hv = self._ground_to_vsa(structured)
        
        return {
            "structured_output": structured,
            "grounded_hv": grounded_hv
        }

    def generate(self, intent_data: Dict[str, Any], context_hv: Optional[SemanticFingerprint] = None) -> str:
        """
        Input: System intent (e.g. {'action': 'move', 'target': 'food'})
        Output: Natural language response.
        """
        if self.mock_mode:
            return self._mock_generate(intent_data)

        # Construct prompt from system state
        action = intent_data.get('action', 'unknown')
        target = intent_data.get('target', 'unknown')
        reason = intent_data.get('reason', 'no reason provided')
        
        prompt = (
            f"<|user|>\nDescribe this action naturally: I am deciding to {action} towards {target} because {reason}.<|end|>\n"
            f"<|assistant|>\n"
        )
        
        response = self.llm(
            prompt,
            max_tokens=64,
            stop=["<|end|>"],
            temperature=0.7
        )
        return response['choices'][0]['text'].strip()

    def _ground_to_vsa(self, structured: Dict) -> Optional[SemanticFingerprint]:
        """
        Convert structured concepts into the Semantic Map's hypervector space.
        This is the crucial 'Understanding' step.
        """
        # Simple additive composition for now
        # V_meaning = V_intent + V_entity1 + V_entity2
        
        combined_fp = None
        
        # Gather keywords
        keywords = [structured.get('intent', '')] + structured.get('entities', [])
        
        for word in keywords:
            if not word: continue
            fp = self.cortex.get_fingerprint(word)
            
            # If word unknown, try to visualize/learn it (omitted for now) or skip
            if fp:
                if combined_fp is None:
                    combined_fp = fp
                else:
                    combined_fp = combined_fp.union(fp) # Superposition
                    
        return combined_fp

    # --- Mocks for development without weights ---
    def _mock_understand(self, text: str) -> Dict[str, Any]:
        """Hardcoded rules for testing without LLM."""
        text = text.lower()
        if "food" in text:
            return {
                "structured_output": {"intent": "seek", "entities": ["food"], "relation": "target"},
                "grounded_hv": self.cortex.get_fingerprint("food")
            }
        elif "run" in text or "move" in text:
            return {
                "structured_output": {"intent": "move", "entities": [], "relation": "none"},
                "grounded_hv": self.cortex.get_fingerprint("move")
            }
        return {"structured_output": {"intent": "none", "entities": []}, "grounded_hv": None}

    def _mock_generate(self, intent_data: Dict) -> str:
        return f"[MOCK LLM] I will {intent_data.get('action')} the {intent_data.get('target')}."
