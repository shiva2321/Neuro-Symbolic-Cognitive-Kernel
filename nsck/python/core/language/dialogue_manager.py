"""
NSCK Dialogue Manager
=====================
Orchestrates conversation, context tracking, and intent routing.
Acts as the bridge between the User and the Cognitive Engine.
"""

from collections import deque
from typing import Optional, Tuple, Any, Dict
import re

# We will need to interact with these systems
# from language_module import LanguageModule (Passed in dependency injection)
# from cognitive_engine import CognitiveEngine (Passed in dependency injection)

class DialogueManager:
    def __init__(self, cognitive_engine, language_module):
        self.engine = cognitive_engine
        self.language = language_module
        
        # Store last 10 turns given as (Sender, Text)
        self.context_window = deque(maxlen=10) 
    
    def reset(self):
        """Clear the dialogue context window."""
        self.context_window.clear()
        print("[DIALOGUE] Context window cleared.")
        
    def process_turn(self, user_input: str) -> str:
        """Process one dialogue turn from the user."""
        # 1. Add to context
        self.context_window.append(("user", user_input))
        
        # 2. Resolve references using context
        # (e.g. "eat it" -> "eat food")
        resolved_input = self.resolve_anaphora(user_input)
        
        # 3. Understand (LLM -> Intent+VSA)
        understanding = self.language.understand(resolved_input)
        structured = understanding.get("structured_output", {})
        grounded_hv = understanding.get("grounded_hv", None)
        
        intent = structured.get("intent", "unknown")
        
        # 4. Route to appropriate handler
        response = ""
        
        # Priority: Specific intents first
        if "why" in resolved_input.lower() or intent == "explain":
             response = self.generate_explanation()
        elif intent == "seek" or intent == "move":
            response = self.handle_command(structured, grounded_hv)
        elif intent == "question" or "?" in user_input:
            response = self.handle_question(structured)
        else:
            response = "I'm not sure what you mean. Can you rephrase?"
        
        # 5. Add response to context
        self.context_window.append(("agent", response))
        return response
    
    def respond(self, text: str) -> str:
        """
        Generate a response to input text (API for real-world eval).
        Alias for process_turn() for compatibility with test harness.
        """
        return self.process_turn(text)
    
    def resolve_anaphora(self, text: str) -> str:
        """
        Simple keyword replacement for pronouns strings like 'it', 'that'.
        Looks at the LAST agent utterance to find entities.
        """
        target_words = ["it", "that", "the same one"]
        text_lower = text.lower()
        
        # Quick check to see if we even need to resolve
        needs_resolution = any(w in text_lower.split() for w in target_words)
        if not needs_resolution:
            return text
            
        # Look backwards in context
        last_entity = None
        
        for sender, message in reversed(self.context_window):
            if sender == "agent":
                # Clean punctuation for matching
                clean_message = re.sub(r'[^\w\s]', '', message)
                words = clean_message.split()
                for w in words:
                    # Very naive: assume 'food', 'wall', 'enemy' are entities
                    if w.lower() in ["food", "wall", "obstacle", "enemy"]:
                        last_entity = w
                        break
            
            if last_entity: break
        
        if last_entity:
            # Replace 'it' with the entity
            # Regex to replace standalone 'it'
            pattern = re.compile(r'\b(it|that)\b', re.IGNORECASE)
            return pattern.sub(last_entity, text)
            
        return text

    def handle_command(self, structured: Dict, grounded_hv: Any) -> str:
        """Execute a command via the Cognitive Engine.

        Delegates to the engine's ``process_dialogue`` when available,
        otherwise generates a response via the language module.
        """
        entities = structured.get("entities", [])
        target = entities[0] if entities else "something"
        intent = structured.get("intent", "act")

        # Delegate to engine if available
        if self.engine and hasattr(self.engine, "set_mission_goal"):
            try:
                self.engine.set_mission_goal(intent, 0)
            except Exception:
                pass

        if self.language:
            return self.language.generate({
                "action": intent,
                "target": target,
                "reason": "user command",
            })
        return f"Executing {intent} towards {target}."

    def handle_question(self, structured: Dict) -> str:
        """Answer a query about current cognitive state."""
        if self.engine and hasattr(self.engine, "get_stats"):
            stats = self.engine.get_stats()
            return (
                f"I have made {stats.get('decisions', 0)} decisions and "
                f"recorded {stats.get('episodes_recorded', 0)} episodes."
            )
        return "I don't have enough context to answer that."

    def generate_explanation(self) -> str:
        """Explain the most recent decision using the engine's explainer."""
        if self.engine and hasattr(self.engine, "explain"):
            return self.engine.explain()
        return "No explanation available — engine not connected."
