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

from python.core.language.nlg import NLGEngine
from python.core.reasoning.causal_interface import CausalQueryService, MockCausalService, Explanation, Prediction

class DialogueManager:
    def __init__(self, cognitive_engine, language_module, causal_service: Optional[CausalQueryService] = None):
        self.engine = cognitive_engine
        self.language = language_module
        self.causal_service = causal_service or MockCausalService()
        self.nlg = NLGEngine()
        
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

        # 4a. Pre-VSA causal/factual pattern detector (runs before VSA routing)
        # Handles cases where VSA parser mis-routes causal questions.
        causal_response = self._try_causal_patterns(user_input)
        if causal_response:
            self.context_window.append(("agent", causal_response))
            return causal_response

        if "QUERY" in structured: # VSA structural intent
            # Input was "What is X?"
            # Structured: {"QUERY": "What", "TARGET": "X", ...}
            # The 'language.understand' should have extracted this from the Tree.
            target = structured.get("TARGET") or structured.get("entities", ["something"])[0]
            response = self.retrieve_knowledge(target)
            
        # Priority: Specific intents first
        elif intent == "explain" or intent == "why":
             response = self.handle_explanation(structured)
        elif intent == "predict" or intent == "what_if":
             response = self.handle_prediction(structured)
        elif intent == "seek" or intent == "move":
            response = self.handle_command(structured, grounded_hv)
        elif intent == "question":
             response = self.handle_question(structured)
        elif intent != "unknown":
            # Generic acknowledgement using Structural NLG
            action = intent
            targets = structured.get("entities", [])
            target = targets[0] if targets else "something"
            
            # Generate: "I noticed that you mentioned [action] related to [target]"
            # Using Frame: (I, noticed, that...)
            # We can use the Realizer for the core proposition
            # But "that you mentioned..." is a sub-clause. 
            # Let's simplify: "I noticed the [action]."
            
            response = self.nlg.realizer.realize_sentence("I", "notice", f"the {action}")
            # Append context
            if target != "something":
                 response += " " + self.nlg.realizer.realize_sentence("It", "involve", target)
        else:
            response = "I am unable to parse that structure."
        
        # 5. Add response to context
        self.context_window.append(("agent", response))
        return response

    def retrieve_knowledge(self, subject: str) -> str:
        """Retrieve knowledge about a subject from Semantic Memory."""
        if not self.engine or not hasattr(self.engine, "semantic_memory"):
            return "I have no memory module."
            
        mem = self.engine.semantic_memory
        
        # 1. Check if concept exists (Case handling by VSA usually capitalizes)
        sub_cap = subject.capitalize()
        
        if sub_cap not in mem.concept_graph:
             # Try to find closest match?
             return self.nlg.realizer.realize_sentence("I", "do not know", sub_cap)
        
        # 2. Get immediate relations
        out_edges = list(mem.concept_graph.out_edges(sub_cap, data=True))
        
        if not out_edges:
             return self.nlg.realizer.realize_sentence("I", "know", sub_cap) + ". But I have no details."
             
        # 3. Generate response using Structural NLG
        # Pick top 2 relations
        responses = []
        for _, neighbor, data in out_edges[:3]:
            relation = data.get("relation", "related_to")
            
            # Map relation to verb if needed
            # "is_a" -> handled by realizer (be)
            # "has_property" -> handled by realizer (have)
            
            sentence = self.nlg.realizer.realize_sentence(sub_cap, relation, neighbor)
            responses.append(sentence)
                 
        return " ".join(responses)
    
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

    def retrieve_knowledge(self, subject: str) -> str:
        """Retrieve knowledge about a subject from Semantic Memory."""
        if not self.engine or not hasattr(self.engine, "semantic_memory"):
            return "I don't have a semantic memory connected."
            
        mem = self.engine.semantic_memory
        
        # 1. Check if concept exists
        # Normalize subject?
        # subject = subject.lower().capitalize() # Or keep as is? Memory is likely case-sensitive or lower
        
        # Try direct lookup
        if subject not in mem.concept_graph:
            # Try lowercase
            if subject.lower() in mem.concept_graph:
                subject = subject.lower()
            elif subject.title() in mem.concept_graph:
                subject = subject.title()
            else:
                 return f"I don't know much about '{subject}' yet."
        
        # 2. Get immediate relations
        out_edges = list(mem.concept_graph.out_edges(subject, data=True))
        
        if not out_edges:
             return f"I know '{subject}' exists, but I haven't learned its relationships yet."
             
        # 3. Generate response using NLG
        # Pick top 2 relations
        responses = []
        for _, neighbor, data in out_edges[:2]:
            relation = data.get("relation", "related_to")
            
            # Use specific template per relation type if possible
            if relation == "is_a":
                 responses.append(self.nlg.generate("fact", {
                     "subject": subject, "object": neighbor, "relation": "is a"
                 }))
            elif relation == "has_property":
                 # neighbor might be "red", relation "has_property"
                 responses.append(self.nlg.generate("fact", {
                     "subject": subject, "property": "characteristic", "value": neighbor
                 }))
            else:
                 responses.append(self.nlg.generate("fact", {
                     "subject": subject, "relation": relation, "object": neighbor
                 }))
                 
        return " ".join(responses)

    def handle_question(self, structured: Dict) -> str:
        """Answer a query about current cognitive state or world knowledge."""
        intent = structured.get("intent")
        
        # Check if it asks about "self" (stats)
        if "self" in structured.get("entities", []) or intent == "status":
            if self.engine and hasattr(self.engine, "get_stats"):
                stats = self.engine.get_stats()
                return (
                    f"I have made {stats.get('decisions', 0)} decisions and "
                    f"recorded {stats.get('episodes_recorded', 0)} episodes."
                )
        
        # Otherwise, try to extract a topic entity
        entities = structured.get("entities", [])
        if entities:
            topic = entities[0]
            return self.retrieve_knowledge(topic)
            
        return "I don't have enough context to answer that."

    def generate_explanation(self) -> str:
        """Explain the most recent decision using the engine's explainer."""
        if self.engine and hasattr(self.engine, "explain"):
            return self.engine.explain()
        return "No explanation available — engine not connected."

    def handle_explanation(self, structured: Dict) -> str:
        """Handle 'Why X?' queries using Causal Service."""
        # Extract target
        entities = structured.get("entities", [])
        target = entities[0] if entities else "the recent event"
        
        # Call service
        explanation = self.causal_service.explain_why(target)
        
        # If text is pre-generated (mock or simple), use it
        if explanation.text:
            return explanation.text
            
        # Otherwise, realize the chain
        if explanation.chain:
            return f"Because {explanation.cause} causes {explanation.effect}. " + \
                   self.nlg.realizer.realize_chain(explanation.chain)
                   
        return f"I am not sure why {target} happened."

    def handle_prediction(self, structured: Dict) -> str:
        """Handle 'What if X?' queries using Causal Service."""
        entities = structured.get("entities", [])
        action = entities[0] if entities else "that"
        
        # Call service
        prediction = self.causal_service.predict_what_if(action, {})
        
        if prediction.text:
            return prediction.text
            
        return f"If you {action}, then {prediction.predicted_outcome} might happen."

    # ------------------------------------------------------------------
    # Smart causal / factual pattern detection
    # ------------------------------------------------------------------

    # Regex patterns matched BEFORE VSA routing to avoid mis-parsing
    _CAUSAL_CAUSE_PAT = re.compile(
        r"what\s+(?:causes?|leads?\s+to|triggers?|produces?)\s+(.+?)[\?\.]?\s*$",
        re.IGNORECASE,
    )
    _CAUSAL_EFFECT_PAT = re.compile(
        r"what\s+(?:does|will|can)\s+(.+?)\s+(?:cause|lead\s+to|produce|trigger)[\?\.]?\s*$",
        re.IGNORECASE,
    )
    _WHAT_IS_PAT = re.compile(
        r"what\s+is\s+(?:a\s+|an\s+)?(.+?)[\?\.]?\s*$",
        re.IGNORECASE,
    )
    _EXPLAIN_PAT = re.compile(
        r"(?:explain|describe|tell\s+me\s+about|how\s+does)\s+(.+?)[\?\.]?\s*$",
        re.IGNORECASE,
    )

    def _try_causal_patterns(self, user_input: str) -> str:
        """
        Return a meaningful response if user_input matches a known causal /
        factual query pattern.  Returns empty string if no match so the
        calling code can fall through to normal VSA routing.
        """
        txt = user_input.strip()

        # --- 1. "What causes X?" ---
        m = self._CAUSAL_CAUSE_PAT.search(txt)
        if m:
            effect = m.group(1).strip()
            return self._answer_causes_of(effect)

        # --- 2. "What does X cause?" ---
        m = self._CAUSAL_EFFECT_PAT.search(txt)
        if m:
            cause = m.group(1).strip()
            return self._answer_effects_of(cause)

        # --- 3. "What is X?" ---
        m = self._WHAT_IS_PAT.search(txt)
        if m:
            topic = m.group(1).strip()
            return self._answer_what_is(topic)

        # --- 4. "Explain X / How does X work?" ---
        m = self._EXPLAIN_PAT.search(txt)
        if m:
            topic = m.group(1).strip()
            return self._answer_explain(topic)

        return ""

    # ------------------------------------------------------------------
    # Individual answer helpers
    # ------------------------------------------------------------------

    def _sem_mem(self):
        """Shorthand: return semantic_memory if available, else None."""
        if self.engine and hasattr(self.engine, "semantic_memory"):
            return self.engine.semantic_memory
        return None

    def _answer_causes_of(self, effect: str) -> str:
        """Return a sentence listing what causes *effect*."""
        mem = self._sem_mem()
        causes = []

        if mem is not None:
            g = mem.concept_graph
            # Try exact match first, then case-insensitive
            for candidate in list(g.nodes()):
                c_lower = str(candidate).lower()
                e_lower = effect.lower()
                if c_lower == e_lower or c_lower.replace("_", " ") == e_lower:
                    effect = candidate
                    break

            if effect in g:
                for src, _, data in g.in_edges(effect, data=True):
                    rel = data.get("relation", "")
                    if "cause" in rel.lower() or "lead" in rel.lower() or "trigger" in rel.lower():
                        causes.append(str(src).replace("_", " "))

        # Also search causal graphs stored on engine
        if not causes and self.engine and hasattr(self.engine, "causal_graphs"):
            for tag, cg in self.engine.causal_graphs.items():
                if hasattr(cg, "graph") and hasattr(cg.graph, "edges"):
                    for src, tgt, data in cg.graph.edges(data=True):
                        if effect.lower() in str(tgt).lower():
                            causes.append(str(src).replace("ACTION_", "").replace("_", " "))

        if causes:
            if len(causes) == 1:
                return f"{causes[0].capitalize()} causes {effect.replace('_', ' ')}."
            else:
                listed = ", ".join(c.capitalize() for c in causes[:-1])
                return (f"{listed} and {causes[-1].capitalize()} "
                        f"all cause {effect.replace('_', ' ')}.")
        return f"I have not yet learned what causes {effect.replace('_', ' ')}."

    def _answer_effects_of(self, cause: str) -> str:
        """Return a sentence listing what *cause* produces."""
        mem = self._sem_mem()
        effects = []

        if mem is not None:
            g = mem.concept_graph
            for candidate in list(g.nodes()):
                if str(candidate).lower().replace("_", " ") == cause.lower():
                    cause = candidate
                    break

            if cause in g:
                for _, tgt, data in g.out_edges(cause, data=True):
                    rel = data.get("relation", "")
                    if "cause" in rel.lower() or "lead" in rel.lower() or "trigger" in rel.lower():
                        effects.append(str(tgt).replace("_", " "))

        if effects:
            if len(effects) == 1:
                return f"{cause.replace('_', ' ').capitalize()} causes {effects[0]}."
            else:
                listed = ", ".join(e for e in effects[:-1])
                return f"{cause.replace('_', ' ').capitalize()} can cause {listed} and {effects[-1]}."
        return f"I do not know what {cause.replace('_', ' ')} causes yet."

    def _answer_what_is(self, topic: str) -> str:
        """Return a brief definition / description of *topic* from semantic memory."""
        mem = self._sem_mem()
        if mem is None:
            return f"I have no information about {topic}."

        g = mem.concept_graph
        # Case-insensitive node look-up
        matched = None
        for node in g.nodes():
            if str(node).lower().replace("_", " ") == topic.lower():
                matched = node
                break

        if matched is None:
            return f"I have not learned about {topic} yet."

        parts = []
        # Gather all outgoing relations
        for _, tgt, data in g.out_edges(matched, data=True):
            rel = data.get("relation", "related to")
            parts.append(f"{rel.replace('_', ' ')} {str(tgt).replace('_', ' ')}")

        if parts:
            return f"{str(matched).capitalize()} is {'; '.join(parts[:3])}."
        return f"I know {str(matched)} but have no detailed relations for it yet."

    def _answer_explain(self, topic: str) -> str:
        """Return a multi-sentence explanation of *topic* using spread activation."""
        mem = self._sem_mem()
        if mem is None:
            return f"I cannot explain {topic} — no memory module."

        # Find matching node
        g = mem.concept_graph
        matched = None
        for node in g.nodes():
            if str(node).lower().replace("_", " ") == topic.lower():
                matched = node
                break

        if matched is None:
            # Try spread activation from word fragments
            return f"I do not have enough information to explain {topic} yet."

        # Collect the immediate neighbourhood
        sentences = []
        for _, tgt, data in g.out_edges(matched, data=True):
            rel = data.get("relation", "relates to").replace("_", " ")
            sentences.append(
                f"{str(matched).capitalize()} {rel} {str(tgt).replace('_', ' ')}"
            )
        # Incoming causal edges
        for src, _, data in g.in_edges(matched, data=True):
            rel = data.get("relation", "relates to").replace("_", " ")
            sentences.append(
                f"{str(src).capitalize()} {rel} {str(matched).replace('_', ' ')}"
            )

        if not sentences:
            return f"I know {str(matched)} exists but have no associated facts yet."

        intro = f"Here is what I know about {topic}: "
        return intro + ". ".join(sentences[:5]) + "."
