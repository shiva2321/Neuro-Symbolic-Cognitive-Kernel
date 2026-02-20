"""
NSCK Natural Language Generation (NLG) Module
==============================================
Pure VSA Structural Realizer + Discourse Planner.

Converts internal semi-structured facts into fluent English using
grammatical rules (Morphology + Syntax), NOT templates, NOT LLMs.

Design
------
* StructuralRealizer  – single S-V-O sentence construction (morphology +
  syntax trees).
* DiscoursePlanner    – multi-sentence generation: orders facts by coherence,
  inserts connectives (therefore, because, however …), applies pronoun
  anaphora, and formats responses by query type (factual / explanatory /
  procedural / comparative).
* NLGEngine           – thin orchestrator used by the rest of NSCK.

No neural networks involved.  Every decision is deterministic rule-based.
"""

from typing import Any, Dict, List, Optional, Tuple
import re

class StructuralRealizer:
    """
    Grammar-based Natural Language Generator.
    Converts structured semantic frames into fluent English sentences using
    morphological rules and syntactic trees.
    """
    def __init__(self):
        self.vowels = set("aeiou")

    def pluralize(self, noun: str) -> str:
        """Simple pluralization rules."""
        noun = noun.lower()
        if noun.endswith("s") or noun.endswith("x") or noun.endswith("ch") or noun.endswith("sh"):
            return noun + "es"
        if noun.endswith("y") and noun[-2] not in self.vowels:
            return noun[:-1] + "ies"
        return noun + "s"

    def conjugate(self, verb: str, person: str = "3rd", number: str = "singular", tense: str = "present") -> str:
        """Conjugate verb based on person, number, and tense."""
        verb = verb.lower()
        if verb == "be":
            if tense == "present":
                if person == "1st" and number == "singular": return "am"
                if number == "plural" or person == "2nd": return "are"
                return "is"
            elif tense == "past":
                if number == "singular" and person != "2nd": return "was"
                return "were"
        
        if verb == "have":
            if tense == "present":
                if person == "3rd" and number == "singular": return "has"
                return "have"
        
        # Regular verbs (3rd person singular present)
        if tense == "present" and person == "3rd" and number == "singular":
            if verb.endswith("y") and verb[-2] not in self.vowels:
                return verb[:-1] + "ies"
            if verb.endswith(("s", "sh", "ch", "x", "z")):
                return verb + "es"
            return verb + "s"
            
        # Regular past
        if tense == "past":
            if verb.endswith("e"): return verb + "d"
            if verb.endswith("y") and verb[-2] not in self.vowels:
                return verb[:-1] + "ied"
            return verb + "ed"
            
        return verb

    def determine_determiner(self, noun: str, definiteness: str = "definite") -> str:
        """Choose a/an/the."""
        if definiteness == "definite":
            return "the"
        if definiteness == "indefinite":
            if noun[0].lower() in self.vowels:
                return "an"
            return "a"
        return ""

    def realize_sentence(self, subject: str, relation: str, object_: str, tense: str = "present") -> str:
        """
        Realize a full S-V-O sentence.
        
        Logic:
        S -> NP VP
        NP -> Det N
        VP -> V NP
        """
        # 1. Subject NP
        subj_det = ""
        # proper nouns don't need determiners usually
        if subject and subject[0].isupper(): 
            subj_np = subject
        else:
            subj_det = self.determine_determiner(subject, "definite")
            subj_np = f"{subj_det} {subject}".strip()

        # 2. Verb Loop
        # Handle "is a", "has_property"
        # Normalize relation
        norm_rel = relation.lower().replace(" ", "_")
        verb = relation
        obj_definiteness = "indefinite" # "is a dog"
        
        if norm_rel == "is_a" or norm_rel == "isa":
            verb = "be"
            # Force object to lower case for generic classes to ensure determiner binding?
            # Or just ignore isupper check for is_a
            # "Valkyria is a Game" is acceptable.
            # But the test wants "Valkyria is a game".
            object_ = object_.lower() 
        elif norm_rel == "has_property":
            verb = "have" # "The dog has fur"
            obj_definiteness = "none" 
        elif norm_rel == "developed_by":
            verb = "develop"
            tense = "past" 
            # Simplified: "Sega developed Valkyria"
            # But "Valkyria developed Sega" is wrong direction.
            # If relation is passive, we might need to swap subject/object or use passive voice?
            # For this demo, let's assumes we are just articulating the edge.
            # "Valkyria is related to Sega (developed_by)"
            pass
        else:
            obj_definiteness = "definite"

        # Conjugate
        realized_verb = self.conjugate(verb, person="3rd", number="singular", tense=tense)

        # 3. Object NP
        # For 'is_a', we always want a determiner if it's countable (implies indefinite).
        if object_ and object_[0].isupper() and verb != "be":
            obj_np = object_
        elif obj_definiteness == "none":
            obj_np = object_
        else:
            obj_det = self.determine_determiner(object_, obj_definiteness)
            obj_np = f"{obj_det} {object_}".strip()
            
        # 4. Assemble
        sentence = f"{subj_np} {realized_verb} {obj_np}."
        
        # Capitalize first letter
        sentence = sentence[0].upper() + sentence[1:]
        
        return sentence

    def realize_chain(self, chain: List[Any]) -> str:
        """
        Realize a causal chain as a compound sentence.
        Input: [Link(cause='A', effect='B', relation='causes'), Link(cause='B', effect='C', relation='causes')]
        OR Simple List: ["Rain", "Wet Grass", "Mud"]
        Output: "Rain causes wet grass, which causes mud."
        """
        if not chain:
            return ""
            
        # Check if it's a simple list of strings ["A", "B", "C"]
        if isinstance(chain[0], str):
            # Convert to link-like structure
            # A -> B, B -> C
            parts = []
            for i in range(len(chain) - 1):
                cause = chain[i]
                effect = chain[i+1]
                
                if i == 0:
                    parts.append(self.realize_sentence(cause, "causes", effect)[:-1]) # Remove period
                else:
                    # "which causes C"
                    parts.append(f", which causes {effect}")
            return "".join(parts) + "."

        parts = []
        
        # Heuristic: Start with the first cause
        first_link = chain[0]
        # "Rain causes wet grass"
        # Access attributes safely (handle dict or object)
        cause = getattr(first_link, 'cause', None) or (first_link.get('cause') if isinstance(first_link, dict) else None)
        effect = getattr(first_link, 'effect', None) or (first_link.get('effect') if isinstance(first_link, dict) else None)
        relation = getattr(first_link, 'relation', None) or (first_link.get('relation', 'causes') if isinstance(first_link, dict) else 'causes')
        
        clause = self.realize_sentence(cause, relation, effect)
        parts.append(clause[:-1]) # Remove period
        
        # Follow the chain
        current_subject = effect
        
        for i in range(1, len(chain)):
            link = chain[i]
            l_cause = getattr(link, 'cause', None) or (link.get('cause') if isinstance(link, dict) else None)
            l_effect = getattr(link, 'effect', None) or (link.get('effect') if isinstance(link, dict) else None)
            l_relation = getattr(link, 'relation', None) or (link.get('relation', 'causes') if isinstance(link, dict) else 'causes')

            # "which causes mud"
            # If the next link's cause matches previous effect, use relative clause
            if l_cause == current_subject:
                # "which causes mud"
                rel_verb = self.conjugate(l_relation, person="3rd", number="singular")
                parts.append(f", which {rel_verb} {l_effect}")
                current_subject = l_effect
            else:
                # Discontinuous chain? Start new sentence logic or conjunction
                parts.append(f". {self.realize_sentence(l_cause, l_relation, l_effect)}")
                current_subject = l_effect
                
        return "".join(parts) + "."

class NLGEngine:
    def __init__(self):
        self.realizer = StructuralRealizer()

    def generate(self, category: str, data: Dict[str, Any]) -> str:
        """Generate response using Grammar Engine."""
        if category == "fact":
            subj = data.get("subject", "it")
            rel = data.get("relation", "is related to")
            obj = data.get("object") or data.get("value", "something")
            
            # Handle property specific
            if "property" in data:
                # "The dog has the property color = red"
                # Simplify to "The dog is red" or "The dog has red color"
                prop = data["property"]
                val = data["value"]
                # Heuristic: if property is 'characteristic', treat as adjective "The dog is red"
                if prop == "characteristic":
                    return self.realizer.realize_sentence(subj, "is_a", val) # "The dog is red" ('is_a' triggers 'be')
                else:
                    return self.realizer.realize_sentence(subj, "have", val) # "The dog has red"
            
            return self.realizer.realize_sentence(subj, rel, obj)

        return "I am unable to articulate that thought yet."


# ---------------------------------------------------------------------------
# Discourse Planner
# ---------------------------------------------------------------------------

class DiscoursePlanner:
    """
    Multi-sentence discourse planner.

    Converts a list of semantic frames (dicts) into a coherent paragraph:
    1. Orders frames by topic centrality and causal ordering.
    2. Selects discourse connectives appropriate to the relation type.
    3. Applies pronoun anaphora to avoid repetition.
    4. Formats output based on query type (factual/explanatory/procedural).

    Input frame schema (each element of *frames* list)
    ---------------------------------------------------
    {
      "subject":  str,
      "relation": str,          # e.g. "is_a", "causes", "has_property"
      "object":   str,
      "tense":    str,          # optional: "present" | "past" | "future"
      "negate":   bool,         # optional
      "importance": float,      # optional 0-1, default 0.5
    }

    Query types
    -----------
    * "factual"      - brief factual answer
    * "explanatory"  - topic sentence + supporting evidence
    * "procedural"   - numbered/sequenced steps
    * "comparative"  - contrast two entities
    * "causal"       - causal chain narrative
    """

    # Connectives keyed by (relation_type, position_in_discourse)
    _CONNECTIVES: Dict[str, Dict[str, str]] = {
        "causes": {
            "first": "",
            "middle": "As a result,",
            "last": "Therefore,",
        },
        "enables": {
            "first": "",
            "middle": "This makes it possible to",
            "last": "Consequently,",
        },
        "contradicts": {
            "first": "",
            "middle": "However,",
            "last": "Nevertheless,",
        },
        "similar_to": {
            "first": "",
            "middle": "Similarly,",
            "last": "In the same way,",
        },
        "precedes": {
            "first": "First,",
            "middle": "Then,",
            "last": "Finally,",
        },
        "has_property": {
            "first": "",
            "middle": "Additionally,",
            "last": "Furthermore,",
        },
        "default": {
            "first": "",
            "middle": "Also,",
            "last": "In addition,",
        },
    }

    def __init__(self):
        self._realizer = StructuralRealizer()

    def plan(
        self,
        frames: List[Dict[str, Any]],
        query_type: str = "factual",
        topic: str = "",
    ) -> str:
        """
        Generate a coherent multi-sentence response from a list of frames.

        Parameters
        ----------
        frames : list of semantic frame dicts
        query_type : "factual" | "explanatory" | "procedural" | "causal" | "comparative"
        topic : the central topic word (for anaphora)

        Returns
        -------
        str : a fluent multi-sentence paragraph
        """
        if not frames:
            return "I don't have enough information to answer that."

        frames = self._sort_frames(frames, query_type)

        sentences: List[str] = []
        seen_subjects: List[str] = []

        for i, frame in enumerate(frames):
            pos = "first" if i == 0 else ("last" if i == len(frames) - 1 else "middle")
            sentence = self._realize_frame(frame, i, seen_subjects, topic, pos)
            if sentence:
                sentences.append(sentence)
                subj = frame.get("subject", "")
                if subj and subj not in seen_subjects:
                    seen_subjects.append(subj)

        if query_type == "procedural":
            return self._format_steps(sentences)

        return " ".join(sentences)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sort_frames(
        self, frames: List[Dict[str, Any]], query_type: str
    ) -> List[Dict[str, Any]]:
        """Order frames: high-importance first; causal chains in order."""
        if query_type in ("causal", "procedural"):
            return frames  # preserve supplied order

        def key(f: Dict[str, Any]) -> float:
            rel = f.get("relation", "")
            if rel in ("is_a", "isa", "is"):
                return 0.0
            if rel in ("causes", "enables", "creates"):
                return 0.1
            return 1.0 - f.get("importance", 0.5)

        return sorted(frames, key=key)

    def _realize_frame(
        self,
        frame: Dict[str, Any],
        index: int,
        seen_subjects: List[str],
        topic: str,
        position: str,
    ) -> str:
        """Realize one frame as a sentence with connective + anaphora."""
        subj = frame.get("subject", "")
        rel = frame.get("relation", "is related to")
        obj = frame.get("object") or frame.get("value", "")
        tense = frame.get("tense", "present")
        negate = frame.get("negate", False)

        if not subj or not obj:
            return ""

        # --- Anaphora: replace repeated subject with pronoun ---
        display_subj = self._apply_anaphora(subj, seen_subjects, topic)

        # --- Negation ---
        if negate:
            sentence = self._realizer.realize_sentence(display_subj, rel, obj, tense)
            # Insert "not" after first finite verb
            sentence = re.sub(
                r"\b(is|are|was|were|has|have|do|does)\b",
                lambda m: m.group(0) + " not",
                sentence,
                count=1,
            )
        else:
            sentence = self._realizer.realize_sentence(display_subj, rel, obj, tense)

        # --- Connective ---
        conn_map = self._CONNECTIVES.get(rel, self._CONNECTIVES["default"])
        connective = conn_map.get(position, "")
        if connective and index > 0:
            sentence = connective + " " + sentence[0].lower() + sentence[1:]

        return sentence

    def _apply_anaphora(
        self, subject: str, seen: List[str], topic: str
    ) -> str:
        """Replace subject with pronoun if it was recently mentioned."""
        if subject not in seen:
            return subject
        return "it"

    def _format_steps(self, sentences: List[str]) -> str:
        """Format as numbered steps for procedural queries."""
        if not sentences:
            return ""
        return "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))


# ---------------------------------------------------------------------------
# Extended NLGEngine
# ---------------------------------------------------------------------------

class NLGEngine:
    def __init__(self):
        self.realizer = StructuralRealizer()
        self.discourse = DiscoursePlanner()

    def generate(self, category: str, data: Dict[str, Any]) -> str:
        """Generate response using Grammar Engine (single-frame)."""
        if category == "fact":
            subj = data.get("subject", "it")
            rel = data.get("relation", "is related to")
            obj = data.get("object") or data.get("value", "something")

            if "property" in data:
                prop = data["property"]
                val = data["value"]
                if prop == "characteristic":
                    return self.realizer.realize_sentence(subj, "is_a", val)
                else:
                    return self.realizer.realize_sentence(subj, "have", val)

            return self.realizer.realize_sentence(subj, rel, obj)

        return "I am unable to articulate that thought yet."

    def generate_discourse(
        self,
        frames: List[Dict[str, Any]],
        query_type: str = "factual",
        topic: str = "",
    ) -> str:
        """
        Generate a multi-sentence response from multiple semantic frames.

        Parameters
        ----------
        frames : list of {"subject", "relation", "object", ...} dicts
        query_type : response style ("factual"|"explanatory"|"procedural"|"causal")
        topic : central topic word (aids anaphora)
        """
        return self.discourse.plan(frames, query_type=query_type, topic=topic)

    def generate_causal_chain(self, chain: List[Any]) -> str:
        """Generate a causal-chain narrative."""
        return self.realizer.realize_chain(chain)
