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

        # Path resolution: check potential locations
        resolved_path = None
        candidates = [
            model_path,                                           # 1. As provided (CWD-relative)
            os.path.join(os.path.dirname(__file__), model_path), # 2. Script-relative (nsck-demo/python/models/...)
            os.path.join(os.path.dirname(__file__), "..", "..", model_path) # 3. Project root (models/...)
        ]
        
        for cand in candidates:
            if os.path.exists(cand):
                resolved_path = cand
                break

        if not resolved_path:
            print(f"WARNING: Model file not found at any of {candidates}. LanguageModule running in MOCK mode.")
            self.mock_mode = True
            return

        model_path = resolved_path


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
        """Rule-based NLU with chart-style phrase chunking + semantic frames.

        Architecture (no LLM, no neural nets):
        1. Tokenise + heuristic POS tagging
        2. Bottom-up phrase chunking (NP, VP, PP, clause boundaries)
        3. Semantic frame extraction (Agent-Action-Patient-Instrument-Location)
        4. Clause segmentation (if/then, because, when, although)
        5. Coreference hints (pronoun → most-recent NP)
        6. Intent classification from syntactic patterns

        Produces:
        - intent    : action verb / question / conditional / command / inform
        - entities  : extracted noun phrases
        - relation  : subject → verb → object triple
        - deps      : full dependency-like structure with semantic roles
        - frames    : semantic frame slots (agent, patient, instrument, location, etc.)
        - clauses   : segmented clause list
        """
        import re
        text_lower = text.lower().strip()
        original_text = text.strip()

        # ── 1. Tokenise ──────────────────────────────────────────────
        tokens = re.findall(r"[A-Za-z0-9]+(?:'[a-z]+)?", text_lower)
        if not tokens:
            return {"structured_output": {"intent": "unknown", "entities": [],
                    "relation": "none", "deps": {}, "frames": {}, "clauses": []},
                    "grounded_hv": None}

        # ── 2. Heuristic POS tagging ─────────────────────────────────
        _DETS = {"a", "an", "the", "this", "that", "these", "those", "my",
                 "your", "his", "her", "its", "our", "their", "some", "any",
                 "no", "every", "each", "all", "both", "few", "many", "much"}
        _PREPS = {"in", "on", "at", "by", "for", "with", "from", "to",
                  "into", "through", "during", "before", "after", "above",
                  "below", "between", "under", "over", "about", "of", "off",
                  "up", "out", "against", "among", "around", "near", "using"}
        _AUXS = {"is", "are", "was", "were", "be", "been", "being",
                 "have", "has", "had", "do", "does", "did",
                 "will", "would", "could", "should", "shall", "can",
                 "may", "might", "must"}
        _PRONS = {"i", "me", "you", "he", "she", "it", "we", "they",
                  "him", "her", "us", "them", "who", "what", "which",
                  "myself", "yourself", "himself", "herself", "itself"}
        _CONJS = {"and", "or", "but", "nor", "yet", "so"}
        _SUBCONJS = {"if", "then", "because", "since", "when", "while",
                     "although", "unless", "until", "before", "after",
                     "whereas", "whenever", "wherever", "whether", "that"}
        _ADVS = {"very", "quite", "rather", "really", "always", "never",
                 "often", "usually", "sometimes", "here", "there", "now",
                 "then", "also", "already", "still", "just", "only", "not",
                 "too", "more", "most", "less", "least", "well"}
        _ADJ_SUFFIXES = ("ful", "less", "ous", "ive", "ble", "ial", "al",
                         "ent", "ant", "ic", "ary", "ory")

        def pos_tag(w):
            if w in _DETS: return "DET"
            if w in _PREPS: return "PREP"
            if w in _AUXS: return "AUX"
            if w in _PRONS: return "PRON"
            if w in _CONJS: return "CONJ"
            if w in _SUBCONJS: return "SCONJ"
            if w in _ADVS: return "ADV"
            if w.endswith(("ing",)) and len(w) > 4: return "VERB"
            if w.endswith(("ify", "ise", "ize")): return "VERB"
            if w.endswith("ed") and len(w) > 3: return "VERB"
            if any(w.endswith(s) for s in _ADJ_SUFFIXES): return "ADJ"
            return "NOUN"  # default: content word → noun

        tags = [pos_tag(t) for t in tokens]

        # ── 3. Clause segmentation ───────────────────────────────────
        clauses = []
        current_clause_tokens = []
        current_clause_type = "main"
        for tok, tag in zip(tokens, tags):
            if tag == "SCONJ" and current_clause_tokens:
                clauses.append({"type": current_clause_type,
                               "tokens": list(current_clause_tokens)})
                current_clause_tokens = [tok]
                current_clause_type = tok  # e.g. "if", "because"
            else:
                current_clause_tokens.append(tok)
        if current_clause_tokens:
            clauses.append({"type": current_clause_type,
                           "tokens": list(current_clause_tokens)})

        # ── 4. Bottom-up phrase chunking ─────────────────────────────
        def chunk_np(toks, tgs, start):
            """Parse NP → (DET)? (ADJ|ADV)* (NOUN|PRON)+"""
            i = start
            n = len(toks)
            det = None
            adjs = []
            nouns = []
            if i < n and tgs[i] == "DET":
                det = toks[i]; i += 1
            while i < n and tgs[i] in ("ADJ", "ADV"):
                adjs.append(toks[i]); i += 1
            while i < n and tgs[i] in ("NOUN", "PRON"):
                nouns.append(toks[i]); i += 1
            if not nouns:
                return None
            np_str = " ".join(([det] if det else []) + adjs + nouns)
            return {"span": np_str, "det": det, "adjs": adjs,
                    "head": nouns[-1], "nouns": nouns, "end": i}

        # Parse the main clause (first clause, or all tokens if no clauses)
        main_tokens = clauses[0]["tokens"] if clauses else tokens
        main_tags = [pos_tag(t) for t in main_tokens]

        # Full dependency structure
        deps: Dict[str, Any] = {
            "subject": None, "verb": None, "aux": None,
            "object": None, "prep_phrases": [], "adverbs": [],
        }

        i = 0
        n = len(main_tokens)

        # Skip leading SCONJ
        if i < n and main_tags[i] == "SCONJ":
            i += 1

        # Subject NP
        subj_np = chunk_np(main_tokens, main_tags, i)
        if subj_np:
            deps["subject"] = subj_np["span"]
            i = subj_np["end"]

        # Adverbs before verb
        while i < n and main_tags[i] == "ADV":
            deps["adverbs"].append(main_tokens[i]); i += 1

        # Aux + Verb
        if i < n and main_tags[i] == "AUX":
            deps["aux"] = main_tokens[i]; i += 1
        # Skip adverbs between aux and verb
        while i < n and main_tags[i] == "ADV":
            deps["adverbs"].append(main_tokens[i]); i += 1
        if i < n and main_tags[i] in ("VERB", "NOUN", "ADJ"):
            deps["verb"] = main_tokens[i]; i += 1

        # If verb not found but aux found, aux might be the verb (is, was, etc.)
        if deps["verb"] is None and deps["aux"]:
            deps["verb"] = deps["aux"]
            deps["aux"] = None

        # Object NP
        obj_np = chunk_np(main_tokens, main_tags, i)
        if obj_np:
            deps["object"] = obj_np["span"]
            i = obj_np["end"]

        # PP* (Prep + NP)
        while i < n:
            if main_tags[i] == "PREP":
                prep = main_tokens[i]; i += 1
                pp_np = chunk_np(main_tokens, main_tags, i)
                if pp_np:
                    deps["prep_phrases"].append({
                        "prep": prep, "obj": pp_np["span"]
                    })
                    i = pp_np["end"]
                else:
                    # Bare prep — grab remaining words
                    remaining = []
                    while i < n and main_tags[i] not in ("PREP", "CONJ", "SCONJ"):
                        remaining.append(main_tokens[i]); i += 1
                    if remaining:
                        deps["prep_phrases"].append({
                            "prep": prep, "obj": " ".join(remaining)
                        })
            else:
                i += 1

        # ── 5. Semantic frame extraction ─────────────────────────────
        frames: Dict[str, Optional[str]] = {
            "agent": deps["subject"],
            "action": deps["verb"],
            "patient": deps["object"],
            "instrument": None,
            "location": None,
            "source": None,
            "destination": None,
            "time": None,
            "cause": None,
            "condition": None,
        }
        for pp in deps["prep_phrases"]:
            p = pp["prep"]
            obj = pp["obj"]
            if p in ("with", "using", "by"):
                frames["instrument"] = obj
            elif p in ("in", "on", "at", "near", "above", "below", "between"):
                frames["location"] = obj
            elif p in ("from", "out"):
                frames["source"] = obj
            elif p in ("to", "into", "towards"):
                frames["destination"] = obj
            elif p in ("during", "before", "after"):
                frames["time"] = obj
            elif p in ("because", "due"):
                frames["cause"] = obj

        # Fill condition from subordinate clauses
        for cl in clauses:
            if cl["type"] in ("if", "unless", "whether"):
                frames["condition"] = " ".join(cl["tokens"])
            elif cl["type"] in ("because", "since"):
                frames["cause"] = " ".join(cl["tokens"])

        # ── 6. Coreference hints (pronoun → most recent NP) ─────────
        last_np = deps["subject"] or deps["object"]
        coref_map = {}
        for tok in tokens:
            if tok in _PRONS and last_np:
                coref_map[tok] = last_np

        # ── 7. Intent classification ─────────────────────────────────
        intent = "inform"
        entities: List[str] = []

        if any(cl["type"] in ("if", "unless") for cl in clauses):
            intent = "conditional"
            for cl in clauses:
                entities.append(" ".join(cl["tokens"]))
        elif text_lower.endswith("?") or (tokens and tokens[0] in
                ("what", "where", "why", "how", "when", "who", "which")):
            intent = "question"
        elif tokens and tokens[0] in ("go", "move", "turn", "eat", "take",
                "find", "search", "explore", "avoid", "retreat", "stop",
                "run", "look", "get", "set", "put", "open", "close"):
            intent = "command"
            if deps["object"]:
                entities = [deps["object"]]
        elif deps["verb"]:
            intent = deps["verb"]

        # Extract entities from NPs if not set
        if not entities:
            for field in ("subject", "object"):
                if deps[field]:
                    entities.append(deps[field])
            for pp in deps["prep_phrases"]:
                entities.append(pp["obj"])
        entities = entities[:8]  # cap

        # Relation triple
        relation = "none"
        if deps["subject"] and deps["verb"] and deps["object"]:
            relation = f"{deps['subject']}→{deps['verb']}→{deps['object']}"
        elif deps["subject"] and deps["verb"]:
            relation = f"{deps['subject']}→{deps['verb']}"

        structured = {
            "intent": intent,
            "entities": entities,
            "relation": relation,
            "deps": deps,
            "frames": frames,
            "clauses": clauses,
            "coref": coref_map,
        }

        return {
            "structured_output": structured,
            "grounded_hv": self._ground_to_vsa(structured),
        }

    def _mock_generate(self, intent_data: Dict) -> str:
        return f"[MOCK LLM] I will {intent_data.get('action')} the {intent_data.get('target')}."
