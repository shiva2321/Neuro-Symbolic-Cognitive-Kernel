"""
VSA Language Module (Neuro-Symbolic)
====================================

A VSA-native replacement for the LLM-based LanguageModule.
Integrates:
1. Left-Corner Parser (Text -> VSA Tree)
2. Resonator Network (VSA Tree -> Structured Intent)

This module demonstrates "Understanding" without a neural network, 
using only algebra and high-dimensional vectors.
"""

from typing import Dict, Any, List, Tuple, Optional
import sys
import os
import nltk # [Phase 3] NLTK for robust POS tagging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import python.core.vsa.hypervec_shim as hv
from python.core.language.parser import LeftCornerParser
from python.core.vsa.resonator import ResonatorNetwork
from python.core.memory.semantic_memory import SemanticMemory

from typing import Optional
class VSALanguageModule:
    def __init__(self, semantic_memory: Optional[SemanticMemory] = None):
        self.parser = LeftCornerParser()
        
        # Initialize Memories via injection or new
        self.memory = semantic_memory if semantic_memory else SemanticMemory()
        
        # In a real system, we'd filter these views (e.g. self.memory.get_view("agent"))
        # For now, we share the global memory.
        self.agent_mem = self.memory
        self.action_mem = self.memory
        
        # [Phase 3] Dynamic Vocabulary via NLTK
        # Ensure NLTK resources are available
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
             print("[VSA] Downloading NLTK resources...")
             nltk.download('averaged_perceptron_tagger_eng', quiet=True)
             nltk.download('punkt', quiet=True)
             nltk.download('punkt_tab', quiet=True)

        self.vocab = {} # Deprecated hardcoded vocab
        
        # Pre-seed memory with core roles if missing
        for role in ["AGENT", "ACTION", "OBJECT", "THEME"]:
             if role not in self.memory.concept_hvs:
                  self.memory.add_concept(role, {}, hv_override=self.parser.get_vector(role))
        
        # [Phase 3] Parser vectors synced on fly
        pass
            
        # Initialize Resonator
        # We want to factorize the Tree structure.
        # Tree = (Subject * AGENT) + (Action * ACTION)
        # We know AGENT and ACTION vectors (Roles).
        # We want to find Subject (which is in agent_mem) and Action (in action_mem).
        # NOTE: Subject is strictly (Word * Tag).
        # But for this demo, let's assume the Resonator can find "dog" even if it's bound with "NN".
        # Or we add (Word * Tag) to memory?
        # Better: We configure Resonator to search for 'dog'.
        # The query vector will be `Tree * inv(AGENT)`.
        # This gives `Subject`.
        # `Subject = dog * NN`.
        # `dog * NN` is NOT `dog`.
        # So we can't find `dog` directly unless we unbind NN.
        # But `vsa_language_module` knows `NN`.
        
        self.resonator = ResonatorNetwork(
            codebooks={
                "agent": self.agent_mem, 
                "action": self.action_mem
            }
        )

        # ── Cached per-call helpers (built once in __init__) ────────────────
        # Creating SemanticMemory + ResonatorNetwork on every understand() call
        # is the single biggest performance bottleneck (~9 s/turn → ~0.05 s/turn).
        self._tag_mem = SemanticMemory()
        self._tag_mem.add_concept("NN", {}, hv_override=self.parser.get_vector("NN"))
        self._tag_mem.add_concept("VB", {}, hv_override=self.parser.get_vector("VB"))
        self._tag_mem.add_concept("DT", {}, hv_override=self.parser.get_vector("DT"))

        self._res_decoder = ResonatorNetwork(
            codebooks={"word": self.agent_mem, "tag": self._tag_mem}
        )
        self._res_action = ResonatorNetwork(
            codebooks={"word": self.action_mem, "tag": self._tag_mem}
        )
        self._res_query = ResonatorNetwork(
            codebooks={"word": self.memory}
        )

    def understand(self, text: str) -> Dict[str, Any]:
        """
        End-to-End VSA Understanding:
        Text -> Parser -> Tree -> Resonator -> Meaning
        """
        # 1. Tokenize (Simple split for demo)
        # Expects: "The dog run" (Mapped to tags)
        # [Phase 3] NLTK Tokenization & Tagging
        text = text.lower() # VSA is case-insensitive for now
        nltk_tokens = nltk.word_tokenize(text)
        nltk_tags = nltk.pos_tag(nltk_tokens)
        
        tokens = []
        for word, tag in nltk_tags:
            # Map NLTK tags to VSA Parser tags
            vsa_tag = "UNK"
            if tag.startswith("NN"): vsa_tag = "NN"
            elif tag.startswith("VB"): vsa_tag = "VB"
            elif tag.startswith("JJ"): vsa_tag = "NN" # Adjectives treated as Nouns for now (bundled)
            # Filter out function words (DT, PRP, IN, CC, etc.) from becoming unique concepts
            # They are structural, not semantic in this VSA model.
            else: vsa_tag = "UNK"
            
            if vsa_tag != "UNK":
                # Ensure concept exists in Memory (Dynamic Learning)
                if word not in self.memory.concept_hvs:
                    # [Learning] Add new concept on the fly
                    # Generate stable vector from name
                    # In future: Bind with context
                    v = self.parser.get_vector(word) 
                    self.memory.add_concept(word, {"learned": True, "pos": vsa_tag}, hv_override=v)
                
                tokens.append((word, vsa_tag))
            else:
                 # Skip unknown mechanics (punctuation, etc)
                 pass
                
        # 2. Parse (Text -> VSA Tree)
        tree_hv = self.parser.parse_sentence(tokens)
        
        # 3. Decode (VSA Tree -> Meaning)
        # We manually query the Tree structure using algebraic unbinding
        
        # Extract Subject Cluster: Tree * inv(AGENT)
        # SubjectCluster = (The*DT + Dog*NN)
        v_agent_role = self.parser.get_vector("AGENT")
        subject_cluster = tree_hv.xor(v_agent_role)
        
        # Extract Action Cluster: Tree * inv(ACTION)
        v_action_role = self.parser.get_vector("ACTION")
        action_cluster = tree_hv.xor(v_action_role)
        
        # [NEW] Extract Query Cluster: Tree * inv(QUERY) (or PROBE)
        # If the sentence was "What...", we bound "What" to QUERY role.
        # We also bound the Target to OBJECT role.
        
        if "QUERY" in self.parser.vectors:
            v_query_role = self.parser.get_vector("QUERY")
            query_cluster = tree_hv.xor(v_query_role)
            # Simply check similarity to "What/Who"
            what_vec = self.parser.get_vector("WP")
            # If (Tree * QUERY) is similar to (WP), then it's a question.
            sim = query_cluster.similarity_robust(what_vec)
            if sim > 0.6: # VSA Threshold
                structured_output = {"intent": "question", "QUERY": "What"}
                
                # Extract Target (Object of the question)
                # In "What is X?", X is bound to OBJECT/THEME
                v_obj_role = self.parser.get_vector("OBJECT")
                target_cluster = tree_hv.xor(v_obj_role)
                
                # Resonate to find the target concept
                # Reuse agent/word memory for target
                factors = self._res_query.factorize(target_cluster)
                target_name = factors.get("word", ("unknown", 0.0))[0]
                
                structured_output["TARGET"] = target_name
                structured_output["entities"] = [target_name]
                
                return {
                    "text": text,
                    "structured_output": structured_output,
                    "grounded_hv": tree_hv
                }

        # Now use Resonator? or just Search?
        # SubjectCluster has (Dog * NN). We want "Dog".
        # We can unbind NN first.
        # Now use Resonator to find (Word, Tag) pair from the cluster?
        # SubjectCluster ~= (Dog * NN).
        # We want to find Word and Tag.
        # We have codebooks for Words and Tags.
        
        # Setup Resonator for Subject
        # But wait, self.resonator was init with 'agent' and 'action'.
        # That was for finding (Agent, Action) from Sentence?
        # No, Sentence is Sum. Resonator is for Product.
        # The Product here is Word * Tag.
        
        # Factorize Subject Cluster using cached resonators (no re-init per call)
        subj_factors = self._res_decoder.factorize(subject_cluster)

        # Factorize Action Cluster
        act_factors = self._res_action.factorize(action_cluster)
        
        # [Fix] Unpack safely
        intent_match = act_factors.get("word", ("unknown", 0.0))
        intent = intent_match[0] if isinstance(intent_match, tuple) else "unknown"
        
        entity_match = subj_factors.get("word", ("unknown", 0.0))
        entity = entity_match[0] if isinstance(entity_match, tuple) else "unknown"

        result = {
            "intent": intent,
            "entities": [entity],
            "grounded_hv": tree_hv,
            "debug_tree": str(tree_hv)
        }
        
        return result

if __name__ == "__main__":
    vsa = VSALanguageModule()
    res = vsa.understand("The dog ran")
    print(f"Input: 'The dog ran'")
    print(f"Result: {res['intent']} {res['entities']}")
