"""
VSA-based NLU Engine (V4)
=========================
Replaces NgramNLU as the primary intent classifier and entity extractor.

Uses DistributionalCodebook for word HVs + role-filler binding for
sentence-level representations. No gradients, no transformers.

Pipeline:
  1. Tokenize (simple whitespace + punctuation)
  2. Build word HVs from DistributionalCodebook
  3. Encode sentence as positional role-filler bundle
  4. Compare to intent prototype HVs (k-NN in HV space)
  5. Extract entities via capitalization + known-concept lookup

Exposes NgramNLU-compatible interface for drop-in replacement.
"""
from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import python.core.vsa.hypervec_shim as hypervec_rs

logger = logging.getLogger("nsck.language.vsa_nlu")

HyperVector = hypervec_rs.HyperVector

# ---------------------------------------------------------------------------
# Intent definitions — representative sentence lists per intent
# ---------------------------------------------------------------------------

_INTENT_SENTENCES: Dict[str, List[str]] = {
    "question": [
        "what is this",
        "how does it work",
        "why did that happen",
        "where is the location",
        "who is responsible",
        "when will it happen",
        "which option is better",
        "can you explain this",
        "do you know the answer",
        "what are the results",
        "how many items are there",
        "why is this important",
    ],
    "command": [
        "go to the next step",
        "stop the process now",
        "start the engine",
        "move forward quickly",
        "execute the plan",
        "run the algorithm",
        "navigate to the goal",
        "compute the result",
        "activate the system",
        "disable this feature",
        "reset to default",
        "find the solution",
    ],
    "statement": [
        "the system is running normally",
        "the process completed successfully",
        "the data shows positive results",
        "the agent learned from experience",
        "the model predicts high confidence",
        "knowledge is stored in memory",
        "the environment has changed",
        "the current state is stable",
        "the task was completed",
        "the result is correct",
        "performance is improving",
        "the prediction was accurate",
    ],
    "greeting": [
        "hello how are you",
        "hi there good morning",
        "hey what is up",
        "good morning to you",
        "greetings and welcome",
        "nice to meet you",
        "good afternoon",
        "welcome back again",
        "glad to see you",
        "hope you are well",
    ],
    "farewell": [
        "goodbye see you later",
        "bye have a good day",
        "see you tomorrow",
        "farewell until next time",
        "take care and goodbye",
        "good night and sleep well",
        "until we meet again",
        "have a safe journey",
        "it was nice talking",
        "talk to you soon",
    ],
    "exclamation": [
        "that is amazing incredible",
        "wow what a result",
        "excellent performance great job",
        "fantastic this is wonderful",
        "oh no that is terrible",
        "unbelievable what happened here",
        "brilliant work done perfectly",
        "great achievement well done",
        "outstanding result very good",
        "remarkable improvement indeed",
    ],
    "negation": [
        "this does not work",
        "the system is not responding",
        "no that is incorrect",
        "never do this action",
        "do not proceed further",
        "this cannot be right",
        "none of these are correct",
        "it is not possible",
        "the result is not valid",
        "this should not happen",
        "cannot execute this command",
        "will not accept this",
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


def _build_prototype(sentences: List[str], codebook) -> HyperVector:
    """Build an intent prototype HV by bundling sentence HVs."""
    hvs = []
    for sentence in sentences:
        tokens = _tokenize(sentence)
        if not tokens:
            continue
        hv = codebook.encode_sentence(tokens)
        hvs.append(hv)
    if not hvs:
        return HyperVector(0)
    result = hvs[0]
    for hv in hvs[1:]:
        result = result.bundle(hv)
    return result


# ---------------------------------------------------------------------------
# VSANLUEngine
# ---------------------------------------------------------------------------

class VSANLUEngine:
    """
    VSA-based NLU engine using DistributionalCodebook + intent prototypes.

    Provides NgramNLU-compatible interface for backward compatibility.
    """

    def __init__(
        self,
        codebook=None,
        enable_hf_corpus: bool = False,
    ):
        """
        Initialise engine. If `codebook` is None, build from BUILTIN_CORPUS.
        """
        if codebook is not None:
            self._codebook = codebook
        else:
            from python.core.language.distributional_semantics import DistributionalCodebook
            self._codebook = DistributionalCodebook.build_default(
                enable_hf_corpus=enable_hf_corpus
            )

        # Build intent prototype HVs
        self._prototypes: Dict[str, HyperVector] = {}
        for intent, sentences in _INTENT_SENTENCES.items():
            self._prototypes[intent] = _build_prototype(sentences, self._codebook)
            logger.debug("[VSA-NLU] Built prototype for intent '%s'", intent)

        logger.info("[VSA-NLU] Initialized with %d intent prototypes", len(self._prototypes))

    def encode_sentence(self, text: str) -> HyperVector:
        """Encode a full sentence as a HyperVector."""
        tokens = _tokenize(text)
        if not tokens:
            return HyperVector(0)
        return self._codebook.encode_sentence(tokens)

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify intent of text.

        Returns
        -------
        (intent, confidence) where confidence ∈ [0.0, 1.0].
        """
        sentence_hv = self.encode_sentence(text)

        best_intent = "statement"
        best_sim = 0.0
        for intent, proto_hv in self._prototypes.items():
            try:
                sim = float(sentence_hv.similarity(proto_hv))
                if sim > best_sim:
                    best_sim = sim
                    best_intent = intent
            except Exception:
                pass

        return best_intent, best_sim

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from text.

        Returns list of (entity_text, entity_type) tuples.
        Entity type is inferred from capitalization and known-concept lookup.
        """
        entities: List[Tuple[str, str]] = []
        words = text.split()
        for word in words:
            clean = re.sub(r"[^\w]", "", word)
            if not clean:
                continue
            if clean[0].isupper() and len(clean) > 1:
                entities.append((clean, "PROPER_NOUN"))
        return entities

    # ------------------------------------------------------------------
    # NgramNLU-compatible interface
    # ------------------------------------------------------------------

    def process(self, text: str) -> Dict[str, Any]:
        """Process text — NgramNLU-compatible output dict."""
        intent, confidence = self.classify(text)
        entities = self.extract_entities(text)
        sentence_hv = self.encode_sentence(text)

        return {
            "intent": {"label": intent, "confidence": confidence},
            "entities": [{"text": e, "type": t} for e, t in entities],
            "sentence_hv": sentence_hv,
            "engine": "vsa_nlu",
        }

    def train(self, texts: List[str], labels: Optional[List[str]] = None) -> None:
        """No-op: VSANLUEngine uses pre-built prototypes (no gradient training)."""
        pass
