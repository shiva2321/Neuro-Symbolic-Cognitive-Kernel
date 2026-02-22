"""Distributional Semantics via VSA context windows for NSCK V3."""
from __future__ import annotations
import logging
import pickle
from typing import List, Dict, Optional
import python.core.vsa.hypervec_shim as hypervec_rs

logger = logging.getLogger(__name__)

HyperVector = hypervec_rs.HyperVector

BUILTIN_CORPUS: List[List[str]] = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "in", "the", "park"],
    ["cats", "and", "dogs", "are", "common", "pets"],
    ["the", "sun", "shines", "in", "the", "sky"],
    ["water", "flows", "in", "rivers", "and", "oceans"],
    ["plants", "need", "water", "and", "sunlight", "to", "grow"],
    ["animals", "eat", "food", "to", "survive"],
    ["humans", "are", "social", "animals"],
    ["science", "studies", "the", "natural", "world"],
    ["physics", "and", "chemistry", "are", "natural", "sciences"],
    ["biology", "studies", "living", "organisms"],
    ["mathematics", "is", "the", "language", "of", "science"],
    ["computers", "process", "data", "and", "information"],
    ["the", "internet", "connects", "people", "worldwide"],
    ["cities", "are", "large", "human", "settlements"],
    ["forests", "provide", "oxygen", "and", "shelter"],
    ["the", "earth", "orbits", "the", "sun"],
    ["the", "moon", "orbits", "the", "earth"],
    ["gravity", "pulls", "objects", "toward", "the", "earth"],
    ["light", "travels", "faster", "than", "sound"],
    ["energy", "cannot", "be", "created", "or", "destroyed"],
    ["atoms", "make", "up", "all", "matter"],
    ["cells", "are", "the", "basic", "unit", "of", "life"],
    ["genes", "carry", "genetic", "information"],
    ["evolution", "explains", "the", "diversity", "of", "life"],
    ["the", "brain", "controls", "the", "nervous", "system"],
    ["language", "enables", "communication", "between", "humans"],
    ["knowledge", "is", "power", "and", "freedom"],
    ["technology", "changes", "human", "society"],
    ["art", "expresses", "human", "emotion", "and", "culture"],
    ["music", "and", "art", "are", "creative", "activities"],
    ["history", "records", "past", "human", "events"],
    ["the", "climate", "affects", "living", "conditions"],
    ["food", "provides", "energy", "for", "living", "organisms"],
    ["water", "is", "essential", "for", "life"],
    ["oxygen", "is", "needed", "for", "breathing"],
    ["the", "heart", "pumps", "blood", "through", "the", "body"],
    ["the", "lungs", "exchange", "oxygen", "and", "carbon", "dioxide"],
    ["trees", "convert", "carbon", "dioxide", "into", "oxygen"],
    ["rivers", "flow", "from", "mountains", "to", "the", "sea"],
    ["clouds", "form", "from", "water", "vapor", "in", "the", "atmosphere"],
    ["rain", "falls", "from", "clouds", "to", "the", "ground"],
    ["the", "ocean", "covers", "most", "of", "the", "earth"],
    ["birds", "can", "fly", "because", "they", "have", "wings"],
    ["fish", "live", "and", "breathe", "in", "water"],
    ["the", "cat", "chased", "the", "mouse", "across", "the", "room"],
    ["students", "learn", "new", "skills", "at", "school"],
    ["teachers", "help", "students", "understand", "concepts"],
    ["doctors", "treat", "patients", "with", "medicine"],
    ["engineers", "design", "and", "build", "structures"],
]


class DistributionalCodebook:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._codebook: Dict[str, HyperVector] = {}

    def build_from_corpus(self, sentences: List[List[str]]) -> None:
        word_contexts: Dict[str, List[HyperVector]] = {}
        for sentence in sentences:
            for i, word in enumerate(sentence):
                w = word.lower()
                if w not in word_contexts:
                    word_contexts[w] = []
                for offset in range(-self.window_size, self.window_size + 1):
                    if offset == 0:
                        continue
                    j = i + offset
                    if 0 <= j < len(sentence):
                        context_word = sentence[j].lower()
                        ctx_hv = HyperVector(hash(context_word) % (2**32))
                        # Permute by offset to encode position (silently skip if unsupported)
                        try:
                            ctx_hv = ctx_hv.permute(offset)
                        except (AttributeError, TypeError) as e:
                            logger.debug("permute unavailable, using unpermuted HV: %s", e)
                        word_contexts[w].append(ctx_hv)
        for word, context_hvs in word_contexts.items():
            if not context_hvs:
                self._codebook[word] = HyperVector(hash(word) % (2**32))
                continue
            accumulated = context_hvs[0]
            for hv in context_hvs[1:]:
                accumulated = accumulated.bundle(hv)
            self._codebook[word] = accumulated

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"window_size": self.window_size, "codebook": self._codebook}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.window_size = data["window_size"]
        self._codebook = data["codebook"]

    def get_hv(self, word: str) -> Optional[HyperVector]:
        return self._codebook.get(word.lower())

    def similarity(self, word1: str, word2: str) -> float:
        hv1 = self.get_hv(word1)
        hv2 = self.get_hv(word2)
        if hv1 is None or hv2 is None:
            return 0.0
        try:
            return float(hv1.similarity(hv2))
        except Exception:
            return 0.0

    @classmethod
    def build_default(cls, window_size: int = 3) -> "DistributionalCodebook":
        cb = cls(window_size=window_size)
        cb.build_from_corpus(BUILTIN_CORPUS)
        return cb
