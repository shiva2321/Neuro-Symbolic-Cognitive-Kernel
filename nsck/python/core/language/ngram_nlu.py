"""
Ngram NLU
=========
Probabilistic NLU layer using n-gram language models with Naive Bayes classification.
"""
from __future__ import annotations

import sys
import os
import re
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

INTENT_LABELS = ["question", "command", "statement", "greeting", "farewell"]

# Simple English noun list for entity extraction
_NOUN_WORDS = {
    "cat", "dog", "car", "house", "computer", "phone", "book", "water", "food",
    "person", "people", "city", "country", "world", "time", "day", "year",
    "man", "woman", "child", "name", "hand", "life", "way", "part", "place",
    "case", "week", "company", "system", "program", "question", "work",
    "government", "state", "fact", "point", "kind", "head", "eye", "side",
    "group", "number", "night", "home", "money", "story", "face", "room",
    "idea", "information", "back", "school", "body", "parent", "area",
    "problem", "line", "end", "air", "thing", "example", "service",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _get_ngrams(tokens: List[str], n: int) -> List[tuple]:
    if n == 1:
        return [(t,) for t in tokens]
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


class NgramNLU:
    """Probabilistic NLU using n-gram Naive Bayes."""

    def __init__(self, n: int = 2):
        self.n = n
        self._labels: List[str] = []
        self._label_priors: Dict[str, float] = {}
        self._ngram_probs: Dict[str, Dict[tuple, float]] = {}
        self._vocab: set = set()

    def train(self, sentences: List[str], labels: List[str]):
        """Train on labeled sentences."""
        assert len(sentences) == len(labels)
        label_counts: Dict[str, int] = defaultdict(int)
        label_ngrams: Dict[str, Dict[tuple, int]] = defaultdict(lambda: defaultdict(int))

        for sent, label in zip(sentences, labels):
            tokens = _tokenize(sent)
            label_counts[label] += 1
            for ng in _get_ngrams(tokens, 1):
                label_ngrams[label][ng] += 1
                self._vocab.add(ng)
            if self.n > 1:
                for ng in _get_ngrams(tokens, self.n):
                    label_ngrams[label][ng] += 1
                    self._vocab.add(ng)

        total = sum(label_counts.values())
        self._labels = list(label_counts.keys())
        self._label_priors = {lbl: cnt / total for lbl, cnt in label_counts.items()}
        self._ngram_probs = {}
        for lbl, ngram_dict in label_ngrams.items():
            total_ng = sum(ngram_dict.values())
            vocab_size = len(self._vocab) + 1
            self._ngram_probs[lbl] = {
                ng: (cnt + 1) / (total_ng + vocab_size)
                for ng, cnt in ngram_dict.items()
            }
            self._ngram_probs[lbl]["__unk__"] = 1 / (total_ng + vocab_size)

    def predict_distribution(self, text: str) -> Dict[str, float]:
        """Return probability distribution over labels."""
        if not self._labels:
            return {}
        tokens = _tokenize(text)
        ngrams = _get_ngrams(tokens, 1)
        if self.n > 1:
            ngrams = ngrams + _get_ngrams(tokens, self.n)

        log_scores: Dict[str, float] = {}
        for lbl in self._labels:
            log_p = math.log(self._label_priors.get(lbl, 1e-9))
            probs = self._ngram_probs.get(lbl, {})
            unk_p = probs.get("__unk__", 1e-9)
            for ng in ngrams:
                log_p += math.log(probs.get(ng, unk_p))
            log_scores[lbl] = log_p

        # Softmax normalization
        max_log = max(log_scores.values())
        exp_scores = {lbl: math.exp(v - max_log) for lbl, v in log_scores.items()}
        total = sum(exp_scores.values())
        return {lbl: v / total for lbl, v in exp_scores.items()}

    def predict(self, text: str) -> Tuple[str, float]:
        """Return (best_label, confidence) using Naive Bayes with Laplace smoothing."""
        dist = self.predict_distribution(text)
        if not dist:
            return ("unknown", 0.0)
        best = max(dist, key=dist.__getitem__)
        return (best, dist[best])

    def extract_intent(self, text: str) -> str:
        """Classify intent from INTENT_LABELS."""
        lower = text.lower().strip()
        # Rule-based heuristics for high accuracy
        if any(lower.startswith(w) for w in ("hi", "hello", "hey", "greetings", "howdy")):
            return "greeting"
        if any(lower.startswith(w) for w in ("bye", "goodbye", "farewell", "see you", "later")):
            return "farewell"
        if lower.endswith("?") or lower.startswith(("what", "who", "where", "when", "why", "how", "is ", "are ", "do ", "does ", "can ", "will ")):
            return "question"
        if any(lower.startswith(w) for w in ("please", "do ", "stop", "start", "go", "run", "make", "give", "tell", "show", "open", "close", "set ")):
            return "command"
        return "statement"

    def extract_entities(self, text: str) -> List[str]:
        """Extract entities: tokens that are nouns (capitalized or in noun list)."""
        tokens = re.findall(r"[a-zA-Z']+", text)
        entities = []
        for tok in tokens:
            if tok[0].isupper() and tok.lower() not in {
                "i", "a", "an", "the", "and", "or", "but", "is", "are",
                "was", "were", "be", "been", "have", "has", "had",
            }:
                entities.append(tok)
            elif tok.lower() in _NOUN_WORDS:
                entities.append(tok)
        return list(dict.fromkeys(entities))  # deduplicate preserving order


class NgramNLUAdapter:
    """Wraps NgramNLU and integrates with NSCK."""

    _BUILTIN_SENTENCES = [
        ("what is this?", "question"),
        ("how does it work?", "question"),
        ("where are you going?", "question"),
        ("please open the door", "command"),
        ("stop the process", "command"),
        ("go to the next step", "command"),
        ("the system is running", "statement"),
        ("this is a test", "statement"),
        ("results are stored here", "statement"),
        ("hello there", "greeting"),
        ("hi how are you", "greeting"),
        ("hey there", "greeting"),
        ("goodbye", "farewell"),
        ("see you later", "farewell"),
        ("bye bye", "farewell"),
    ]

    def __init__(self, config=None):
        self._config = config
        self._nlu = NgramNLU(n=2)
        sentences = [s for s, _ in self._BUILTIN_SENTENCES]
        labels = [l for _, l in self._BUILTIN_SENTENCES]
        try:
            from python.core.language.distributional_semantics import BUILTIN_CORPUS
            for toks in BUILTIN_CORPUS[:50]:
                sent = " ".join(toks)
                sentences.append(sent)
                labels.append("statement")
        except Exception:
            pass
        self._nlu.train(sentences, labels)

    def process(self, text: str) -> dict:
        intent = self._nlu.extract_intent(text)
        entities = self._nlu.extract_entities(text)
        label, confidence = self._nlu.predict(text)
        label_probs = self._nlu.predict_distribution(text)
        return {
            "intent": intent,
            "entities": entities,
            "confidence": float(confidence),
            "label_probs": label_probs,
        }
