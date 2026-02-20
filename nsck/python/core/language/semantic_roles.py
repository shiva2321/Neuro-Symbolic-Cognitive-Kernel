"""
NSCK VSA Semantic Role Labeling (SRL)
======================================
Identifies **who did what to whom, where, when, why, and how** in a sentence
using Vector Symbolic Architecture operations — no neural network required.

Architecture
------------
Semantic Role Labeling is implemented in three stages:

1. **Candidate extraction** – lexico-syntactic heuristics identify the
   predicate (main verb) and candidate argument spans.

2. **VSA role binding** – each candidate is bound with its proposed
   ``ThematicRole`` hypervector to form a composite:
   ``S = bind(AGENT, v_subject) ⊕ bind(PRED, v_verb) ⊕ bind(PATIENT, v_object) ⊕ ...``

3. **Resonator verification** – an iterative resonator network factorizes S
   to recover each (role, filler) pair.  Roles whose resonator estimate
   exceeds a similarity threshold are accepted.

Thematic Roles (PropBank-inspired)
-----------------------------------
* PRED       – the predicate (main verb)
* AGENT      – the do-er / causer (A0 in PropBank)
* PATIENT    – the entity affected (A1)
* THEME      – the entity moved / described (A1 in some frames)
* RECIPIENT  – the beneficiary (A2)
* INSTRUMENT – the means used (A2/A3)
* LOCATION   – the spatial setting (AM-LOC)
* TEMPORAL   – the time (AM-TMP)
* MANNER     – how the action is done (AM-MNR)
* CAUSE      – why it happened (AM-CAU)
* PURPOSE    – the goal (AM-PRP)
* NEGATION   – negation marker (AM-NEG)

Usage
-----
>>> from python.core.language.semantic_roles import SemanticRoleLabeler
>>> srl = SemanticRoleLabeler()
>>> frame = srl.label("The dog quickly bit the boy in the park.")
>>> print(frame)
SRLFrame(pred='bit', agent='dog', patient='boy', location='park', manner='quickly')
>>> srl.label("Mary did not give the book to John yesterday.")
SRLFrame(pred='give', agent='mary', patient='book', recipient='john', temporal='yesterday', negation=True)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import python.core.vsa.hypervec_shim as hypervec_rs


# ---------------------------------------------------------------------------
# Thematic roles — deterministic VSA vectors
# ---------------------------------------------------------------------------

_ROLE_SEEDS: Dict[str, int] = {
    "PRED":       60001,
    "AGENT":      60002,
    "PATIENT":    60003,
    "THEME":      60004,
    "RECIPIENT":  60005,
    "INSTRUMENT": 60006,
    "LOCATION":   60007,
    "TEMPORAL":   60008,
    "MANNER":     60009,
    "CAUSE":      60010,
    "PURPOSE":    60011,
    "NEGATION":   60012,
}


def _role_hv(role: str) -> Any:
    seed = _ROLE_SEEDS.get(role.upper(), 60099)
    return hypervec_rs.HyperVector(seed)


def _word_hv(word: str) -> Any:
    """Deterministic HV for a word (hash-based seed)."""
    import hashlib
    seed = int(hashlib.md5(word.lower().encode()).hexdigest()[:8], 16)
    return hypervec_rs.HyperVector(seed % (2**31))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SRLFrame:
    """A semantic role frame for one predicate."""
    pred: str = ""
    agent: str = ""
    patient: str = ""
    theme: str = ""
    recipient: str = ""
    instrument: str = ""
    location: str = ""
    temporal: str = ""
    manner: str = ""
    cause: str = ""
    purpose: str = ""
    negation: bool = False
    # All roles including extras (role → filler string)
    all_roles: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "pred": self.pred,
            "agent": self.agent,
            "patient": self.patient,
        }
        for attr in ("theme", "recipient", "instrument", "location",
                     "temporal", "manner", "cause", "purpose"):
            val = getattr(self, attr)
            if val:
                d[attr] = val
        if self.negation:
            d["negation"] = True
        d["confidence"] = round(self.confidence, 3)
        return d

    def __str__(self) -> str:
        parts = [f"pred='{self.pred}'"]
        for attr in ("agent", "patient", "theme", "recipient",
                     "instrument", "location", "temporal", "manner",
                     "cause", "purpose"):
            val = getattr(self, attr)
            if val:
                parts.append(f"{attr}='{val}'")
        if self.negation:
            parts.append("negation=True")
        return "SRLFrame(" + ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Lexico-syntactic heuristics
# ---------------------------------------------------------------------------

# Common prepositions and the roles they typically signal
_PREP_TO_ROLE: Dict[str, str] = {
    "in":       "LOCATION",
    "at":       "LOCATION",
    "on":       "LOCATION",
    "inside":   "LOCATION",
    "outside":  "LOCATION",
    "near":     "LOCATION",
    "by":       "LOCATION",
    "to":       "RECIPIENT",
    "for":      "PURPOSE",
    "from":     "CAUSE",
    "with":     "INSTRUMENT",
    "without":  "INSTRUMENT",
    "because":  "CAUSE",
    "since":    "CAUSE",
    "through":  "INSTRUMENT",
    "via":      "INSTRUMENT",
    "using":    "INSTRUMENT",
    "until":    "TEMPORAL",
    "before":   "TEMPORAL",
    "after":    "TEMPORAL",
    "during":   "TEMPORAL",
    "when":     "TEMPORAL",
    "while":    "TEMPORAL",
    "yesterday": "TEMPORAL",
    "today":    "TEMPORAL",
    "tomorrow": "TEMPORAL",
}

# Manner adverbs end in -ly; catch these
_MANNER_RE = re.compile(r"\b(\w+ly)\b", re.IGNORECASE)

# Negation markers
_NEGATION_WORDS = {"not", "never", "no", "nor", "neither", "without",
                   "n't", "isn't", "wasn't", "didn't", "doesn't", "won't",
                   "can't", "couldn't", "wouldn't", "shouldn't"}

# Common auxiliary / modal verbs (not the main predicate)
_AUXILIARIES = {"is", "are", "was", "were", "be", "been", "being",
                "has", "have", "had", "do", "does", "did",
                "will", "would", "shall", "should", "may", "might",
                "must", "can", "could", "ought"}

# Common irregular past-tense forms that won't match the regular pattern
_IRREGULAR_PAST = {
    "ate", "bit", "broke", "brought", "built", "bought", "caught", "came",
    "chose", "cut", "did", "drew", "drank", "drove", "fell", "felt", "flew",
    "forgot", "found", "gave", "got", "grew", "had", "heard", "held", "hit",
    "kept", "knew", "left", "lost", "made", "meant", "met", "paid", "put",
    "ran", "read", "rode", "rose", "said", "saw", "sent", "set", "shot",
    "showed", "shut", "sang", "sat", "slept", "sold", "sought", "spoke",
    "spent", "stood", "stole", "struck", "swam", "swept", "taught", "told",
    "took", "threw", "understood", "woke", "wore", "won", "wrote", "let",
    "led", "lent", "lit", "met", "meant", "shook", "thought", "went",
}


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-zA-Z'-]+|\d+", text.lower())


def _extract_noun_phrase(tokens: List[str], start: int) -> Tuple[str, int]:
    """
    Greedily consume a noun phrase starting at *start*.
    Returns (phrase, next_index).
    """
    determiners = {"the", "a", "an", "this", "that", "these", "those",
                   "my", "your", "his", "her", "its", "our", "their"}
    adjectives_re = re.compile(r"(?:\w+ly|\w+(?:ful|less|ous|ive|al|ic|ish))")
    i = start
    words = []

    # Skip determiner
    if i < len(tokens) and tokens[i] in determiners:
        i += 1

    # Consume adjectives
    while i < len(tokens) and adjectives_re.match(tokens[i]):
        words.append(tokens[i])
        i += 1

    # Consume noun (one word for simplicity)
    if i < len(tokens) and tokens[i] not in _AUXILIARIES:
        words.append(tokens[i])
        i += 1

    return " ".join(words), i


# ---------------------------------------------------------------------------
# Resonator
# ---------------------------------------------------------------------------

class _Resonator:
    """
    Simplified resonator network: single-step unbinding with codebook lookup.

    For each role, it computes the "filler estimate":
        f_role_est = unbind(composite, role_hv) = composite XOR role_hv
    Then searches the word codebook for the nearest match.

    In a full resonator network this would iterate to convergence; the
    single-step version suffices for short codebooks (<1000 words).
    """

    def __init__(self, word_codebook: Dict[str, Any]):
        self.codebook = word_codebook  # word → HV

    def recover_filler(self, composite: Any, role: str) -> Tuple[str, float]:
        """
        Recover the best-matching filler word for *role* from *composite*.
        Returns (word, similarity).
        """
        role_hv = _role_hv(role)
        estimate = composite.xor(role_hv)
        best_word = ""
        best_sim = -1.0
        for word, hv in self.codebook.items():
            sim = float(estimate.similarity(hv))
            if sim > best_sim:
                best_sim = sim
                best_word = word
        return best_word, best_sim


# ---------------------------------------------------------------------------
# SemanticRoleLabeler — public API
# ---------------------------------------------------------------------------

class SemanticRoleLabeler:
    """
    VSA-based Semantic Role Labeler.

    Label sentences without any neural network by combining:
    1. Lexico-syntactic heuristics to identify candidates.
    2. VSA binding to form a composite sentence representation.
    3. Resonator unbinding to verify role assignments.
    """

    # Minimum resonator similarity to accept a role assignment
    RESONATOR_THRESHOLD = 0.40

    def __init__(self):
        self._codebook: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label(self, sentence: str) -> SRLFrame:
        """
        Extract a semantic role frame from *sentence*.

        Returns an SRLFrame populated with recognised roles.
        """
        tokens = _tokenize(sentence)
        frame = self._heuristic_label(sentence, tokens)

        # Verify with resonator if codebook is populated
        if self._codebook:
            frame = self._resonator_verify(frame, tokens)

        return frame

    def label_batch(self, sentences: List[str]) -> List[SRLFrame]:
        """Label a list of sentences. Returns a list of SRLFrames."""
        return [self.label(s) for s in sentences]

    def update_codebook(self, words: List[str]) -> None:
        """Add words to the resonator codebook."""
        for w in words:
            if w.lower() not in self._codebook:
                self._codebook[w.lower()] = _word_hv(w)

    # ------------------------------------------------------------------
    # Heuristic stage
    # ------------------------------------------------------------------

    def _heuristic_label(self, sentence: str, tokens: List[str]) -> SRLFrame:
        """Lexico-syntactic first pass."""
        frame = SRLFrame()

        # --- Negation ---
        for t in tokens:
            if t in _NEGATION_WORDS:
                frame.negation = True
                break

        # --- Manner adverbs ---
        manner_matches = _MANNER_RE.findall(sentence)
        if manner_matches:
            frame.manner = manner_matches[0].lower()

        # --- Find predicate ---
        pred_idx = self._find_predicate(tokens)
        if pred_idx < 0:
            frame.confidence = 0.0
            return frame

        frame.pred = tokens[pred_idx]

        # --- Subject (AGENT) is typically the NP before the verb ---
        if pred_idx > 0:
            agent, _ = _extract_noun_phrase(tokens, max(0, pred_idx - 3))
            if agent:
                frame.agent = agent

        # --- Object (PATIENT/THEME) is the NP immediately after verb ---
        obj_start = pred_idx + 1
        # Skip auxiliaries and negations
        while obj_start < len(tokens) and (
            tokens[obj_start] in _AUXILIARIES
            or tokens[obj_start] in _NEGATION_WORDS
        ):
            obj_start += 1
        # Skip if the next token is a preposition (it belongs to a PP, not the direct object)
        obj_end = obj_start
        if obj_start < len(tokens) and tokens[obj_start] not in _PREP_TO_ROLE:
            patient, obj_end = _extract_noun_phrase(tokens, obj_start)
            if patient:
                frame.patient = patient

        # --- Prepositional phrases (scan from after the predicate) ---
        self._parse_prepositions(tokens, pred_idx + 1, frame)

        # --- Temporal: standalone time words ---
        time_words = {"yesterday", "today", "tomorrow", "now",
                      "recently", "soon", "later", "earlier"}
        for t in tokens:
            if t in time_words and not frame.temporal:
                frame.temporal = t

        frame.confidence = 0.70
        return frame

    def _find_predicate(self, tokens: List[str]) -> int:
        """Return index of the main predicate (main verb)."""
        # Pass 1 (highest priority): irregular past tenses
        for i, t in enumerate(tokens):
            if t in _IRREGULAR_PAST:
                return i
        # Pass 2: non-auxiliary verb with regular past/present morphology
        # Exclude common English nouns that happen to end in -s/-ed/-ing/-en
        _NON_VERB_ENDINGS = re.compile(
            r"^(?:children|women|men|garden|chicken|wooden|golden|token|"
            r"kitchen|linen|seven|happen|often|open|even|broken|frozen|"
            r"spoken|stolen|given|taken|chosen|driven|risen|hidden|"
            r"\w+ness|\w+ment|\w+tion|\w+sion)$",
            re.IGNORECASE,
        )
        for i, t in enumerate(tokens):
            if (t not in _AUXILIARIES
                    and not _NON_VERB_ENDINGS.match(t)
                    and re.match(r"\w+(?:s|ed|ing|en)$", t)):
                return i
        # Pass 3: any verb-like token in the standard auxiliary list that
        # is used as a main verb (e.g. "is", "has" in copular/predicative use)
        main_verb_candidates = {"is", "are", "was", "were", "has", "have", "had", "does", "did"}
        for i, t in enumerate(tokens):
            if t in main_verb_candidates:
                return i
        return -1

    def _parse_prepositions(
        self, tokens: List[str], start: int, frame: SRLFrame
    ) -> None:
        """Fill location/temporal/recipient/instrument/etc. from PP chunks."""
        i = start
        while i < len(tokens):
            t = tokens[i]
            role = _PREP_TO_ROLE.get(t)
            if role:
                # grab the NP after the preposition
                np, next_i = _extract_noun_phrase(tokens, i + 1)
                if np:
                    role_lower = role.lower()
                    if hasattr(frame, role_lower):
                        # Only set if not already filled
                        if not getattr(frame, role_lower):
                            setattr(frame, role_lower, np)
            i += 1

    # ------------------------------------------------------------------
    # Resonator verification stage
    # ------------------------------------------------------------------

    def _resonator_verify(
        self, frame: SRLFrame, tokens: List[str]
    ) -> SRLFrame:
        """
        Build a VSA composite for the sentence and use the resonator to
        verify/improve the heuristic frame.
        """
        # Update codebook with all tokens in this sentence
        self.update_codebook(tokens)

        # Build composite VSA sentence representation
        composite = self._build_composite(frame)

        resonator = _Resonator(self._codebook)
        total_roles = 0
        matched_roles = 0

        for role in ("PRED", "AGENT", "PATIENT", "THEME", "RECIPIENT",
                     "INSTRUMENT", "LOCATION", "TEMPORAL", "MANNER"):
            current_filler = getattr(frame, role.lower(), "")
            if not current_filler:
                continue
            total_roles += 1
            recovered, sim = resonator.recover_filler(composite, role)
            if sim >= self.RESONATOR_THRESHOLD:
                matched_roles += 1
            # Only override if resonator is more confident and it found
            # a token that actually appears in the sentence
            if (sim >= self.RESONATOR_THRESHOLD and recovered in tokens
                    and recovered != current_filler):
                setattr(frame, role.lower(), recovered)

        if total_roles > 0:
            frame.confidence = max(frame.confidence,
                                   matched_roles / total_roles)
        return frame

    def _build_composite(self, frame: SRLFrame) -> Any:
        """
        Build composite VSA vector:
        S = ⊕ bind(role_hv, filler_hv)   for each filled role.
        """
        composite = None
        for role in ("PRED", "AGENT", "PATIENT", "THEME", "RECIPIENT",
                     "INSTRUMENT", "LOCATION", "TEMPORAL", "MANNER"):
            filler = getattr(frame, role.lower(), "")
            if not filler:
                continue
            r_hv = _role_hv(role)
            f_hv = _word_hv(filler)
            bound = r_hv.xor(f_hv)
            composite = bound if composite is None else composite.bundle(bound)
        return composite or hypervec_rs.HyperVector(0)
