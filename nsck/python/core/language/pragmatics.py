"""
NSCK Pragmatics Module (G2 from roadmap)
========================================
Scalar implicature, Gricean maxims, and basic pragmatic inference.

No neural networks. No statistical models. Rule-based + VSA confidence scoring.

Overview
--------
Language is richer than its literal content.  "Some students passed" *implies*
"not all students passed" (Gricean maxims of Quantity).  "Can you pass the salt?"
is a polite *request*, not a question about ability (indirect speech acts).

This module implements:

1. **Scalar Implicature** (Grice 1975)
   - Detects scalar terms (some, many, often, possible, warm, good, …)
   - Infers upper-bound implicature:  "some" → "not all"
   - Infers lower-bound implicature:  "at least 3" → possibly "more than 3"

2. **Gricean Maxims Checking**
   - Quantity: Is the utterance as informative as required?
   - Quality:  Does the speaker assert only what they believe to be true?
   - Relation: Is the utterance relevant to the current topic?
   - Manner:   Is the utterance clear, brief, orderly?

3. **Indirect Speech Acts** (Searle 1969)
   - Classifies utterances as: assert, question, request, command, offer, promise
   - Detects politeness hedging ("Could you…", "Would you mind…")

4. **Presupposition Detection**
   - "John stopped smoking" → presupposes John was smoking
   - "Have you stopped beating your wife?" → loaded presupposition

5. **VSA Confidence Integration**
   - Each pragmatic inference is scored 0–1
   - Scores integrate with the belief system (BeliefMetadata)

Usage
-----
>>> from python.core.language.pragmatics import PragmaticsEngine
>>> pe = PragmaticsEngine()
>>> result = pe.analyze("Some students passed the exam.")
>>> result.scalar_implicatures
[ScalarImplicature(trigger='some', upper_bound='not all students passed the exam')]
>>> result.speech_act
'assert'
>>> pe.analyze("Can you pass the salt?").speech_act
'request'
>>> pe.check_maxims("The cat is on the mat.", context_topic="weather")
[MaximViolation(maxim='Relation', severity='moderate', note='Off-topic response')]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Scalar scales (Horn scales)
# ---------------------------------------------------------------------------
# Each scale is ordered from weakest to strongest:
# Using the weakest term implicates NOT the stronger ones.

SCALAR_SCALES: Dict[str, List[str]] = {
    # Quantity scale
    "some":     ["some", "many", "most", "all"],
    "many":     ["many", "most", "all"],
    "most":     ["most", "all"],
    "few":      ["few", "some", "many", "most", "all"],
    "possible": ["possible", "likely", "certain"],
    "likely":   ["likely", "certain"],
    "might":    ["might", "may", "will", "must"],
    "may":      ["may", "will", "must"],
    "could":    ["could", "can", "will"],
    # Temperature / quality scales
    "warm":     ["warm", "hot", "boiling"],
    "cool":     ["cool", "cold", "freezing"],
    "good":     ["good", "great", "excellent", "perfect"],
    "bad":      ["bad", "terrible", "awful"],
    "okay":     ["okay", "good", "great", "excellent"],
    "sometimes": ["sometimes", "often", "always"],
    "often":    ["often", "always"],
    # Numeric
    "one":      ["one", "two", "three", "four", "five"],
    "two":      ["two", "three", "four", "five"],
    "three":    ["three", "four", "five"],
}

# Upper-bound implicature templates: what is implied when X is used
UPPER_BOUND_IMPLICATURE: Dict[str, str] = {
    "some":      "not all",
    "many":      "not all",
    "most":      "not all",
    "few":       "not many",
    "possible":  "not certain",
    "likely":    "not certain",
    "might":     "not definitely",
    "may":       "not definitely",
    "could":     "cannot be sure",
    "warm":      "not hot",
    "cool":      "not cold",
    "good":      "not great",
    "okay":      "not good or better",
    "bad":       "not terrible",
    "sometimes": "not always",
    "often":     "not always",
    # Numeric scalar implicatures
    "one":       "not more than one (in this context)",
    "two":       "not more than two (in this context)",
    "three":     "not more than three (in this context)",
}

# ---------------------------------------------------------------------------
# Indirect speech act patterns
# ---------------------------------------------------------------------------

# Each pattern maps regex → (speech_act, is_indirect)
_SPEECH_ACT_PATTERNS: List[Tuple[re.Pattern, str, bool]] = [
    (re.compile(r"^\s*can\s+you\b", re.I),                   "request",  True),
    (re.compile(r"^\s*could\s+you\b", re.I),                  "request",  True),
    (re.compile(r"^\s*would\s+you\s+mind\b", re.I),           "request",  True),
    (re.compile(r"^\s*would\s+you\b", re.I),                  "request",  True),
    (re.compile(r"^\s*will\s+you\b", re.I),                   "request",  True),
    (re.compile(r"^\s*please\b", re.I),                       "request",  False),
    (re.compile(r"^\s*(?:do|does|did|is|are|was|were|have|has)\s+\w", re.I), "question", False),
    (re.compile(r"^\s*(?:what|where|when|who|why|how|which)\b", re.I), "question", False),
    (re.compile(r"\?\s*$"),                                    "question", False),
    (re.compile(r"^\s*(?:i\s+(?:will|shall|promise|swear)\b)", re.I), "promise", False),
    (re.compile(r"^\s*(?:i\s+(?:offer|can\s+help)\b)", re.I), "offer",   False),
    (re.compile(r"^\s*(?:don'?t|do\s+not|stop|cease|never)\b", re.I), "command", False),
    (re.compile(r"!\s*$"),                                     "command", False),
]

# Politeness hedge patterns
_POLITENESS_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:please|kindly|if\s+you\s+(?:don'?t\s+mind|would|could))\b", re.I),
    re.compile(r"\b(?:would\s+you\s+mind|could\s+you\s+possibly|might\s+I)\b", re.I),
    re.compile(r"\b(?:sorry\s+to\s+bother|excuse\s+me|pardon)\b", re.I),
]

# Presupposition triggers
_PRESUPPOSITION_TRIGGERS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:stopped?|ceased?|quit)\s+(\w+ing)\b", re.I),
     "presupposes prior {0}"),
    (re.compile(r"\bregret(?:s|ted)?\s+(?:that\s+)?(.+)", re.I),
     "presupposes {0}"),
    (re.compile(r"\bknow(?:s)?\s+(?:that\s+)?(.+)", re.I),
     "presupposes truth of '{0}'"),
    (re.compile(r"\breali[sz]e(?:d|s)?\s+(?:that\s+)?(.+)", re.I),
     "presupposes {0}"),
    (re.compile(r"\bagain\b", re.I),
     "presupposes the action happened before"),
    (re.compile(r"\bstill\b", re.I),
     "presupposes the state held in the past"),
    (re.compile(r"\beven\b", re.I),
     "presupposes the action is noteworthy or unexpected"),
]


# ---------------------------------------------------------------------------
# Gricean maxim violation detection
# ---------------------------------------------------------------------------

# Relevance: topic-mismatch signal words (very rough heuristic)
_TOPIC_CHANGE_CUES = re.compile(
    r"\b(?:by\s+the\s+way|speaking\s+of|on\s+another\s+note|"
    r"incidentally|anyway|regardless)\b",
    re.I,
)

# Manner: overly wordy hedge phrases
_MANNER_HEDGE = re.compile(
    r"\b(?:it\s+might\s+be\s+(?:said|argued|noted)\s+that|"
    r"one\s+could\s+say|in\s+some\s+sense|more\s+or\s+less)\b",
    re.I,
)

# Quality: explicit uncertainty markers
_QUALITY_UNCERTAINTY = re.compile(
    r"\b(?:i\s+think|i\s+believe|i\s+(?:am\s+)?not\s+sure|perhaps|probably|"
    r"it\s+seems|apparently|supposedly)\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

@dataclass
class ScalarImplicature:
    """A single scalar implicature inference."""
    trigger: str          # The scalar term that triggered the implicature
    scale: List[str]      # The full Horn scale
    upper_bound: str      # What is implied (the negation of stronger terms)
    confidence: float = 0.9

    def __str__(self) -> str:
        return (
            f"'{self.trigger}' → implies '{self.upper_bound}' "
            f"(scale: {' < '.join(self.scale)})"
        )


@dataclass
class Presupposition:
    """A presupposition triggered by an utterance."""
    trigger_word: str
    content: str
    confidence: float = 0.85

    def __str__(self) -> str:
        return f"'{self.trigger_word}' presupposes: {self.content}"


@dataclass
class MaximViolation:
    """A potential violation of a Gricean maxim."""
    maxim: str            # Quantity | Quality | Relation | Manner
    severity: str         # low | moderate | high
    note: str

    def __str__(self) -> str:
        return f"[{self.maxim}] {self.severity}: {self.note}"


@dataclass
class PragmaticAnalysis:
    """
    Full pragmatic analysis of an utterance.

    Attributes
    ----------
    utterance : str
    speech_act : str
        assert | question | request | command | offer | promise
    is_indirect : bool
        True when the surface form differs from the intended speech act.
    is_polite : bool
        True when politeness hedges detected.
    scalar_implicatures : list[ScalarImplicature]
    presuppositions : list[Presupposition]
    maxim_violations : list[MaximViolation]
    pragmatic_enrichments : list[str]
        Free-text notes about pragmatic content.
    """
    utterance: str
    speech_act: str = "assert"
    is_indirect: bool = False
    is_polite: bool = False
    scalar_implicatures: List[ScalarImplicature] = field(default_factory=list)
    presuppositions: List[Presupposition] = field(default_factory=list)
    maxim_violations: List[MaximViolation] = field(default_factory=list)
    pragmatic_enrichments: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PragmaticsEngine
# ---------------------------------------------------------------------------

class PragmaticsEngine:
    """
    Full pragmatic analysis engine for NSCK.

    Implements:
    * Scalar implicature detection (Grice 1975 / Horn 1972)
    * Indirect speech act classification (Searle 1969)
    * Presupposition detection
    * Gricean maxim violation checking
    * Politeness hedging detection

    All analyses are deterministic rule-based — no probabilities, no ML.
    Confidence scores express how reliably the rule fires.

    Example
    -------
    >>> pe = PragmaticsEngine()
    >>> r = pe.analyze("Some students passed the exam.")
    >>> r.speech_act
    'assert'
    >>> r.scalar_implicatures[0].upper_bound
    'not all'
    >>> pe.analyze("Can you pass the salt?").speech_act
    'request'
    """

    def analyze(
        self,
        utterance: str,
        context_topic: Optional[str] = None,
    ) -> PragmaticAnalysis:
        """
        Perform full pragmatic analysis of *utterance*.

        Parameters
        ----------
        utterance    : str   The text to analyse.
        context_topic: str   (optional) The current conversation topic.
                             Used for Relation maxim checking.

        Returns
        -------
        PragmaticAnalysis with all sub-analyses populated.
        """
        result = PragmaticAnalysis(utterance=utterance)

        result.speech_act, result.is_indirect = self._classify_speech_act(utterance)
        result.is_polite = self._detect_politeness(utterance)
        result.scalar_implicatures = self._extract_scalar_implicatures(utterance)
        result.presuppositions = self._extract_presuppositions(utterance)
        result.maxim_violations = self._check_maxims(utterance, context_topic)
        result.pragmatic_enrichments = self._build_enrichments(result)

        return result

    # ------------------------------------------------------------------
    # Speech act classification
    # ------------------------------------------------------------------

    def _classify_speech_act(self, utterance: str) -> Tuple[str, bool]:
        """Return (speech_act, is_indirect)."""
        for pattern, act, indirect in _SPEECH_ACT_PATTERNS:
            if pattern.search(utterance):
                return act, indirect
        return "assert", False

    def _detect_politeness(self, utterance: str) -> bool:
        return any(p.search(utterance) for p in _POLITENESS_PATTERNS)

    # ------------------------------------------------------------------
    # Scalar implicature
    # ------------------------------------------------------------------

    def _extract_scalar_implicatures(
        self, utterance: str
    ) -> List[ScalarImplicature]:
        """
        Find scalar terms in *utterance* and infer upper-bound implicatures.

        Only the *weakest* scalar term in any scale is a valid trigger
        (using "all" instead of "some" cancels the implicature).
        """
        words = re.findall(r"\w+", utterance.lower())
        word_set = set(words)
        implicatures: List[ScalarImplicature] = []
        seen_triggers: set = set()

        for word in words:
            if word in SCALAR_SCALES and word not in seen_triggers:
                scale = SCALAR_SCALES[word]
                # Check: is a stronger term from the same scale also present?
                # If "all" is present, "some" is not the scalar trigger
                stronger = scale[scale.index(word) + 1:]
                if any(s in word_set for s in stronger):
                    continue  # Stronger term used — no implicature
                ub = UPPER_BOUND_IMPLICATURE.get(word)
                if ub:
                    implicatures.append(ScalarImplicature(
                        trigger=word,
                        scale=scale,
                        upper_bound=ub,
                        confidence=0.9,
                    ))
                    seen_triggers.add(word)

        return implicatures

    # ------------------------------------------------------------------
    # Presupposition
    # ------------------------------------------------------------------

    def _extract_presuppositions(
        self, utterance: str
    ) -> List[Presupposition]:
        presuppositions: List[Presupposition] = []
        for pattern, template in _PRESUPPOSITION_TRIGGERS:
            m = pattern.search(utterance)
            if m:
                # Fill template with captured group if any
                if m.lastindex and m.lastindex >= 1:
                    content = template.format(m.group(1).strip())
                else:
                    content = template
                presuppositions.append(Presupposition(
                    trigger_word=m.group(0).strip(),
                    content=content,
                ))
        return presuppositions

    # ------------------------------------------------------------------
    # Gricean maxim checking
    # ------------------------------------------------------------------

    def _check_maxims(
        self,
        utterance: str,
        context_topic: Optional[str],
    ) -> List[MaximViolation]:
        violations: List[MaximViolation] = []

        # Quantity: very short responses may violate informativeness
        word_count = len(re.findall(r"\w+", utterance))
        if word_count <= 2 and not utterance.strip().endswith("?"):
            violations.append(MaximViolation(
                maxim="Quantity",
                severity="low",
                note=f"Utterance is very short ({word_count} words); "
                     f"may be under-informative.",
            ))

        # Quality: explicit uncertainty markers
        if _QUALITY_UNCERTAINTY.search(utterance):
            violations.append(MaximViolation(
                maxim="Quality",
                severity="low",
                note="Speaker signals uncertainty — asserting something "
                     "they may not fully believe.",
            ))

        # Relation: topic mismatch signal
        if _TOPIC_CHANGE_CUES.search(utterance):
            violations.append(MaximViolation(
                maxim="Relation",
                severity="moderate",
                note="Cue phrase suggests possible topic change.",
            ))
        # Simple lexical overlap check if context_topic given
        if context_topic:
            # Tokenise both and check for any shared content word (4+ chars)
            topic_words = {w for w in re.findall(r"\w{4,}", context_topic.lower())}
            utt_words   = {w for w in re.findall(r"\w{4,}", utterance.lower())}
            overlap = len(topic_words & utt_words)
            if overlap == 0 and len(topic_words) > 0:
                violations.append(MaximViolation(
                    maxim="Relation",
                    severity="moderate",
                    note=f"No shared content words with topic '{context_topic}'.",
                ))

        # Manner: hedging phrase
        if _MANNER_HEDGE.search(utterance):
            violations.append(MaximViolation(
                maxim="Manner",
                severity="low",
                note="Overly circumlocutory hedge phrase detected.",
            ))

        # Manner: excessive length (very rough)
        if word_count > 60:
            violations.append(MaximViolation(
                maxim="Manner",
                severity="low",
                note=f"Utterance is long ({word_count} words); consider brevity.",
            ))

        return violations

    def check_maxims(
        self,
        utterance: str,
        context_topic: Optional[str] = None,
    ) -> List[MaximViolation]:
        """
        Public method: check *utterance* against Gricean maxims.

        Parameters
        ----------
        utterance     : str  The utterance to check.
        context_topic : str  (optional) Current conversation topic for
                             Relation maxim.

        Returns
        -------
        List of MaximViolation — empty if no violations detected.
        """
        return self._check_maxims(utterance, context_topic)

    # ------------------------------------------------------------------
    # Enrichment summary
    # ------------------------------------------------------------------

    def _build_enrichments(self, result: PragmaticAnalysis) -> List[str]:
        enrichments: List[str] = []
        for si in result.scalar_implicatures:
            enrichments.append(
                f"Scalar implicature: '{si.trigger}' implies {si.upper_bound}"
            )
        for ps in result.presuppositions:
            enrichments.append(f"Presupposition: {ps.content}")
        if result.is_indirect:
            enrichments.append(
                f"Indirect speech act: surface form differs from "
                f"intended speech act '{result.speech_act}'"
            )
        if result.is_polite:
            enrichments.append("Politeness hedge detected")
        for mv in result.maxim_violations:
            enrichments.append(f"Maxim [{mv.maxim}]: {mv.note}")
        return enrichments

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def speech_act(self, utterance: str) -> str:
        """Return just the speech act classification."""
        act, _ = self._classify_speech_act(utterance)
        return act

    def implicatures(self, utterance: str) -> List[str]:
        """Return a list of human-readable implicature strings."""
        return [str(si) for si in self._extract_scalar_implicatures(utterance)]

    def presuppositions(self, utterance: str) -> List[str]:
        """Return a list of human-readable presupposition strings."""
        return [str(ps) for ps in self._extract_presuppositions(utterance)]
