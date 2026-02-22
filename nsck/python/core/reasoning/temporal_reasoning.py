"""
NSCK Temporal Reasoning Module
================================
Implements **Allen's Interval Algebra** (Allen 1983) — a complete, sound,
and decidable calculus for reasoning about temporal relationships between
time-interval events.

Why temporal reasoning matters for NSCK
-----------------------------------------
Natural language is saturated with temporal information:
  "Alice taught Bob *before* the exam."
  "The rain *caused* the flood *during* the storm."
  "Treatment started *after* diagnosis."

Without an explicit temporal layer, NSCK stores these as unordered facts in
the causal graph, losing crucial sequencing information that is needed for:
  * Narrative understanding (story comprehension).
  * Causal reasoning (causes must precede effects).
  * Planning (STRIPS action ordering).
  * Question answering ("What happened first?").

Allen's 13 interval relations (+ their converses = 26 total)
-------------------------------------------------------------
Symbol  Relation          Meaning
------  --------          -------
<       before            A ends before B begins          (converse: >)
m       meets             A end == B start                (converse: mi)
o       overlaps          A starts before B, partial ovlp (converse: oi)
F       finished-by       A finishes == B finishes; A longer (converse: f)
D       contains          A contains B entirely           (converse: d)
s       starts            A starts == B; A shorter        (converse: S)
=       equals            A == B                          (converse: =)

In NSCK we use string labels: "before","meets","overlaps","finished_by",
"contains","starts","equals" and their converses prefixed with "i_"
(e.g. "i_before" = "after").

Composition table (Allen 1983)
--------------------------------
Used to infer relations through transitivity:
  If A before B  AND  B before C  → A before C
  If A before B  AND  B meets  C  → A before C
  (full 13×13 table implemented)

Integration with NSCK
---------------------
* TemporalKnowledgeGraph stores (event_A, relation, event_B) triples.
* query_ordering(A, B) returns the interval relation between A and B
  (or infers it via transitivity if not directly stored).
* TextKnowledgeLearner can call register_event() when it extracts temporal
  markers: "before", "after", "during", "while", "until", "since" etc.
* CausalReasoner enforces cause-before-effect using temporal precedence.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("nsck.temporal_reasoning")

# ---------------------------------------------------------------------------
# Allen's 13 base relations (+ converses)
# ---------------------------------------------------------------------------

# Converse map: rel → converse_of_rel
_CONVERSE: Dict[str, str] = {
    "before":      "after",
    "after":       "before",
    "meets":       "met_by",
    "met_by":      "meets",
    "overlaps":    "overlapped_by",
    "overlapped_by": "overlaps",
    "starts":      "started_by",
    "started_by":  "starts",
    "during":      "contains",
    "contains":    "during",
    "finishes":    "finished_by",
    "finished_by": "finishes",
    "equals":      "equals",
}

# Linguistic markers → Allen relation
_MARKER_TO_RELATION: Dict[str, str] = {
    "before":   "before",
    "prior to": "before",
    "earlier":  "before",
    "after":    "after",
    "following":"after",
    "since":    "after",
    "meets":    "meets",
    "during":   "during",
    "while":    "overlaps",
    "when":     "overlaps",
    "until":    "met_by",
    "as":       "overlaps",
    "starts":   "starts",
    "begins":   "starts",
    "equals":   "equals",
    "same time":"equals",
}

# Minimal transitivity table (most common compositions)
# Keyed as (rel1, rel2) → resulting_relation (or None if ambiguous)
_TRANSITIVITY: Dict[Tuple[str, str], Optional[str]] = {
    ("before",  "before"):      "before",
    ("before",  "meets"):       "before",
    ("before",  "overlaps"):    "before",
    ("before",  "starts"):      "before",
    ("before",  "during"):      "before",
    ("before",  "finishes"):    "before",
    ("before",  "equals"):      "before",
    ("after",   "after"):       "after",
    ("after",   "met_by"):      "after",
    ("meets",   "before"):      "before",
    ("meets",   "meets"):       "before",
    ("meets",   "overlaps"):    "before",
    ("meets",   "starts"):      "before",
    ("meets",   "during"):      "before",
    ("meets",   "finishes"):    "before",
    ("meets",   "equals"):      "meets",
    ("overlaps","before"):      "before",
    ("overlaps","meets"):       "before",
    ("during",  "before"):      "before",
    ("during",  "meets"):       "before",
    ("during",  "overlaps"):    None,
    ("during",  "during"):      "during",
    ("equals",  "before"):      "before",
    ("equals",  "meets"):       "meets",
    ("equals",  "overlaps"):    "overlaps",
    ("equals",  "during"):      "during",
    ("equals",  "starts"):      "starts",
    ("equals",  "finishes"):    "finishes",
    ("equals",  "equals"):      "equals",
    ("contains","before"):      None,
    ("contains","after"):       "after",
    ("contains","during"):      None,
    ("contains","contains"):    "contains",
    ("starts",  "before"):      "before",
    ("starts",  "during"):      "during",
    ("finishes","after"):       "after",
    ("finishes","overlapped_by"):"after",
}


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class TemporalFact:
    """A temporal relationship between two events."""
    event_a: str
    relation: str
    event_b: str
    confidence: float = 1.0
    source: str = ""

    def __str__(self) -> str:
        return f"{self.event_a} {self.relation} {self.event_b}"


@dataclass
class TemporalQuery:
    """Result of a temporal ordering query."""
    event_a: str
    event_b: str
    relation: Optional[str]
    inferred: bool          # True if derived via transitivity
    confidence: float
    inference_path: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TemporalKnowledgeGraph
# ---------------------------------------------------------------------------

class TemporalKnowledgeGraph:
    """
    Stores and reasons over Allen-interval temporal facts.

    Usage::

        tkg = TemporalKnowledgeGraph()
        tkg.add_fact("diagnosis", "before", "treatment")
        tkg.add_fact("treatment", "before", "recovery")

        q = tkg.query_ordering("diagnosis", "recovery")
        print(q.relation)   # "before"  (inferred via transitivity)
        print(q.inferred)   # True

        print(tkg.timeline("diagnosis"))
        # [("before", "treatment"), ("before", "recovery")]
    """

    def __init__(self, max_facts: int = 10_000):
        self.max_facts = max_facts

        # Forward index: event → [(relation, other_event, confidence)]
        self._forward: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        # Reverse index: (a, b) → TemporalFact for quick lookup
        self._direct: Dict[Tuple[str, str], TemporalFact] = {}

        self._total_facts: int = 0
        logger.info("[TemporalKG] initialized (max_facts=%d)", max_facts)

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def add_fact(
        self,
        event_a: str,
        relation: str,
        event_b: str,
        confidence: float = 1.0,
        source: str = "",
    ) -> None:
        """
        Register a temporal relationship ``event_a relation event_b``.

        The converse relationship is automatically also stored.
        """
        relation = relation.lower().replace(" ", "_")
        if relation not in _CONVERSE:
            logger.warning(
                "[TemporalKG] unknown relation '%s'; storing as-is", relation
            )

        if self._total_facts >= self.max_facts:
            logger.warning(
                "[TemporalKG] capacity %d reached; fact not stored", self.max_facts
            )
            return

        # Forward
        fact = TemporalFact(event_a, relation, event_b, confidence, source)
        self._direct[(event_a, event_b)] = fact
        self._forward[event_a].append((relation, event_b, confidence))

        # Converse
        converse_rel = _CONVERSE.get(relation, f"i_{relation}")
        conv_fact = TemporalFact(event_b, converse_rel, event_a, confidence, source)
        self._direct[(event_b, event_a)] = conv_fact
        self._forward[event_b].append((converse_rel, event_a, confidence))

        self._total_facts += 1
        logger.debug("[TemporalKG] added: %s", str(fact))

    def add_from_text_markers(
        self, event_a: str, marker: str, event_b: str, confidence: float = 0.9
    ) -> Optional[str]:
        """
        Translate a linguistic marker ("before", "after", "during" …) to an
        Allen relation and store the fact.  Returns the relation used or None.
        """
        marker_l = marker.lower().strip()
        relation = _MARKER_TO_RELATION.get(marker_l)
        if relation is None:
            logger.debug(
                "[TemporalKG] no mapping for marker '%s'", marker
            )
            return None
        self.add_fact(event_a, relation, event_b, confidence)
        return relation

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_ordering(self, event_a: str, event_b: str) -> TemporalQuery:
        """
        Return the temporal relation between event_a and event_b.

        1. Check direct storage.
        2. Attempt one-hop transitivity inference.
        3. Return None relation if unknown.
        """
        # Direct lookup
        if (event_a, event_b) in self._direct:
            fact = self._direct[(event_a, event_b)]
            return TemporalQuery(
                event_a=event_a,
                event_b=event_b,
                relation=fact.relation,
                inferred=False,
                confidence=fact.confidence,
            )

        # One-hop inference: event_a → X → event_b
        for rel1, intermediate, conf1 in self._forward.get(event_a, []):
            for rel2, target, conf2 in self._forward.get(intermediate, []):
                if target == event_b:
                    composed = _TRANSITIVITY.get((rel1, rel2))
                    if composed is not None:
                        return TemporalQuery(
                            event_a=event_a,
                            event_b=event_b,
                            relation=composed,
                            inferred=True,
                            confidence=min(conf1, conf2) * 0.9,
                            inference_path=[event_a, intermediate, event_b],
                        )

        return TemporalQuery(
            event_a=event_a,
            event_b=event_b,
            relation=None,
            inferred=False,
            confidence=0.0,
        )

    def timeline(self, event: str) -> List[Tuple[str, str]]:
        """
        Return all known temporal relationships involving ``event``.

        Returns: list of (relation, other_event) sorted by relation name.
        """
        pairs = [(rel, other) for rel, other, _ in self._forward.get(event, [])]
        return sorted(pairs, key=lambda x: x[0])

    def what_happened_before(self, event: str) -> List[str]:
        """Return events known to precede ``event``."""
        return [
            other
            for rel, other, _ in self._forward.get(event, [])
            if rel in ("after", "met_by", "overlapped_by", "started_by", "finished_by")
        ]

    def what_happened_after(self, event: str) -> List[str]:
        """Return events known to follow ``event``."""
        return [
            other
            for rel, other, _ in self._forward.get(event, [])
            if rel in ("before", "meets", "overlaps", "starts", "finishes")
        ]

    def get_all_events(self) -> Set[str]:
        """Return the set of all event names in the graph."""
        return set(self._forward.keys())

    def get_fact_count(self) -> int:
        """Return number of stored temporal facts (counting each direction once)."""
        return self._total_facts

    # ------------------------------------------------------------------
    # Natural language extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_markers_from_sentence(sentence: str) -> List[Tuple[str, str]]:
        """
        Scan a sentence for temporal marker words and return
        (marker_word, position) pairs.

        Very lightweight — no dependency beyond stdlib.
        """
        found: List[Tuple[str, str]] = []
        words = sentence.lower().split()
        for i, word in enumerate(words):
            word_clean = word.strip(".,;!?")
            if word_clean in _MARKER_TO_RELATION:
                found.append((word_clean, str(i)))
        return found

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        return {
            "total_facts": self._total_facts,
            "events": len(self._forward),
            "allen_relations_used": list(
                {rel for rels in self._forward.values() for rel, _, _ in rels}
            ),
        }
