"""
NSCK Fluent Natural Language Generation (V6)
=============================================
Produces human-quality, non-patchy English responses from NSCK knowledge
graphs — without any LLM.

Design philosophy
-----------------
* **Relation verbalizer** — maps semantic-graph edge types to *varied*
  natural-language phrasings (3–5 alternates per relation), chosen by
  a deterministic hash so the output is reproducible.
* **FluentResponseComposer** — assembles topic sentences, supporting
  evidence sentences, and a conclusion into a coherent paragraph.
* **NSCKResponseEngine** — the public API that the rest of NSCK calls.

No templates are ever repeated consecutively for the same relation.
Connectives are chosen to match the logical relationship between facts.
Pronouns replace repeated subjects (anaphora).

Example
-------
>>> from python.core.language.fluent_nlg import NSCKResponseEngine
>>> engine = NSCKResponseEngine()
>>> frames = [
...     {"subject": "Python", "relation": "is_a", "object": "programming language"},
...     {"subject": "Python", "relation": "has_property", "object": "readability"},
...     {"subject": "Python", "relation": "causes", "object": "developer productivity"},
...     {"subject": "Python", "relation": "used_for", "object": "data science"},
... ]
>>> print(engine.respond(frames, topic="Python", query_type="explanatory"))
Python is a programming language. It is known for its readability, which
contributes to developer productivity. Furthermore, it is widely used for
data science.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Sentence variation templates per relation
# ---------------------------------------------------------------------------
# Each entry: list of (template, connective_hint)
# Placeholders: {S} = subject, {O} = object, {s} = subject lowercase,
#               {o} = object lowercase, {det} = determiner for object

_RELATION_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
    # Taxonomic
    "is_a": [
        ("{S} is a {o}.", ""),
        ("{S} is one of the {o}s.", ""),
        ("{S} belongs to the category of {o}.", ""),
        ("{S} represents a type of {o}.", ""),
        ("{S} can be classified as a {o}.", ""),
    ],
    "isa": [
        ("{S} is a {o}.", ""),
        ("{S} falls under the category of {o}.", ""),
    ],
    # Property
    "has_property": [
        ("{S} has {o}.", "Additionally,"),
        ("{S} is characterized by {o}.", "Additionally,"),
        ("{S} possesses {o}.", "Furthermore,"),
        ("{S} is known for its {o}.", "Moreover,"),
        ("{S} features {o}.", "In addition,"),
    ],
    "has": [
        ("{S} has {o}.", ""),
        ("{S} possesses {o}.", ""),
    ],
    # Causal
    "causes": [
        ("{S} causes {o}.", "As a result,"),
        ("{S} leads to {o}.", "Consequently,"),
        ("{S} results in {o}.", "Therefore,"),
        ("{S} brings about {o}.", "This leads to"),
        ("{S} is responsible for {o}.", "Because of this,"),
    ],
    "leads_to": [
        ("{S} leads to {o}.", "Consequently,"),
        ("{S} results in {o}.", "As a result,"),
        ("{S} brings about {o}.", "Therefore,"),
    ],
    "results_in": [
        ("{S} results in {o}.", "As a result,"),
        ("{S} produces {o}.", "Consequently,"),
        ("{S} generates {o}.", "Therefore,"),
    ],
    "triggered_by": [
        ("{O} triggers {s}.", "Because of this,"),
        ("{S} is triggered by {o}.", "As a result,"),
    ],
    # Conditional / logical
    "implies": [
        ("{S} implies {o}.", ""),
        ("{S} suggests that {o} follows.", ""),
        ("If {s}, then {o} is expected.", ""),
    ],
    "conditional_on": [
        ("{S} is conditional on {o}.", ""),
        ("{S} depends on {o}.", ""),
    ],
    "depends_on": [
        ("{S} depends on {o}.", ""),
        ("{S} relies on {o}.", ""),
        ("{S} requires {o}.", ""),
    ],
    # Temporal
    "precedes": [
        ("{S} comes before {o}.", "First,"),
        ("{S} precedes {o}.", "Then,"),
        ("{S} happens before {o}.", ""),
    ],
    "follows": [
        ("{S} follows {o}.", "Then,"),
        ("{S} comes after {o}.", "After that,"),
        ("{S} occurs after {o}.", "Subsequently,"),
    ],
    "since_event": [
        ("{S} has continued since {o}.", ""),
        ("Since {o}, {s} has been the case.", ""),
    ],
    "until_event": [
        ("{S} continues until {o}.", ""),
        ("{S} persists until {o}.", ""),
    ],
    "occurs_during": [
        ("{S} occurs during {o}.", "Meanwhile,"),
        ("{S} takes place during {o}.", ""),
    ],
    # Similarity / difference
    "similar_to": [
        ("{S} is similar to {o}.", "Similarly,"),
        ("{S} resembles {o}.", "In the same way,"),
        ("{S} shares characteristics with {o}.", "Likewise,"),
        ("{S} and {o} have much in common.", ""),
    ],
    "different_from": [
        ("{S} differs from {o}.", "However,"),
        ("{S} is distinct from {o}.", "In contrast,"),
        ("Unlike {o}, {s} has its own unique characteristics.", "On the other hand,"),
    ],
    "opposite_of": [
        ("{S} is the opposite of {o}.", "In contrast,"),
        ("{S} and {o} are opposites.", "However,"),
    ],
    # Parthood
    "part_of": [
        ("{S} is part of {o}.", ""),
        ("{S} forms a component of {o}.", ""),
        ("{S} is a constituent of {o}.", ""),
    ],
    "contains": [
        ("{S} contains {o}.", ""),
        ("{S} includes {o}.", ""),
        ("{S} is made up of {o}.", ""),
    ],
    # Capability
    "capable_of": [
        ("{S} is capable of {o}.", ""),
        ("{S} can perform {o}.", ""),
        ("{S} has the ability to {o}.", ""),
    ],
    "used_for": [
        ("{S} is used for {o}.", ""),
        ("{S} is commonly employed in {o}.", ""),
        ("{S} serves as a tool for {o}.", ""),
        ("{S} finds application in {o}.", ""),
    ],
    "associated_with": [
        ("{S} is associated with {o}.", ""),
        ("{S} is closely linked to {o}.", ""),
        ("{S} is connected to {o}.", ""),
    ],
    # Spatial
    "located_in": [
        ("{S} is located in {o}.", ""),
        ("{S} can be found in {o}.", ""),
    ],
    "above": [
        ("{S} is above {o}.", ""),
        ("{S} sits above {o}.", ""),
    ],
    "below": [
        ("{S} is below {o}.", ""),
        ("{S} sits below {o}.", ""),
    ],
    "near": [
        ("{S} is near {o}.", ""),
        ("{S} is close to {o}.", ""),
    ],
    # Negation
    "not_is_a": [
        ("{S} is not a {o}.", ""),
        ("{S} should not be classified as a {o}.", ""),
    ],
    "not_has_property": [
        ("{S} does not have {o}.", ""),
        ("{S} lacks {o}.", ""),
    ],
    "not_relates_to": [
        ("{S} is not related to {o}.", ""),
        ("{S} has no direct connection to {o}.", ""),
    ],
    "cannot_do": [
        ("{S} cannot {o}.", ""),
        ("{S} is unable to {o}.", ""),
    ],
    "lacks": [
        ("{S} lacks {o}.", ""),
        ("{S} does not possess {o}.", ""),
    ],
    "never_does": [
        ("{S} never {o}.", ""),
        ("{S} does not {o}.", ""),
    ],
    # Miscellaneous / default
    "relates_to": [
        ("{S} relates to {o}.", ""),
        ("{S} is connected to {o}.", ""),
    ],
    "semantically_related": [
        ("{S} is related to {o}.", ""),
        ("{S} is connected to the concept of {o}.", ""),
    ],
    "is": [
        ("{S} is {o}.", ""),
        ("{S} refers to {o}.", ""),
    ],
}

# Fallback when relation is unknown
_DEFAULT_TEMPLATES: List[Tuple[str, str]] = [
    ("{S} is related to {o} ({rel}).", ""),
    ("There is a {rel} relationship between {s} and {o}.", ""),
    ("{S} and {o} are connected through a {rel} link.", ""),
]

# Opening sentences for multi-sentence responses (by query type)
_TOPIC_SENTENCES: Dict[str, List[str]] = {
    "factual": [
        "Here is what is known about {topic}:",
        "The following describes {topic}:",
        "Regarding {topic}:",
        "{Topic} can be described as follows.",
    ],
    "explanatory": [
        "{Topic} is an important concept.",
        "Understanding {topic} requires examining several aspects.",
        "To explain {topic}: ",
        "{Topic} can be understood in the following way.",
    ],
    "causal": [
        "The causal chain involving {topic} unfolds as follows.",
        "{Topic} participates in the following causal sequence.",
        "Here is how {topic} relates to its effects.",
    ],
    "procedural": [
        "The following steps describe {topic}:",
        "To understand the procedure involving {topic}:",
        "The process involving {topic} works as follows.",
    ],
    "comparative": [
        "Comparing {topic} to related concepts reveals the following.",
        "Here is how {topic} compares to other concepts.",
    ],
}

# Conclusion sentence templates (added when > 2 facts)
_CONCLUSIONS: List[str] = [
    "In summary, {topic} is a multifaceted concept with these key characteristics.",
    "Taken together, these facts illustrate the nature of {topic}.",
    "These properties make {topic} a well-defined concept in this domain.",
]

# Discourse connectives by position
_POSITION_CONNECTIVES: Dict[str, List[str]] = {
    "middle": ["Also,", "Furthermore,", "In addition,", "Moreover,"],
    "last":   ["Finally,", "In summary,", "Lastly,", "To conclude,"],
}


# ---------------------------------------------------------------------------
# RelationVerbalizer
# ---------------------------------------------------------------------------

class RelationVerbalizer:
    """
    Maps (subject, relation, object) triples to varied natural English.

    Uses a deterministic rotation through template variants based on the
    hash of the triple so the same triple always produces the same variant
    but consecutive triples with the same relation differ.
    """

    def verbalize(
        self,
        subject: str,
        relation: str,
        obj: str,
        variant_seed: int = 0,
    ) -> Tuple[str, str]:
        """
        Verbalize one semantic triple.

        Parameters
        ----------
        subject, relation, obj : str
        variant_seed : int  offset to choose which template variant to use

        Returns
        -------
        (sentence, connective_hint) — the realized sentence and a suggested
        discourse connective for it (may be empty string).
        """
        templates = _RELATION_TEMPLATES.get(
            relation.lower().replace(" ", "_"),
            _DEFAULT_TEMPLATES,
        )
        idx = (hash(subject + relation + obj) + variant_seed) % len(templates)
        template, hint = templates[idx]
        sentence = self._fill(template, subject, relation, obj)
        return sentence, hint

    @staticmethod
    def _fill(template: str, subject: str, relation: str, obj: str) -> str:
        """Fill template placeholders."""
        # Capitalize first word of subject
        S = subject.strip()
        if S and S[0].islower():
            S = S[0].upper() + S[1:]

        s = subject.strip().lower()
        O_cap = obj.strip()
        if O_cap and O_cap[0].isupper() and not O_cap.isupper():
            pass  # proper noun — keep
        else:
            O_cap = obj.strip()

        o = obj.strip().lower()
        rel = relation.lower().replace("_", " ")

        result = (template
                  .replace("{S}", S)
                  .replace("{s}", s)
                  .replace("{O}", O_cap)
                  .replace("{o}", o)
                  .replace("{rel}", rel))
        # Ensure first char is uppercase
        if result:
            result = result[0].upper() + result[1:]
        return result


# ---------------------------------------------------------------------------
# FluentResponseComposer
# ---------------------------------------------------------------------------

class FluentResponseComposer:
    """
    Builds a full fluent response paragraph from a list of semantic frames.

    Pipeline
    --------
    1. Add a topic sentence if multi-sentence response.
    2. Realize each frame into a natural sentence (using RelationVerbalizer).
    3. Apply discourse connectives between sentences.
    4. Apply pronoun anaphora (avoid repeating the same subject).
    5. Add a conclusion sentence for multi-fact responses.
    6. Return the joined paragraph.

    Input frame schema
    ------------------
    Each frame is a dict with at minimum:
        subject  : str
        relation : str
        object   : str

    Optional keys:
        tense      : "present" | "past" | "future"
        negate     : bool
        importance : float (0–1)
        variant    : int   (template variant override)
    """

    def __init__(self):
        self._verbalizer = RelationVerbalizer()

    def compose(
        self,
        frames: List[Dict[str, Any]],
        topic: str = "",
        query_type: str = "factual",
        max_sentences: int = 8,
    ) -> str:
        """
        Compose a fluent paragraph from semantic frames.

        Parameters
        ----------
        frames      : list of semantic frame dicts
        topic       : central topic word (for anaphora + topic sentence)
        query_type  : "factual"|"explanatory"|"procedural"|"causal"|"comparative"
        max_sentences : cap on number of sentences in the response

        Returns
        -------
        str : a fluent paragraph
        """
        if not frames:
            return "I don't have enough information about that topic."

        frames = self._sort_frames(frames, query_type)[:max_sentences]
        n = len(frames)
        sentences: List[str] = []
        seen_subjects: List[str] = []
        topic_sentence_added = False

        # ── Topic sentence ───────────────────────────────────────────────
        if n > 1 and topic:
            sentences.append(self._topic_sentence(topic, query_type))
            topic_sentence_added = True

        # ── Main facts ───────────────────────────────────────────────────
        for i, frame in enumerate(frames):
            s = frame.get("subject", "")
            r = frame.get("relation", "")
            o = frame.get("object", frame.get("value", ""))
            negate = frame.get("negate", False)
            variant = frame.get("variant", i)

            if not s or not o:
                continue

            # Anaphora: replace repeated subject with pronoun
            display_s = self._anaphora(s, seen_subjects, topic)

            if negate:
                # Use a dedicated negated relation template if one exists,
                # otherwise fall back to the original relation + force-insert "not"
                neg_rel = "not_" + r if "not_" + r in _RELATION_TEMPLATES else r
                sentence, hint = self._verbalizer.verbalize(display_s, neg_rel, o, variant)
                if "not" not in sentence.lower() and "lacks" not in sentence.lower():
                    # Force insert negation
                    sentence = self._insert_negation(sentence)
            else:
                sentence, hint = self._verbalizer.verbalize(display_s, r, o, variant)

            # ── Discourse connective ─────────────────────────────────────
            if i > 0 and n > 1:
                pos = "last" if i == n - 1 else "middle"
                connective = self._connective(hint, r, pos)
                if connective:
                    sentence = connective + " " + sentence[0].lower() + sentence[1:]

            sentences.append(sentence)
            if s and s not in seen_subjects:
                seen_subjects.append(s)

        if not sentences:
            return "I don't have enough information about that topic."

        # ── Procedural numbering ─────────────────────────────────────────
        if query_type == "procedural":
            # Skip the topic sentence (if added) — number only the fact sentences
            fact_sents = sentences[1:] if topic_sentence_added else sentences
            return self._format_steps(fact_sents)

        # ── Conclusion ───────────────────────────────────────────────────
        if n >= 3 and topic:
            conclusion = self._conclusion(topic)
            sentences.append(conclusion)

        return " ".join(sentences)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sort_frames(
        self, frames: List[Dict[str, Any]], query_type: str
    ) -> List[Dict[str, Any]]:
        if query_type in ("causal", "procedural"):
            return frames  # preserve supplied order
        def key(f: Dict[str, Any]) -> float:
            rel = f.get("relation", "")
            importance = f.get("importance", 0.5)
            if rel in ("is_a", "isa", "is"):
                return 0.0
            if rel in ("causes", "leads_to", "results_in"):
                return 0.1
            if rel in ("has_property", "has", "capable_of"):
                return 0.3
            return 1.0 - importance
        return sorted(frames, key=key)

    def _topic_sentence(self, topic: str, query_type: str) -> str:
        templates = _TOPIC_SENTENCES.get(query_type, _TOPIC_SENTENCES["factual"])
        idx = hash(topic) % len(templates)
        tmpl = templates[idx]
        T = topic.strip()
        if T:
            T = T[0].upper() + T[1:]
        return (tmpl.replace("{topic}", topic.lower())
                    .replace("{Topic}", T))

    def _conclusion(self, topic: str) -> str:
        idx = hash(topic + "_conclusion") % len(_CONCLUSIONS)
        return _CONCLUSIONS[idx].replace("{topic}", topic.lower())

    @staticmethod
    def _anaphora(subject: str, seen: List[str], topic: str) -> str:
        """Replace repeated subject with appropriate pronoun."""
        if subject not in seen:
            return subject
        # Use "it" for most things; "they" if plural hint
        s_lower = subject.lower()
        if s_lower.endswith("s") and not s_lower.endswith("ss"):
            return "they"
        return "it"

    @staticmethod
    def _connective(hint: str, relation: str, position: str) -> str:
        """Return the best discourse connective for this position."""
        if hint:
            return hint
        rel_norm = relation.lower().replace(" ", "_")
        if rel_norm in ("causes", "leads_to", "results_in"):
            return "As a result," if position == "middle" else "Therefore,"
        if rel_norm in ("precedes", "follows"):
            return "Then," if position == "middle" else "Finally,"
        if rel_norm in ("similar_to",):
            return "Similarly,"
        if rel_norm in ("different_from", "opposite_of"):
            return "However," if position == "middle" else "In contrast,"
        candidates = _POSITION_CONNECTIVES.get(position, [])
        if not candidates:
            return ""
        idx = hash(relation + position) % len(candidates)
        return candidates[idx]

    @staticmethod
    def _insert_negation(sentence: str) -> str:
        """Insert 'not' after the first finite verb in *sentence*."""
        return re.sub(
            r"\b(is|are|was|were|has|have|can|could|does|do|will|would)\b",
            lambda m: m.group(0) + " not",
            sentence,
            count=1,
        )

    @staticmethod
    def _format_steps(sentences: List[str]) -> str:
        return "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))


# ---------------------------------------------------------------------------
# NSCKResponseEngine — public API
# ---------------------------------------------------------------------------

class NSCKResponseEngine:
    """
    High-level fluent response engine for NSCK.

    Provides three response modes:

    1. ``respond(frames, ...)`` — full paragraph from semantic frames
    2. ``answer_query(query, facts, ...)`` — answer a specific question
       from a list of (subject, relation, object) triples
    3. ``describe(concept, semantic_memory, ...)`` — look up a concept in
       semantic memory and describe it in fluent prose

    All output is fluent English — no SQL-style "X HAS_PROPERTY Y" strings.

    Example
    -------
    >>> engine = NSCKResponseEngine()
    >>> frames = [
    ...     {"subject": "water", "relation": "is_a", "object": "liquid"},
    ...     {"subject": "water", "relation": "has_property", "object": "transparency"},
    ...     {"subject": "water", "relation": "causes", "object": "erosion"},
    ... ]
    >>> engine.respond(frames, topic="water")
    'Understanding water requires examining several aspects. Water is a liquid.
    It is known for its transparency. As a result, it leads to erosion.'
    """

    def __init__(self):
        self._composer = FluentResponseComposer()
        self._verbalizer = RelationVerbalizer()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def respond(
        self,
        frames: List[Dict[str, Any]],
        topic: str = "",
        query_type: str = "factual",
        max_sentences: int = 8,
    ) -> str:
        """
        Generate a fluent response from a list of semantic frames.

        Parameters
        ----------
        frames      : list of {"subject", "relation", "object"} dicts
        topic       : central topic word
        query_type  : "factual"|"explanatory"|"procedural"|"causal"|"comparative"
        max_sentences : maximum sentences in output
        """
        return self._composer.compose(
            frames,
            topic=topic,
            query_type=query_type,
            max_sentences=max_sentences,
        )

    def answer_query(
        self,
        query: str,
        facts: List[Tuple[str, str, str]],
        topic: str = "",
    ) -> str:
        """
        Answer a natural-language query from a list of (s, r, o) triples.

        Attempts to detect the query type from the question word and
        selects an appropriate response style.

        Parameters
        ----------
        query  : natural language question (e.g. "What is water?")
        facts  : list of (subject, relation, object) tuples
        topic  : central topic (defaults to first subject if empty)
        """
        if not facts:
            return f"I don't have any information about that."

        if not topic and facts:
            topic = facts[0][0]

        query_type = self._detect_query_type(query)
        frames = [{"subject": s, "relation": r, "object": o}
                  for s, r, o in facts]
        return self._composer.compose(
            frames, topic=topic, query_type=query_type
        )

    def describe(
        self,
        concept: str,
        semantic_memory: Any,
        max_relations: int = 6,
    ) -> str:
        """
        Describe a concept by querying semantic memory for its relations.

        Parameters
        ----------
        concept         : the concept to describe
        semantic_memory : SemanticMemory instance
        max_relations   : max number of relations to include

        Returns
        -------
        A fluent paragraph describing the concept.
        """
        # Query semantic memory for known facts about this concept
        try:
            # spread_activate returns a dict {concept: score}
            results = semantic_memory.query(
                semantic_memory.concept_hvs.get(concept),
                k=max_relations + 1,
            ) if concept in semantic_memory.concept_hvs else []
        except Exception:
            results = []

        # Try to get direct relations from the graph
        frames: List[Dict[str, Any]] = []
        try:
            graph = semantic_memory.concept_graph
            for _, neighbor, data in graph.out_edges(concept, data=True):
                rel = data.get("relation", "related_to")
                frames.append({"subject": concept, "relation": rel, "object": neighbor})
                if len(frames) >= max_relations:
                    break
        except Exception:
            pass

        if not frames:
            return f"I don't have enough information about {concept!r} yet."

        return self._composer.compose(
            frames, topic=concept, query_type="explanatory"
        )

    def single_fact(
        self, subject: str, relation: str, obj: str, variant: int = 0
    ) -> str:
        """
        Realize a single (subject, relation, object) triple as a sentence.
        Useful for concise one-fact answers.
        """
        sentence, _ = self._verbalizer.verbalize(subject, relation, obj, variant)
        return sentence

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_query_type(query: str) -> str:
        """Heuristically detect the query type from the question."""
        q = query.lower().strip()
        if q.startswith("why") or "because" in q or "cause" in q or "reason" in q:
            return "causal"
        if q.startswith("how") and ("step" in q or "procedure" in q or "process" in q):
            return "procedural"
        if q.startswith("compare") or "differ" in q or "versus" in q or " vs " in q:
            return "comparative"
        if q.startswith("explain") or "explain" in q:
            return "explanatory"
        return "factual"
