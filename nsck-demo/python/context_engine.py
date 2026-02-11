"""
NSCK Context Engine
===================
Contextual disambiguation and situated understanding.

Handles the problem of polysemy and context-dependent meaning:
- "red" in traffic context → danger
- "red" in flower context → beauty/romance
- "red" in health context → blood/emergency

Uses VSA role-filler bindings and spreading activation in semantic memory
to disambiguate concepts based on surrounding context.
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np

try:
    import hypervec_shim as hypervec_rs
except ImportError:
    from hypervec_py import HyperVector as _HV

    class _Shim:
        HyperVector = _HV
    hypervec_rs = _Shim()


@dataclass
class ContextFrame:
    """A snapshot of the current interpretive context."""
    domain: str                          # e.g. "traffic", "garden", "health"
    active_concepts: List[str]           # concepts currently in focus
    environment_cues: Dict[str, Any]     # raw sensory/environmental cues
    emotional_state: str = "neutral"     # current emotion
    timestamp: float = 0.0


@dataclass
class DisambiguatedMeaning:
    """Result of contextual disambiguation."""
    concept: str                 # original concept (e.g., "red")
    meaning: str                 # resolved meaning (e.g., "danger")
    context_domain: str          # which domain was used
    confidence: float            # how confident the disambiguation is
    alternative_meanings: List[Tuple[str, float]]  # other possible meanings
    explanation: str             # human-readable explanation


class ContextEngine:
    """
    Contextual disambiguation engine using VSA and semantic associations.

    Core idea: Every concept is represented as a base HV. Its *meaning* in a
    given context is the bundle of the concept HV with a context-role HV.
    By comparing the contextualised HV against known meaning prototypes we
    can select the most appropriate interpretation.
    """

    def __init__(self, semantic_memory=None):
        self.semantic_memory = semantic_memory

        # Concept → { context_domain → meaning_label }
        self.context_meanings: Dict[str, Dict[str, str]] = defaultdict(dict)

        # Concept → { meaning_label → prototype HV }
        self.meaning_prototypes: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Context domain → role HV (deterministic seed per domain)
        self.domain_roles: Dict[str, Any] = {}

        # Association strength: (concept, meaning) → float
        self.association_strength: Dict[Tuple[str, str], float] = defaultdict(float)

        # Observation counts for learning
        self._obs_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)

        # Default / fallback meanings
        self.default_meanings: Dict[str, str] = {}

        # Bootstrap common-sense associations
        self._bootstrap_common_sense()

    def reset(self):
        """Clear all contextual mappings and re-bootstrap."""
        self.context_meanings = defaultdict(dict)
        self.meaning_prototypes = defaultdict(dict)
        self.association_strength = defaultdict(float)
        self._obs_counts = defaultdict(int)
        self.default_meanings = {}
        self._bootstrap_common_sense()
        print("[CONTEXT] Engine reset complete.")

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def _bootstrap_common_sense(self):
        """Seed the engine with basic common-sense contextual meanings."""
        associations = {
            "red": {
                "traffic": "danger",
                "garden": "beauty",
                "health": "emergency",
                "emotion": "passion",
                "politics": "ideology",
                "default": "danger",
            },
            "blue": {
                "emotion": "sadness",
                "sky": "calm",
                "health": "cold",
                "default": "calm",
            },
            "green": {
                "traffic": "safe",
                "environment": "nature",
                "health": "healthy",
                "finance": "profit",
                "default": "safe",
            },
            "hot": {
                "temperature": "high_temperature",
                "food": "spicy",
                "popularity": "trending",
                "default": "high_temperature",
            },
            "cold": {
                "temperature": "low_temperature",
                "emotion": "unfriendly",
                "health": "illness",
                "default": "low_temperature",
            },
            "light": {
                "physics": "illumination",
                "weight": "not_heavy",
                "color": "pale",
                "default": "illumination",
            },
            "run": {
                "exercise": "physical_movement",
                "software": "execute",
                "politics": "campaign",
                "default": "physical_movement",
            },
            "bank": {
                "finance": "financial_institution",
                "river": "riverbank",
                "aviation": "aircraft_turn",
                "default": "financial_institution",
            },
        }

        for concept, domains in associations.items():
            default = domains.pop("default", None)
            if default:
                self.default_meanings[concept] = default
            for domain, meaning in domains.items():
                self.register_meaning(concept, domain, meaning)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _get_domain_role(self, domain: str) -> Any:
        """Get or create a deterministic role HV for a domain."""
        if domain not in self.domain_roles:
            seed = hash(f"domain_role_{domain}") % (2**32)
            self.domain_roles[domain] = hypervec_rs.HyperVector(seed)
        return self.domain_roles[domain]

    def register_meaning(
        self,
        concept: str,
        context_domain: str,
        meaning: str,
        strength: float = 1.0,
    ):
        """
        Register that *concept* means *meaning* when interpreted
        within *context_domain*.
        """
        self.context_meanings[concept][context_domain] = meaning
        self.association_strength[(concept, meaning)] = strength

        # Build a prototype HV = concept_hv XOR domain_role
        concept_seed = hash(f"concept_{concept}") % (2**32)
        concept_hv = hypervec_rs.HyperVector(concept_seed)
        domain_role = self._get_domain_role(context_domain)
        meaning_seed = hash(f"meaning_{meaning}") % (2**32)
        meaning_hv = hypervec_rs.HyperVector(meaning_seed)

        # Prototype = bundle(concept XOR domain_role, meaning_hv)
        bound = concept_hv.xor(domain_role)
        prototype = bound.bundle(meaning_hv)
        self.meaning_prototypes[concept][meaning] = prototype

    # ------------------------------------------------------------------
    # Disambiguation
    # ------------------------------------------------------------------

    def disambiguate(
        self,
        concept: str,
        context: ContextFrame,
    ) -> DisambiguatedMeaning:
        """
        Resolve the meaning of *concept* given the current *context*.

        Strategy (ordered by priority):
        1. Direct domain match — if context.domain is registered for the concept.
        2. Cue overlap — score each known domain by how many active_concepts
           overlap with that domain's typical cues.
        3. Semantic proximity — if semantic_memory is available, use spreading
           activation from active_concepts to find the closest meaning.
        4. Default meaning.
        """
        known_domains = self.context_meanings.get(concept, {})

        if not known_domains:
            return DisambiguatedMeaning(
                concept=concept,
                meaning=concept,  # identity — no disambiguation data
                context_domain=context.domain,
                confidence=0.3,
                alternative_meanings=[],
                explanation=f"No contextual meanings registered for '{concept}'.",
            )

        # --- Strategy 1: Direct domain match ---
        if context.domain in known_domains:
            meaning = known_domains[context.domain]
            alts = [
                (m, self.association_strength.get((concept, m), 0.5))
                for d, m in known_domains.items()
                if d != context.domain
            ]
            return DisambiguatedMeaning(
                concept=concept,
                meaning=meaning,
                context_domain=context.domain,
                confidence=0.9,
                alternative_meanings=alts,
                explanation=(
                    f"'{concept}' in {context.domain} context → '{meaning}' "
                    f"(direct domain match)."
                ),
            )

        # --- Strategy 2: Cue overlap scoring ---
        domain_scores: Dict[str, float] = defaultdict(float)
        active_set = set(c.lower() for c in context.active_concepts)
        env_values = set(str(v).lower() for v in context.environment_cues.values())
        combined_cues = active_set | env_values

        for domain in known_domains:
            # Score by keyword overlap between domain name and context cues
            domain_words = set(domain.lower().split("_"))
            overlap = len(combined_cues & domain_words)
            domain_scores[domain] += overlap * 0.3

            # Boost if any active concept contains the domain name
            for cue in combined_cues:
                if domain.lower() in cue:
                    domain_scores[domain] += 0.4

        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0:
                meaning = known_domains[best_domain]
                conf = min(0.85, 0.5 + domain_scores[best_domain])
                alts = [
                    (known_domains[d], domain_scores[d])
                    for d in known_domains
                    if d != best_domain
                ]
                return DisambiguatedMeaning(
                    concept=concept,
                    meaning=meaning,
                    context_domain=best_domain,
                    confidence=conf,
                    alternative_meanings=alts,
                    explanation=(
                        f"'{concept}' matched domain '{best_domain}' via context "
                        f"cues {combined_cues & set(best_domain.lower().split('_'))} "
                        f"→ '{meaning}'."
                    ),
                )

        # --- Strategy 3: Semantic proximity (if available) ---
        if self.semantic_memory and context.active_concepts:
            activation = self.semantic_memory.spread_activation(
                context.active_concepts, steps=2, decay=0.6
            )
            best_meaning = None
            best_score = -1.0
            for domain, meaning in known_domains.items():
                score = activation.get(meaning, 0.0) + activation.get(domain, 0.0)
                if score > best_score:
                    best_score = score
                    best_meaning = meaning
                    best_domain_sem = domain

            if best_meaning and best_score > 0:
                return DisambiguatedMeaning(
                    concept=concept,
                    meaning=best_meaning,
                    context_domain=best_domain_sem,
                    confidence=min(0.8, 0.4 + best_score),
                    alternative_meanings=[
                        (m, activation.get(m, 0.0))
                        for d, m in known_domains.items()
                        if m != best_meaning
                    ],
                    explanation=(
                        f"'{concept}' resolved via semantic activation: "
                        f"'{best_meaning}' (score={best_score:.2f})."
                    ),
                )

        # --- Strategy 4: Default ---
        default = self.default_meanings.get(concept, list(known_domains.values())[0])
        return DisambiguatedMeaning(
            concept=concept,
            meaning=default,
            context_domain="default",
            confidence=0.4,
            alternative_meanings=[
                (m, 0.3) for m in known_domains.values() if m != default
            ],
            explanation=f"'{concept}' → default meaning '{default}' (no context match).",
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn_from_feedback(
        self,
        concept: str,
        correct_meaning: str,
        context: ContextFrame,
    ):
        """
        Update associations after receiving feedback on a disambiguation.
        Strengthens the link between the concept, domain, and meaning.
        """
        domain = context.domain
        self._obs_counts[(concept, domain, correct_meaning)] += 1
        count = self._obs_counts[(concept, domain, correct_meaning)]

        # Register or strengthen
        self.register_meaning(concept, domain, correct_meaning, strength=min(2.0, 0.5 + 0.1 * count))

    def infer_context_domain(
        self,
        cues: List[str],
        environment: Dict[str, Any],
    ) -> str:
        """
        Infer the most likely context domain from available cues.
        Uses frequency of domain keywords in the cue set.
        """
        domain_votes: Dict[str, float] = defaultdict(float)
        all_tokens = set(c.lower() for c in cues)
        all_tokens.update(str(v).lower() for v in environment.values())

        for concept, domains in self.context_meanings.items():
            for domain in domains:
                for token in all_tokens:
                    if domain.lower() in token or token in domain.lower():
                        domain_votes[domain] += 1.0

        if domain_votes:
            return max(domain_votes, key=domain_votes.get)
        return "general"

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def disambiguate_all(
        self,
        concepts: List[str],
        context: ContextFrame,
    ) -> List[DisambiguatedMeaning]:
        """Disambiguate a list of concepts in the same context."""
        return [self.disambiguate(c, context) for c in concepts]

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine statistics."""
        total_concepts = len(self.context_meanings)
        total_meanings = sum(len(d) for d in self.context_meanings.values())
        return {
            "registered_concepts": total_concepts,
            "total_meanings": total_meanings,
            "domains": list(self.domain_roles.keys()),
            "observations": sum(self._obs_counts.values()),
        }
