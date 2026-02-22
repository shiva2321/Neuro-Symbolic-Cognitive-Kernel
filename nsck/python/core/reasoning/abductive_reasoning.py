"""
NSCK Abductive Reasoning Module
================================
Implements **Inference to the Best Explanation (IBE)**, the logical form of
abduction first formalised by Charles Sanders Peirce (1839-1914):

    Deduction:  All A are B; X is A  →  X is B.           (forward, certain)
    Induction:  X is A and B, many times  →  All A are B.  (forward, probable)
    Abduction:  X is B; All A are B  →  X is A.            (backward, best guess)

Practical role in NSCK
-----------------------
When the agent observes an *effect* it did not predict, it needs to explain
**why** that effect occurred.  The agent searches the causal knowledge graph
*backwards* from the observation to find the most parsimonious, coherent
set of causes that, if true, would *explain* the observation.

Scoring function for candidate explanations
--------------------------------------------
The best explanation maximises a composite score:

    score(H) = w_cov * coverage(H, O)
             + w_sim * prior_similarity(H)
             + w_prs * parsimony(H)
             - w_inc * incoherence(H)

where:
  * coverage(H, O)    = fraction of observations explained by hypothesis H
  * prior_similarity  = how well H's HV matches existing belief state
  * parsimony(H)      = 1 / (1 + len(H))  ← Occam's razor
  * incoherence(H)    = number of known contradictions with H

Integration with NSCK
---------------------
* ``CausalGraph`` (causal_reasoning.py) supplies the backward-search graph.
* ``SemanticMemory`` supplies prior similarity and belief coherence.
* ``CognitiveState.trace`` gets a new ``abductive_explanation`` field.
* ``ExplanationGenerator`` can use IBE results to provide natural-language
  justifications via ``explain_observation()``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import python.core.vsa.hypervec_shim as hypervec_rs

logger = logging.getLogger("nsck.abductive_reasoning")


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """
    A candidate explanation produced by abductive reasoning.

    ``causes`` is the ordered causal chain from root cause to observation.
    ``score``  is the IBE composite score (higher = better explanation).
    """
    causes: List[str]           # causal chain, e.g. ["rain", "wet_road"]
    score: float
    coverage: float             # fraction of observations explained
    parsimony: float
    coherence: float            # 1 - incoherence
    prior_sim: float            # similarity to current belief state

    def to_dict(self) -> Dict:
        return {
            "causes": self.causes,
            "score": round(self.score, 4),
            "coverage": round(self.coverage, 4),
            "parsimony": round(self.parsimony, 4),
            "coherence": round(self.coherence, 4),
            "prior_sim": round(self.prior_sim, 4),
        }

    def __str__(self) -> str:
        chain = " → ".join(self.causes) if self.causes else "(empty)"
        return (
            f"Hypothesis({chain}, score={self.score:.3f}, "
            f"cov={self.coverage:.2f}, pars={self.parsimony:.2f})"
        )


@dataclass
class AbductionResult:
    """
    Result of an abductive query.

    ``best`` is the highest-scoring Hypothesis or None if no explanation
    was found.  ``all_hypotheses`` contains the top-K ranked alternatives.
    """
    observation: str
    best: Optional[Hypothesis]
    all_hypotheses: List[Hypothesis]
    search_depth: int
    candidates_examined: int


# ---------------------------------------------------------------------------
# AbductiveReasoner
# ---------------------------------------------------------------------------

class AbductiveReasoner:
    """
    Inference-to-Best-Explanation engine for NSCK.

    Performs backward chaining over a ``CausalGraph`` (or any graph with a
    ``get_causes(effect)`` method returning ``[(cause, strength)]``) to find
    causal chains that best explain an observed concept.

    Usage::

        from python.core.reasoning.abductive_reasoning import AbductiveReasoner
        from python.core.reasoning.causal_reasoning import CausalGraph

        cg = CausalGraph()
        cg.add_causes("rain",       "wet_road",  strength=0.9)
        cg.add_causes("wet_road",   "accident",  strength=0.7)
        cg.add_causes("ice",        "accident",  strength=0.8)

        ar = AbductiveReasoner(causal_graph=cg)
        result = ar.explain("accident")
        print(result.best)
        # Hypothesis(rain → wet_road → accident, score=0.712, ...)
    """

    def __init__(
        self,
        causal_graph=None,             # CausalGraph instance
        semantic_memory=None,          # SemanticMemory (optional, for prior_sim)
        max_depth: int = 4,
        top_k: int = 5,
        w_coverage: float = 0.4,
        w_parsimony: float = 0.25,
        w_prior: float = 0.2,
        w_coherence: float = 0.15,
    ):
        """
        Args:
            causal_graph:   Source of causal knowledge.
            semantic_memory: Optional SM for belief coherence scoring.
            max_depth:      Maximum backward-chain depth (Occam guard).
            top_k:          Return top-K hypotheses.
            w_coverage:     Weight for coverage term in score.
            w_parsimony:    Weight for parsimony (shorter = better).
            w_prior:        Weight for prior belief alignment.
            w_coherence:    Weight for incoherence penalty.
        """
        self.causal_graph = causal_graph
        self.semantic_memory = semantic_memory
        self.max_depth = max_depth
        self.top_k = top_k
        self.w_coverage = w_coverage
        self.w_parsimony = w_parsimony
        self.w_prior = w_prior
        self.w_coherence = w_coherence

        logger.info(
            "[AbductiveReasoner] init: max_depth=%d, top_k=%d", max_depth, top_k
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        observation: str,
        context_hv: Optional[Any] = None,
        known_observations: Optional[List[str]] = None,
    ) -> AbductionResult:
        """
        Find the best causal explanation(s) for ``observation``.

        Args:
            observation:        The concept / effect to explain.
            context_hv:         Optional HV of current context (for prior_sim).
            known_observations: Other observations that must also be covered
                                (conjunctive abduction).

        Returns:
            An ``AbductionResult`` with ranked hypotheses.
        """
        if known_observations is None:
            known_observations = [observation]
        elif observation not in known_observations:
            known_observations = [observation] + list(known_observations)

        # Backward-chain search from observation
        raw_chains: List[List[str]] = []
        candidates_examined = [0]
        self._backward_chain(
            observation, [observation], raw_chains, set(), candidates_examined
        )

        if not raw_chains:
            logger.debug("[ABD] no chains found for '%s'", observation)
            return AbductionResult(
                observation=observation,
                best=None,
                all_hypotheses=[],
                search_depth=self.max_depth,
                candidates_examined=candidates_examined[0],
            )

        # Score every chain
        hypotheses: List[Hypothesis] = []
        for chain in raw_chains:
            h = self._score_hypothesis(chain, known_observations, context_hv)
            hypotheses.append(h)

        # Rank by score (descending)
        hypotheses.sort(key=lambda h: h.score, reverse=True)
        top = hypotheses[: self.top_k]

        logger.debug(
            "[ABD] '%s': %d chains → best=%s",
            observation,
            len(raw_chains),
            str(top[0]) if top else "none",
        )

        return AbductionResult(
            observation=observation,
            best=top[0] if top else None,
            all_hypotheses=top,
            search_depth=self.max_depth,
            candidates_examined=candidates_examined[0],
        )

    def explain_text(self, observation: str) -> str:
        """
        Return a single human-readable explanation string for ``observation``.
        """
        result = self.explain(observation)
        if result.best is None:
            return f"No causal explanation found for '{observation}'."
        h = result.best
        chain_str = " → ".join(h.causes)
        return (
            f"Best explanation for '{observation}': {chain_str} "
            f"(score={h.score:.2f}, parsimony={h.parsimony:.2f})"
        )

    def explain_multiple(
        self, observations: List[str], context_hv: Optional[Any] = None
    ) -> AbductionResult:
        """
        Find the single hypothesis that best explains ALL observations jointly
        (joint / conjunctive abduction).
        """
        if not observations:
            raise ValueError("observations list must not be empty")

        # Use the first observation as the primary query
        primary = observations[0]
        return self.explain(primary, context_hv, known_observations=observations)

    # ------------------------------------------------------------------
    # Internal backward-chain search
    # ------------------------------------------------------------------

    def _backward_chain(
        self,
        current: str,
        chain: List[str],
        results: List[List[str]],
        visited: Set[str],
        counter: List[int],
    ) -> None:
        """
        Recursive DFS backward over causal graph.

        Stops when:
        - No causes found for ``current`` (root cause reached) → add chain
          only if it has at least 2 nodes (a root cause + the observation).
          Single-node chains (the observation alone) are excluded because
          they do not constitute an explanation — an explanation requires at
          least one causal predecessor.
        - Max depth exceeded.
        - Cycle detected.
        """
        counter[0] += 1
        causes = self._get_causes(current)

        if not causes or len(chain) > self.max_depth:
            # Only record chains that contain at least one causal predecessor
            # (chain has length >= 2 after reversal: [root, ..., observation])
            if len(chain) >= 2:
                results.append(list(reversed(chain)))
            return

        for cause, _strength in causes:
            if cause in visited:
                continue
            visited.add(cause)
            self._backward_chain(
                cause, chain + [cause], results, visited, counter
            )
            visited.discard(cause)

    def _get_causes(self, effect: str) -> List[Tuple[str, float]]:
        """
        Query causal graph for causes of ``effect``.
        Returns list of (cause, strength) tuples.
        """
        if self.causal_graph is None:
            return []
        # Preferred: get_immediate_causes() returns list of cause strings
        try:
            causes = self.causal_graph.get_immediate_causes(effect)
            if isinstance(causes, list) and causes:
                # Each element may be a string or a tuple
                result = []
                for c in causes:
                    if isinstance(c, tuple):
                        result.append(c)
                    else:
                        result.append((str(c), 1.0))
                return result
        except AttributeError:
            pass
        # Fallback: backward dict: effect → [(cause, CausalLink), ...]
        try:
            backward = getattr(self.causal_graph, "backward", None)
            if backward is not None and effect in backward:
                return [
                    (cause, link.strength if hasattr(link, "strength") else 1.0)
                    for cause, link in backward[effect]
                ]
        except Exception:
            pass
        # Last resort: networkx-style predecessor traversal
        try:
            g = getattr(self.causal_graph, "_graph", None) or getattr(
                self.causal_graph, "graph", None
            )
            if g is not None and effect in g:
                return [
                    (pred, g[pred][effect].get("strength", 1.0))
                    for pred in g.predecessors(effect)
                ]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # Hypothesis scoring (IBE composite)
    # ------------------------------------------------------------------

    def _score_hypothesis(
        self,
        chain: List[str],
        observations: List[str],
        context_hv: Optional[Any],
    ) -> Hypothesis:
        """Compute the IBE composite score for a causal chain."""
        # Coverage: how many observations appear in the chain?
        obs_set = set(obs.lower() for obs in observations)
        chain_set = set(c.lower() for c in chain)
        covered = len(obs_set & chain_set)
        coverage = covered / max(1, len(observations))

        # Parsimony: shorter chain is simpler
        parsimony = 1.0 / (1.0 + max(0, len(chain) - 1))

        # Prior similarity: how close is the root cause to current beliefs?
        prior_sim = self._prior_similarity(chain[0] if chain else "", context_hv)

        # Coherence: 1 - incoherence (contradictions with known facts)
        coherence = self._coherence(chain)

        # Composite score
        score = (
            self.w_coverage * coverage
            + self.w_parsimony * parsimony
            + self.w_prior * prior_sim
            + self.w_coherence * coherence
        )

        return Hypothesis(
            causes=chain,
            score=score,
            coverage=coverage,
            parsimony=parsimony,
            coherence=coherence,
            prior_sim=prior_sim,
        )

    def _prior_similarity(self, concept: str, context_hv: Optional[Any]) -> float:
        """Similarity of concept to current context belief state."""
        if context_hv is None:
            # Check SemanticMemory for concept existence (0.5 if known, 0.2 if not)
            if self.semantic_memory is not None:
                try:
                    known = self.semantic_memory.concept_hvs.get(concept)
                    return 0.6 if known is not None else 0.2
                except Exception:
                    pass
            return 0.5

        concept_hv = hypervec_rs.HyperVector(hash(concept) & 0x7FFFFFFF)
        try:
            return float(context_hv.similarity(concept_hv))
        except Exception:
            try:
                return float(concept_hv.similarity(context_hv))
            except Exception:
                return 0.5

    def _coherence(self, chain: List[str]) -> float:
        """
        Return coherence score (0–1) based on how many chain links exist
        in SemanticMemory.  More grounded links → higher coherence.
        """
        if self.semantic_memory is None or not chain:
            return 0.8     # optimistic default when no SM available
        known = 0
        for concept in chain:
            if self.semantic_memory.concept_hvs.get(concept) is not None:
                known += 1
        return known / len(chain)
