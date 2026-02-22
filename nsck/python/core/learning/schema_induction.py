"""
NSCK Schema Induction Module
==============================
Generalizes concrete episodic experiences into abstract, reusable schemas.

Cognitive science foundation
------------------------------
Piaget (1952) described two complementary mechanisms:
  * **Assimilation** – fitting new experience into an existing schema.
  * **Accommodation** – updating a schema when the new experience cannot fit.

Bartlett (1932) showed that memory is reconstructive: people remember the
*gist* (schema) not the verbatim episode, filling gaps with prior knowledge.

In NSCK terms:
  * An **episode** is a set of (subject, relation, object) triples with a
    situation HyperVector and a task tag.
  * A **schema** is the prototype of a cluster of similar episodes: the most
    representative (subject_type, relation, object_type) pattern plus a
    centroid HV and a confidence score.

Implementation strategy
------------------------
1. Collect episodes that share the **same relation** (same edge type in the
   causal / semantic graph) and whose situation HVs are mutually similar
   above ``similarity_threshold``.
2. Compute the centroid HV (bundle all situational HVs, normalize) to form
   the schema's representative vector.
3. Infer abstract slot labels via the most-common word or a placeholder such
   as ``?X`` / ``?Y`` when diversity is too high to pick one word.
4. Store the schema in a ``SchemaLibrary`` with usage statistics.
5. During recall, any new episode can be **assimilated** into the closest
   matching schema (if similarity > threshold) or used to induce a new one.

Why it helps
------------
- Enables **systematic generalization**: once "X causes Y" is learned for
  specific (rain, flooding), the agent can apply the causal schema to
  (stress, illness) without re-learning causation from scratch.
- Reduces memory footprint: 1 schema replaces N near-identical episodes.
- Improves reasoning speed: schema look-up is O(S) where S << N episodes.
- Provides human-readable "why" explanations: the agent can say
  "this follows the CAUSATION schema".
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import python.core.vsa.hypervec_shim as hypervec_rs

logger = logging.getLogger("nsck.schema_induction")


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """A single learned fact / experience triple."""
    subject: str
    relation: str
    object: str
    situation_hv: Optional[Any] = None   # hypervec_rs.HyperVector
    task_tag: str = "default"
    confidence: float = 1.0


@dataclass
class Schema:
    """
    An abstract template induced from a cluster of similar episodes.

    Slots use ``?X`` / ``?Y`` when the filler diversity across the cluster
    is too high to commit to a single word.
    """
    schema_id: str
    relation: str
    subject_slot: str    # concrete word or "?SUBJECT"
    object_slot: str     # concrete word or "?OBJECT"
    centroid_hv: Optional[Any] = None
    support: int = 0            # number of episodes that match
    confidence: float = 1.0
    exemplars: List[Episode] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "schema_id": self.schema_id,
            "relation": self.relation,
            "subject_slot": self.subject_slot,
            "object_slot": self.object_slot,
            "support": self.support,
            "confidence": self.confidence,
        }


@dataclass
class AssimilationResult:
    """Result of trying to assimilate a new episode into an existing schema."""
    matched: bool
    schema: Optional[Schema] = None
    similarity: float = 0.0
    # 'assimilation' = fit into existing schema
    # 'accommodation' = new schema was created
    process: str = "none"


# ---------------------------------------------------------------------------
# SchemaInducer
# ---------------------------------------------------------------------------

class SchemaInducer:
    """
    Induces abstract schemas from episodic triples using VSA similarity
    clustering and prototype extraction.

    Usage::

        inducer = SchemaInducer(similarity_threshold=0.6)

        # Feed episodes (learned facts)
        ep = Episode("rain", "causes", "flooding", situation_hv=hv)
        inducer.add_episode(ep)

        # After batch learning, induce schemas
        inducer.induce()

        # Find closest schema for a new episode
        result = inducer.assimilate(Episode("stress", "causes", "illness"))
        if result.matched:
            print("Follows schema:", result.schema.schema_id)

        # Retrieve all schemas for a relation
        schemas = inducer.get_schemas_for_relation("causes")
    """

    def __init__(
        self,
        similarity_threshold: float = 0.45,
        min_support: int = 2,
        max_slot_diversity: float = 0.8,
    ):
        """
        Args:
            similarity_threshold: Min HV similarity to add episode to existing cluster.
            min_support: Minimum cluster size to promote to schema.
            max_slot_diversity: Fraction of distinct words allowed before slot becomes ``?X``.
        """
        self.similarity_threshold = similarity_threshold
        self.min_support = min_support
        self.max_slot_diversity = max_slot_diversity

        # Episodes grouped by relation
        self._episodes_by_relation: Dict[str, List[Episode]] = defaultdict(list)

        # Induced schemas library
        self.schemas: Dict[str, Schema] = {}          # schema_id -> Schema
        self._schemas_by_relation: Dict[str, List[Schema]] = defaultdict(list)

        # Running episode counter for deterministic IDs
        self._episode_count: int = 0

        logger.info(
            "[SchemaInducer] initialized (sim_threshold=%.2f, min_support=%d)",
            similarity_threshold,
            min_support,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_episode(self, episode: Episode) -> None:
        """Register a new episode for later schema induction."""
        self._episodes_by_relation[episode.relation].append(episode)
        self._episode_count += 1

    def add_episodes(self, episodes: List[Episode]) -> None:
        """Batch-register multiple episodes."""
        for ep in episodes:
            self.add_episode(ep)

    def induce(self) -> List[Schema]:
        """
        Cluster all buffered episodes by relation and HV similarity,
        then promote clusters with sufficient support to schemas.

        Returns the list of newly created / updated schemas.
        """
        new_schemas: List[Schema] = []
        for relation, episodes in self._episodes_by_relation.items():
            clusters = self._cluster_episodes(episodes)
            for cluster in clusters:
                if len(cluster) < self.min_support:
                    continue
                schema = self._promote_cluster(relation, cluster)
                if schema.schema_id not in self.schemas:
                    self.schemas[schema.schema_id] = schema
                    self._schemas_by_relation[relation].append(schema)
                    new_schemas.append(schema)
                    logger.debug(
                        "[SchemaInducer] new schema '%s': %s %s %s (support=%d)",
                        schema.schema_id,
                        schema.subject_slot,
                        schema.relation,
                        schema.object_slot,
                        schema.support,
                    )
                else:
                    # Update support count
                    existing = self.schemas[schema.schema_id]
                    existing.support = max(existing.support, schema.support)
        return new_schemas

    def assimilate(self, episode: Episode) -> AssimilationResult:
        """
        Try to fit a new episode into an existing schema (assimilation).
        If no schema matches closely enough, create a new single-episode
        seed (accommodation pending more support).

        Returns an ``AssimilationResult``.
        """
        candidates = self._schemas_by_relation.get(episode.relation, [])
        if not candidates:
            # No schemas for this relation yet — buffer for future induction
            self.add_episode(episode)
            return AssimilationResult(matched=False, process="accommodation")

        best_schema: Optional[Schema] = None
        best_sim: float = 0.0

        for schema in candidates:
            sim = self._schema_episode_similarity(schema, episode)
            if sim > best_sim:
                best_sim = sim
                best_schema = schema

        if best_schema is not None and best_sim >= self.similarity_threshold:
            best_schema.support += 1
            best_schema.exemplars.append(episode)
            return AssimilationResult(
                matched=True,
                schema=best_schema,
                similarity=best_sim,
                process="assimilation",
            )

        # Accommodation: buffer the episode for future induction
        self.add_episode(episode)
        return AssimilationResult(matched=False, process="accommodation")

    def get_schemas_for_relation(self, relation: str) -> List[Schema]:
        """Return all schemas whose relation matches."""
        return list(self._schemas_by_relation.get(relation, []))

    def get_all_schemas(self) -> List[Schema]:
        """Return all induced schemas."""
        return list(self.schemas.values())

    def explain_episode(self, episode: Episode) -> str:
        """
        Return a human-readable explanation of which schema an episode
        follows (or 'no matching schema found').
        """
        result = self.assimilate(episode)
        if result.matched and result.schema:
            s = result.schema
            return (
                f"Follows schema '{s.schema_id}': "
                f"{s.subject_slot} {s.relation} {s.object_slot} "
                f"(support={s.support}, confidence={s.confidence:.2f}, "
                f"similarity={result.similarity:.2f})"
            )
        return f"No matching schema for '{episode.subject} {episode.relation} {episode.object}'"

    def summary(self) -> Dict:
        """Return a summary dict of current schema library."""
        return {
            "total_schemas": len(self.schemas),
            "total_episodes": self._episode_count,
            "relations_covered": list(self._schemas_by_relation.keys()),
            "schemas": [s.to_dict() for s in self.schemas.values()],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cluster_episodes(self, episodes: List[Episode]) -> List[List[Episode]]:
        """
        Greedy single-linkage clustering of episodes by HV cosine similarity.
        Episodes without a situation_hv cluster only by exact (relation,
        subject, object) match.
        """
        clusters: List[List[Episode]] = []

        for ep in episodes:
            placed = False
            for cluster in clusters:
                if self._episode_fits_cluster(ep, cluster):
                    cluster.append(ep)
                    placed = True
                    break
            if not placed:
                clusters.append([ep])

        return clusters

    def _episode_fits_cluster(
        self, episode: Episode, cluster: List[Episode]
    ) -> bool:
        """
        An episode fits a cluster if it is sufficiently similar to at least
        one existing cluster member (single-linkage criterion).
        """
        for member in cluster:
            sim = self._episode_similarity(episode, member)
            if sim >= self.similarity_threshold:
                return True
        return False

    def _episode_similarity(self, a: Episode, b: Episode) -> float:
        """
        Compute similarity between two episodes.

        Strategy (in order of preference):
        1. If both have situation HVs → cosine similarity of HVs.
        2. Exact string match on (subject, object) → 1.0.
        3. Levenshtein-inspired token overlap on subject+object words.
        """
        if a.situation_hv is not None and b.situation_hv is not None:
            try:
                return float(a.situation_hv.similarity(b.situation_hv))
            except Exception:
                pass

        # Fallback: token Jaccard over subject+object words, with a base
        # "same-relation" bonus of 0.5 so that episodes already grouped by
        # the same relation can cluster even when filler words differ.
        words_a = set(a.subject.lower().split() + a.object.lower().split())
        words_b = set(b.subject.lower().split() + b.object.lower().split())
        if not words_a and not words_b:
            return 1.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        jaccard = intersection / union if union > 0 else 0.0
        # Base 0.5 reflects same-relation membership; word overlap adds up to 1.0
        return 0.5 + 0.5 * jaccard

    def _schema_episode_similarity(
        self, schema: Schema, episode: Episode
    ) -> float:
        """
        Similarity between a schema (centroid HV) and a new episode.
        """
        if schema.centroid_hv is not None and episode.situation_hv is not None:
            try:
                return float(schema.centroid_hv.similarity(episode.situation_hv))
            except Exception:
                pass

        # Fallback: slot match
        # Base similarity = 0.5 (same relation is guaranteed by the caller;
        # this represents "possibly the same schema with different fillers").
        subj_match = (
            schema.subject_slot == episode.subject
            or schema.subject_slot.startswith("?")
        )
        obj_match = (
            schema.object_slot == episode.object
            or schema.object_slot.startswith("?")
        )
        if subj_match and obj_match:
            return 0.9
        if subj_match or obj_match:
            return 0.7
        return 0.5  # same relation, different fillers — still a schema match

    def _promote_cluster(self, relation: str, cluster: List[Episode]) -> Schema:
        """
        Create a Schema from a cluster of episodes.

        - Centroid HV = bundle of all situation HVs (if available).
        - subject_slot / object_slot = most-common word, or ``?SUBJECT``/``?OBJECT``
          if diversity exceeds ``max_slot_diversity``.
        """
        # --- Centroid HV ---
        hvs = [ep.situation_hv for ep in cluster if ep.situation_hv is not None]
        centroid_hv = None
        if hvs:
            centroid_hv = hvs[0]
            for hv in hvs[1:]:
                try:
                    centroid_hv = centroid_hv.bundle(hv)
                except Exception:
                    pass

        # --- Slot labels ---
        subject_slot = self._best_slot_label(
            [ep.subject for ep in cluster], "?SUBJECT"
        )
        object_slot = self._best_slot_label(
            [ep.object for ep in cluster], "?OBJECT"
        )

        # --- Confidence = mean episode confidence ---
        confidence = float(np.mean([ep.confidence for ep in cluster]))

        schema_id = f"schema_{relation}_{subject_slot}_{object_slot}".replace(
            " ", "_"
        ).lower()

        schema = Schema(
            schema_id=schema_id,
            relation=relation,
            subject_slot=subject_slot,
            object_slot=object_slot,
            centroid_hv=centroid_hv,
            support=len(cluster),
            confidence=confidence,
            exemplars=list(cluster),
        )
        return schema

    def _best_slot_label(self, words: List[str], placeholder: str) -> str:
        """
        Return the most common word if it appears in at least
        (1 - max_slot_diversity) of the list, else return placeholder.
        """
        if not words:
            return placeholder
        from collections import Counter
        counts = Counter(words)
        most_common, freq = counts.most_common(1)[0]
        diversity = 1.0 - (freq / len(words))
        if diversity <= self.max_slot_diversity:
            return most_common
        return placeholder
