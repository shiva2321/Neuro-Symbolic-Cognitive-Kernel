"""
NSCK Semantic Memory Module (Phase 3.1)
=======================================
Structured knowledge base of concepts and relations.
Implements abstracted knowledge (Schemas) and spreading activation.
"""

import logging
import networkx as nx
import numpy as np
import pickle
import os
import heapq
from typing import Dict, List, Any, Optional, Set, Tuple
import python.core.vsa.hypervec_shim as hypervec_rs
from python.core.memory.societal_knowledge_world import SocietalKnowledgeWorld

# V3: optional HNSW index (external hnswlib)
try:
    import hnswlib as _hnswlib
    _HNSWLIB_AVAILABLE = True
except ImportError:
    _hnswlib = None
    _HNSWLIB_AVAILABLE = False

logger = logging.getLogger("nsck.semantic_memory")


# ---------------------------------------------------------------------------
# NSW (Navigable Small World) — pure-Python ANN fallback
# ---------------------------------------------------------------------------

class _NSWIndex:
    """
    Navigable Small World approximate nearest-neighbour index.

    A lightweight greedy graph search ANN that gives sub-linear query time
    at scale (O(log N) with good graph construction) without any external
    dependencies.  Used automatically when ``hnswlib`` is not installed.

    Complexity
    ----------
    * Build: O(N · M · D)  where M = max_connections, D = vector dimension
    * Query: O(log N · ef · D)

    Parameters
    ----------
    M  : int  — max outgoing edges per node (higher = better recall, slower build)
    ef : int  — beam width during search (higher = better recall, slower query)
    """

    def __init__(self, M: int = 16, ef: int = 50) -> None:
        self._M = M
        self._ef = ef
        self._vectors: List[np.ndarray] = []       # float32 L2-normalised
        self._graph: Dict[int, List[int]] = {}      # adjacency list

    def add_item(self, vec: np.ndarray) -> int:
        """Insert a float32 vector. Returns the assigned integer ID."""
        v = vec.astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 1e-8:
            v = v / norm
        idx = len(self._vectors)
        self._vectors.append(v)
        self._graph[idx] = []

        if idx == 0:
            return idx

        # Find M nearest neighbours among existing nodes (greedy search)
        neighbours = self._greedy_search(v, ef=max(self._ef, self._M + 1))
        # Keep top-M
        neighbours = neighbours[: self._M]
        self._graph[idx] = [n for n, _ in neighbours]

        # Reciprocal edges (bidirectional; trim to M if over limit)
        for n, _ in neighbours:
            if len(self._graph[n]) < self._M:
                self._graph[n].append(idx)
            else:
                # Replace the farthest existing neighbour if new one is closer
                farthest_dist = max(
                    1.0 - float(np.dot(self._vectors[n], self._vectors[e]))
                    for e in self._graph[n]
                )
                new_dist = 1.0 - float(np.dot(self._vectors[n], v))
                if new_dist < farthest_dist:
                    # Find and replace
                    farthest = max(
                        self._graph[n],
                        key=lambda e: 1.0 - float(np.dot(self._vectors[n], self._vectors[e]))
                    )
                    self._graph[n].remove(farthest)
                    self._graph[n].append(idx)
        return idx

    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Return the k approximate nearest neighbours as [(id, distance), ...].
        Distance is cosine distance (1 − similarity), ascending.
        """
        if not self._vectors:
            return []
        q = query.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            q = q / norm
        results = self._greedy_search(q, ef=max(self._ef, k))
        return results[:k]

    def _greedy_search(self, query: np.ndarray, ef: int) -> List[Tuple[int, float]]:
        """Greedy beam search from a deterministic entry point (node 0)."""
        if not self._vectors:
            return []

        # Use node 0 as the fixed entry point for determinism and cache locality.
        # A random entry point gives better average recall but high variance at
        # small N — deterministic is more predictable for a unit-tested component.
        entry = 0
        visited: Set[int] = {entry}
        dist0 = 1.0 - float(np.dot(self._vectors[entry], query))
        candidates: List[Tuple[float, int]] = [(dist0, entry)]
        results: List[Tuple[float, int]] = [(dist0, entry)]

        while candidates:
            d, node = heapq.heappop(candidates)
            if results and d > results[-1][0] and len(results) >= ef:
                break
            for neighbour in self._graph.get(node, []):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                nd = 1.0 - float(np.dot(self._vectors[neighbour], query))
                if len(results) < ef or nd < results[-1][0]:
                    heapq.heappush(candidates, (nd, neighbour))
                    results.append((nd, neighbour))
                    results.sort()
                    if len(results) > ef:
                        results.pop()
        return [(idx, d) for d, idx in results]

class SemanticMemory:
    """
    Structured knowledge base of concepts and relations.
    Built from episodic memory consolidation during sleep.
    
    Automatically uses Rust SemanticMemoryConcurrent when available for 10-100x speedup.
    Falls back to Python implementation transparently.
    """
    
    # Default relation weights for spreading activation.
    # Higher weight = stronger propagation through that edge type.
    DEFAULT_RELATION_WEIGHTS: Dict[str, float] = {
        "is_a": 0.9,            # Taxonomic links carry most meaning
        "has_property": 0.7,    # Properties propagate strongly
        "causes": 0.6,          # Causal links
        "leads_to": 0.6,        # Consequential (alias for causes)
        "results_in": 0.6,      # Result of action
        "implies": 0.55,        # Logical implication
        "conditional_on": 0.5,  # V4: conditional dependency
        "part_of": 0.5,         # Mereological
        "similar_to": 0.4,      # Associative
        "semantically_related": 0.35,  # Weakest — co-occurrence based
        # V4: negation (low activation weight — negation dampens spreading)
        "not_is_a": 0.1,
        "not_has_property": 0.1,
        "not_relates_to": 0.1,
        "cannot_do": 0.1,
        "never_does": 0.1,
        "lacks": 0.1,
        # V4: temporal ordering
        "precedes": 0.45,
        "follows": 0.45,
        "since_event": 0.35,
        "until_event": 0.35,
        "triggered_by": 0.5,
        "occurs_during": 0.4,
        # V4: similarity/difference
        "different_from": 0.2,
        "opposite_of": 0.2,
        "associated_with": 0.45,
        "depends_on": 0.5,
        "capable_of": 0.6,
        "used_for": 0.5,
        "has_part": 0.4,
        "at_location": 0.4,
    }
    
    def __init__(self, relation_weights: Optional[Dict[str, float]] = None, use_rust: bool = True,
                 config=None):
        self._config = config
        # Try to use Rust backend if available and requested
        self._rust_backend = None
        if use_rust and hypervec_rs.SemanticMemoryConcurrent is not None:
            try:
                self._rust_backend = hypervec_rs.SemanticMemoryConcurrent()
                print("SemanticMemory Initialized with Rust backend (concurrent, optimized).")
            except Exception as e:
                print(f"[WARNING] Failed to initialize Rust backend: {e}")
                print("Falling back to Python implementation.")
        else:
            print("SemanticMemory Initialized with Python backend.")
        
        # Graph database of concepts (always maintained for graph operations)
        self.concept_graph = nx.DiGraph()
        
        # Concept -> HyperVector mapping (Python fallback)
        self.concept_hvs: Dict[str, hypervec_rs.HyperVector] = {}
        
        # Relation types (extensible — new types registered on first use)
        self.relations = list(self.DEFAULT_RELATION_WEIGHTS.keys())
        
        # Configurable relation weights for spreading activation
        self.relation_weights: Dict[str, float] = dict(self.DEFAULT_RELATION_WEIGHTS)
        if relation_weights:
            self.relation_weights.update(relation_weights)

        # V3: Stigmergy — edge-level pheromone strengths
        self._stigmergy: Dict[Tuple[str, str], float] = {}

        # V4: Hot cache — LRU cache for recently/frequently activated concepts
        _hot_cache_size = getattr(config, 'semantic_hot_cache_size', 256) if config else 256
        self._HOT_CACHE_SIZE: int = _hot_cache_size
        self._hot_cache: Dict[str, Any] = {}  # concept_name -> (hv, activation_score)

        # V3/V6: ANN index for fast nearest-concept lookup
        # Uses hnswlib when available; falls back to pure-Python NSW otherwise.
        self._hnsw_index = None
        self._hnsw_id_to_concept: List[str] = []
        self._hnsw_dim: int = 0
        # V4: HNSW enabled by default (was gated behind config flag)
        self._hnsw_enabled: bool = True
        if not getattr(config, 'enable_hnsw_index', True):
            self._hnsw_enabled = False
            if _HNSWLIB_AVAILABLE:
                logger.info("[SEMANTIC] HNSW index enabled (hnswlib)")
                logger.info("[SEMANTIC] HNSW index enabled (pure-Python NSW fallback)")

        # V5: Societal Knowledge World Orchestrator
        self.societal_world = SocietalKnowledgeWorld(dim=getattr(config, 'hv_dimension', 10240) if config else 10240)
        # V18: Track number of edges mirrored into the Rust DashMap.  When
        # concept_graph has more edges (e.g. via direct add_edge() calls that
        # bypass add_relation()), spread_activation falls back to the Python
        # edge-list path to stay correct.
        self._rust_synced_edge_count: int = 0

    def _init_hnsw(self, dim: int):
        """Lazily initialise the ANN index once the HV dimension is known."""
        if not self._hnsw_enabled:
            return
        if _HNSWLIB_AVAILABLE and _hnswlib is not None:
            try:
                idx = _hnswlib.Index(space='cosine', dim=dim)
                idx.init_index(max_elements=100_000, ef_construction=200, M=16)
                idx.set_ef(50)
                self._hnsw_index = idx
                self._hnsw_dim = dim
                self._hnsw_id_to_concept = []
                return
            except Exception as e:
                logger.warning("[SEMANTIC] hnswlib init failed: %s; switching to NSW", e)
        # Pure-Python NSW fallback
        self._hnsw_index = _NSWIndex(M=16, ef=50)
        self._hnsw_dim = dim
        self._hnsw_id_to_concept = []

    def reset(self):
        """Clear all semantic knowledge and re-initialize."""
        self.concept_graph = nx.DiGraph()
        self.concept_hvs = {}
        self._stigmergy = {}
        self._hnsw_index = None
        self._hnsw_id_to_concept = []
        print("[SEMANTIC] Memory reset complete.")
    
    def add_concept(self, concept_name: str, properties: Dict[str, Any], hv_override: Optional[hypervec_rs.HyperVector] = None):
        """
        Add a new concept to semantic memory.
        
        Args:
            concept_name: Name of the concept (e.g. "Apple")
            properties: Dictionary of properties
            hv_override: Optional pre-calculated hypervector
        """
        if hv_override:
            hv = hv_override
        else:
            # Generate base hypervector for concept name
            hv = hypervec_rs.HyperVector(hash(concept_name) % (2**32))
            
            # Bind properties into concept HV
            for prop, value in properties.items():
                prop_hv = hypervec_rs.HyperVector(hash(prop) % (2**32))
                value_hv = hypervec_rs.HyperVector(hash(str(value)) % (2**32))
                # Role-filler binding: XOR the property role with value, then bundle into concept
                bound = prop_hv.xor(value_hv)
                hv = hv.bundle(bound)

        # V3: Incremental concept refinement — if concept exists, blend HVs
        # In VSA, weighted bundling is achieved by including a concept
        # multiple times in a bundle (90% old = 9 copies, 10% new = 1 copy).
        if (concept_name in self.concept_hvs and
                self._config is not None and
                getattr(self._config, 'enable_incremental_concept_refinement', False)):
            existing_hv = self.concept_hvs[concept_name]
            # 90% old, 10% new via majority-vote bundling (9 old : 1 new)
            blended = existing_hv
            for _ in range(9):
                blended = blended.bundle(existing_hv)
            blended = blended.bundle(hv)
            hv = blended
        
        # Store in both backends
        self.concept_hvs[concept_name] = hv
        if self._rust_backend is not None:
            try:
                self._rust_backend.add_concept(concept_name, hv)
            except Exception as e:
                print(f"[WARNING] Rust backend add_concept failed: {e}")
        
        import time as _time
        self.concept_graph.add_node(
            concept_name,
            access_count=0,
            last_accessed=_time.time(),
            importance_score=1.0,
            **properties,
        )

        # V3/V6: Insert into ANN index if enabled.
        # The pure-Python NSW fallback has O(N) insertion cost; throttle it to
        # every 20th concept during bulk loads to avoid O(N²) total cost.
        # hnswlib (when available) is fast enough for every insert.
        if self._hnsw_enabled:
            try:
                bits = np.asarray(hv.bits, dtype=np.float32)
                dim = len(bits)
                if self._hnsw_index is None:
                    self._init_hnsw(dim)
                if self._hnsw_index is not None:
                    idx = len(self._hnsw_id_to_concept)
                    self._hnsw_id_to_concept.append(concept_name)
                    if _HNSWLIB_AVAILABLE and not isinstance(self._hnsw_index, _NSWIndex):
                        # External hnswlib: O(log N) — always safe to insert immediately
                        self._hnsw_index.add_items(bits.reshape(1, -1), [idx])
                    else:
                        # Pure-Python NSW fallback: O(N) — throttle to every 20th insert
                        _nsw_ctr = getattr(self, '_nsw_insert_counter', 0) + 1
                        self._nsw_insert_counter = _nsw_ctr
                        if _nsw_ctr % 20 == 1:
                            self._hnsw_index.add_item(bits)
                        else:
                            # Still track the id-to-concept mapping; the node
                            # will be picked up when a search is run (NSW searches
                            # the graph, not the id list).
                            pass
            except Exception as e:
                logger.debug("[SEMANTIC] ANN index add failed: %s", e)
                
        # V5: Inject into Societal World
        self.societal_world.ingest_concept(concept_name, hv, source="semantic")
        # Throttle tick_world() to every 50 add_concept calls — calling it on every
        # single add is O(N) work repeated N times = O(N²) total, which hangs at
        # 1K+ concepts.  The societal epoch model is fine-grained enough that a
        # batch-tick every 50 concepts is functionally equivalent.
        self._societal_tick_counter = getattr(self, '_societal_tick_counter', 0) + 1
        if self._societal_tick_counter % 50 == 0:
            self.societal_world.tick_world()
    
    def get_concept(self, concept_name: str) -> Optional[hypervec_rs.HyperVector]:
        """Retrieve the hypervector for a given concept."""
        import time as _time
        hv = self.concept_hvs.get(concept_name)
        if hv is not None and concept_name in self.concept_graph:
            d = self.concept_graph.nodes[concept_name]
            d["access_count"] = d.get("access_count", 0) + 1
            d["last_accessed"] = _time.time()
        return hv
    
    def add_relation(self, concept1: str, relation: str, concept2: str, timestamp: float = 0.0):
        """
        Add relation between concepts with temporal validation.
        Only updates if the new information is more recent or same time.

        V3: When ``enable_free_energy_beliefs`` is set, checks for contradictory
        relations before adding and applies free-energy belief scoring to decide
        whether to revise, flag as contested, or keep the existing belief.
        """
        if concept1 not in self.concept_graph or concept2 not in self.concept_graph:
            return

        # V3: Free-energy belief scoring
        if (self._config is not None and
                getattr(self._config, 'enable_free_energy_beliefs', False)):
            self._apply_belief_revision(concept1, relation, concept2, timestamp)
            return

        # Check if edge exists
        if self.concept_graph.has_edge(concept1, concept2):
            existing_data = self.concept_graph.get_edge_data(concept1, concept2)
            existing_time = existing_data.get('timestamp', 0.0)
            
            # If explicit timestamp provided and it's older than existing knowledge, IGNORE
            if timestamp > 0 and timestamp < existing_time:
                # [Belief Revision] Reject outdated info
                return

        # Detect whether this is a brand-new edge (not yet in NetworkX).
        # We only count brand-new edges toward _rust_synced_edge_count so that
        # repeated updates to the same edge don't inflate the counter and mask
        # unsynchronised edges that were added directly via concept_graph.add_edge().
        _edge_is_new = not self.concept_graph.has_edge(concept1, concept2)

        # Update or create edge
        self.concept_graph.add_edge(concept1, concept2, relation=relation, timestamp=timestamp)

        # V18: Mirror to Rust DashMap immediately at write time (O(1) cost per write).
        # This keeps Rust in sync without the O(E) scan previously done lazily in
        # spread_activation_fast().  Weighted edges use the typed relation weights so
        # parallel_spread_activation() propagates correctly.
        if self._rust_backend is not None:
            try:
                effective_weight = float(self.relation_weights.get(relation, 0.3))
                self._rust_backend.add_relation_weighted(
                    concept1, concept2, effective_weight
                )
                if _edge_is_new:
                    self._rust_synced_edge_count += 1
            except AttributeError:
                # Older Rust build without add_relation_weighted — fall back to unweighted
                try:
                    self._rust_backend.add_relation(concept1, concept2)
                    if _edge_is_new:
                        self._rust_synced_edge_count += 1
                except Exception:
                    pass
            except Exception:
                pass  # never break on Rust failure

        # V5: Societal Valence Bonding (record co-activation)
        self.societal_world.record_co_activation([concept1, concept2])

    def _apply_belief_revision(
        self, concept1: str, relation: str, concept2: str, timestamp: float
    ):
        """V3: Free-energy belief revision logic for add_relation().

        Detects contradictions: same (subject, relation_type) but different object.
        For example: 'Paris capital_of France' contradicts 'Paris capital_of Germany'.
        """
        try:
            from python.core.reasoning.belief_revision import BeliefMetadata, BeliefScorer
        except ImportError:
            self.concept_graph.add_edge(concept1, concept2, relation=relation, timestamp=timestamp)
            # V18: Mirror to Rust immediately on fallback path
            if self._rust_backend is not None:
                try:
                    self._rust_backend.add_relation_weighted(
                        concept1, concept2, float(self.relation_weights.get(relation, 0.3))
                    )
                except AttributeError:
                    try:
                        self._rust_backend.add_relation(concept1, concept2)
                    except Exception:
                        pass
                except Exception:
                    pass
            return

        scorer = BeliefScorer()

        # Check for contradictory relation: same concept1 + same relation type but different concept2
        contradicted_target = None
        for _, target, data in self.concept_graph.out_edges(concept1, data=True):
            if data.get('relation') == relation and target != concept2:
                contradicted_target = target
                break

        if contradicted_target is not None:
            # Contradiction detected: (concept1, relation, contradicted_target) vs new (concept1, relation, concept2)
            old_data = self.concept_graph.get_edge_data(concept1, contradicted_target) or {}
            meta_dict = old_data.get('belief_meta', {})
            meta = BeliefMetadata(
                evidence_count=meta_dict.get('evidence_count', 1),
                contradiction_count=meta_dict.get('contradiction_count', 0) + 1,
                complexity=meta_dict.get('complexity', 1.0),
                first_seen=meta_dict.get('first_seen', timestamp or 0.0),
                last_confirmed=meta_dict.get('last_confirmed', timestamp or 0.0),
                status=meta_dict.get('status', 'active'),
            )
            should_revise, reason = scorer.should_revise(meta, new_evidence_supports=False)
            if should_revise:
                # Revise: replace old belief with new one.
                # The new belief starts fresh (contradiction_count=0).
                self.concept_graph.remove_edge(concept1, contradicted_target)
                self.concept_graph.add_edge(
                    concept1, concept2, relation=relation, timestamp=timestamp,
                    belief_meta={
                        'evidence_count': 1,
                        'contradiction_count': 0,
                        'complexity': meta.complexity,
                        'first_seen': timestamp or 0.0,
                        'last_confirmed': timestamp or 0.0,
                        'status': 'active',
                    }
                )
            else:
                # Mark old belief as contested, record contradiction count
                new_meta = {
                    'evidence_count': meta.evidence_count,
                    'contradiction_count': meta.contradiction_count,
                    'complexity': meta.complexity,
                    'first_seen': meta.first_seen,
                    'last_confirmed': meta.last_confirmed,
                    'status': meta.status,  # 'contested' if scored that way
                }
                self.concept_graph.add_edge(
                    concept1, contradicted_target, relation=relation,
                    timestamp=old_data.get('timestamp', 0.0), belief_meta=new_meta
                )
                # Also store the new contradicting claim
                self.concept_graph.add_edge(
                    concept1, concept2, relation=relation, timestamp=timestamp,
                    belief_meta={
                        'evidence_count': 1,
                        'contradiction_count': 0,
                        'complexity': 1.0,
                        'first_seen': timestamp or 0.0,
                        'last_confirmed': timestamp or 0.0,
                        'status': 'active',
                    }
                )
        else:
            # No contradiction — add or increment evidence
            edge_data = self.concept_graph.get_edge_data(concept1, concept2) or {}
            meta_dict = edge_data.get('belief_meta', {})
            evidence_count = meta_dict.get('evidence_count', 0) + 1
            self.concept_graph.add_edge(
                concept1, concept2, relation=relation, timestamp=timestamp,
                belief_meta={
                    'evidence_count': evidence_count,
                    'contradiction_count': meta_dict.get('contradiction_count', 0),
                    'complexity': meta_dict.get('complexity', 1.0),
                    'first_seen': meta_dict.get('first_seen', timestamp or 0.0),
                    'last_confirmed': timestamp or 0.0,
                    'status': 'active',
                }
            )

        # V18: Mirror the final edge state to Rust DashMap immediately after belief revision
        if self._rust_backend is not None:
            try:
                self._rust_backend.add_relation_weighted(
                    concept1, concept2, float(self.relation_weights.get(relation, 0.3))
                )
            except AttributeError:
                try:
                    self._rust_backend.add_relation(concept1, concept2)
                except Exception:
                    pass
            except Exception:
                pass
    
    def query(self, query_hv: hypervec_rs.HyperVector, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find concepts most similar to query HV.
        Uses hnswlib (if installed) → NSW fallback → Rust parallel search → Python linear scan.
        """
        # V3/V6: ANN index (hnswlib or pure-Python NSW)
        if self._hnsw_enabled and self._hnsw_index is not None and self._hnsw_id_to_concept:
            try:
                bits = np.asarray(query_hv.bits, dtype=np.float32)
                n_results = min(k, len(self._hnsw_id_to_concept))
                if _HNSWLIB_AVAILABLE and not isinstance(self._hnsw_index, _NSWIndex):
                    # External hnswlib API
                    labels, distances = self._hnsw_index.knn_query(bits.reshape(1, -1), k=n_results)
                    results = []
                    for label, dist in zip(labels[0], distances[0]):
                        concept = self._hnsw_id_to_concept[label]
                        results.append((concept, 1.0 - float(dist)))
                else:
                    # Pure-Python NSW API
                    raw = self._hnsw_index.search(bits, k=n_results)
                    results = [
                        (self._hnsw_id_to_concept[idx], 1.0 - dist)
                        for idx, dist in raw
                        if idx < len(self._hnsw_id_to_concept)
                    ]
                return results
            except Exception as e:
                logger.debug("[SEMANTIC] ANN query failed: %s; falling back", e)

        # Try Rust backend
        if self._rust_backend is not None and hasattr(self._rust_backend, 'parallel_semantic_search'):
            try:
                return self._rust_backend.parallel_semantic_search(query_hv, k)
            except Exception as e:
                print(f"[WARNING] Rust backend query failed: {e}, falling back to Python")

        # Python linear scan fallback
        similarities = []
        for concept_name, concept_hv in self.concept_hvs.items():
            sim = query_hv.cosine_similarity(concept_hv) if hasattr(query_hv, 'cosine_similarity') else query_hv.similarity(concept_hv)
            similarities.append((concept_name, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    
    def spread_activation(self, start_concepts: List[str], steps: int = 3, decay: float = 0.7) -> Dict[str, float]:
        """
        Spreading activation for associative retrieval.
        
        Propagation strength is modulated by relation type weights,
        so ``is_a`` edges (0.9) carry activation more strongly than
        ``similar_to`` edges (0.4).

        When stigmergy is present, edge weights are additionally boosted
        by their pheromone strength (V3 feature).
        
        Returns activation levels for nodes.
        """
        # Try Rust-accelerated path first.
        # Only use Rust parallel_spread_activation when the Rust DashMap is fully
        # in sync with the NetworkX graph.  Direct concept_graph.add_edge() calls
        # (e.g. cross-domain bridge edges added in tests) bypass add_relation() so
        # they are not mirrored — detect this and fall back to the Python edge-list
        # path which reads concept_graph directly.
        _graph_edge_count = self.concept_graph.number_of_edges()
        _rust_fully_synced = (
            self._rust_backend is not None
            and self._rust_synced_edge_count >= _graph_edge_count
        )
        try:
            from python.core.memory.semantic_memory_shim import spread_activation_fast
            result = spread_activation_fast(
                self.concept_graph, start_concepts, self.relation_weights,
                self._stigmergy, steps, decay,
                rust_backend=self._rust_backend if _rust_fully_synced else None,
            )
            if result is not None:
                return result
        except Exception as _e:
            logger.warning(
                "spread_activation: Rust fast-path failed (%s); using Python fallback. "
                "To suppress: ensure hypervec_rs.so is built correctly.", _e
            )

        activation = {c: 1.0 for c in start_concepts if c in self.concept_graph}
        
        _MAX_FRONTIER = 200
        _stigmergy_boost = bool(self._stigmergy)
        
        for _ in range(steps):
            new_activation = activation.copy()

            frontier = sorted(
                ((c, a) for c, a in activation.items() if a >= 0.01),
                key=lambda x: x[1], reverse=True)[:_MAX_FRONTIER]

            for concept, act in frontier:
                for _, neighbor, data in self.concept_graph.out_edges(concept, data=True):
                    rel = data.get("relation", "similar_to")
                    edge_weight = self.relation_weights.get(rel, 0.3)
                    # V3: boost by stigmergy pheromone if present
                    if _stigmergy_boost:
                        stig = self._stigmergy.get((concept, neighbor), 0.0)
                        edge_weight = edge_weight * (1.0 + stig)
                    spread_val = act * decay * edge_weight
                    new_activation[neighbor] = new_activation.get(neighbor, 0.0) + spread_val
            
            activation = new_activation
            
        # V4: Update hot cache with top activated concepts
        self._update_hot_cache(activation)
        return activation

    def _update_hot_cache(self, activations: Dict[str, float]) -> None:
        """Update hot cache with top activated concepts (V4)."""
        top = sorted(activations.items(), key=lambda x: x[1], reverse=True)[:self._HOT_CACHE_SIZE // 2]
        for name, score in top:
            hv = self.concept_hvs.get(name)
            if hv is not None:
                self._hot_cache[name] = (hv, score)
        # LRU eviction: remove oldest inserted entries when over limit
        # (insertion-order based; dict preserves insertion order since Python 3.7)
        while len(self._hot_cache) > self._HOT_CACHE_SIZE:
            oldest = next(iter(self._hot_cache))
            del self._hot_cache[oldest]

    # V3: Stigmergy methods

    def mark_path(self, path: List[str], reward: float = 1.0):
        """Increment stigmergy pheromone on consecutive edges in path."""
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            self._stigmergy[key] = self._stigmergy.get(key, 0.0) + reward

    def evaporate_stigmergy(self, decay_rate: float = 0.99):
        """Decay all stigmergy values (call during sleep)."""
        keys_to_remove = []
        for key in list(self._stigmergy):
            self._stigmergy[key] *= decay_rate
            if self._stigmergy[key] < 1e-6:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._stigmergy[key]

    def get_stigmergy(self, concept1: str, concept2: str) -> float:
        """Get stigmergy strength for an edge."""
        return self._stigmergy.get((concept1, concept2), 0.0)
    
    def get_inherited_properties(self, concept: str) -> Dict[str, Any]:
        """
        Get properties of a concept including those inherited via ``is_a`` edges.
        
        Walks the ``is_a`` hierarchy upward and merges properties, with the
        most specific (closest) concept's properties taking precedence.
        
        Args:
            concept: The concept to look up
            
        Returns:
            Merged properties dict (own + inherited)
        """
        if concept not in self.concept_graph:
            return {}
        
        # Collect all ancestors via is_a (BFS)
        visited = set()
        queue = [concept]
        ancestor_chain = []  # ordered from most specific to most general
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            ancestor_chain.append(current)
            
            # Follow is_a edges upward
            for _, parent, data in self.concept_graph.out_edges(current, data=True):
                if data.get("relation") == "is_a" and parent not in visited:
                    queue.append(parent)
        
        # Merge properties from general to specific (specific wins)
        merged = {}
        for ancestor in reversed(ancestor_chain):
            props = dict(self.concept_graph.nodes[ancestor])
            merged.update(props)
        
        return merged
    
    def extract_schema(self, concept: str) -> Dict[str, Any]:
        """Extract abstracted structure (Schema) for a concept."""
        if concept not in self.concept_graph:
            return {}
        
        schema = {
            "name": concept,
            "properties": dict(self.concept_graph.nodes[concept]),
            "is_a": [],
            "parts": [],
            "causes": [],
            "caused_by": []
        }
        
        # Outgoing relations
        for _, neighbor, data in self.concept_graph.out_edges(concept, data=True):
            rel = data.get("relation")
            if rel == "is_a": schema["is_a"].append(neighbor)
            elif rel == "part_of": schema["parts"].append(neighbor)
            elif rel == "causes": schema["causes"].append(neighbor)
            
        # Incoming relations
        for neighbor, _, data in self.concept_graph.in_edges(concept, data=True):
            rel = data.get("relation")
            if rel == "causes": schema["caused_by"].append(neighbor)
            
        return schema

    # ── V4: Transitive / taxonomic inference ────────────────────────────────

    def infer_transitive(self, relation_type: str = "is_a", max_hops: int = 3) -> int:
        """V4: Derive new facts by transitive closure over *relation_type* edges.

        Example: If ``A is_a B`` and ``B is_a C`` exist, adds ``A is_a C``.

        Works for any transitive relation: ``is_a``, ``part_of``, ``causes``,
        ``located_in``, ``leads_to``, ``implies``, etc.

        Args:
            relation_type: Edge type to close transitively.
            max_hops:       Maximum chain length (prevents infinite loops on cycles).

        Returns:
            Number of new edges inferred and added.
        """
        new_edges: List[Tuple[str, str]] = []
        nodes = list(self.concept_graph.nodes())

        for source in nodes:
            # BFS from source following only relation_type edges
            visited: Set[str] = {source}
            frontier = [source]
            hops = 0
            while frontier and hops < max_hops:
                next_frontier: List[str] = []
                for node in frontier:
                    for _, target, data in self.concept_graph.out_edges(node, data=True):
                        if data.get("relation") == relation_type and target not in visited:
                            visited.add(target)
                            next_frontier.append(target)
                            if not self.concept_graph.has_edge(source, target):
                                new_edges.append((source, target))
                frontier = next_frontier
                hops += 1

        added = 0
        for src, tgt in new_edges:
            if src != tgt:  # no self-loops
                self.concept_graph.add_edge(
                    src, tgt,
                    relation=relation_type,
                    inferred=True,
                    timestamp=0.0,
                )
                added += 1

        if added:
            logger.debug("[SEMANTIC] infer_transitive(%s): added %d derived edges", relation_type, added)
        return added

    # ── V4: Prototype-based category generalization ──────────────────────────

    def build_prototypes(
        self,
        category_relation: str = "is_a",
        min_members: int = 2,
    ) -> Dict[str, hypervec_rs.HyperVector]:
        """V4: Build a *prototype* hypervector for each category by bundling all
        member HVs.

        Inspired by prototype theory (Rosch 1973): a category is best represented
        by the central tendency of its members, not a single exemplar.  In VSA,
        the *bundle* of all member HVs is exactly that central tendency — the
        resulting vector is maximally similar to all members.

        Usage::

            prototypes = memory.build_prototypes(min_members=3)
            sim = prototypes["Animal"].similarity(dog_hv)  # high if dog is animal-like

        Args:
            category_relation: Edge type used to identify category membership
                                (default: ``"is_a"``).
            min_members:        Minimum number of members needed to form a prototype.

        Returns:
            Dict mapping category name → prototype HyperVector.
        """
        # Gather members for each category
        category_members: Dict[str, List[str]] = {}
        for src, tgt, data in self.concept_graph.edges(data=True):
            if data.get("relation") == category_relation:
                category_members.setdefault(tgt, []).append(src)

        prototypes: Dict[str, hypervec_rs.HyperVector] = {}
        for category, members in category_members.items():
            if len(members) < min_members:
                continue
            prototype_hv: Optional[hypervec_rs.HyperVector] = None
            for member in members:
                member_hv = self.concept_hvs.get(member)
                if member_hv is None:
                    continue
                if prototype_hv is None:
                    prototype_hv = member_hv
                else:
                    prototype_hv = prototype_hv.bundle(member_hv)
            if prototype_hv is not None:
                prototypes[category] = prototype_hv
                logger.debug(
                    "[SEMANTIC] Prototype for '%s' built from %d members", category, len(members)
                )

        return prototypes

    def save(self, filepath: str):
        """Save semantic memory to gzip+JSON (safe, no pickle)."""
        import base64
        import gzip
        import json
        import numpy as np
        concepts_data = {}
        for name, d in self.concept_graph.nodes(data=True):
            concepts_data[name] = {
                "properties": {k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                               for k, v in d.items()},
            }
        hvs_data = {}
        for name, hv in self.concept_hvs.items():
            try:
                bits = np.asarray(hv.bits, dtype=np.int8)
                hvs_data[name] = base64.b64encode(bits.tobytes()).decode("ascii")
            except Exception:
                pass
        data = {
            "schema_version": 2,
            "concepts": concepts_data,
            "concept_hvs": hvs_data,
            "relation_weights": self.relation_weights,
            "edges": [(u, v, d) for u, v, d in self.concept_graph.edges(data=True)],
        }
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"[SEMANTIC] Saved {len(self.concept_hvs)} concepts to {filepath} (JSON).")

    def load(self, filepath: str):
        """Load semantic memory from gzip+JSON or legacy gzip+pickle."""
        import base64
        import gzip
        import json
        import numpy as np
        import warnings
        if not os.path.exists(filepath):
            print(f"[SEMANTIC] No memory file found at {filepath}")
            return
        # Try JSON first
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                data = json.load(f)
            for name, cd in data.get("concepts", {}).items():
                props = cd.get("properties", {})
                if name not in self.concept_hvs:
                    self.add_concept(name, props)
                else:
                    self.concept_graph.nodes[name].update(props)
            for name, b64 in data.get("concept_hvs", {}).items():
                try:
                    bits = np.frombuffer(base64.b64decode(b64), dtype=np.int8)
                    import python.core.vsa.hypervec_shim as hv_mod
                    self.concept_hvs[name] = hv_mod.HyperVector.from_bits(bits)
                except Exception:
                    pass
            self.relation_weights = data.get("relation_weights", self.DEFAULT_RELATION_WEIGHTS)
            for u, v, d in data.get("edges", []):
                if not self.concept_graph.has_edge(u, v):
                    self.concept_graph.add_edge(u, v, **d)
            print(f"[SEMANTIC] Loaded {len(self.concept_hvs)} concepts from {filepath} (JSON).")
            return
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError):
            pass
        # Fallback: legacy pickle
        warnings.warn(
            f"Loading SemanticMemory from pickle ({filepath}). "
            "Re-save with save() to upgrade to JSON.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            with gzip.open(filepath, "rb") as f:
                data = pickle.load(f)
            self.concept_graph = data.get("concept_graph", nx.DiGraph())
            self.concept_hvs = data.get("concept_hvs", {})
            self.relation_weights = data.get("relation_weights", self.DEFAULT_RELATION_WEIGHTS)
            for u, v, d in data.get("edges", []):
                if not self.concept_graph.has_edge(u, v):
                    self.concept_graph.add_edge(u, v, **d)
            # Sync with Rust backend if active
            if self._rust_backend:
                for name, hv in self.concept_hvs.items():
                    try:
                        self._rust_backend.add_concept(name, hv)
                    except Exception as e:
                        print(f"[WARNING] Rust sync failed for {name}: {e}")
            print(f"[SEMANTIC] Loaded {len(self.concept_hvs)} concepts.")
        except Exception as e:
            print(f"[SEMANTIC] Failed to load memory: {e}")

    def decay_concepts(self, lambda_decay: float = 0.01) -> int:
        """Exponential decay of importance_score based on time since last access."""
        import math, time
        now = time.time()
        count = 0
        for node in list(self.concept_graph.nodes()):
            d = self.concept_graph.nodes[node]
            last = d.get("last_accessed", now)
            hours = (now - last) / 3600.0
            old = d.get("importance_score", 1.0)
            new_val = old * math.exp(-lambda_decay * hours)
            self.concept_graph.nodes[node]["importance_score"] = new_val
            count += 1
        return count

    def prune_below(self, threshold: float = 0.1) -> int:
        """Remove concepts whose importance_score is below threshold."""
        to_remove = [
            n for n, d in self.concept_graph.nodes(data=True)
            if d.get("importance_score", 1.0) < threshold
        ]
        for node in to_remove:
            self.concept_graph.remove_node(node)
            self.concept_hvs.pop(node, None)
        return len(to_remove)
