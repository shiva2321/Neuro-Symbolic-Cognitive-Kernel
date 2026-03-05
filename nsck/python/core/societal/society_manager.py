"""
SocietyManager — orchestrator for the societal hypervector knowledge graph.

The SocietyManager maintains a population of LivingHyperVectors and provides:

1. **Registration** — add/remove concepts with domain paths.
2. **Bond management** — auto-form bonds when similarity > threshold; dissolve
   weak bonds; enforce max-bonds-per-node.
3. **Leiden-style clustering** — greedy modularity optimisation that assigns
   every LHV to a community (cluster_id).
4. **Percolation** — find the giant connected component and measure the
   percolation threshold.
5. **Multi-resolution clusters** — run clustering at several resolution
   parameters γ and cache all results.
6. **Hierarchical domain structure** — `domain_tree()` returns a nested dict
   of domains → subdomains → concept ids.
7. **Epoch stepping** — advance the logical clock, decay activations and
   bonds, run optional auto-clustering.
"""
from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Set, Tuple

import numpy as np

from python.core.societal.living_hypervector import LivingHyperVector, Bond, _hv_cosine_sim

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _modularity_gain(
    node_degree: int,
    community_degree: int,
    k_i_in: int,
    m2: float,
    resolution: float,
) -> float:
    """Compute the modularity gain δQ for moving a node into a community."""
    return (k_i_in / m2) - resolution * (community_degree * node_degree) / (m2 ** 2)


# ---------------------------------------------------------------------------
# ClusterResult
# ---------------------------------------------------------------------------

class ClusterResult:
    """Result of a single clustering pass.

    Attributes
    ----------
    resolution:
        The γ resolution parameter used.
    communities:
        Mapping of cluster_id → frozenset of concept_ids.
    modularity:
        Achieved modularity score Q ∈ [-0.5, 1].
    n_communities:
        Number of non-empty communities.
    """

    def __init__(
        self,
        resolution: float,
        communities: Dict[int, FrozenSet[str]],
        modularity: float,
    ) -> None:
        self.resolution = resolution
        self.communities = communities
        self.modularity = modularity
        self.n_communities = len(communities)

    def community_of(self, concept_id: str) -> Optional[int]:
        """Return the cluster_id containing *concept_id*, or None."""
        for cid, members in self.communities.items():
            if concept_id in members:
                return cid
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<ClusterResult γ={self.resolution:.2f} "
            f"Q={self.modularity:.4f} k={self.n_communities}>"
        )


# ---------------------------------------------------------------------------
# SocietyManager
# ---------------------------------------------------------------------------

class SocietyManager:
    """Manage a dynamic society of LivingHyperVectors.

    Parameters
    ----------
    bond_threshold:
        Minimum similarity for automatic bond formation.
    max_bonds:
        Maximum bonds per node (enforce hub-and-spoke sparsity).
    bond_decay_rate:
        Per-epoch bond strength decay factor.
    activation_decay_rate:
        Per-epoch activation decay factor.
    activation_spread_factor:
        Fraction of activation propagated along each bond per epoch.
    auto_cluster_interval:
        Run Leiden clustering every N epochs (0 = never auto-cluster).
    """

    def __init__(
        self,
        bond_threshold: float = 0.65,
        max_bonds: int = 8,
        bond_decay_rate: float = 0.01,
        activation_decay_rate: float = 0.05,
        activation_spread_factor: float = 0.4,
        auto_cluster_interval: int = 10,
    ) -> None:
        self.bond_threshold = float(bond_threshold)
        self.max_bonds = int(max_bonds)
        self.bond_decay_rate = float(bond_decay_rate)
        self.activation_decay_rate = float(activation_decay_rate)
        self.activation_spread_factor = float(activation_spread_factor)
        self.auto_cluster_interval = int(auto_cluster_interval)

        self._concepts: Dict[str, LivingHyperVector] = {}
        self._epoch: int = 0
        self._cluster_cache: Dict[float, ClusterResult] = {}
        self._percolation_threshold: Optional[float] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, lhv: LivingHyperVector) -> None:
        """Add a LivingHyperVector to the society."""
        self._concepts[lhv.concept_id] = lhv
        self._cluster_cache.clear()
        self._percolation_threshold = None

    def register_many(self, lhvs: List[LivingHyperVector]) -> None:
        """Bulk-register a list of LivingHyperVectors."""
        for lhv in lhvs:
            self._concepts[lhv.concept_id] = lhv
        self._cluster_cache.clear()
        self._percolation_threshold = None

    def unregister(self, concept_id: str) -> bool:
        """Remove a concept from the society and dissolve its bonds.

        Returns True if the concept existed.
        """
        if concept_id not in self._concepts:
            return False
        # Dissolve inbound bonds from all other nodes
        for other in self._concepts.values():
            other.dissolve_bond(concept_id)
        del self._concepts[concept_id]
        self._cluster_cache.clear()
        self._percolation_threshold = None
        return True

    def get(self, concept_id: str) -> Optional[LivingHyperVector]:
        """Return the LHV for *concept_id*, or None."""
        return self._concepts.get(concept_id)

    def __len__(self) -> int:
        return len(self._concepts)

    def __iter__(self) -> Iterator[LivingHyperVector]:
        return iter(self._concepts.values())

    def __contains__(self, concept_id: str) -> bool:
        return concept_id in self._concepts

    # ------------------------------------------------------------------
    # Bond formation / dissolution
    # ------------------------------------------------------------------

    def auto_bond(
        self,
        candidates: Optional[List[str]] = None,
        bond_type: str = "similarity",
    ) -> int:
        """Scan all (or a subset of) concept pairs and form bonds.

        Parameters
        ----------
        candidates:
            If supplied, only consider pairs within this subset.  Otherwise
            scan all O(N²) pairs (use with caution for large societies).
        bond_type:
            Bond type assigned to newly created bonds.

        Returns
        -------
        int
            Number of new bonds formed.
        """
        ids = list(candidates or self._concepts.keys())
        formed = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = self._concepts.get(ids[i])
                b = self._concepts.get(ids[j])
                if a is None or b is None:
                    continue
                # Skip if already bonded
                if ids[j] in a._bonds:
                    continue
                sim = _hv_cosine_sim(a.hv, b.hv)
                if sim >= self.bond_threshold:
                    a.form_bond(b, epoch=self._epoch, bond_type=bond_type,
                                strength_override=sim)
                    b.form_bond(a, epoch=self._epoch, bond_type=bond_type,
                                strength_override=sim)
                    formed += 1

        # Enforce max-bonds constraint (keep strongest)
        self._enforce_max_bonds()
        return formed

    def _enforce_max_bonds(self) -> None:
        """Trim each node to at most max_bonds strongest bonds."""
        for lhv in self._concepts.values():
            if len(lhv._bonds) > self.max_bonds:
                # Sort by descending strength and keep top max_bonds
                sorted_bonds = sorted(
                    lhv._bonds.items(), key=lambda kv: kv[1].strength, reverse=True
                )
                keep = {pid for pid, _ in sorted_bonds[: self.max_bonds]}
                for pid in list(lhv._bonds):
                    if pid not in keep:
                        del lhv._bonds[pid]

    def decay_all_bonds(self) -> int:
        """Decay bonds across all nodes.  Returns total bonds dissolved."""
        total = 0
        for lhv in self._concepts.values():
            dissolved = lhv.decay_bonds(
                rate=self.bond_decay_rate, min_strength=0.05
            )
            total += len(dissolved)
        return total

    def form_bond_explicit(
        self,
        concept_a: str,
        concept_b: str,
        bond_type: str = "similarity",
        strength: Optional[float] = None,
    ) -> Optional[Bond]:
        """Explicitly form a bond between two named concepts.

        Returns the Bond object, or None if either concept is missing.
        """
        a = self._concepts.get(concept_a)
        b = self._concepts.get(concept_b)
        if a is None or b is None:
            return None
        bond = a.form_bond(b, epoch=self._epoch, bond_type=bond_type,
                           strength_override=strength)
        b.form_bond(a, epoch=self._epoch, bond_type=bond_type,
                    strength_override=strength)
        self._cluster_cache.clear()
        return bond

    # ------------------------------------------------------------------
    # Leiden-style clustering
    # ------------------------------------------------------------------

    def leiden_cluster(self, resolution: float = 1.0) -> ClusterResult:
        """Run a greedy Leiden-style community detection pass.

        The algorithm is a single-pass greedy modularity maximisation:

        1. Assign each node to its own community.
        2. For each node in random order, compute modularity gain of moving
           it to each neighbour's community.
        3. Accept the best improvement; repeat until no node moves.

        Parameters
        ----------
        resolution:
            Higher values → more, smaller communities.

        Returns
        -------
        ClusterResult
        """
        if resolution in self._cluster_cache:
            return self._cluster_cache[resolution]

        ids = list(self._concepts.keys())
        n = len(ids)
        if n == 0:
            result = ClusterResult(resolution, {}, 0.0)
            self._cluster_cache[resolution] = result
            return result

        idx = {cid: i for i, cid in enumerate(ids)}
        community = list(range(n))  # initial: each node in its own community

        # Build adjacency as sim weights
        adj: Dict[int, Dict[int, float]] = defaultdict(dict)
        m2 = 0.0  # 2 * total weight
        for i, cid in enumerate(ids):
            lhv = self._concepts[cid]
            for bond in lhv._bonds.values():
                j = idx.get(bond.peer_id)
                if j is not None:
                    adj[i][j] = bond.strength
                    m2 += bond.strength

        if m2 == 0.0:
            # No edges — each node is its own community
            communities: Dict[int, FrozenSet[str]] = {
                i: frozenset([ids[i]]) for i in range(n)
            }
            result = ClusterResult(resolution, communities, 0.0)
        else:
            # Node degrees
            degree = [sum(adj[i].values()) for i in range(n)]

            improved = True
            max_iter = n * 5
            step = 0
            while improved and step < max_iter:
                improved = False
                order = list(range(n))
                random.shuffle(order)
                for i in order:
                    current_c = community[i]
                    # Sum of weights from i to each community
                    weight_to_c: Dict[int, float] = defaultdict(float)
                    for j, w in adj[i].items():
                        weight_to_c[community[j]] += w
                    # Community degrees
                    c_degree: Dict[int, float] = defaultdict(float)
                    for k in range(n):
                        c_degree[community[k]] += degree[k]

                    # Find best community to move to
                    best_gain = 0.0
                    best_c = current_c
                    for c, k_i_in in weight_to_c.items():
                        if c == current_c:
                            continue
                        gain = _modularity_gain(
                            node_degree=int(degree[i]),
                            community_degree=int(c_degree[c]),
                            k_i_in=int(k_i_in),
                            m2=m2,
                            resolution=resolution,
                        )
                        if gain > best_gain:
                            best_gain = gain
                            best_c = c
                    if best_c != current_c:
                        community[i] = best_c
                        improved = True
                step += 1

            # Re-index communities to contiguous ints
            raw_communities: Dict[int, List[str]] = defaultdict(list)
            for i, c in enumerate(community):
                raw_communities[c].append(ids[i])
            communities = {
                new_id: frozenset(members)
                for new_id, (_, members) in enumerate(raw_communities.items())
            }

            # Assign cluster_id back to each LHV
            for cid_int, members in communities.items():
                for cid in members:
                    lhv = self._concepts.get(cid)
                    if lhv is not None:
                        lhv._cluster_id = cid_int

            # Compute modularity Q
            q = 0.0
            for i in range(n):
                for j, w in adj[i].items():
                    if community[i] == community[j]:
                        q += w - resolution * degree[i] * degree[j] / m2
            q /= m2

            result = ClusterResult(resolution, communities, float(q))

        self._cluster_cache[resolution] = result
        return result

    def multi_resolution_cluster(
        self, resolutions: Optional[List[float]] = None
    ) -> Dict[float, ClusterResult]:
        """Run Leiden at several resolutions and cache all results.

        Default resolutions: [0.5, 1.0, 1.5, 2.0].
        """
        if resolutions is None:
            resolutions = [0.5, 1.0, 1.5, 2.0]
        return {r: self.leiden_cluster(r) for r in resolutions}

    # ------------------------------------------------------------------
    # Percolation
    # ------------------------------------------------------------------

    def percolation_threshold(self, n_steps: int = 20) -> float:
        """Find the bond-strength threshold at which a giant component emerges.

        Sweep from 0 → 1; return the epsilon at which the largest connected
        component first exceeds N/2 nodes.

        Returns
        -------
        float
            Percolation threshold ε* ∈ [0, 1].
        """
        if self._percolation_threshold is not None:
            return self._percolation_threshold

        n = len(self._concepts)
        if n == 0:
            self._percolation_threshold = 1.0
            return 1.0

        ids = list(self._concepts.keys())
        eps_vals = np.linspace(0.0, 1.0, n_steps)

        threshold = 1.0
        for eps in reversed(eps_vals):
            lcc = self._largest_component_size(float(eps))
            if lcc >= n / 2:
                threshold = float(eps)
                break

        self._percolation_threshold = threshold
        return threshold

    def _largest_component_size(self, min_bond_strength: float) -> int:
        """BFS to find size of the largest connected component."""
        ids = list(self._concepts.keys())
        visited: Set[str] = set()
        max_size = 0

        for start in ids:
            if start in visited:
                continue
            queue = [start]
            component: Set[str] = set()
            while queue:
                cid = queue.pop()
                if cid in component:
                    continue
                component.add(cid)
                lhv = self._concepts.get(cid)
                if lhv is None:
                    continue
                for bond in lhv._bonds.values():
                    if (bond.strength >= min_bond_strength
                            and bond.peer_id not in component
                            and bond.peer_id in self._concepts):
                        queue.append(bond.peer_id)
            visited.update(component)
            max_size = max(max_size, len(component))

        return max_size

    def connected_components(
        self, min_bond_strength: float = 0.0
    ) -> List[Set[str]]:
        """Return all connected components at *min_bond_strength*."""
        ids = list(self._concepts.keys())
        visited: Set[str] = set()
        components: List[Set[str]] = []

        for start in ids:
            if start in visited:
                continue
            queue = [start]
            component: Set[str] = set()
            while queue:
                cid = queue.pop()
                if cid in component:
                    continue
                component.add(cid)
                lhv = self._concepts.get(cid)
                if lhv is None:
                    continue
                for bond in lhv._bonds.values():
                    if (bond.strength >= min_bond_strength
                            and bond.peer_id not in component
                            and bond.peer_id in self._concepts):
                        queue.append(bond.peer_id)
            visited.update(component)
            components.append(component)

        return sorted(components, key=len, reverse=True)

    # ------------------------------------------------------------------
    # Hierarchical domain structure
    # ------------------------------------------------------------------

    def domain_tree(self) -> Dict[str, Any]:
        """Return the hierarchical domain tree.

        Returns a nested dict::

            {
              "science": {
                "physics": {"quantum": ["concept_a", "concept_b"]},
                "biology": {"genetics": ["gene_x"]}
              },
              "__unassigned__": ["concept_z"]
            }
        """
        tree: Dict[str, Any] = {}
        for lhv in self._concepts.values():
            node = tree
            path = lhv.domain_path or ["__unassigned__"]
            for i, segment in enumerate(path):
                if i == len(path) - 1:
                    # Leaf: store list of concept_ids
                    node.setdefault(segment, [])
                    if isinstance(node[segment], list):
                        node[segment].append(lhv.concept_id)
                    else:
                        node[segment].setdefault("__concepts__", []).append(
                            lhv.concept_id
                        )
                else:
                    node.setdefault(segment, {})
                    if isinstance(node[segment], list):
                        # Promote to dict (mixed depth)
                        node[segment] = {"__concepts__": node[segment]}
                    node = node[segment]
        return tree

    def concepts_in_domain(self, *domain_path: str) -> List[LivingHyperVector]:
        """Return all LHVs whose domain_path starts with *domain_path*."""
        prefix = list(domain_path)
        return [
            lhv for lhv in self._concepts.values()
            if lhv.domain_path[: len(prefix)] == prefix
        ]

    # ------------------------------------------------------------------
    # Activation & epoch stepping
    # ------------------------------------------------------------------

    def step_epoch(self, run_cluster: bool = True) -> Dict[str, Any]:
        """Advance the logical clock by one epoch.

        1. Decay all activations.
        2. Spread activation along bonds.
        3. Decay all bonds (dissolve weak ones).
        4. Optionally run Leiden clustering.

        Returns
        -------
        Dict[str, Any]
            Epoch statistics dict.
        """
        self._epoch += 1

        # 1. Decay activations
        for lhv in self._concepts.values():
            lhv.decay_activation(rate=self.activation_decay_rate)

        # 2. Spread activation
        all_ids = list(self._concepts.keys())
        # Snapshot activations before spreading (avoid order effects)
        activations = {cid: self._concepts[cid].activation for cid in all_ids}
        for cid in all_ids:
            lhv = self._concepts[cid]
            if lhv.activation > 0.01:
                lhv.spread_activation(
                    self._concepts,
                    spread_factor=self.activation_spread_factor,
                    epoch=self._epoch,
                )

        # 3. Decay bonds
        bonds_dissolved = self.decay_all_bonds()

        # 4. Auto-cluster
        cluster_run = False
        if (run_cluster and self.auto_cluster_interval > 0
                and self._epoch % self.auto_cluster_interval == 0):
            self.leiden_cluster(resolution=1.0)
            cluster_run = True

        return {
            "epoch": self._epoch,
            "n_concepts": len(self._concepts),
            "bonds_dissolved": bonds_dissolved,
            "cluster_run": cluster_run,
            "mean_activation": float(
                np.mean([lhv.activation for lhv in self._concepts.values()])
            ) if self._concepts else 0.0,
        }

    # ------------------------------------------------------------------
    # Query / retrieval
    # ------------------------------------------------------------------

    def nearest_neighbors(
        self,
        query_hv: Any,
        k: int = 5,
        min_activation: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Return top-k concepts by similarity to *query_hv*.

        Parameters
        ----------
        query_hv:
            Any HV (Rust or Python) to search against.
        k:
            Number of results to return.
        min_activation:
            Only consider concepts with activation >= this threshold.

        Returns
        -------
        List[Tuple[str, float]]
            Sorted list of (concept_id, similarity).
        """
        scores: List[Tuple[str, float]] = []
        for cid, lhv in self._concepts.items():
            if lhv.activation < min_activation:
                continue
            sim = _hv_cosine_sim(query_hv, lhv.hv)
            scores.append((cid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def activate_concept(
        self, concept_id: str, delta: float = 0.5, spread: bool = True
    ) -> bool:
        """Activate a concept and optionally spread to its bonds.

        Returns True if the concept exists.
        """
        lhv = self._concepts.get(concept_id)
        if lhv is None:
            return False
        lhv.activate(delta, epoch=self._epoch)
        if spread:
            lhv.spread_activation(
                self._concepts,
                spread_factor=self.activation_spread_factor,
                epoch=self._epoch,
            )
        return True

    # ------------------------------------------------------------------
    # Topological health
    # ------------------------------------------------------------------

    def topological_health(self) -> Dict[str, Any]:
        """Compute society-level topological health metrics.

        Returns
        -------
        Dict with keys:
            n_concepts, n_bonds, mean_bond_strength, percolation_threshold,
            n_components, mean_topo_persistence, mean_activation,
            hub_concepts (top-5 by bond count)
        """
        n = len(self._concepts)
        if n == 0:
            return {
                "n_concepts": 0, "n_bonds": 0, "mean_bond_strength": 0.0,
                "percolation_threshold": 1.0, "n_components": 0,
                "mean_topo_persistence": 0.0, "mean_activation": 0.0,
                "hub_concepts": [],
            }

        all_bonds = [
            bond
            for lhv in self._concepts.values()
            for bond in lhv._bonds.values()
        ]
        n_bonds = len(all_bonds) // 2  # undirected
        mean_strength = float(np.mean([b.strength for b in all_bonds])) if all_bonds else 0.0

        components = self.connected_components()
        n_components = len(components)

        mean_activation = float(
            np.mean([lhv.activation for lhv in self._concepts.values()])
        )

        # Hubs: concepts with most bonds
        hub_concepts = sorted(
            self._concepts.values(), key=lambda x: len(x._bonds), reverse=True
        )[:5]

        return {
            "n_concepts": n,
            "n_bonds": n_bonds,
            "mean_bond_strength": round(mean_strength, 4),
            "percolation_threshold": self.percolation_threshold(),
            "n_components": n_components,
            "mean_topo_persistence": 0.0,  # compute on demand
            "mean_activation": round(mean_activation, 4),
            "hub_concepts": [h.concept_id for h in hub_concepts],
        }

    # ------------------------------------------------------------------
    # Statistics / export
    # ------------------------------------------------------------------

    @property
    def epoch(self) -> int:
        """Current logical epoch."""
        return self._epoch

    def snapshot(self) -> "SocietalSnapshot":
        """Return a full observational snapshot of the current society state (V30).

        Returns
        -------
        SocietalSnapshot
            A fully populated snapshot dataclass.
        """
        from python.core.societal.snapshots import SocietalSnapshot

        n = len(self._concepts)

        all_bonds = [
            bond
            for lhv in self._concepts.values()
            for bond in lhv._bonds.values()
        ]
        n_bonds = len(all_bonds) // 2  # undirected
        avg_bond_strength = (
            float(sum(b.strength for b in all_bonds) / len(all_bonds))
            if all_bonds else 0.0
        )
        avg_activation = (
            float(sum(lhv.activation for lhv in self._concepts.values()) / n)
            if n > 0 else 0.0
        )

        # Communities from last clustering pass (resolution=1.0)
        cluster_result = self._cluster_cache.get(1.0)
        n_communities = cluster_result.n_communities if cluster_result is not None else 0

        perc_threshold = self._percolation_threshold if self._percolation_threshold is not None else 1.0

        giant = self._largest_component_size(0.0)

        top_activated = sorted(
            [(cid, lhv.activation) for cid, lhv in self._concepts.items()],
            key=lambda x: x[1], reverse=True
        )[:10]

        top_bonded = sorted(
            [(cid, len(lhv._bonds)) for cid, lhv in self._concepts.items()],
            key=lambda x: x[1], reverse=True
        )[:10]

        domain_tree_summary = {}
        try:
            dtree = self.domain_tree()
            domain_tree_summary = {
                k: len(v) if isinstance(v, list) else sum(
                    len(sub) if isinstance(sub, list) else 1
                    for sub in v.values()
                ) if isinstance(v, dict) else 1
                for k, v in dtree.items()
            }
        except Exception:
            pass

        return SocietalSnapshot(
            epoch=self._epoch,
            n_concepts=n,
            n_bonds=n_bonds,
            avg_bond_strength=round(avg_bond_strength, 4),
            avg_activation=round(avg_activation, 4),
            n_communities=n_communities,
            percolation_threshold=round(perc_threshold, 4),
            giant_component_size=giant,
            top_activated=top_activated,
            top_bonded=top_bonded,
            domain_tree_summary=domain_tree_summary,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full society state (excluding raw HV bits)."""
        return {
            "epoch": self._epoch,
            "bond_threshold": self.bond_threshold,
            "max_bonds": self.max_bonds,
            "n_concepts": len(self._concepts),
            "concepts": [lhv.to_dict() for lhv in self._concepts.values()],
        }

    def summary(self) -> str:
        """One-line human-readable summary."""
        health = self.topological_health()
        return (
            f"SocietyManager(epoch={self._epoch} "
            f"N={health['n_concepts']} bonds={health['n_bonds']} "
            f"components={health['n_components']} "
            f"perc_thr={health['percolation_threshold']:.3f})"
        )
