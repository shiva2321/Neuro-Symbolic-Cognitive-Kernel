"""
NSCK Global Workspace Module
Global Workspace Theory implementation with mental rehearsal and veto mechanisms.

Implements the Global Workspace Theory (GWT) architecture where specialized modules
compete for access to a global broadcast channel (consciousness-like information sharing).

Mental rehearsal feature: before committing to an action, the winning proposal is
simulated through the WorldModel. If the predicted next state is similar to
a known danger vector, the proposal is vetoed and alternatives are evaluated.
Provides safe decision-making through predictive simulation.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
import logging
import time

# Configure logging
logger = logging.getLogger(__name__)

class WorkspaceModule(ABC):
    """Abstract base class for modules that connect to the Global Workspace."""
    
    @abstractmethod
    def receive_broadcast(self, content: Any):
        """Receive content broadcast from the global workspace."""
        pass

@dataclass
class Coalition:
    """A bundle of information competing for consciousness."""
    source: str
    content: Any
    base_salience: float      # Intrinsic loudness (0.0 - 1.0)
    relevance: float = 0.0    # Match with current context/goal
    affect_match: float = 0.0 # Match with current Drives (e.g. "Food" matches "Hunger")
    sender_confidence: float = 0.5 # [Phase 3.3] Metadata from module
    
    @property
    def activation(self) -> float:
        return self.base_salience + self.relevance + self.affect_match + (self.sender_confidence * 0.5)


@dataclass
class RehearsalEvent:
    """Record of a single mental rehearsal veto event (for telemetry)."""
    timestamp: float
    vetoed_source: str
    vetoed_action: str
    danger_similarity: float
    cycle: int


class GlobalWorkspace:
    """
    The central executive that manages the 'stream of consciousness'.
    Implementation: LIDA-Lite.

    Phase 8 additions:
    - Danger vector registry for veto logic
    - ``compete_with_rehearsal()`` for simulate-before-acting
    """
    
    def __init__(self, attention_threshold: float = 0.5):
        self.modules: Dict[str, WorkspaceModule] = {}
        self.workspace_content: Optional[Any] = None
        self.current_winner: Optional[str] = None
        self.attention_threshold = attention_threshold
        self.history: List[Tuple[str, Any, float]] = [] 
        self.mission_focus: Optional[str] = None # e.g. "EXPLORATION"
        self.latest_coalitions: List[Coalition] = [] # For telemetry

        # --- Phase 8: Mental Rehearsal ---
        self._danger_vectors: list = []          # List of HyperVectors representing dangerous states
        self.veto_threshold: float = 0.75        # Similarity above this → veto
        self.max_rehearsal_cycles: int = 3        # Max deliberation rounds
        self.max_danger_vectors: int = 200        # Cap to prevent memory bloat
        self.rehearsal_log: List[RehearsalEvent] = []  # For dashboard telemetry
        self._rehearsal_log_max: int = 50

    def register_module(self, name: str, module: WorkspaceModule):
        self.modules[name] = module
        logger.info(f"[GWT] Registered module: {name}")
        
    def compete(self, proposals: List[Coalition]) -> Optional[Coalition]:
        """
        LIDA Competition Cycle.
        Calculates Activation = Salience + Relevance + Affect.
        """
        self.latest_coalitions = proposals # Save for telemetry
        if not proposals:
            return None
            
        # 1. Apply Mission Focus Bias
        if self.mission_focus:
             for c in proposals:
                 if c.source == self.mission_focus:
                     c.base_salience += 0.2 # Priority boost

        # 2. Score all coalitions
        ranked = sorted(proposals, key=lambda c: c.activation, reverse=True)
        winner = ranked[0]
        
        # 2. Check threshold
        if winner.activation < self.attention_threshold:
            return None
            
        # 3. Broadcast
        self.current_winner = winner.source
        self.workspace_content = winner.content
        self.broadcast(winner.content)
        
        # Log history
        self.history.append((winner.source, winner.content, winner.activation))
        if len(self.history) > 100: self.history.pop(0)
            
        return winner
        
    def broadcast(self, content: Any):
        """Send content to all registered modules."""
        for name, module in self.modules.items():
            try:
                module.receive_broadcast(content)
            except Exception as e:
                logger.error(f"[GWT] Error broadcasting to {name}: {e}")

    # ------------------------------------------------------------------
    # Phase 8: Mental Rehearsal & Veto
    # ------------------------------------------------------------------
    def register_danger(self, danger_hv) -> None:
        """Register a state hypervector as dangerous (e.g. death state).

        Parameters
        ----------
        danger_hv : HyperVector
            Situation HV at the moment of a catastrophic outcome.
        """
        self._danger_vectors.append(danger_hv)
        # LRU eviction: drop oldest if over cap
        if len(self._danger_vectors) > self.max_danger_vectors:
            self._danger_vectors = self._danger_vectors[-self.max_danger_vectors:]

    def _is_dangerous(self, predicted_hv) -> Tuple[bool, float]:
        """Check whether *predicted_hv* is similar to any known danger vector.

        Returns (is_dangerous, max_similarity).
        """
        if not self._danger_vectors:
            return False, 0.0

        max_sim = 0.0
        for dv in self._danger_vectors:
            try:
                sim = predicted_hv.similarity(dv)
                if sim > max_sim:
                    max_sim = sim
            except Exception:
                continue

        return max_sim >= self.veto_threshold, max_sim

    def compete_with_rehearsal(
        self,
        proposals: List[Coalition],
        current_state_hv,
        world_model,
        get_action_hv_fn,
        n_cycles: Optional[int] = None,
    ) -> Optional[Coalition]:
        """Mental rehearsal: simulate-then-act with veto.

        For each deliberation cycle the top-ranked proposal is passed through
        the WorldModel.  If the predicted next state is *dangerous*, the
        proposal is vetoed (salience halved) and the next-best is tried.

        Parameters
        ----------
        proposals : list[Coalition]
        current_state_hv : HyperVector – current situation encoding
        world_model : object with ``imagine(state_hv, action_hv)`` method
        get_action_hv_fn : callable(action_str) → HyperVector
        n_cycles : override for ``max_rehearsal_cycles``

        Returns
        -------
        Optional[Coalition] – winner (or emergency fallback if all vetoed)
        """
        self.latest_coalitions = proposals
        if not proposals:
            return None

        cycles = n_cycles or self.max_rehearsal_cycles

        # Apply mission focus bias (same as compete)
        if self.mission_focus:
            for c in proposals:
                if c.source == self.mission_focus:
                    c.base_salience += 0.2

        # Working copy for deliberation
        remaining = list(proposals)
        vetoed_sources: List[str] = []

        for cycle in range(cycles):
            if not remaining:
                break

            # Rank by activation
            remaining.sort(key=lambda c: c.activation, reverse=True)
            candidate = remaining[0]

            # Threshold check
            if candidate.activation < self.attention_threshold:
                break  # Nobody strong enough → no winner

            # --- Mental simulation ---
            try:
                action_hv = get_action_hv_fn(candidate.content)
                pred_state_bits, pred_reward = world_model.imagine(
                    current_state_hv, action_hv
                )

                # Convert predicted bits back to HV for similarity check
                # pred_state_bits is a numpy float array; threshold at 0.5 → binary
                import numpy as np
                import python.core.vsa.hypervec_shim as hv_mod

                binary = (np.array(pred_state_bits) > 0.5).astype(np.int8)
                # Build a temporary HV from the predicted bits
                from python.core.vsa.hypervec_py import HyperVectorPy
                pred_hv = HyperVectorPy.from_bits(binary)

                is_danger, sim = self._is_dangerous(pred_hv)
            except Exception as e:
                # If imagination fails, trust the proposal
                logger.warning(f"[REHEARSAL] Imagination failed for {candidate.source}: {e}")
                is_danger, sim = False, 0.0

            if is_danger:
                # VETO: penalize and remove
                event = RehearsalEvent(
                    timestamp=time.time(),
                    vetoed_source=candidate.source,
                    vetoed_action=str(candidate.content),
                    danger_similarity=sim,
                    cycle=cycle,
                )
                self.rehearsal_log.append(event)
                if len(self.rehearsal_log) > self._rehearsal_log_max:
                    self.rehearsal_log.pop(0)

                logger.info(
                    f"[REHEARSAL] VETO cycle={cycle}: {candidate.source} "
                    f"action={candidate.content} danger_sim={sim:.3f}"
                )
                vetoed_sources.append(candidate.source)
                candidate.base_salience *= 0.5  # Penalize
                remaining.remove(candidate)
                continue
            else:
                # Safe – commit
                self.current_winner = candidate.source
                self.workspace_content = candidate.content
                self.broadcast(candidate.content)
                self.history.append(
                    (candidate.source, candidate.content, candidate.activation)
                )
                if len(self.history) > 100:
                    self.history.pop(0)
                return candidate

        # All proposals vetoed → emergency fallback
        logger.warning("[REHEARSAL] DEADLOCK: all proposals vetoed – emergency ACTION_STAY")
        emergency = Coalition(
            source="EMERGENCY",
            content="ACTION_STAY",
            base_salience=0.1,
        )
        self.current_winner = "EMERGENCY"
        self.workspace_content = "ACTION_STAY"
        self.broadcast("ACTION_STAY")
        return emergency

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """Get current workspace status."""
        return {
            "current_winner": self.current_winner,
            "current_content_type": type(self.workspace_content).__name__ if self.workspace_content else "None",
            "history_len": len(self.history),
            "danger_vectors": len(self._danger_vectors),
            "rehearsal_vetoes": len(self.rehearsal_log),
        }

    def get_recent_vetoes(self, n: int = 5) -> List[dict]:
        """Return recent rehearsal veto events for dashboard."""
        return [
            {
                "timestamp": e.timestamp,
                "source": e.vetoed_source,
                "action": e.vetoed_action,
                "danger_sim": round(e.danger_similarity, 3),
                "cycle": e.cycle,
            }
            for e in self.rehearsal_log[-n:]
        ]
