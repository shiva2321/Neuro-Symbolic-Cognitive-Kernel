# NCGN Technical Reference

This document details the algorithmic core of the Neuromorphic Cognitive Graph Network.

## System 1: The Dynamical Engine

The `System1Engine` executes a strictly serialized 8-phase tick pipeline. This ensures deterministic behavior and prevents race conditions in signal propagation.

### The 8-Phase Pipeline

1.  **Phase 0: Transduction**: External energy injection.
    *   `E_new = min(1.0, E_old + Input)`
    *   *Only phase where external world touches the graph.*

2.  **Phase 1: Passive Decay (The Leak)**:
    *   `E_new = E_old * α` (where α ≈ 0.9)
    *   Nodes with `E < 0.001` are deactivated.

3.  **Phase 2: Firing Determination**:
    *   Criteria: `E > Threshold` AND `RefractoryTimer == 0`
    *   *State is captured here, effectively "spending" the energy.*

4.  **Phase 3: Refractory Reset**:
    *   Firing nodes are reset to `RestingPotential` (0.0).
    *   `RefractoryTimer` set to 3 ticks.

5.  **Phase 4: Propagation**:
    *   Spikes travel along synapses.
    *   `Signal = Weight * SpikeMagnitude`
    *   Signals are buffered, NOT applied immediately (preventing infinite loops in one tick).

6.  **Phase 5: Integration**:
    *   Buffered signals applied to target nodes.
    *   `E_target += Signal`

7.  **Phase 6: Lateral Inhibition (k-WTA)**:
    *   **k-Winners-Take-All**: Only the top `k` most energetic nodes stay active.
    *   All others are hard-suppressed to `E = 0`.
    *   *Implementation: Min-Heap selection O(N log k).*

8.  **Phase 7: Surprise Monitor**:
    *   Calculates mismatch between Expected and Observed state.
    *   If `Surprise > Gate`, triggers System 2 interrupt.

9.  **Phase 8: Maintenance**:
    *   Decrement refractory timers.
    *   Decay novelty scores.

### Surprise Calculation (v5.1 Patch)

Surprise is the RMS (Root Mean Square) variation between prediction and observation, weighted by confidence.

$$ S = \sqrt{ \sum ( (E_{pred} - E_{obs}) \cdot Confidence )^2 } $$

*   **Prediction**: Captured at start of tick.
*   **Observation**: Captured after Phase 6 (Inhibition).
*   **Logic**: High surprise only occurs when a *high-confidence* expectation is violated. Novel inputs (low confidence) do not trigger surprise.

## System 2: The Deliberative Controller

System 2 is a symbolic logic engine that intervenes on System 1.

### Intervention Protocol

1.  **Interrupt**: Pause System 1 tick loop.
2.  **Schema Check**: Retrieve `EventSchema` for the attempted action.
    *   *Example Schema*: "Action: Eat, Roles: {Target: Edible}"
3.  **Diagnosis**:
    *   **Constraint Violation**: Object has property `IsEdible = False`.
    *   **Missing Knowledge**: Object has no `IsEdible` property.
4.  **Planning**:
    *   *Violation*: Apply **LTD** (Long-Term Depression) to weaken the causative synapse. Inject energy into "Query User" goal.
    *   *Missing*: Inject energy into "Ask Property" goal.
5.  **Execute**: Modify graph weights/energies.
6.  **Resume**: Unpause System 1.

## Staging: The Hippocampus

The `StagingBuffer` prevents "hallucinations" from unverified bulk data (like file uploads).

### Merge Logic
When committing staging buffer to main memory:

1.  **Nodes**: If exists, decrease `NoveltyScore` (reinforcement). If new, create with `NoveltyScore = 0.5`.
2.  **Edges**: If exists, `NewConfidence = (OldConf + NewConf) / 2`.
3.  **Conflicts**:
    *   **Contradiction**: New edge negates existing edge. Flagged.
    *   **Schema Violation**: New edge violates System 2 schema. Flagged.
