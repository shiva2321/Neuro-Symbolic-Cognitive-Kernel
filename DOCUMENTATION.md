# NCGN Technical Documentation v5.0

## 1. System Architecture: The Dual-Process Model

NCGN implements a neuro-symbolic architecture divided into two distinct but interacting systems:

### System 1: The Reactive Engine (`core/system1.py`)
A dynamical system where concepts are nodes in a graph. It operates on "ticks." Every tick, energy flows through synapses based on weights. It is "Thinking Fast"—associative, automatic, and energy-efficient.

### System 2: The Deliberative Controller (`core/system2.py`)
A symbolic validator that monitors System 1. It is "Thinking Slow"—expensive, logical, and only wakes up when System 1 is "surprised."

---

## 2. The Immutable Tick Pipeline

Every discrete time step ($t$) in System 1 follows a strictly serialized 8-phase execution order:

1.  **Transduction (Phase 0)**: External energy enters the system via the `sensory_buffer`.
2.  **Passive Decay (Phase 1)**: All nodes lose a fraction of their energy ($E = E \times 0.9$). This clears "working memory."
3.  **Firing Determination (Phase 2)**: Nodes exceeding their threshold (default 0.75) are marked for firing.
4.  **Refractory & Reset (Phase 3)**: Firing nodes spend their energy and enter a refractory period where they cannot fire again for $N$ ticks.
5.  **Propagation (Phase 4)**: Spike transmission is calculated ($Input = Weight \times Spike$). Updates are buffered to prevent infinite cascades.
6.  **Integration (Phase 5)**: Buffered energy is applied to target nodes.
7.  **Lateral Inhibition (k-WTA) (Phase 6)**: A "Winner-Take-All" mechanism using a Min-Heap. Only the top $K$ most energetic nodes survive; all others are set to zero. This forces the system to make a "decision."
8.  **Surprise Monitor (Phase 7)**: The system compares the current state to the state it predicted earlier.
9.  **Maintenance (Phase 8)**: Housekeeping tasks like decaying novelty and decrementing timers.

---

## 3. Data Structures

### ConceptNode (`core/memory.py`)
The atomic unit of state.
* `energy`: Current membrane potential (0.0 to 1.0).
* `threshold`: Limit at which the node "fires."
* `novelty_score`: High for new concepts, decays over time.

### Synapse (`core/memory.py`)
The unit of association.
* `weight`: Strength of association (dictates energy flow speed).
* `confidence`: Epistemic truth value (used by System 2 to weigh violations).

---

## 4. Bridge & Surprise Math

**PATCHED v5.1**: Uses Root Mean Square (RMS) for proper Euclidean distance calculation.

Surprise is the penalty for **Violated High-Confidence Expectations**. It is calculated as:

$$S = \sqrt{\sum_{n \in Predicted} ((E_{pred}(n) - E_{obs}(n)) \times Confidence(n))^2}$$

*   If the system predicts something with high confidence and it *doesn't* happen, Surprise goes up.
*   If the system sees something completely new (Novelty), Surprise stays low (because there was no prediction to violate).
*   **Gate Threshold**: 0.45 (Lowered from 0.5 for higher sensitivity).

---

## 5. Intervention Protocol

When Surprise exceeds a threshold, System 2 pauses System 1 and executes:
1.  **Diagnosis**: Fetches an `EventSchema` (e.g., `eat.json`) and checks if the current action triple (Agent, Action, Object) violates constraints.
2.  **Intervention**: Modifies System 1 state. It might:
    *   **LTD (Long-Term Depression)**: Weaken the synapse that made the bad prediction.
    *   **Goal Injection**: Inject energy into a `Goal_Node` (like "Query User").
3.  **Resume**: System 1 continues, now directed by the new goal energy.

---

## 6. How to Extend

To add new knowledge to the system:
1.  Add a JSON schema to `core/schemas/`.
2.  Define roles (agent, target) and constraints (is_edible, is_animate).
3.  Use the `System2Controller.set_property()` method to define what the system knows about specific objects.
