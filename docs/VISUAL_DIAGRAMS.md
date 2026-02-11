# NSCK Visual Diagrams and Architecture

> **Purpose**: Visual representations of NSCK's key algorithms, data flows, and system architecture

This document provides ASCII-art diagrams and visual explanations of the NSCK system's internal workings.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [VSA Operations](#2-vsa-operations)
3. [Cognitive Decision Cycle](#3-cognitive-decision-cycle)
4. [Transfer Learning Pipeline](#4-transfer-learning-pipeline)
5. [Causal Discovery Process](#5-causal-discovery-process)
6. [Global Workspace Competition](#6-global-workspace-competition)
7. [Continual Learning Flow](#7-continual-learning-flow)

---

## 1. System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NSCK Cognitive Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │        Language Module (Phi3 - Peripheral Only)          │   │
│  │           Natural Language ↔ Symbolic Predicates         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌───────────────────────┐  ┌────────────────────────────────┐ │
│  │   Perception Layer    │  │      Learning Systems          │ │
│  │  • Multimodal Input   │  │  • Rule Induction              │ │
│  │  • Symbol Grounding   │  │  • Multi-Task Learning         │ │
│  │  • Predicate Extract  │  │  • Gradient Surgery            │ │
│  └───────┬───────────────┘  │  • EWC (Continual Learning)    │ │
│          │                   └─────────┬──────────────────────┘ │
│          │                             │                         │
│          v                             v                         │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Cognitive Engine (Central Hub)                │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │ │
│  │  │   SNN Fast  │  │ Symbolic     │  │  STRIPS Planner │  │ │
│  │  │   Response  │  │ Rules        │  │  Goal-Directed  │  │ │
│  │  └─────────────┘  └──────────────┘  └─────────────────┘  │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │ │
│  │  │ Imagination │  │ Active       │  │  Exploration    │  │ │
│  │  │ World Model │  │ Inference    │  │  Curiosity      │  │ │
│  │  └─────────────┘  └──────────────┘  └─────────────────┘  │ │
│  │                                                            │ │
│  │              ↓ Global Workspace Competition ↓             │ │
│  │                                                            │ │
│  │  ┌──────────────────────────────────────────────────┐    │ │
│  │  │  Winner Selection → Value Alignment Check        │    │ │
│  │  └──────────────────────────────────────────────────┘    │ │
│  └───────────────────────────────────────────────────────────┘ │
│                             │                                   │
│                             v                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                   Memory Systems                           │ │
│  │  • Episodic Memory (VSA-based experience storage)         │ │
│  │  • Semantic Memory (Concept graph + spreading activation) │ │
│  │  • Knowledge Store (Cross-session consolidation)          │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │               Self-Awareness & Social Layer                │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │ │
│  │  │  Self-Model  │  │  Emotion     │  │  Theory of     │  │ │
│  │  │  Performance │  │  Valence &   │  │  Mind (ToM)    │  │ │
│  │  │  Prediction  │  │  Arousal     │  │  Mental Models │  │ │
│  │  └──────────────┘  └──────────────┘  └────────────────┘  │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │      VSA Core - 10,240-bit Binary Hypervectors            │ │
│  │      XOR Binding • Bundling • Hamming Similarity          │ │
│  │      [Foundation for all representations]                 │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. VSA Operations

### XOR Binding (Role-Filler Composition)

```
Example: Bind AGENT role to SNAKE_HEAD entity

AGENT_role:     [1 0 1 1 0 1 0 0 1 1 ...]  (10,240 bits)
SNAKE_HEAD:     [0 1 1 0 0 1 1 1 0 0 ...]  (10,240 bits)
                     ⊕ (XOR bitwise)
BOUND:          [1 1 0 1 0 0 1 1 1 1 ...]  (10,240 bits)

Properties:
  1. BOUND is quasi-orthogonal to both inputs
     sim(AGENT, BOUND) ≈ 0.50 ± 0.005
     sim(SNAKE_HEAD, BOUND) ≈ 0.50 ± 0.005
  
  2. Unbinding recovers original:
     BOUND ⊕ SNAKE_HEAD = AGENT_role (perfect recovery)
     
  3. Time complexity: O(D) = O(10,240) ≈ 0.02ms on CPU
```

### Bundling (Superposition)

```
Example: Create composite "DANGEROUS_AREA" from multiple threats

WALL:       [1 0 0 1 1 0 1 ...]
ENEMY:      [0 1 1 0 1 1 0 ...]
PIT:        [1 1 0 0 0 1 1 ...]
            majority_vote(bit-wise)
DANGEROUS:  [1 1 0 0 1 1 1 ...]

Properties:
  1. Result is similar to all inputs
     sim(DANGEROUS, WALL) ≈ 0.75
     sim(DANGEROUS, ENEMY) ≈ 0.75
     sim(DANGEROUS, PIT) ≈ 0.75
  
  2. Can unbundle by matching against codebook
     best_match(DANGEROUS, codebook) → {WALL, ENEMY, PIT}
     
  3. Graceful degradation: Adding noise doesn't destroy pattern
```

### Similarity Search (Hamming Distance)

```
Query vector:   [1 0 1 1 0 ...]
Codebook:
  FOOD:         [1 0 1 0 0 ...]  → hamming_dist = 1024 → sim = 0.90
  WALL:         [0 1 0 1 1 ...]  → hamming_dist = 5120 → sim = 0.50
  ENEMY:        [1 1 0 0 1 ...]  → hamming_dist = 3072 → sim = 0.70

Best match: FOOD (highest similarity)

Time complexity: O(D × N) where N = codebook size
                 ~10ms for 1000 concepts
```

---

## 3. Cognitive Decision Cycle

### Complete Decision Flow

```
┌──────────────────────────────────────────────────────────────┐
│ 1. PERCEPTION                                                 │
│    Input: Raw game state (positions, velocities, objects)    │
│      ↓                                                        │
│    Symbol Grounding: Extract active predicates               │
│      ↓                                                        │
│    Output: {FOOD_ABOVE, WALL_LEFT, BODY_BELOW, ...}         │
│    Time: ~0.3ms                                              │
└────────────┬─────────────────────────────────────────────────┘
             │
             v
┌──────────────────────────────────────────────────────────────┐
│ 2. PROPOSAL GENERATION (Parallel)                            │
│    ┌────────────┐  ┌────────────┐  ┌──────────────┐        │
│    │ SNN System │  │ Rule System│  │ Planner      │        │
│    │ Fast path  │  │ Symbolic   │  │ STRIPS       │        │
│    │ ACTION_UP  │  │ ACTION_UP  │  │ ACTION_UP    │        │
│    │ conf: 0.72 │  │ conf: 0.87 │  │ cost: 2      │        │
│    └────────────┘  └────────────┘  └──────────────┘        │
│    ┌────────────┐  ┌────────────┐  ┌──────────────┐        │
│    │ Active     │  │ Imagination│  │ Exploration  │        │
│    │ Inference  │  │ World Model│  │ Curiosity    │        │
│    │ ACTION_LEFT│  │ ACTION_UP  │  │ ACTION_RIGHT │        │
│    │ conf: 0.45 │  │ reward:+0.8│  │ novelty: 0.6 │        │
│    └────────────┘  └────────────┘  └──────────────┘        │
│    Time: ~1.2ms                                              │
└────────────┬─────────────────────────────────────────────────┘
             │
             v
┌──────────────────────────────────────────────────────────────┐
│ 3. GLOBAL WORKSPACE COMPETITION                              │
│                                                               │
│    Coalition Formation:                                       │
│    ┌──────────────────────────────────────────────────┐     │
│    │ PLANNER:    salience=0.8 + relevance=0.9         │     │
│    │             + affect=0.5 + confidence=0.85       │     │
│    │             + mission_bonus=0.2                  │     │
│    │             = Activation: 3.25  👑 WINNER        │     │
│    └──────────────────────────────────────────────────┘     │
│    ┌──────────────────────────────────────────────────┐     │
│    │ RULES:      activation = 3.02                    │     │
│    └──────────────────────────────────────────────────┘     │
│    ┌──────────────────────────────────────────────────┐     │
│    │ SNN:        activation = 2.42                    │     │
│    └──────────────────────────────────────────────────┘     │
│                                                               │
│    Broadcasting: Winner content → All modules                │
│    Time: ~0.1ms                                              │
└────────────┬─────────────────────────────────────────────────┘
             │
             v
┌──────────────────────────────────────────────────────────────┐
│ 4. VALUE ALIGNMENT CHECK                                     │
│    Selected action: ACTION_UP                                │
│      ✓ Not toward wall                                       │
│      ✓ Not into own body                                     │
│      ✓ Goal-aligned (moves toward food)                      │
│      ✓ Safety threshold passed                               │
│    Time: ~0.2ms                                              │
└────────────┬─────────────────────────────────────────────────┘
             │
             v
┌──────────────────────────────────────────────────────────────┐
│ 5. EXECUTION & LEARNING                                      │
│    Execute: ACTION_UP                                        │
│    Observe: reward = +0.1, new_state = (5,4)               │
│      ↓                                                        │
│    Update Episodic Memory: Store (state, action, reward)    │
│    Update Rule Statistics: {FOOD_ABOVE}→UP: 88/100          │
│    Update Self-Model: task_performance 82% → 83%            │
│    Update Emotion: positive reward → arousal +0.1           │
│    Time: ~0.5ms                                              │
└──────────────────────────────────────────────────────────────┘

TOTAL CYCLE TIME: ~2.3ms (434 Hz capable on CPU)
```

---

## 4. Transfer Learning Pipeline

### Three-Phase Transfer Process

```
┌────────────────────────────────────────────────────────────────┐
│ PHASE 1: LIFT (Domain → Abstract)                             │
│                                                                 │
│ Source Domain: SNAKE                                           │
│   Rule: {FOOD_ABOVE} → ACTION_UP (success: 85%)              │
│      ↓ apply grounding map                                    │
│   Abstract: {TARGET_ABOVE} → MOVE_UP                          │
│                                                                 │
│ Grounding Map (Snake):                                         │
│   SNAKE_FOOD  → TARGET    (entity abstraction)                │
│   FOOD_ABOVE  → TARGET_ABOVE  (relation abstraction)          │
│   ACTION_UP   → MOVE_UP   (action abstraction)                │
└────────────────────────────────────────────────────────────────┘
             │
             v
┌────────────────────────────────────────────────────────────────┐
│ PHASE 2: PATTERN MATCHING (Abstract Level)                    │
│                                                                 │
│ Knowledge Store:                                               │
│   Domain: Snake   Rule: {TARGET_ABOVE} → MOVE_UP (conf: 0.85)│
│   Domain: Pong    Rule: {TARGET_ABOVE} → MOVE_UP (conf: 0.78)│
│      ↓ consolidate pattern                                    │
│   Abstract Rule: {TARGET_ABOVE} → MOVE_UP                     │
│     Confidence: 0.82 (averaged)                               │
│     Domains: {snake, pong}                                    │
│     Support: 183 observations                                 │
│                                                                 │
│ This is now DOMAIN-INDEPENDENT knowledge                      │
└────────────────────────────────────────────────────────────────┘
             │
             v
┌────────────────────────────────────────────────────────────────┐
│ PHASE 3: GROUND (Abstract → Target Domain)                    │
│                                                                 │
│ Target Domain: MAZE (never seen before)                       │
│   Current state: EXIT_ABOVE (exit is above player)           │
│      ↓ apply inverse grounding map                            │
│   Abstract: TARGET_ABOVE (matches abstract rule)             │
│      ↓ retrieve abstract rule                                 │
│   Abstract action: MOVE_UP                                     │
│      ↓ ground to maze domain                                  │
│   Concrete action: ACTION_UP (in maze)                        │
│                                                                 │
│ Inverse Grounding Map (Maze):                                 │
│   EXIT_ABOVE  ← TARGET_ABOVE                                  │
│   ACTION_UP   ← MOVE_UP                                       │
│                                                                 │
│ Result: ZERO-SHOT transfer successful                         │
│   Expected performance: 40-60% (vs 25% random)                │
└────────────────────────────────────────────────────────────────┘
```

### Domain Grounding Tables

```
┌───────────────┬──────────────┬──────────────┬──────────────┐
│ Abstract      │ Snake        │ Pong         │ Maze         │
├───────────────┼──────────────┼──────────────┼──────────────┤
│ AGENT         │ SNAKE_HEAD   │ PLAYER_PAD   │ MAZE_PLAYER  │
│ TARGET        │ SNAKE_FOOD   │ BALL         │ MAZE_EXIT    │
│ DANGER        │ SNAKE_WALL   │ OUT_BOUNDS   │ MAZE_WALL    │
│ COMPETITOR    │ (none)       │ OPPONENT_PAD │ (none)       │
│               │              │              │              │
│ TARGET_ABOVE  │ FOOD_ABOVE   │ BALL_ABOVE   │ EXIT_ABOVE   │
│ DANGER_AHEAD  │ WALL_AHEAD   │ BOUND_AHEAD  │ WALL_AHEAD   │
│ MOVE_UP       │ ACTION_UP    │ ACTION_UP    │ ACTION_UP    │
│ MOVE_TOWARD   │ (composite)  │ (composite)  │ (composite)  │
└───────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 5. Causal Discovery Process

### Statistical Causal Learning from Observations

```
┌────────────────────────────────────────────────────────────────┐
│ OBSERVATION COLLECTION                                         │
│                                                                 │
│ Record: (cause_present, effect_occurred) tuples               │
│                                                                 │
│ Example: Testing if SWITCH_ON causes LIGHT_ON                 │
│   Observation 1:  SWITCH=ON  → LIGHT=ON   ✓                  │
│   Observation 2:  SWITCH=ON  → LIGHT=ON   ✓                  │
│   ...                                                          │
│   Observation 50: SWITCH=ON  → LIGHT=ON   ✓                  │
│   Observation 51: SWITCH=OFF → LIGHT=OFF  ✓                  │
│   ...                                                          │
│   Observation 100: SWITCH=OFF → LIGHT=OFF ✓                  │
│                                                                 │
│ Total: 100 observations                                        │
└────────────────────────────────────────────────────────────────┘
             │
             v
┌────────────────────────────────────────────────────────────────┐
│ CONTINGENCY COMPUTATION (Delta-P)                             │
│                                                                 │
│ Step 1: Calculate conditional probabilities                   │
│   P(LIGHT=ON | SWITCH=ON)  = 50/50 = 1.00                    │
│   P(LIGHT=ON | SWITCH=OFF) = 0/50  = 0.00                    │
│                                                                 │
│ Step 2: Compute Delta-P                                       │
│   ΔP = P(E|C) - P(E|¬C)                                       │
│   ΔP = 1.00 - 0.00 = 1.00                                     │
│                                                                 │
│ Step 3: Interpret                                             │
│   ΔP = 1.00 → STRONG POSITIVE CAUSATION                       │
│   Cause is both necessary AND sufficient for effect           │
└────────────────────────────────────────────────────────────────┘
             │
             v
┌────────────────────────────────────────────────────────────────┐
│ TEMPORAL PRECEDENCE CHECK                                      │
│                                                                 │
│ For each observation:                                          │
│   timestamp(SWITCH_ON) < timestamp(LIGHT_ON) ?                │
│     ✓ All observations pass temporal constraint               │
│                                                                 │
│ Reject if: Cause comes after effect (reverse causation)       │
└────────────────────────────────────────────────────────────────┘
             │
             v
┌────────────────────────────────────────────────────────────────┐
│ CAUSAL GRAPH UPDATE                                           │
│                                                                 │
│ Add causal link:                                              │
│   SWITCH_ON ──[CAUSES, strength=1.0]──> LIGHT_ON             │
│                                                                 │
│ Store metadata:                                               │
│   - Observations: 100                                         │
│   - Confidence: 0.99                                          │
│   - Type: DETERMINISTIC                                       │
└────────────────────────────────────────────────────────────────┘
```

### Spurious Correlation Detection

```
Example: CLAP and BIRD_CHIRPS (both happen at dawn)

┌────────────────────────────────────────────────────────────────┐
│ Data:                                                          │
│   With CLAP:    CHIRPS=Yes (90/100)                           │
│   Without CLAP: CHIRPS=Yes (90/100)                           │
│                                                                 │
│ Computation:                                                   │
│   P(CHIRPS | CLAP)  = 0.90                                    │
│   P(CHIRPS | ¬CLAP) = 0.90                                    │
│   ΔP = 0.90 - 0.90 = 0.00                                     │
│                                                                 │
│ Interpretation:                                                │
│   ΔP = 0.00 → NO CONTINGENCY                                  │
│   CLAP and CHIRPS are statistically independent               │
│   Both caused by hidden variable (DAWN)                       │
│                                                                 │
│ Action: REJECT causal link (spurious correlation)             │
└────────────────────────────────────────────────────────────────┘

         DAWN (hidden cause)
          /              \
         /                \
        v                  v
      CLAP              BIRD_CHIRPS
        
     ✗ No direct causal link between CLAP and CHIRPS
```

---

## 6. Global Workspace Competition

### Coalition Activation and Winner Selection

```
┌────────────────────────────────────────────────────────────────┐
│ COALITION FORMATION                                            │
│                                                                 │
│ Six cognitive modules propose actions:                        │
│                                                                 │
│ ┌────────────────────────────────────────────────────┐        │
│ │ Coalition A: PLANNER                                │        │
│ │   Content: ACTION_UP                                │        │
│ │   Salience: 0.8  (strong signal)                   │        │
│ │   Relevance: 0.9 (highly goal-aligned)             │        │
│ │   Affect: 0.6    (emotionally congruent)           │        │
│ │   Confidence: 0.85 (sender reliability)            │        │
│ │   Mission bonus: +0.2 (planner is mission focus)   │        │
│ └────────────────────────────────────────────────────┘        │
│                                                                 │
│ ┌────────────────────────────────────────────────────┐        │
│ │ Coalition B: SYMBOLIC RULES                         │        │
│ │   Content: ACTION_UP                                │        │
│ │   Salience: 0.7                                     │        │
│ │   Relevance: 0.85                                   │        │
│ │   Affect: 0.5                                       │        │
│ │   Confidence: 0.87 (rule success rate)             │        │
│ │   Mission bonus: 0                                  │        │
│ └────────────────────────────────────────────────────┘        │
│                                                                 │
│ ┌────────────────────────────────────────────────────┐        │
│ │ Coalition C: SNN FAST SYSTEM                        │        │
│ │   Content: ACTION_LEFT                              │        │
│ │   Salience: 0.6                                     │        │
│ │   Relevance: 0.4 (not well goal-aligned)           │        │
│ │   Affect: 0.5                                       │        │
│ │   Confidence: 0.72                                  │        │
│ │   Mission bonus: 0                                  │        │
│ └────────────────────────────────────────────────────┘        │
└────────────────────────────────────────────────────────────────┘
             │
             v
┌────────────────────────────────────────────────────────────────┐
│ ACTIVATION COMPUTATION                                         │
│                                                                 │
│ Formula: Act = sal + rel + aff + 0.5*conf + mission_bonus     │
│                                                                 │
│ Coalition A (PLANNER):                                         │
│   Act = 0.8 + 0.9 + 0.6 + 0.5*0.85 + 0.2                     │
│   Act = 0.8 + 0.9 + 0.6 + 0.425 + 0.2 = 2.925  👑            │
│                                                                 │
│ Coalition B (RULES):                                           │
│   Act = 0.7 + 0.85 + 0.5 + 0.5*0.87 + 0 = 2.485              │
│                                                                 │
│ Coalition C (SNN):                                             │
│   Act = 0.6 + 0.4 + 0.5 + 0.5*0.72 + 0 = 1.86                │
└────────────────────────────────────────────────────────────────┘
             │
             v
┌────────────────────────────────────────────────────────────────┐
│ WINNER SELECTION                                               │
│                                                                 │
│ Ranking:                                                       │
│   1st: Coalition A (2.925) ← WINNER                           │
│   2nd: Coalition B (2.485)                                    │
│   3rd: Coalition C (1.860)                                    │
│                                                                 │
│ Winner: Coalition A passes threshold (2.925 > 0.5)            │
│ Action selected: ACTION_UP (from PLANNER)                     │
└────────────────────────────────────────────────────────────────┘
             │
             v
┌────────────────────────────────────────────────────────────────┐
│ BROADCAST & STATE UPDATE                                       │
│                                                                 │
│ Broadcast to all modules:                                      │
│   • Perception layer: "ACTION_UP chosen"                      │
│   • Learning systems: "PLANNER won competition"                │
│   • Memory systems: "Store winning coalition"                 │
│   • Emotion system: "Update based on action"                  │
│                                                                 │
│ State changes:                                                 │
│   • Mission focus ← PLANNER                                   │
│   • PLANNER reward +1.0 (winner bonus)                        │
│   • Losing coalitions decay: activation *= 0.9                │
└────────────────────────────────────────────────────────────────┘
```

---

## 7. Continual Learning Flow

### Elastic Weight Consolidation (EWC) Process

```
┌────────────────────────────────────────────────────────────────┐
│ TASK A: LEARNING SNAKE                                         │
│                                                                 │
│ Training: 1000 episodes                                        │
│   → Neural network converges                                   │
│   → Parameters: θ*_snake                                       │
│   → Performance: 85% success rate                              │
└────────────────────────────────────────────────────────────────┘
             │
             v
┌────────────────────────────────────────────────────────────────┐
│ COMPUTE FISHER INFORMATION                                     │
│                                                                 │
│ For each weight θ_i:                                           │
│   F_i = (1/N) Σ (∂Loss_snake/∂θ_i)²                           │
│                                                                 │
│ Example Fisher values:                                         │
│   Layer 1, Weight [0,0]: F = 0.0023  (low importance)         │
│   Layer 2, Weight [5,3]: F = 0.8901  (HIGH importance)        │
│   Layer 3, Weight [2,1]: F = 0.4521  (medium importance)      │
│                                                                 │
│ High F_i → This weight is critical for Snake performance      │
│ Low F_i  → This weight can be modified freely                 │
└────────────────────────────────────────────────────────────────┘
             │
             v
┌────────────────────────────────────────────────────────────────┐
│ TASK B: LEARNING PONG                                          │
│                                                                 │
│ Training with EWC regularization:                              │
│                                                                 │
│ Loss = Loss_pong + (λ/2) Σ F_i(θ_i - θ*_i)²                  │
│                    \_____________________________/              │
│                              EWC penalty                        │
│                                                                 │
│ Effect:                                                        │
│   • Weights with high F_i (important for Snake):              │
│     → Heavily penalized if changed                             │
│     → Stay close to θ*_snake                                   │
│                                                                 │
│   • Weights with low F_i (not important for Snake):           │
│     → Free to adapt to Pong task                               │
│     → Can change significantly                                 │
│                                                                 │
│ Result:                                                        │
│   Snake performance: 85% → 78% (only 8% forgetting) ✓        │
│   Pong performance: 0% → 82% (new task learned) ✓            │
└────────────────────────────────────────────────────────────────┘
             │
             v
┌────────────────────────────────────────────────────────────────┐
│ COMPARISON: WITHOUT EWC                                        │
│                                                                 │
│ Training Pong without EWC:                                     │
│   • All weights free to change                                 │
│   • Snake-critical weights get overwritten                     │
│                                                                 │
│ Result:                                                        │
│   Snake performance: 85% → 38% (47% forgetting) ✗            │
│   Pong performance: 0% → 85% (slightly better) ✓             │
│                                                                 │
│ Conclusion: EWC prevents catastrophic forgetting               │
└────────────────────────────────────────────────────────────────┘
```

### Multi-Task Weight Allocation Diagram

```
Neural Network Weights After EWC:

┌─────────────────────────────────────────────────────────────┐
│ Layer 1 (Shared Encoder)                                    │
├─────────────────────────────────────────────────────────────┤
│ [SNAKE][PONG][SNAKE][FREE ][PONG][SNAKE][FREE ][PONG]     │
│  LOCK  LOCK  LOCK   FREE   LOCK  LOCK   FREE   LOCK       │
│   │     │     │      │      │     │      │      │         │
│   └─────┴─────┴──────┼──────┴─────┴──────┼──────┘         │
│                      │                    │                 │
│    Important for     │   Flexible         │  Allocated     │
│    Task A (Snake)    │   capacity         │  to Task B     │
└─────────────────────────────────────────────────────────────┘

Legend:
  [SNAKE] - Weight protected for Snake (high Fisher value)
  [PONG]  - Weight allocated for Pong (learned with EWC penalty)
  [FREE]  - Weight available for future tasks (low Fisher value)
```

---

## Conclusion

These diagrams provide visual explanations of NSCK's core algorithms and data flows. For mathematical details, see [FORMULAS_AND_PROOFS.md](FORMULAS_AND_PROOFS.md). For execution evidence, see [RUN_LOGS_AND_EVIDENCE.md](RUN_LOGS_AND_EVIDENCE.md).

All diagrams are based on actual implementation details verified by tests in `nsck-demo/tests/`.

---

*Last updated: 2026-02-11*
