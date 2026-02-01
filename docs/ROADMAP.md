# NSCK Implementation Roadmap: Journey to Sentience

This document outlines the multi-session journey to transform NSCK v1 (Snake Agent) into the Sentient NSCK (Specification in `NSCK_SPECIFICATION.md`).

> **CRITICAL FOR ALL AGENTS**: Do not skip steps. Do not implement "fake" versions. Building the foundation takes time.

---

## Phase 1: The "Self" & Language Foundation
*Goal: Give the agent an Identity (Proto-Self) and a Voice (Semantic Folding).*

*   [x] **1.1 VSA Language Engine (`lingua_cortex.py`)**
    *   Implement Semantic Folding (2D Grid SDRs).
    *   Create a simple "Snippet Trainer" to learn vector representations from text.
    *   Implement Boolean Logic for word disambiguation.
    *   **Verified**: `test_lingua.py` confirms topological similarity and intersection logic.
*   [x] **1.2 The Proto-Self (`homeostasis.py`)**
    *   Create `HomeostaticMonitor`.
    *   Track variables: `Energy` (simulated battery), `Integrity` (error rates), `Latency`.
    *   Generate "Drives" (Urgency scores) that influence the Global Workspace.
    *   **Verified**: `test_homeostasis.py` confirms hunger drive increases with energy decay.
*   [x] **1.3 LIDA-Lite Global Workspace (`global_workspace.py` refactor)**
    *   Update existing GWT to support "Coalitions".
    *   Implement the "Competition" formula: $Activation = Salience + Relevance + Affect$.
    *   **Verified**: `test_workspace.py` confirms "Hunger" drive forces "Eat" action to win over "Play".

## Phase 2: Multimodal Perception
*Goal: "See" and "Hear" using the same algebraic language.*

*   [x] **2.1 Audio Encoder (`voice_hd.py`)**
    *   Implement MFCC extraction (Custom SciPy, no librosa).
    *   Implement ID-Level Hypervector quantization.
    *   Implement N-gram permutation for temporal sequences.
    *   **Verified**: `test_voice_hd.py` confirms **100% classification accuracy** (30/30) on "Up" vs "Down" chirps using Nearest-Prototype classifier.
*   [x] **2.2 Sensor Fusion (`perception.py`)**
    *   Bind Audio vectors with Visual vectors (from CNN/SNN).
    *   Formula: $V_{Fusion} = MajorityVote( (V_{Audio} \oplus R_{Audio}) + (V_{Visual} \oplus R_{Visual}) )$
    *   **Verified**: `test_fusion.py` demonstrates "Cross-modal retrieval": Query the fused vector with "Sound" XOR "Role" recovers the "Label".

## Phase 3: Structural Plasticity
*Goal: A brain that evolves.*

*   [x] **3.1 Deep Rewiring Layer (`SparseLinear`)**
    *   Implement "Deep R" algorithm (STDP-like pruning/regrowth based on weight magnitude).
    *   **Verified**: `test_plasticity.py` confirms Mask Drift (connectivity evolves) while maintaining strict Target Sparsity (30%).
*   [x] **3.2 Plastic SNN Architecture (`plastic_snn.py`)**
    *   Implement `Neurogenesis` trigger (Plateau detection -> Add Neurons).
    *   **Verified**: System correctly detects loss plateau and physically resizes the hidden layer (20->30 neurons) on the fly.

## Phase 4: Agency & Curiosity
*Goal: Self-generated purpose.*

*   [x] **4.1 Active Inference Loop (`agency.py`)**
    *   Replace standard RL logic with `ActiveAgent` (Free Energy Minimization).
    *   Define the "Expected Free Energy" (G) function.
    *   **Verified**: `test_agency.py` confirms two behaviors:
        1.  **Pure Curiosity**: W/o reward, explored 39 tiles vs 17 random ($d=7.2$).
        2.  **Goal Seeking**: With reward, reliable navigation to food (G=-20.9).

## Phase 5: System Integration (The "Body")
*Goal: Connect the Brains to the Body.*

*   [x] **5.1 Snake Integration (`snake_headless.py`)**
    *   Replace random/heuristic logic with `ActiveAgent`.
    *   Map Game Grid -> Agent Belief Map ($Q[x,y]$).
    *   **Verified**: Agent plays autonomously. Switching modes based on Hunger.
*   [x] **5.2 Biological Integration (`homeostasis.py` link)**
    *   Connect `Energy` decay to Game Ticks.
    *   Connect `Integrity` damage to Wall Hits.
    *   **Verified**: "Hunger" drive rises during play (Step 94: Hunger 0.44), Agent eats, Energy restores, Drive falls (Step 95: Hunger 0.03). Agent switches to Curiosity.

---

## Current Status
**Current Phase**: Phase 5 Complete.
**Next Immediate Task**: Project Concluded. NSCK v2 is operational.
