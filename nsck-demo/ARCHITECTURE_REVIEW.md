# Architecture Review: NSCK v1.0

## System Overview
The **Neuro-Symbolic Cognitive Kernel (NSCK)** is a hybrid AI architecture designed to solve the "Tripartite Bottleneck" of modern AI: Energy, Grounding, and Catastrophic Forgetting. It implements the **Integrated Neuro-Symbolic Graph Architecture (INSGA)**.

## Component Hierarchy

### 1. The Kernel (Rust Core)
**Location:** `rust_vsa/src/lib.rs` -> `hypervec_rs`
*   **Role:** The "Reptilian Brain". Handles high-speed, low-power operations.
*   **Key Tech:** Vector Symbolic Architectures (VSA) / Hyperdimensional Computing.
*   **Performance:** Uses bitwise XOR/Population Count operations (SIMD optimized) instead of floating-point matrix multiplication.
*   **Functions:**
    *   `bind(A, B)`: Associates concepts (XOR).
    *   `bundle([A, B, C])`: Creates superposition (Majority Vote).
    *   `similarity(A, B)`: Hamming distance measurement.

### 2. The Cortex (Visual Sensory)
**Location:** `python/snn_qat.py` -> `TaskAwareSNN`
*   **Role:** The "Visual Cortex". Extracts features from raw pixel data.
*   **Architecture:** Convolutional Spiking Neural Network (CSNN).
*   **Innovation: Late Fusion**
    *   Shared Backbone: Extracts generalized edges/motion.
    *   Task Injection: A "Context Vector" is injected into the latent space to switch the network's behavior between "Snake" and "Pong" instantly.

### 3. The Executive (Python Orchestrator)
**Location:** `python/python_server.py`
*   **Role:** The "Frontal Lobe". Manages the loop between Perception, Logic, and Action.
*   **Key Logic: Active Inference**
    *   Calculates `Free Energy` (Surprise).
    *   If Surprise is HIGH -> Triggers "Curiosity" or "VSA Rescue" (System 2).
    *   If Surprise is LOW -> Allows Habitual SNN response (System 1).

## Data Flow Pipeline

1.  **Input:** Game Frame (10x10 Grid) -> received via ZMQ.
2.  **Transduction:** Frame converted to Spikes -> Processed by `TaskAwareSNN`.
3.  **Proposal:** SNN proposes an Action (e.g., "UP").
4.  **Veto Check (System 2):**
    *   Rust VSA Engine simulates "UP".
    *   Checks safeguards (e.g., "Will UP hit a wall?").
    *   **Decision:**
        *   Safe? -> Execute SNN Action.
        *   Unsafe? -> `[VSA RESCUE]` -> Override with VSA Safe Move.
5.  **Learning:**
    *   If SNN was wrong, generate an error signal (Dopamine).
    *   Update SNN weights via Hebbian-like plasticity.
    *   Update VSA Graph edges.

## Future Roadmap (v2.0)
*   **Federated Graph Distillation:** Merging knowledge graphs from multiple parallel agents.
*   **Sleep Replay:** Optimizing the "Sleep" cycle to consolidate memories more efficiently.
*   **Hardware Acceleration:** Porting the Rust Core to FPGA/Neuromorphic hardware.
