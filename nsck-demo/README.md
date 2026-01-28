# Neuro-Symbolic Cognitive Kernel (NSCK) v1.0

> **Energy-Efficient, Continuous Autonomous Cognition**

The NSCK is a hybrid AI system implementing the **Integrated Neuro-Symbolic Graph Architecture (INSGA)**. It fuses a neuromorphic sensory frontend (System 1) with a robust, Rust-accelerated symbolic logic core (System 2) to achieve highly efficient, safe, and verifiable autonomous behavior.

## Key Features
*   **System 1 (Fast):** Convolutional Spiking Neural Network (`snn_qat.py`) for millisecond-latency perception.
*   **System 2 (Slow):** Vector Symbolic Architecture (`hypervec_rs`) for logical reasoning and safety checks.
*   **Alphanumeric Support:** Expanded core associative layer to 62 classes (A-Z, a-z, 0-9) using **Weight Surgery** to preserve existing training.
*   **Multi-Modal Letter Lab:** Support for both **Handwritten (EMNIST)** and **Typed (Synthetic)** autonomous character training.
*   **Active Inference:** Unified objective function minimizing "Surprise" (Free Energy).
*   **Transfer Learning:** Demonstrated capability to learn multiple tasks (Snake, Pong) without catastrophic forgetting using "Late Fusion".

## Documentation
*   [**Deep Dive Analysis & Experiment Proofs**](./DEEP_DIVE_ANALYSIS.md): Detailed breakdown of the 10-minute validation run, including video and log analysis.
*   [**Architecture Review**](./ARCHITECTURE_REVIEW.md): Technical mapping of code components to the conceptual layers (Sensory, Holographic, Associative, Executive).
*   [**Technical Report**](./NSCK_Technical_Report.md): Mathematical foundations.

## Installation

### Prerequisites
*   Python 3.11+
*   Rust (1.70+)
*   Visual Studio Build Tools (Windows)

### Setup
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/shiva2321/Node_network.git
    cd Node_network/nsck-demo
    ```
2.  **Build Rust Core:**
    ```bash
    cd rust_vsa
    maturin develop --release
    cd ..
    ```
3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1.  **Start the Dashboard:**
    ```bash
    python python/dashboard.py
    ```
2.  **Launch Server:** Click "START Server" on the dashboard.
3.  **Start Games:** Click "Start Snake" or "Start Pong".

## License
MIT License
