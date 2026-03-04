# SOCIETAL_MATH_PROOFS.md: Topological & Spectral Foundations

## 1. Spectral Renormalization Group (Spectral RG)

The NSCK V5 uses a spectral coarse-graining method to simplify the knowledge graph $G = (V, E)$.

### Laplacian Definition
We define the normalized graph Laplacian as:
$$L_{norm} = I - D^{-1/2} A D^{-1/2}$$
where $A$ is the adjacency (similarity) matrix and $D$ is the degree matrix.

### Coarse-Graining Transformation
The transformation $R$ maps the fine-grained nodes $V$ to super-nodes $V'$. This is achieved by projecting onto the $k$ slowest modes (smallest non-zero eigenvalues) of $L_{norm}$:
$$\Phi_k = [\phi_1, \phi_2, \dots, \phi_k]$$
The clustering in this spectral embedding space identifies functional macrostates that preserve the global diffusion properties of the original graph.

## 2. Higher-Order Hodge Laplacians

To capture triadic relationships (meaning combinations) beyond simple binary edges, we implement the combinatorial Hodge Laplacian.

### Boundary Operators
- $B_1$: Maps edges to nodes (0-simplices).
- $B_2$: Maps triangles (2-simplices) to edges (1-simplices).

### Hodge Laplacians
- **0-Laplacian (Node Level)**: $L_0 = B_1 B_1^T$ (The standard graph Laplacian).
- **1-Laplacian (Edge Level)**: $L_1 = B_1^T B_1 + B_2 B_2^T$.
  - $B_1^T B_1$: "Upward" flow from nodes.
  - $B_2^T B_2$: "Downward" flow from triangles (triadic closure).
- **2-Laplacian (Triangle Level)**: $L_2 = B_2^T B_2$.

Spectral analysis of $L_1$ and $L_2$ allows us to detect "latent flows" of information and the stability of triadic concepts.

## 3. Topological Health Monitoring (TDA)

We monitor the structural integrity of the knowledge world using Persistent Homology.

### Vietoris-Rips Complex
For a set of hypervectors $X$ and a distance threshold $\epsilon$, we construct the simplicial complex $VR(X, \epsilon)$.

### Betti Numbers
We compute the rank of the homology groups $H_k$:
- $\beta_0$: Number of connected components (Knowledge Islands).
- $\beta_1$: Number of 1-dimensional "holes" (Semantic Gaps or inconsistencies).

### Zipf Health Metric
The distribution of cluster sizes follows a Zipfian power law $P(s) \propto s^{-\alpha}$. A healthy world maintains $\alpha \approx 1.0$. Deviations indicate either fragmentation ($\alpha > 1.5$) or pathological centralization ($\alpha < 0.5$).
