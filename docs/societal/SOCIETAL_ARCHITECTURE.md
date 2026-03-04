# SOCIETAL_ARCHITECTURE.md: NSCK V5 "Human Society" Model

## Overview
The **Societal Knowledge World** in NSCK V5 represents information as a living, hierarchical society. Instead of static nodes and edges, data units (hypervectors) behave like autonomous citizens who form families, neighborhoods, and expansive city-scale structures.

## Hierarchical Structure
The system employs a recursive nesting strategy to scale from individual concepts to global knowledge continents:

1.  **People (Individual Hypervectors)**: The fundamental unit of knowledge. Each `LivingHyperVector` has internal state (energy, valence, stability) and interacts with neighbors via similarity-based bonding.
2.  **Neighborhoods (Districts)**: Tightly coupled clusters of People. Defined by a centroid (anchor) and a shared semantic context.
3.  **Towns/Cities (Domains)**: Emergent clusters of neighborhoods detected via **Topological Percolation**. This is where functional specialization occurs.
4.  **States/Countries (Super-Domains)**: Groupings of Domains based on higher-order spectral relationships.
5.  **Continents (Global World)**: The total set of all knowledge, unified by the `SocietalKnowledgeWorld`.

## Core Mechanisms

### 1. Autonomous Ingestion & Bonding
When a new concept is ingested, it is assigned a `LivingHyperVector`. It performs an HNSW search to find the nearest "social circle" and establishes bonds based on semantic similarity.

### 2. Percolation-Driven Emergence
The `PercolationMonitor` continuously watches the connectivity graph. When a cluster reaches a critical density (Phase Transition), it is upgraded into a formal **KnowledgeDomain**.
- **Town**: >10 concepts, >2 neighborhoods.
- **City**: >50 concepts, >5 neighborhoods.
- **State**: >200 concepts, multiple sub-domains.

### 3. Spectral Renormalization
To maintain efficiency at scale, the system uses **Spectral RG** to coarse-grain large clusters. It identifies dominant spectral modes (macrostates) and collapses complex local topologies into simplified "super-nodes" for higher-level reasoning.

### 4. Societal Context Routing
The `SocietalContextRouter` uses the hierarchical state to guide active inference. Reasoning is no longer a simple search; it is a traversal through the societal hierarchy:
- **Local Reasoning**: Querying within a specific Neighborhood.
- **Global Reasoning**: Querying across Domains via the Continental Backbone.

## Visual Representation
The V5 Dashboard provides a **Galaxy View** where:
- **Hubs** represent Domain centroids.
- **Satellites** represent individual concepts.
- **Links** represent triadic closures and spectral flows.
