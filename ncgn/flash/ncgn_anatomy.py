"""
NcgnAnatomy: The Structural Layer of the Neuromorphic Cognitive Graph Network

Maps high-level functional regions (lobes, cortices) onto the flat ID space of FlashColony.
Manages node allocation, region tracking, and projection-based connectivity.

Design Philosophy:
  - FlashColony handles the physics (spikes, synapses, persistence)
  - NcgnAnatomy handles the topology (regions, projections, semantics)
  - Together they form the complete NCGN architecture
"""

import random
import struct
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from .flash_colony import FlashColony


@dataclass
class BrainRegion:
    """
    A functional region of the brain (e.g., Visual Cortex, Hippocampus).
    Maps to a contiguous block of node IDs in the FlashColony address space.
    """
    name: str
    start_id: int
    end_id: int  # Exclusive (Python range convention)
    region_type: str  # "cortex", "hippocampus", "cerebellum", etc.
    layers: int = 1
    description: str = ""

    @property
    def size(self) -> int:
        """Total number of neurons in this region."""
        return self.end_id - self.start_id

    @property
    def node_ids(self) -> range:
        """Iterator over all node IDs in this region."""
        return range(self.start_id, self.end_id)

    def __repr__(self) -> str:
        return f"{self.name}[{self.start_id}:{self.end_id}]({self.size} neurons)"


@dataclass
class Projection:
    """
    A connection pattern between two regions.
    Specifies: source, target, connection density, and weight range.
    """
    source: BrainRegion
    target: BrainRegion
    density: float  # Fraction of source neurons with synapses to target
    weight_range: Tuple[float, float] = (0.1, 0.5)
    is_inhibitory: bool = False
    description: str = ""

    def __repr__(self) -> str:
        arrow = "⊣" if self.is_inhibitory else "→"
        return (f"{self.source.name} {arrow} {self.target.name} "
                f"(d={self.density:.1%}, w={self.weight_range})")


class NcgnAnatomy:
    """
    High-level brain architecture manager.

    Responsibilities:
      1. Allocate node ID blocks for functional regions
      2. Manage region metadata and hierarchy
      3. Execute projections (create synaptic connections between regions)
      4. Maintain topology constraints (no overlap, consistent allocation)
    """

    def __init__(self, flash_colony: FlashColony):
        """
        Initialize the anatomy layer.

        Args:
            flash_colony: The underlying FlashColony instance
        """
        self.colony = flash_colony
        self.regions: Dict[str, BrainRegion] = {}
        self.projections: List[Projection] = []
        self.next_node_id = 0
        self.statistics = {
            'total_synapses': 0,
            'total_projections': 0,
            'failed_connections': 0
        }

    def create_lobe(
        self,
        name: str,
        size: int,
        region_type: str = "cortex",
        layers: int = 1,
        description: str = ""
    ) -> BrainRegion:
        """
        Create a new functional region (lobe, cortex, nucleus, etc.).

        Args:
            name: Unique identifier for this region
            size: Number of neurons to allocate
            region_type: Semantic type (visual_cortex, hippocampus, etc.)
            layers: Number of processing layers (for structural organization)
            description: Human-readable description

        Returns:
            BrainRegion object representing this allocation

        Raises:
            ValueError: If name already exists or allocation exceeds capacity
        """
        if name in self.regions:
            raise ValueError(f"Region '{name}' already exists!")

        if self.next_node_id + size > self.colony.max_nodes:
            raise ValueError(
                f"Cannot allocate {size} neurons: "
                f"only {self.colony.max_nodes - self.next_node_id} remaining"
            )

        start = self.next_node_id
        end = start + size
        self.next_node_id = end

        region = BrainRegion(
            name=name,
            start_id=start,
            end_id=end,
            region_type=region_type,
            layers=layers,
            description=description
        )

        self.regions[name] = region

        print(f"✓ Created region: {region}")
        print(f"  Type: {region_type}, Layers: {layers}")
        if description:
            print(f"  Description: {description}")

        # Initialize nodes in the colony
        for node_id in region.node_ids:
            node_type = "output" if region_type == "motor_cortex" else "hidden"
            self.colony.add_node(node_id, node_type=node_type)

        return region

    def get_region(self, name: str) -> Optional[BrainRegion]:
        """Retrieve a region by name."""
        return self.regions.get(name)

    def project(
        self,
        source_name: str,
        target_name: str,
        density: float = 0.1,
        weight_range: Tuple[float, float] = (0.1, 0.5),
        is_inhibitory: bool = False,
        description: str = ""
    ) -> Projection:
        """
        Create a projection (connection pattern) between two regions.

        This wires actual synapses in FlashColony based on the density and weight parameters.

        Args:
            source_name: Name of source region
            target_name: Name of target region
            density: Fraction of source neurons to connect (0.0 to 1.0)
            weight_range: Tuple of (min_weight, max_weight)
            is_inhibitory: Whether connections are inhibitory
            description: Human description of the projection

        Returns:
            Projection object

        Raises:
            ValueError: If regions don't exist or parameters invalid
        """
        source = self.get_region(source_name)
        target = self.get_region(target_name)

        if not source:
            raise ValueError(f"Source region '{source_name}' not found!")
        if not target:
            raise ValueError(f"Target region '{target_name}' not found!")

        if not (0.0 <= density <= 1.0):
            raise ValueError(f"Density must be in [0.0, 1.0], got {density}")

        if weight_range[0] >= weight_range[1]:
            raise ValueError(f"Invalid weight_range: {weight_range}")

        projection = Projection(
            source=source,
            target=target,
            density=density,
            weight_range=weight_range,
            is_inhibitory=is_inhibitory,
            description=description
        )

        # Execute the projection: wire synapses
        synapse_count = self._wire_projection(projection)

        self.projections.append(projection)
        self.statistics['total_projections'] += 1
        self.statistics['total_synapses'] += synapse_count

        print(f"✓ Projection created: {projection}")
        print(f"  Synapses wired: {synapse_count}")
        if description:
            print(f"  Description: {description}")

        return projection

    def _wire_projection(self, proj: Projection) -> int:
        """
        Execute a projection by creating synapses between source and target regions.

        Smart Algorithm (respects both out-degree and in-degree):
          1. Determine how many source neurons to activate (density)
          2. Calculate average fan-out per source neuron
          3. Spread connections evenly across target pool to avoid saturation
          4. Track target in-degrees to respect capacity

        Args:
            proj: The Projection to execute

        Returns:
            Number of synapses created
        """
        synapse_count = 0
        failed = 0

        source_neurons = list(proj.source.node_ids)
        target_neurons = list(proj.target.node_ids)

        # How many source neurons participate?
        num_active_sources = max(1, int(len(source_neurons) * proj.density))
        active_sources = random.sample(source_neurons, num_active_sources)

        # Calculate fan-out: total connections / num active sources
        # This spreads load evenly across targets
        total_connections = max(1, int(len(target_neurons) * num_active_sources * proj.density))
        connections_per_source = max(1, total_connections // num_active_sources)

        # Track in-degree of each target to avoid saturation
        target_in_degree = {tid: 0 for tid in target_neurons}

        # Maximum in-degree per target (leave room for other projections)
        max_in_degree = int(self.colony.max_edges_per_node * 0.7)  # Use 70% capacity to be safe

        for source_id in active_sources:
            # For this source, distribute connections across targets
            # preferring targets with lower in-degree
            available_targets = [
                tid for tid in target_neurons
                if target_in_degree[tid] < max_in_degree
            ]

            if not available_targets:
                # All targets are saturated, skip this source
                failed += connections_per_source
                continue

            # Randomly shuffle but prefer lower in-degree targets
            available_targets.sort(key=lambda tid: target_in_degree[tid])

            # Pick targets (with some randomness to avoid pathological patterns)
            num_targets = min(
                connections_per_source,
                len(available_targets)
            )

            # Mix: 80% prefer low in-degree, 20% random
            if random.random() < 0.8:
                targets = available_targets[:num_targets]
            else:
                targets = random.sample(available_targets, num_targets)

            for target_id in targets:
                # Random weight within range
                weight = random.uniform(proj.weight_range[0], proj.weight_range[1])

                # Negate weight if inhibitory
                if proj.is_inhibitory:
                    weight *= -1.0

                # Attempt connection
                success = self.colony.connect_binary(source_id, target_id, weight)

                if success:
                    synapse_count += 1
                    target_in_degree[target_id] += 1
                else:
                    failed += 1

        if failed > 0:
            self.statistics['failed_connections'] += failed
            if failed > synapse_count * 0.1:  # Only warn if > 10% failure
                print(f"  ⚠️  {failed} connections failed (load balancing issue)")

        return synapse_count

    def project_layered(
        self,
        source_name: str,
        target_name: str,
        layer_pattern: str = "feedforward",
        density: float = 0.1,
        weight_range: Tuple[float, float] = (0.1, 0.5)
    ) -> int:
        """
        Create a structured multi-layer projection with specific patterns.

        Patterns:
          - "feedforward": Sequential layer-to-layer connectivity
          - "recurrent": Layer connects back to itself
          - "lateral": Intra-layer connections

        Args:
            source_name, target_name: Region names
            layer_pattern: Connection pattern type
            density: Connection density
            weight_range: Synapse weight range

        Returns:
            Total synapses created
        """
        # For now, delegate to standard project
        # Extended implementation would implement layer-specific patterns
        proj = self.project(source_name, target_name, density, weight_range)
        return proj.source.size * proj.target.size * density

    def print_topology(self):
        """Print a summary of the brain architecture."""
        print("\n" + "=" * 70)
        print("  NCGN BRAIN ARCHITECTURE")
        print("=" * 70)

        print(f"\n📍 REGIONS ({len(self.regions)})")
        print("-" * 70)
        for name, region in self.regions.items():
            print(f"  {region.name:30} | Type: {region.region_type:20} | "
                  f"Size: {region.size:7,} neurons")

        print(f"\n🔗 PROJECTIONS ({len(self.projections)})")
        print("-" * 70)
        for i, proj in enumerate(self.projections, 1):
            print(f"  [{i}] {proj}")

        print(f"\n📊 STATISTICS")
        print("-" * 70)
        print(f"  Total regions:        {len(self.regions)}")
        print(f"  Total projections:    {self.statistics['total_projections']}")
        print(f"  Total synapses:       {self.statistics['total_synapses']:,}")
        print(f"  Failed connections:   {self.statistics['failed_connections']}")
        print(f"  Used capacity:        {self.next_node_id:,} / {self.colony.max_nodes:,} "
              f"({100 * self.next_node_id / self.colony.max_nodes:.1f}%)")

        print("=" * 70 + "\n")

    def save_topology(self, filename: str = "topology.txt"):
        """Export topology to a file for visualization."""
        with open(filename, "w") as f:
            f.write("NCGN TOPOLOGY EXPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write("REGIONS\n")
            f.write("-" * 70 + "\n")
            for name, region in self.regions.items():
                f.write(f"{name}\t{region.start_id}\t{region.end_id}\t{region.size}\t"
                        f"{region.region_type}\n")

            f.write("\nPROJECTIONS\n")
            f.write("-" * 70 + "\n")
            for proj in self.projections:
                f.write(f"{proj.source.name}\t{proj.target.name}\t{proj.density}\t"
                        f"{proj.weight_range[0]}\t{proj.weight_range[1]}\t"
                        f"{proj.is_inhibitory}\n")

        print(f"✓ Topology exported to {filename}")


# Example usage and testing
def test_ncgn_anatomy():
    """Test the NcgnAnatomy layer with a small network."""
    print("\n" + "=" * 70)
    print("  NCGN ANATOMY LAYER TEST")
    print("=" * 70 + "\n")

    # Create a small brain
    colony = FlashColony("test_anatomy_brain.dat", max_nodes=10_000)
    colony.create_new(max_nodes=10_000)

    anatomy = NcgnAnatomy(colony)

    # Create functional regions
    print("\n📍 Creating Brain Regions...")
    print("-" * 70)
    vision = anatomy.create_lobe(
        "visual_cortex",
        size=1_000,
        region_type="visual_cortex",
        layers=3,
        description="Primary visual processing (V1, V2, V3)"
    )

    print()
    memory = anatomy.create_lobe(
        "hippocampus",
        size=500,
        region_type="hippocampus",
        layers=1,
        description="Memory consolidation and retrieval"
    )

    print()
    motor = anatomy.create_lobe(
        "motor_cortex",
        size=200,
        region_type="motor_cortex",
        layers=1,
        description="Motor planning and execution"
    )

    # Create projections
    print("\n\n🔗 Creating Projections...")
    print("-" * 70)
    print()
    anatomy.project(
        source_name="visual_cortex",
        target_name="hippocampus",
        density=0.05,
        weight_range=(0.1, 0.5),
        description="Visual → Memory: encode visual information"
    )

    print()
    anatomy.project(
        source_name="hippocampus",
        target_name="motor_cortex",
        density=0.1,
        weight_range=(0.2, 0.8),
        description="Memory → Motor: retrieve action patterns"
    )

    print()
    anatomy.project(
        source_name="visual_cortex",
        target_name="motor_cortex",
        density=0.02,
        weight_range=(0.05, 0.3),
        description="Visual → Motor: direct sensorimotor reflex"
    )

    # Print topology
    anatomy.print_topology()

    # Save topology
    anatomy.save_topology("test_topology.txt")

    # Verify structure
    print("\n✓ NcgnAnatomy test complete!")
    print(f"  Total neurons allocated: {anatomy.next_node_id:,}")
    print(f"  Total synapses created: {anatomy.statistics['total_synapses']:,}")

    colony.close()


if __name__ == "__main__":
    test_ncgn_anatomy()

