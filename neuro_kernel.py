"""
neuro_kernel.py
Fusion engine: event-driven physics + dynamic memory allocation.
"""

import mmap
import os
import struct
import time
from block_manager import BlockManager

# Binary specs
NODE_FMT = 'Iff f Q I I'
NODE_SIZE = struct.calcsize(NODE_FMT)

SYNAPSE_FMT = 'I f f f'
SYNAPSE_SIZE = struct.calcsize(SYNAPSE_FMT)

HEADER_FMT = '4s I Q Q'
HEADER_SIZE = 256


class NeuroKernel:
    def __init__(self, filepath="cortex.dat", max_nodes=1_000_000):
        self.filepath = filepath
        self.max_nodes = max_nodes
        self.nodes_cached = {}  # Active node cache

        if not os.path.exists(filepath):
            self.create_cortex(max_nodes)

        self.f = open(self.filepath, "r+b")
        self.mem = mmap.mmap(self.f.fileno(), 0)

        heap_start = HEADER_SIZE + (max_nodes * NODE_SIZE)
        self.allocator = BlockManager(self.mem, heap_start)
        self.node_capacities = {}

        print(f"NeuroKernel online. Cortex: {filepath}")
        print(f"Capacity: {max_nodes:,} neurons")
        print("Architecture: Event-Driven Push + Dynamic Vectors")

    def create_cortex(self, max_nodes):
        print(f"Formatting new cortex ({max_nodes:,} nodes)...")
        total_size = HEADER_SIZE + (max_nodes * NODE_SIZE) + (1024 * 1024 * 500)
        with open(self.filepath, "wb") as f:
            header = struct.pack(HEADER_FMT, b'NCGN', 1, 0, max_nodes)
            f.write(header)
            f.seek(total_size - 1)
            f.write(b'\0')

    def connect(self, source_id, target_id, weight):
        node_offset = HEADER_SIZE + (source_id * NODE_SIZE)
        raw_node = self.mem[node_offset: node_offset + NODE_SIZE]
        nid, pot, thr, rate, edge_ptr, edge_count, ntype = struct.unpack(NODE_FMT, raw_node)

        # Initialize missing metadata
        if nid == 0:
            nid = source_id
        if ntype == 0 and source_id != 0:
            ntype = 1  # default to hidden/output style
        if thr == 0 and ntype != 0:
            thr = 1.0

        current_capacity = self.node_capacities.get(source_id, edge_count)

        if edge_count >= current_capacity:
            new_capacity = max(4, current_capacity * 2)
            if edge_ptr == 0:
                new_ptr = self.allocator.malloc(new_capacity)
            else:
                new_ptr = self.allocator.realloc(edge_ptr, current_capacity, new_capacity)
            edge_ptr = new_ptr
            current_capacity = new_capacity
            self.node_capacities[source_id] = new_capacity

        write_offset = edge_ptr + (edge_count * SYNAPSE_SIZE)
        syn_data = struct.pack(SYNAPSE_FMT, target_id, weight, 0.0, 1.0)
        self.mem[write_offset: write_offset + SYNAPSE_SIZE] = syn_data

        updated_node = struct.pack(NODE_FMT, nid, pot, thr, rate, edge_ptr, edge_count + 1, ntype)
        self.mem[node_offset: node_offset + NODE_SIZE] = updated_node

        # Keep cache in sync if present
        cached = self.nodes_cached.get(source_id)
        if cached:
            cached.edge_ptr = edge_ptr
            cached.edge_count = edge_count + 1
            cached.threshold = thr
            cached.ntype = ntype
            cached.write_back()

    def set_input(self, node_id, value):
        self.get_node(node_id).external_input = value

    def get_node(self, node_id):
        if node_id not in self.nodes_cached:
            self.nodes_cached[node_id] = ActiveNode(node_id, self)
        return self.nodes_cached[node_id]

    def step(self, global_dopamine=0.0):
        active_nodes = list(self.nodes_cached.values())

        for node in active_nodes:
            node.update_potential()

        synaptic_currents = {}

        for node in active_nodes:
            if node.is_firing:
                nid, pot, thr, rate, edge_ptr, edge_count, ntype = struct.unpack(
                    NODE_FMT, self.mem[node.offset: node.offset + NODE_SIZE]
                )
                if edge_count > 0 and edge_ptr > 0:
                    block_size = edge_count * SYNAPSE_SIZE
                    raw_synapses = self.mem[edge_ptr: edge_ptr + block_size]
                    for i in range(edge_count):
                        off = i * SYNAPSE_SIZE
                        tid, w, trace, perm = struct.unpack(SYNAPSE_FMT, raw_synapses[off: off + SYNAPSE_SIZE])
                        synaptic_currents[tid] = synaptic_currents.get(tid, 0.0) + w
                        if global_dopamine > 0:
                            pass

        for tid, current in synaptic_currents.items():
            target_node = self.get_node(tid)
            target_node.potential += current
            target_node.write_back()

    def close(self):
        if self.mem:
            self.mem.close()
        if self.f:
            self.f.close()


class ActiveNode:
    def __init__(self, node_id, kernel):
        self.id = node_id
        self.kernel = kernel
        self.offset = HEADER_SIZE + (node_id * NODE_SIZE)

        raw = kernel.mem[self.offset: self.offset + NODE_SIZE]
        (self.nid, self.potential, self.threshold, self.rate,
         self.edge_ptr, self.edge_count, self.ntype) = struct.unpack(NODE_FMT, raw)

        if self.nid == 0:
            self.nid = node_id
        if self.ntype == 0 and node_id != 0:
            self.ntype = 1
        if self.threshold == 0 and self.ntype != 0:
            self.threshold = 1.0

        self.external_input = 0.0
        self.is_firing = False
        self.write_back()

    def update_potential(self):
        if self.ntype == 0:
            if self.external_input > 0:
                self.potential = self.external_input
                self.is_firing = True
            else:
                self.potential = 0.0
                self.is_firing = False
            self.external_input = 0.0
        else:
            self.potential *= 0.9
            self.potential += self.external_input
            if self.potential >= self.threshold:
                self.is_firing = True
                self.potential = 0.0
            else:
                self.is_firing = False
            self.external_input = 0.0
        self.write_back()

    def write_back(self):
        data = struct.pack(NODE_FMT, self.nid, self.potential, self.threshold, self.rate,
                           self.edge_ptr, self.edge_count, self.ntype)
        self.kernel.mem[self.offset: self.offset + NODE_SIZE] = data


if __name__ == "__main__":
    brain = NeuroKernel("cortex.dat", max_nodes=10_000)

    print("\n--- Building structure ---")
    brain.connect(0, 1, weight=1.5)
    print("Synapse created (0 -> 1)")

    node1 = brain.get_node(1)
    node1.threshold = 1.0
    node1.ntype = 2  # output
    node1.write_back()

    print("\n--- Running physics ---")
    print("Stimulating Node 0...")
    brain.set_input(0, 2.0)

    brain.step()
    print(f"Tick 1: Node 0 fired? {brain.get_node(0).is_firing}")
    print(f"        Node 1 potential: {brain.get_node(1).potential:.2f} (charge for next tick)")

    brain.step()
    print(f"Tick 2: Node 1 fired? {brain.get_node(1).is_firing} (expected True)")
    print(f"        Node 1 potential: {brain.get_node(1).potential:.2f}")

    brain.close()

