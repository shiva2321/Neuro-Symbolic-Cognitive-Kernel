import mmap
import os
import struct
from typing import Set

# --- 🧬 BINARY SPECS ---
NODE_FMT = 'Iff f Q I I'
NODE_SIZE = struct.calcsize(NODE_FMT)  # 32 bytes

SYNAPSE_FMT = 'I f f f'
SYNAPSE_SIZE = struct.calcsize(SYNAPSE_FMT)  # 16 bytes

HEADER_FMT = '4s I Q Q'
HEADER_SIZE = 256

class FlashNode:
    """Proxy object for Interleaved Layout."""
    __slots__ = ('node_id', 'mem', 'node_offset', 'synapse_start_offset',
                 'potential', 'is_firing', 'last_spike_tick', 'refractory_counter',
                 'external_input', 'edge_count', 'node_type', 'threshold',
                 'refractory_period', 'edge_ptr')

    def __init__(self, node_id, mem, node_offset, synapse_start_offset):
        self.node_id = node_id
        self.mem = mem
        self.node_offset = node_offset
        self.synapse_start_offset = synapse_start_offset

        # Defaults
        self.potential = 0.0
        self.is_firing = False
        self.last_spike_tick = -999
        self.refractory_counter = 0
        self.external_input = 0.0
        self.edge_count = 0
        self.node_type = 1
        self.threshold = 1.0
        self.refractory_period = 3
        self.edge_ptr = 0

    def read_from_disk(self):
        (self.node_id, self.potential, self.threshold, avg_rate,
         _, self.edge_count, self.node_type) = struct.unpack_from(NODE_FMT, self.mem, self.node_offset)

    def write_to_disk(self):
        struct.pack_into(NODE_FMT, self.mem, self.node_offset,
                         self.node_id, self.potential, self.threshold, 0.0,
                         0, self.edge_count, self.node_type)

    def iter_synapses(self):
        curr = self.synapse_start_offset
        end = curr + (self.edge_count * SYNAPSE_SIZE)
        while curr < end:
            yield struct.unpack_from(SYNAPSE_FMT, self.mem, curr)
            curr += SYNAPSE_SIZE

class FlashColony:
    def __init__(self, filepath="flash_brain.dat", max_edges_per_node=250):
        self.filepath = filepath
        self.max_edges_per_node = max_edges_per_node

        self.f = None
        self.mem = None
        self.nodes = {}
        self.max_nodes = 0
        self.current_tick = 0

        # INTERLEAVED STRIDE: Node + Synapses
        self.node_stride = NODE_SIZE + (self.max_edges_per_node * SYNAPSE_SIZE)

        self.active_next: Set[int] = set()
        self.active_now: Set[int] = set()

        if not os.path.exists(self.filepath):
            self.create_new(initial_nodes=100)
        else:
            self.connect_to_file()

    def create_new(self, initial_nodes=100):
        if self.mem:
            self.mem.close()
            self.mem = None
        if self.f:
            self.f.close()
            self.f = None

        print(f"💾 Creating Dynamic Brain (Interleaved)...")
        total_size = HEADER_SIZE + (initial_nodes * self.node_stride)
        with open(self.filepath, "wb") as f:
            f.write(struct.pack(HEADER_FMT, b'NCGN', 2, 0, initial_nodes))
            f.seek(total_size - 1)
            f.write(b'\0')
        self.connect_to_file()

    def connect_to_file(self):
        if self.f: self.f.close()
        self.f = open(self.filepath, "r+b")
        self.mem = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_WRITE)
        _, _, _, self.max_nodes = struct.unpack(HEADER_FMT, self.mem[:struct.calcsize(HEADER_FMT)])

        # Rebind cached nodes to new memory
        for node in self.nodes.values():
            node.mem = self.mem

    def ensure_capacity(self, node_id):
        if node_id < self.max_nodes: return

        new_max = max(node_id + 1, self.max_nodes + 100)
        print(f"📈 Expanding Brain: {self.max_nodes} -> {new_max} neurons...")

        new_size = HEADER_SIZE + (new_max * self.node_stride)
        self.mem.close()
        self.f.close()

        with open(self.filepath, "r+b") as f:
            f.seek(0)
            f.write(struct.pack(HEADER_FMT, b'NCGN', 2, 0, new_max))
            f.seek(new_size - 1)
            f.write(b'\0')
        self.connect_to_file()

    def get_node(self, node_id):
        if node_id in self.nodes: return self.nodes[node_id]

        self.ensure_capacity(node_id)

        base_offset = HEADER_SIZE + (node_id * self.node_stride)
        synapse_offset = base_offset + NODE_SIZE

        node = FlashNode(node_id, self.mem, base_offset, synapse_offset)
        node.read_from_disk()

        # Initialize fresh slot
        if node.node_id == 0 and node_id != 0:
            node.node_id = node_id
            node.write_to_disk()

        self.nodes[node_id] = node
        return node

    def add_node(self, node_id, node_type="hidden", threshold=1.0):
        node = self.get_node(node_id)
        node.node_type = {"input":0, "hidden":1, "output":2}.get(node_type, 1)
        node.threshold = threshold
        node.write_to_disk()
        self.active_next.add(node_id)

    def connect_binary(self, source_id, target_id, weight):
        src = self.get_node(source_id)
        if src.edge_count >= self.max_edges_per_node:
            print(f"⚠️ Node {source_id} full!")
            return False

        offset = src.synapse_start_offset + (src.edge_count * SYNAPSE_SIZE)
        struct.pack_into(SYNAPSE_FMT, self.mem, offset, target_id, weight, 0.0, 1.0)

        src.edge_count += 1
        src.write_to_disk() # Critical: Update edge count on disk
        return True

    def update_synapse_weight(self, source_id, synapse_index, new_weight):
        src = self.nodes.get(source_id) or self.get_node(source_id)
        if not src or synapse_index >= src.edge_count: return False
        offset = src.synapse_start_offset + (synapse_index * SYNAPSE_SIZE)
        struct.pack_into('f', self.mem, offset + 4, new_weight)
        return True

    def get_synapse_weight(self, source_id, synapse_index):
        src = self.nodes.get(source_id) or self.get_node(source_id)
        if not src or synapse_index >= src.edge_count: return 0.0
        offset = src.synapse_start_offset + (synapse_index * SYNAPSE_SIZE)
        _, weight, _, _ = struct.unpack_from(SYNAPSE_FMT, self.mem, offset)
        return weight

    def set_input(self, node_id, val):
        n = self.get_node(node_id)
        n.external_input = val
        self.active_next.add(node_id)

    def reset(self):
        for node in self.nodes.values():
            node.potential = 0.0
            node.is_firing = False
            node.last_spike_tick = -999
            node.refractory_counter = 0
            node.external_input = 0.0
            node.write_to_disk()
        self.active_now.clear()
        self.active_next.clear()

    def step(self, global_dopamine=0.0, learning_rate=0.01):
        self.active_now, self.active_next = self.active_next, set()
        synaptic_accumulator = {}

        # Phase 1: Fire
        for nid in self.active_now:
            node = self.get_node(nid)
            # Refractory check
            if node.refractory_counter > 0:
                node.refractory_counter -= 1
                node.is_firing = False
                continue

            node.potential += node.external_input
            if node.potential >= node.threshold:
                node.is_firing = True
                node.potential = 0.0
                node.refractory_counter = node.refractory_period
            else:
                node.is_firing = False
                node.potential *= 0.9 # Decay

            node.external_input = 0.0
            if node.is_firing: self.active_next.add(nid)
            elif node.potential > 0.01: self.active_next.add(nid)

        # Phase 2: Propagate
        for nid in self.active_now:
            node = self.get_node(nid)
            if node.is_firing:
                for idx, (tid, w, t, p) in enumerate(node.iter_synapses()):
                    synaptic_accumulator[tid] = synaptic_accumulator.get(tid, 0.0) + w
                    self.active_next.add(tid)

        # Phase 3: Apply
        for tid, force in synaptic_accumulator.items():
            tgt = self.get_node(tid)
            tgt.potential += force
            tgt.write_to_disk()

        self.current_tick += 1

    def close(self):
        if self.mem: self.mem.close()
        if self.f: self.f.close()

    def sample_for_replay(self, k: int, priority_mode: str = "error") -> list:
        """
        Sample k items for replay based on priority mode.
        Modes: 'error', 'novelty', 'confidence_penalty'
        """
        # Placeholder: integrate with actual block scoring arrays
        # For now, return empty list or a mock structure
        return []
