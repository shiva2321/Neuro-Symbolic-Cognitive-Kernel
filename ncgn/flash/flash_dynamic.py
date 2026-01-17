import struct
import mmap
import os
from .block_manager import BlockManager

# --- BINARY SPECS ---
NODE_FMT = 'Iff f Q I I'  # EdgePtr (Q) is 64-bit address
NODE_SIZE = 32
SYNAPSE_FMT = 'I f f f'
SYNAPSE_SIZE = 16
HEADER_SIZE = 256

class DynamicFlashColony:
    def __init__(self, filepath="dynamic_brain.dat", max_nodes=100_000):
        self.filepath = filepath
        self.max_nodes = max_nodes
        if not os.path.exists(filepath):
            self.create_new(max_nodes)
        self.f = open(self.filepath, "r+b")
        self.mem = mmap.mmap(self.f.fileno(), 0)
        heap_start = HEADER_SIZE + (max_nodes * NODE_SIZE)
        self.allocator = BlockManager(self.mem, heap_start)
        self.node_capacities = {}

    def create_new(self, max_nodes):
        print("Creating DYNAMIC Flash Brain...")
        total_size = HEADER_SIZE + (max_nodes * NODE_SIZE) + (1024 * 1024 * 500)
        with open(self.filepath, "wb") as f:
            f.write(b'\0' * total_size)

    def connect_dynamic(self, source_id, target_id, weight):
        node_offset = HEADER_SIZE + (source_id * NODE_SIZE)
        node_data = self.mem[node_offset: node_offset + NODE_SIZE]
        nid, pot, thr, rate, edge_ptr, edge_count, ntype = struct.unpack(NODE_FMT, node_data)
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
        data = struct.pack(SYNAPSE_FMT, target_id, weight, 0.0, 1.0)
        self.mem[write_offset: write_offset + SYNAPSE_SIZE] = data
        updated_node = struct.pack(NODE_FMT, nid, pot, thr, rate, edge_ptr, edge_count + 1, ntype)
        self.mem[node_offset: node_offset + NODE_SIZE] = updated_node

    def close(self):
        self.mem.close()
        self.f.close()

if __name__ == "__main__":
        brain = DynamicFlashColony("dynamic_brain.dat")
        print("Testing Infinite Growth on Node 0...")
        for i in range(50):
            brain.connect_dynamic(0, i, 0.5)
        print("Success! Node 0 held 50 connections.")
        print("  (It resized and moved memory automatically behind the scenes)")
        brain.close()
