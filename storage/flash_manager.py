import mmap
import struct

# --- 🧬 BINARY SPECS ---
# Node: ID (I), Potential (f), Threshold (f), AvgRate (f), EdgePtr (Q), EdgeCount (I), Type (I)
# Size: 4+4+4+4+8+4+4 = 32 bytes
NODE_FMT = 'Iff f Q I I'
NODE_SIZE = struct.calcsize(NODE_FMT)

# Synapse: TargetID (I), Weight (f), Trace (f), Perm (f)
# Size: 4+4+4+4 = 16 bytes
SYNAPSE_FMT = 'I f f f'
SYNAPSE_SIZE = struct.calcsize(SYNAPSE_FMT)

# Header: Magic(4s), Version(I), NodeCount(Q), MaxNodes(Q)
HEADER_FMT = '4s I Q Q'
HEADER_SIZE = 256 # Reserved space for future metadata

class FlashBrain:
    def __init__(self, filepath="brain_v1.dat"):
        self.filepath = filepath
        self.f = None
        self.mem = None
        self.max_nodes = 0

    def create_new(self, max_nodes=100_000):
        """Creates a blank brain file pre-allocated on disk."""
        print(f"💾 Allocating Flash Brain for {max_nodes:,} neurons...")

        # Calculate file size
        # Header + (Nodes * 32 bytes) + (Estimated Synapses Buffer)
        # We reserve huge space for synapses (e.g., 10x nodes)
        synapse_pool_size = max_nodes * 10 * SYNAPSE_SIZE
        total_size = HEADER_SIZE + (max_nodes * NODE_SIZE) + synapse_pool_size

        with open(self.filepath, "wb") as f:
            # 1. Write Header
            # Magic="NCGN", Version=1, NodeCount=0, MaxNodes=max_nodes
            header_data = struct.pack(HEADER_FMT, b'NCGN', 1, 0, max_nodes)
            f.write(header_data)

            # 2. Stretch file to full size (Sparse file usually)
            f.seek(total_size - 1)
            f.write(b'\0')

        print(f"✓ Created {self.filepath} ({total_size / 1024 / 1024:.2f} MB)")
        self.connect()

    def connect(self):
        """Connects to the existing brain file via mmap."""
        self.f = open(self.filepath, "r+b")
        # Map the entire file into memory
        self.mem = mmap.mmap(self.f.fileno(), 0)

        # Read Header
        magic, ver, count, max_n = struct.unpack(HEADER_FMT, self.mem[:struct.calcsize(HEADER_FMT)])
        if magic != b'NCGN':
            raise ValueError("Invalid Brain File Format!")

        self.max_nodes = max_n
        print(f"🔗 Connected to Flash Brain. Capacity: {self.max_nodes:,} nodes.")

    def write_node(self, node_id, potential, threshold, avg_rate, edge_count, node_type):
        """Writes a node directly to the binary slot."""
        if node_id >= self.max_nodes:
            raise ValueError(f"Node ID {node_id} out of bounds!")

        offset = HEADER_SIZE + (node_id * NODE_SIZE)

        # Pack data
        # EdgePtr is currently 0 (we will handle connectivity later)
        data = struct.pack(NODE_FMT, node_id, potential, threshold, avg_rate, 0, edge_count, node_type)

        # Write to memory map
        self.mem[offset:offset+NODE_SIZE] = data

    def read_node(self, node_id):
        """Reads a node from binary."""
        offset = HEADER_SIZE + (node_id * NODE_SIZE)
        data = self.mem[offset:offset+NODE_SIZE]
        return struct.unpack(NODE_FMT, data)

    def close(self):
        if self.mem: self.mem.close()
        if self.f: self.f.close()
        print("🔌 Brain disconnected.")

class FlashManager:
    def __init__(self, *args, **kwargs):
        self._enable_confidence_consolidation: bool = True

    def enable_confidence_consolidation(self, enabled: bool) -> None:
        """Enable/disable confidence-weighted consolidation."""
        self._enable_confidence_consolidation = bool(enabled)

    def consolidate(self, confidence: float, priority: float = 0.0) -> None:
        """
        Confidence-weighted consolidation of memory blocks.

        Args:
            confidence: Confidence score [0..1]
            priority: External priority [0..1]
        """
        if not getattr(self, '_enable_confidence_consolidation', False):
            return
        # Heuristic: consolidate when high confidence and moderate priority
        # Actual implementation should select blocks and persist snapshots
        threshold = 0.7
        if confidence >= threshold:
            # Placeholder hook: call underlying block manager / snapshot logic
            pass

# --- 🧪 TEST DRIVE ---
if __name__ == "__main__":
    brain = FlashBrain("my_first_brain.dat")

    # 1. Create a 1 Million Node Brain
    brain.create_new(max_nodes=1_000_000)

    # 2. Write some "Ghost" Neurons
    print("Writing Neuron #42 (The Answer)...")
    # ID=42, Pot=0.0, Thresh=1.0, Rate=0.25, Edges=0, Type=1(Exc)
    brain.write_node(42, 0.0, 1.0, 0.25, 0, 1)

    # 3. Read it back
    node_data = brain.read_node(42)
    print(f"📖 Read Back Node #42: {node_data}")

    # 4. Verify Persistence
    brain.close()
    print("Test Complete. File 'my_first_brain.dat' is ready.")
