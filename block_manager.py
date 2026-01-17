import struct

# Simplified append-only block allocator for synapse storage.
class BlockManager:
    def __init__(self, mmap_obj, heap_start_offset):
        self.mem = mmap_obj
        self.heap_ptr = heap_start_offset
        self.synapse_fmt = 'I f f f'  # Target, Weight, Trace, Perm
        self.synapse_size = 16

    def malloc(self, num_slots):
        """Allocates a block for N synapses and returns the offset."""
        address = self.heap_ptr
        size_bytes = num_slots * self.synapse_size
        self.heap_ptr += size_bytes
        if self.heap_ptr > len(self.mem):
            raise MemoryError("Flash Brain is full! Expand the file size.")
        return address

    def realloc(self, old_ptr, old_count, new_count):
        """Moves data to a bigger block and returns the new pointer."""
        if new_count <= old_count:
            return old_ptr
        new_ptr = self.malloc(new_count)
        old_size_bytes = old_count * self.synapse_size
        data = self.mem[old_ptr: old_ptr + old_size_bytes]
        self.mem[new_ptr: new_ptr + old_size_bytes] = data
        return new_ptr

