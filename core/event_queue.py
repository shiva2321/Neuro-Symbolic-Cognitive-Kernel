"""
EventQueue: Spike event ordering and temporal causality.

Maintains a time-ordered queue of spike events to enforce:
- Event causality (no retrograde transmission)
- Deterministic ordering (same sequence each run if seeded)
- Minimal memory overhead (O(active_spikes), not O(all_neurons))
"""

import heapq
from typing import List, Tuple, Optional
from .spike import Spike


class EventQueue:
    """
    Priority queue for spike events ordered by timestamp.

    Guarantees:
    - Spikes are processed in strict temporal order
    - No spike can be injected with timestamp < current_time
    - Determinism: seeded RNG (if used) produces identical orderings

    Public Interface:
    - inject(spike) -> None
    - pop_next() -> Optional[Spike]
    - current_time -> int
    - is_empty() -> bool
    """

    def __init__(self):
        """Initialize empty event queue."""
        self._queue: List[Tuple[int, int, Spike]] = []  # (timestamp, sequence, spike)
        self._sequence_counter = 0  # For stable ordering of same-time events
        self._current_time = 0
        self._events_processed = 0

    def inject(self, spike: Spike) -> None:
        """
        Add a spike to the queue.

        Enforces causality: spike timestamp must not be in the past.

        Args:
            spike: The spike event to queue

        Raises:
            ValueError: If spike timestamp < current_time (causality violation)
        """
        if spike.timestamp < self._current_time:
            raise ValueError(
                f"Causality violation: spike from past "
                f"(t={spike.timestamp} < now={self._current_time})"
            )

        # Use sequence counter for stable ordering of simultaneous events
        heapq.heappush(self._queue, (spike.timestamp, self._sequence_counter, spike))
        self._sequence_counter += 1

    def pop_next(self) -> Optional[Spike]:
        """
        Extract the next spike from the queue.

        Advances the logical clock to the spike's timestamp.

        Returns:
            Next spike in temporal order, or None if queue is empty
        """
        if not self._queue:
            return None

        timestamp, _, spike = heapq.heappop(self._queue)
        self._current_time = max(self._current_time, timestamp)
        self._events_processed += 1
        return spike

    @property
    def current_time(self) -> int:
        """Get the current logical time (timestamp of last processed spike)."""
        return self._current_time

    def is_empty(self) -> bool:
        """Check if queue has pending events."""
        return len(self._queue) == 0

    def size(self) -> int:
        """Get number of pending events in queue."""
        return len(self._queue)

    def reset(self) -> None:
        """Clear queue and reset time counter."""
        self._queue.clear()
        self._current_time = 0
        self._sequence_counter = 0
        self._events_processed = 0

    def __repr__(self):
        return f"EventQueue(size={self.size()}, t={self.current_time})"
