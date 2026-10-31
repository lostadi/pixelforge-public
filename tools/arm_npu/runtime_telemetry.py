"""Bounded recent invocation/eviction details with explicit retention evidence.

Runtime operation totals remain separate exact counters. These tails are
diagnostic samples, never complete histories. Compilation/load records retain
their existing full-history semantics because validation consumes their graph
inventory and compile/load counts.
"""
from collections import deque
from collections.abc import Sequence

DETAIL_TRACE_LIMIT = 1024


class TraceTail(Sequence):
    """Keep recent details without retaining one event for every model operation."""

    def __init__(self, limit=DETAIL_TRACE_LIMIT):
        if type(limit) is not int or limit < 1:
            raise ValueError("Detailed trace limit must be a positive integer")
        self.limit = limit
        self.total = 0
        self._entries = deque(maxlen=limit)

    def append(self, event):
        self._entries.append(event)
        self.total += 1

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self._entries)[index]
        return self._entries[index]

    def snapshot(self):
        return list(self._entries)

    def evidence(self):
        return {"policy": "bounded-most-recent", "limit": self.limit,
                "total": self.total, "retained": len(self),
                "dropped": self.total - len(self)}
