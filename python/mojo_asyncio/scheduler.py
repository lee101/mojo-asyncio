"""Timer-scheduler state built on the Mojo kernels.

`asyncio` has no numeric core: it is a selector loop over file descriptors. The
only array-scale work it does is on `BaseEventLoop._scheduled`, the min-heap of
pending `TimerHandle`s keyed on `_when`. This module owns that heap as three
parallel NumPy arrays plus a cancellation byte array, and Mojo owns the sift,
heapify, filter and drain loops that run over it.

The semantics implemented here are those of `asyncio.base_events.BaseEventLoop`:

* key order is `(when, seq)`; `asyncio` compares `_when` alone and calls ties
  undefined, so the monotonic insertion counter is a total-order refinement,
* `select_timeout` is `scheduled[0]._when - time()` clamped by
  `> MAXIMUM_SELECT_TIMEOUT` first and `< 0` second,
* a tick drops cancelled handles off the head, then drains every handle with
  `_when < now + _clock_resolution`.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from asyncio import base_events
from typing import Iterable, Sequence

import numpy as np

from . import _lib

MAXIMUM_SELECT_TIMEOUT = 24 * 3600
MIN_SCHEDULED_TIMER_HANDLES = 100
MIN_CANCELLED_TIMER_HANDLES_FRACTION = 0.5

# `BaseEventLoop.__init__` sets this from the monotonic clock, verbatim.
CLOCK_RESOLUTION = float(time.get_clock_info("monotonic").resolution)


def clamp_timeout(when: float, now: float,
                  max_timeout: float = MAXIMUM_SELECT_TIMEOUT) -> float:
    timeout = when - now
    if timeout > max_timeout:
        return max_timeout
    if timeout < 0.0:
        return 0.0
    return timeout


class TimerHeap:
    """A min-heap of pending timers keyed on `(when, seq)`."""

    __slots__ = ("when", "seq", "tok", "canc", "n", "_seq", "cancelled_count",
                 "_aw", "_as", "_at", "_ac")

    def __init__(self, capacity: int = 16):
        cap = max(int(capacity), 1)
        self.when = np.empty(cap, dtype=np.float64)
        self.seq = np.empty(cap, dtype=np.int64)
        self.tok = np.empty(cap, dtype=np.int64)
        self.canc = np.zeros(cap, dtype=np.int8)
        self.n = 0
        self._seq = 0
        self.cancelled_count = 0
        self._refresh()

    def _refresh(self):
        """Cache the buffer addresses; `.ctypes.data` is not free and a single
        `call_later` has to stay under a few microseconds."""
        self._aw = self.when.ctypes.data
        self._as = self.seq.ctypes.data
        self._at = self.tok.ctypes.data
        self._ac = self.canc.ctypes.data

    def __len__(self) -> int:
        return self.n

    def _grow(self, extra: int):
        need = self.n + extra
        if self.when.size >= need:
            return
        cap = max(need, self.when.size * 2, 1)
        for name, dtype in (
            ("when", np.float64), ("seq", np.int64),
            ("tok", np.int64), ("canc", np.int8),
        ):
            old = getattr(self, name)
            new = np.zeros(cap, dtype=dtype)
            new[: self.n] = old[: self.n]
            setattr(self, name, new)
        self._refresh()

    def push(self, when: float, token: int) -> int:
        """Register one timer, as `loop.call_at(when, ...)` does.

        The single-timer path writes the key and runs the sift directly, so a
        `call_later` costs one kernel call rather than a batch of NumPy
        bookkeeping arrays.
        """
        self._grow(1)
        n = self.n
        self.when[n] = when
        self.seq[n] = self._seq
        self.tok[n] = token
        self.canc[n] = 0
        self._seq += 1
        _lib.lib.asz_sift_up(self._aw, self._as, self._at, n + 1, n)
        self.n = n + 1
        return self.n

    def extend(self, whens: Sequence[float], tokens: Sequence[int],
               cancelled: Sequence[int] | None = None) -> int:
        """Register a batch of timers in a single kernel call."""
        w = np.ascontiguousarray(whens, dtype=np.float64)
        t = np.ascontiguousarray(tokens, dtype=np.int64)
        if w.size != t.size:
            raise ValueError("whens and tokens must have the same length")
        seq = np.arange(self._seq, self._seq + w.size, dtype=np.int64)
        c = (np.zeros(w.size, dtype=np.int8) if cancelled is None
             else np.ascontiguousarray(cancelled, dtype=np.int8))
        if c.size != w.size:
            raise ValueError("cancelled must match the number of timers")
        self._seq += w.size
        self.cancelled_count += int(np.count_nonzero(c))
        self._grow(w.size)
        n, self.when, self.seq, self.tok, self.canc = _lib.schedule_many(
            self.n, self.when, self.seq, self.tok, self.canc, w, seq, t, c
        )
        self._refresh()
        self.n = n
        return n

    def cancel(self, token: int) -> bool:
        """Mark a pending timer cancelled, leaving it in the heap as asyncio does."""
        hits = np.flatnonzero((self.tok[: self.n] == token) & (self.canc[: self.n] == 0))
        if hits.size == 0:
            return False
        self.canc[hits] = 1
        self.cancelled_count += int(hits.size)
        return True

    def pop(self) -> tuple[float, int] | None:
        """Remove and return the earliest `(when, token)`, or None when empty."""
        if self.n == 0:
            return None
        of = np.empty(1, dtype=np.float64)
        oi = np.empty(2, dtype=np.int64)
        self.n = _lib.pop(self.when, self.seq, self.tok, self.canc, self.n, of, oi)
        return float(of[0]), int(oi[1])

    def peek(self) -> float | None:
        return float(self.when[0]) if self.n else None

    def heapify(self) -> None:
        """Re-establish the heap invariant over the live prefix, in Mojo."""
        _lib.heapify(self.when, self.seq, self.tok, self.n)

    def entries(self) -> list[tuple[float, int, int]]:
        """All live entries as `(when, seq, token)`, in pop order."""
        w = self.when[: self.n][self.canc[: self.n] == 0]
        s = self.seq[: self.n][self.canc[: self.n] == 0]
        t = self.tok[: self.n][self.canc[: self.n] == 0]
        order = np.lexsort((t, s, w))
        return [(float(w[i]), int(s[i]), int(t[i])) for i in order]

    def count_due(self, end_time: float) -> int:
        """How many heap entries have `when < end_time`, via binary search."""
        return _lib.count_due(self.when, self.n, end_time)

    def should_compact(self, n: int | None = None,
                       cancelled: int | None = None) -> bool:
        """The `_run_once` cancelled-handle compaction predicate."""
        n = self.n if n is None else n
        c = self.cancelled_count if cancelled is None else cancelled
        return _lib.should_compact(n, c, MIN_SCHEDULED_TIMER_HANDLES,
                                   MIN_CANCELLED_TIMER_HANDLES_FRACTION)

    def compact(self) -> int:
        """Drop every cancelled handle and heapify, as `_run_once` does.

        Returns the new size. Handles that `cancel()` already removed from the
        head are gone; this is the O(n) pass that fires once the cancelled
        fraction passes the threshold.
        """
        if self.n == 0:
            return 0
        w, s, t = _lib.purge_cancelled(
            self.when, self.seq, self.tok, self.canc, self.n
        )
        live = w.size
        if self.when.size < max(live, 1):
            cap = max(live, 1)
            self.when = np.empty(cap, dtype=np.float64)
            self.seq = np.empty(cap, dtype=np.int64)
            self.tok = np.empty(cap, dtype=np.int64)
            self.canc = np.zeros(cap, dtype=np.int8)
        self.when[:live] = w
        self.seq[:live] = s
        self.tok[:live] = t
        self.canc[:live] = 0
        self._refresh()
        self.n = live
        self.cancelled_count = 0
        return live

    def tick(self, now: float, clock_res: float = None,
             max_timeout: float = MAXIMUM_SELECT_TIMEOUT) -> dict:
        """One `_run_once` timer step. See `asz_timer_tick`.

        Returns a dict with the clamped select `timeout` (`None` for an empty
        heap), the `(when, seq, token)` triples drained this step, and the heap
        state afterwards.
        """
        res = _lib.timer_tick(
            self.when, self.seq, self.tok, self.canc, self.n,
            self.cancelled_count, now,
            CLOCK_RESOLUTION if clock_res is None else clock_res,
            max_timeout,
        )
        self.n = res["size"]
        self.cancelled_count = res["cancelled_count"]
        drained = [
            (float(res["popped_when"][i]), int(res["popped_seq"][i]),
             int(res["popped_tok"][i]))
            for i in range(res["pops"])
        ]
        return {
            "timeout": None if res["timeout"] < 0.0 else res["timeout"],
            "drained": drained,
            "size": self.n,
            "cancelled_count": self.cancelled_count,
            "head_tok": res["head_tok"],
            "head_when": None if res["head_tok"] < 0 else res["head_when"],
        }


def heap_from_pairs(pairs: Iterable[tuple[float, int]]) -> TimerHeap:
    """Build a heap from `(when, token)` pairs in one kernel call."""
    items = list(pairs)
    h = TimerHeap(capacity=max(len(items), 1))
    h.extend([w for w, _ in items], [t for _, t in items])
    return h


def heap_from_loop(loop: asyncio.AbstractEventLoop) -> TimerHeap:
    """Snapshot a real event loop's pending timers into a `TimerHeap`."""
    sched = list(loop._scheduled)
    h = TimerHeap(capacity=max(len(sched), 1))
    h.extend([hd._when for hd in sched], [id(hd) for hd in sched],
             [1 if hd._cancelled else 0 for hd in sched])
    return h


def min_live(loop: asyncio.AbstractEventLoop) -> float | None:
    """The earliest live deadline in a real loop, computed by the Mojo kernel.

    `_run_once` reaches this handle by popping cancelled entries off the head,
    so the effective head is the minimum key among the live handles.
    """
    h = heap_from_loop(loop)
    v = _lib.min_live(h.when, h.canc, h.n)
    return None if v < 0.0 else float(v)


def select_timeout(loop: asyncio.AbstractEventLoop, now: float | None = None,
                   ready: bool | None = None,
                   max_timeout: float = MAXIMUM_SELECT_TIMEOUT) -> float | None:
    """The timeout `_run_once` will hand to `selector.select`, computed by Mojo.

    This is a non-mutating replica of the branch in `BaseEventLoop._run_once`
    that produces `timeout`, and it is what the parity tests compare against the
    timeouts a real event loop actually passes to its selector.
    """
    if now is None:
        now = loop.time()
    if ready is None:
        ready = bool(loop._ready) or loop._stopping
    if ready:
        return 0.0
    if not loop._scheduled:
        return None
    when = min_live(loop)
    if when is None:
        return None
    return clamp_timeout(when, now, max_timeout)


def heapq_reference(whens: Sequence[float], tokens: Sequence[int]) -> list[int]:
    """The same pop order, computed with the stdlib `heapq` the loop itself uses.

    `asyncio` orders on `_when` alone, so this is the authoritative reference for
    distinct deadlines. Ties are documented as undefined there; a heap that
    orders ties by insertion is a valid refinement, and `entries()` is checked
    against the reference only where deadlines differ.
    """
    items = list(zip(whens, tokens))
    heapq.heapify(items)
    out = []
    while items:
        out.append(heapq.heappop(items)[1])
    return out
