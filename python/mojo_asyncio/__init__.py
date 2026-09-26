"""mojo-asyncio: Mojo kernels for the `asyncio` timer scheduler.

`asyncio` is an IO event loop with no arithmetic core, so this port does not
pretend to accelerate coroutines, the selector, or task scheduling. It ports
the one part that genuinely loops over an array: the pending-timer min-heap in
`asyncio.base_events.BaseEventLoop._scheduled` and the per-iteration bookkeeping
`_run_once` performs on it.
"""

from .scheduler import (
    CLOCK_RESOLUTION,
    MAXIMUM_SELECT_TIMEOUT,
    MIN_CANCELLED_TIMER_HANDLES_FRACTION,
    MIN_SCHEDULED_TIMER_HANDLES,
    TimerHeap,
    clamp_timeout,
    heap_from_loop,
    heap_from_pairs,
    heapq_reference,
    min_live,
    select_timeout,
)

__all__ = [
    "CLOCK_RESOLUTION",
    "MAXIMUM_SELECT_TIMEOUT",
    "MIN_CANCELLED_TIMER_HANDLES_FRACTION",
    "MIN_SCHEDULED_TIMER_HANDLES",
    "TimerHeap",
    "clamp_timeout",
    "heap_from_loop",
    "heap_from_pairs",
    "heapq_reference",
    "min_live",
    "select_timeout",
]
__version__ = "0.1.0"
