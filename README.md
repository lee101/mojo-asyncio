# mojo-asyncio

`mojo-asyncio` ports the one part of CPython's `asyncio` that actually runs a
loop over an array: the pending-timer min-heap in
`asyncio.base_events.BaseEventLoop._scheduled` and the bookkeeping
`BaseEventLoop._run_once` performs on it every iteration.

**`asyncio` has no numeric core.** It is a selector loop over file descriptors
plus a coroutine scheduler; there is no arithmetic to accelerate and no array to
vectorise. Inventing one would be dishonest, so this port does not. What it does
port is real: `_run_once` walks a heap of `TimerHandle`s on every single loop
iteration, and a server with tens of thousands of scheduled timers pays for
that walk in Python.

The Python package is `mojo_asyncio`. It imports alongside the stdlib `asyncio`
(it is, after all, stdlib) and the tests drive a *real* `BaseEventLoop` and
compare against what the loop actually does.

```python
import mojo_asyncio as ma

h = ma.TimerHeap()
h.extend([10.0, 10.5, 12.0], tokens=[1, 2, 3])   # one kernel call
h.peek()                                          # 10.0
h.tick(now=9.0)                                   # -> {"timeout": 1.0, "drained": [], ...}
h.pop()                                           # (10.0, 1)
```

## Covered subset

| area | what is ported | kernel |
| --- | --- | --- |
| Timer heap | push (single and batch), pop, sift-up, sift-down, heapify | `asz_sift_up`, `asz_sift_down`, `asz_heapify`, `asz_schedule_many`, `asz_pop` |
| `_run_once` step | cancelled-head removal, select-timeout clamp, drain of everything with `_when < now + _clock_resolution` | `asz_timer_tick` |
| Compaction | `cancelled_count / sched_count > 0.5` predicate and the filter + heapify pass it guards | `asz_should_compact`, `asz_purge_cancelled` |
| Select timeout | the `[0, MAXIMUM_SELECT_TIMEOUT]` clamp, non-mutating, against a live loop | `asz_min_live` + `select_timeout` |
| Queries | number of entries due at a deadline (binary search over the leftmost prefix) | `asz_count_due` |

Not implemented, and not pretended at: the selector itself (`epoll`/`kqueue`),
`call_soon` and the `_ready` deque, coroutines, `Task` and its step counter,
`Future`, transports, the subprocess and signal machinery, `run_forever` /
`run_until_complete`, `wait_for` itself, and `staggered_race`. None of those is
array work; they are protocol and control flow, and they belong to the real
`asyncio`. `mojo_asyncio` is a component library, not a drop-in event loop: it
computes the answers, and you decide what to do with them.

## Tie ordering

`TimerHandle.__lt__` compares `_when` only, and `call_later`'s docstring says
which of two callbacks at the same time runs first is undefined. This port
orders on `(when, seq)` with a monotonic insertion counter. That is a *refinement*
of CPython's order: it never reorders two handles CPython would keep ordered, and
it makes the port's own output deterministic. Tests that compare against
`heapq` use distinct deadlines, and exact ties are pinned separately to FIFO.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-asyncio.so`. Set `PYTHONPATH=python`
when using the package outside a Pixi task.

## Performance

Best-of-N wall clock in one process. Every case checks its result against the
`heapq` reference *before* timing, so a kernel regression surfaces as a
correctness failure rather than a suspiciously good number. The baselines are
the algorithms `asyncio` itself runs: `call_at` is `heapq.heappush` in a Python
loop, and the `_run_once` compaction pass is a list comprehension followed by
`heapq.heapify`.

| case | reference | mojo-asyncio | result |
| --- | ---: | ---: | ---: |
| schedule_many n=200000 | 418.73 ms | 58.50 ms | 7.16x faster |
| compact n=200000 (60% cancelled) | 360.70 ms | 10.78 ms | 33.46x faster |
| drain n=200000, 100000 due | 985.96 ms | 94.45 ms | 10.44x faster |
| push x20000, one call each | 16.53 ms | 97.57 ms | **0.17x, i.e. 5.9x slower** |

The last row is the honest bad case and it is the one that matters for
`call_later`. A single heap push is a handful of instructions inside
`heapq.heappush`; going through a shared library adds a ctypes crossing, and no
amount of Mojo can win that. The port is built for the *batch* shapes a real
server produces — arming hundreds of timers, compacting a heap that has
accumulated cancellations, draining a deadline spike — and on those it is
between 7x and 33x faster because the Python-level per-element work disappears
into one call.

Reproduce with:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-asyncio.so`.

The heap is three parallel NumPy arrays owned by the Python layer — `when`
(`float64`), `seq` and `tok` (`int64`) — plus an `int8` cancellation flag, and
Mojo owns every loop that walks them. Buffers cross the C ABI as 64-bit
addresses and are reconstructed in Mojo as
`Pointer[T, AnyOrigin[mut=True]]`, which keeps the exported symbols
non-parametric. `@export` rejects parametric functions and an inferred pointer
origin would make a symbol parametric.

`asz_timer_tick` is the interesting one: it performs the whole `_run_once` timer
step in a single pass — drop cancelled handles off the head, compute the clamped
select timeout from the resulting head, then drain everything due — and returns
the drained `(when, seq, token)` triples along with the new heap state. That is
the shape a real loop wants, because each of those steps would otherwise be a
separate Python-level call per iteration.

`asz_count_due` exploits the heap property: the due entries are exactly the
leftmost prefix of the array, so the count is a binary search in `O(log n)`
rather than a scan.

Nothing here is floating-point arithmetic beyond `now + clock_res` and
`when - now`, so there is no FMA to worry about; the tests use exact equality
wherever the operation is exact, and a 1 ms tolerance only where they compare
against a live loop that re-reads the clock between the two measurements.

## Tests

```
tests/test_heap.py         parity with the stdlib heapq over randomised workloads
tests/test_timeout.py      the clamp, the compaction predicate, the drain rule
tests/test_loop_parity.py  a running BaseEventLoop, instrumented via its selector
```

`test_loop_parity.py` wraps `loop._selector.select`, runs a real loop, and
requires this port to predict every timeout the loop actually passed to the
selector, with and without cancelled timers in the heap. It also requires the
order in which the loop fires its timers to match the order this port drains
them, and does the same for the deadlines `asyncio.wait_for` arms.

## License

MIT
