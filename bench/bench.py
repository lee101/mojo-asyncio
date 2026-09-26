"""Correctness-gated benchmark for mojo-asyncio.

Every case checks the Mojo result against the stdlib `heapq` reference before
timing, so a regression in the kernels shows up as a correctness failure rather
than a suspiciously good number. The baselines are the algorithms `asyncio`
itself runs, not a strawman: `call_at` is `heapq.heappush` in a Python loop, and
the compaction pass in `_run_once` is a list comprehension plus
`heapq.heapify`.
"""

from __future__ import annotations

import heapq
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_asyncio as ma  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _sample(n, seed):
    rng = random.Random(seed)
    return [float(v) for v in rng.sample(range(1, 10 * n), n)]


def bench_schedule_many(n: int = 200_000):
    """Registering a batch of timers.

    Reference: one `heapq.heappush` per timer, which is what `call_at` does.
    Port: the whole batch in a single kernel call.
    """
    whens = _sample(n, 1)
    toks = list(range(n))

    heap = ma.TimerHeap(capacity=n)
    heap.extend(whens, toks)
    got = [t for _, _, t in heap.entries()]
    expect = ma.heapq_reference(whens, toks)
    assert got == expect, "schedule_many mismatch"

    def ref():
        h = []
        for w, t in zip(whens, toks):
            heapq.heappush(h, (w, t))
        return h

    return f"schedule_many n={n}", _time(ref, 3), _time(
        lambda: ma.TimerHeap(capacity=n).extend(whens, toks)
    )


def bench_compact(n: int = 200_000):
    """The `_run_once` compaction pass with 60% of the handles cancelled.

    Reference is the literal CPython code: filter into a new list, then
    `heapq.heapify`. Port does the same stable filter and heapify in one pass.
    """
    whens = _sample(n, 2)
    live = [i for i in range(n) if i % 5 != 0]

    heap = ma.heap_from_pairs(zip(whens, range(n)))
    for i in range(n):
        if i % 5 == 0:
            heap.cancel(i)
    got = [t for _, _, t in heap.entries()]
    ref = ma.heapq_reference([whens[i] for i in live], live)
    assert got == ref, "compaction mismatch"

    cancelled = {i for i in range(n) if i % 5 == 0}
    flags = [1 if i in cancelled else 0 for i in range(n)]

    def ref_pass():
        new = [(w, i) for w, i, c in zip(whens, range(n), flags) if not c]
        heapq.heapify(new)
        return new

    import numpy as np
    arr = np.array(whens, dtype=np.float64)
    seq = np.arange(n, dtype=np.int64)
    tarr = np.arange(n, dtype=np.int64)
    canc = np.array(flags, dtype=np.int8)

    def mojo_pass():
        from mojo_asyncio import _lib
        return _lib.purge_cancelled(arr, seq, tarr, canc, n)

    return f"compact n={n}", _time(ref_pass, 3), _time(mojo_pass)


def bench_drain(n: int = 200_000, frac: float = 0.5):
    """One `_run_once` drain of the timers that are due.

    Reference: the `while scheduled and scheduled[0]._when < end_time: heappop`
    loop. Port: a single kernel call.
    """
    whens = _sample(n, 3)
    toks = list(range(n))
    due = int(n * frac)
    end_time = float(sorted(whens)[due - 1]) + 1e-9

    ref = [(w, t) for w, t in zip(whens, toks)]
    heapq.heapify(ref)
    popped_ref = []
    while ref and ref[0][0] < end_time:
        popped_ref.append(heapq.heappop(ref)[1])

    def ref_pass():
        h = [(w, t) for w, t in zip(whens, toks)]
        heapq.heapify(h)
        out = []
        while h and h[0][0] < end_time:
            out.append(heapq.heappop(h)[1])
        return out

    import numpy as np
    from mojo_asyncio import _lib
    built = ma.heap_from_pairs(zip(whens, toks))
    arr, seq, tarr = built.when.copy(), built.seq.copy(), built.tok.copy()
    canc = np.zeros(n, dtype=np.int8)

    def mojo_pass():
        # the kernel drains in place, so each call starts from the same heap
        return _lib.timer_tick(arr.copy(), seq.copy(), tarr.copy(), canc.copy(),
                               n, 0, end_time, 0.0, ma.MAXIMUM_SELECT_TIMEOUT)

    out = mojo_pass()
    assert [int(x) for x in out["popped_tok"]] == popped_ref, "drain mismatch"

    return f"drain n={n} due={due}", _time(ref_pass, 3), _time(mojo_pass)


def bench_single_push(n: int = 20_000):
    """One timer at a time, which is what `call_later` does per call.

    This is the case the port cannot win: a single heap push is a handful of
    instructions, and it now costs a ctypes crossing.
    """
    whens = _sample(n, 4)
    toks = list(range(n))

    def ref():
        h = []
        for w, t in zip(whens, toks):
            heapq.heappush(h, (w, t))
        return h

    def mojo():
        heap = ma.TimerHeap(capacity=1)
        for w, t in zip(whens, toks):
            heap.push(w, t)
        return heap

    got = [t for _, _, t in mojo().entries()]
    assert got == ma.heapq_reference(whens, toks), "single push mismatch"

    return f"push x{n} (one call each)", _time(ref, 3), _time(mojo)


def main():
    print(f"{'case':<28}{'reference':>12}{'mojo-asyncio':>16}{'ratio':>10}")
    print("-" * 68)
    for fn in (bench_schedule_many, bench_compact, bench_drain, bench_single_push):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<28}{ref*1e3:>10.2f}ms{got*1e3:>14.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
