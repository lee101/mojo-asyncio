"""Parity of the Mojo timer heap against the stdlib `heapq` the loop itself uses.

Deadlines are drawn distinct so the reference has no ties to break: `asyncio`
orders `TimerHandle` on `_when` alone and documents ties as undefined, so a
tie would not be a disagreement with CPython. Ordering of exact ties is pinned
separately in `test_ties_break_by_insertion_order`.
"""

import heapq
import random

import numpy as np
import pytest

import mojo_asyncio as ma


def _distinct_deadlines(n, seed=0, lo=0.0, hi=100000.0):
    rng = random.Random(seed)
    return [float(v) for v in rng.sample(range(int(lo), int(hi)), n)]


def test_matches_heapq_on_a_single_batch():
    whens = _distinct_deadlines(500, seed=1)
    toks = list(range(500))
    h = ma.heap_from_pairs(zip(whens, toks))
    assert len(h) == 500
    assert h.peek() == min(whens)
    got = [t for _, _, t in h.entries()]
    assert got == ma.heapq_reference(whens, toks)
    assert got == [t for _, t in sorted(zip(whens, toks))]


def test_matches_heapq_for_sizes_that_break_simd_tails():
    """A dropped SIMD tail shows up only at sizes that are not a multiple of the
    vector width, so every awkward size is exercised."""
    for n in (1, 2, 3, 5, 7, 9, 15, 17, 31, 33, 63, 65, 127, 129, 1000, 1023):
        whens = _distinct_deadlines(n, seed=n)
        toks = list(range(n))
        h = ma.heap_from_pairs(zip(whens, toks))
        got = [t for _, _, t in h.entries()]
        assert got == ma.heapq_reference(whens, toks), f"size {n}"


def test_interleaved_push_and_pop_matches_heapq():
    """A sift that only ever runs on append, or a pop that skips the final
    sift-down, shows up here and nowhere else."""
    rng = random.Random(7)
    whens = _distinct_deadlines(900, seed=11)
    h = ma.TimerHeap()
    ref = []
    cursor = 0
    tok = 0
    for step in range(400):
        if ref and (len(ref) > 60 or rng.random() < 0.45):
            got = h.pop()
            ref_when, ref_tok = heapq.heappop(ref)
            assert got is not None
            assert got[0] == ref_when
            assert got[1] == ref_tok
        else:
            w = whens[cursor]
            cursor += 1
            h.push(w, tok)
            heapq.heappush(ref, (w, tok))
            tok += 1
        assert len(h) == len(ref)
    while ref:
        got = h.pop()
        ref_when, ref_tok = heapq.heappop(ref)
        assert (got[0], got[1]) == (ref_when, ref_tok)
    assert h.pop() is None
    assert len(h) == 0


def test_duplicate_deadlines_pop_in_insertion_order():
    """`asyncio` leaves exact ties undefined; this port pins them to FIFO."""
    h = ma.TimerHeap()
    toks = [11, 22, 33, 44, 55]
    h.extend([5.0] * 5, toks)
    assert [t for _, _, t in h.entries()] == toks
    popped = [h.pop()[1] for _ in range(5)]
    assert popped == toks


def test_negatives_and_large_magnitudes():
    whens = [-1e9, -5.0, -0.0, 0.0, 1e9, 1.0, -1.0, 1e-300]
    h = ma.heap_from_pairs(zip(whens, range(len(whens))))
    got = [t for _, _, t in h.entries()]
    assert got == ma.heapq_reference(whens, list(range(len(whens))))


def test_heapify_restores_invariant_after_manual_reordering():
    """`_run_once` heapifies a freshly built list; the same has to hold here."""
    rng = np.random.default_rng(3)
    n = 257
    whens = list(rng.permutation(np.arange(n, dtype=np.float64)))
    h = ma.heap_from_pairs(zip(whens, range(n)))
    rng.shuffle(h.when[: h.n])  # scribble on the array the kernel owns
    h.heapify()
    assert [t for _, _, t in h.entries()] == ma.heapq_reference(
        [float(x) for x in whens], list(range(n))
    )


def test_count_due_is_strict_and_finds_the_leftmost_prefix():
    whens = [float(v) for v in range(10)]
    h = ma.heap_from_pairs(zip(whens, range(10)))
    for end in np.arange(-0.5, 11.0, 0.5):
        expected = sum(1 for w in whens if w < end)
        assert h.count_due(float(end)) == expected, end
    # exactly on a key: the comparison is `<`, not `<=`
    assert h.count_due(3.0) == 3
    assert h.count_due(3.5) == 4


def test_purge_keeps_every_live_handle_and_drops_cancelled():
    n = 300
    whens = _distinct_deadlines(n, seed=5)
    h = ma.heap_from_pairs(zip(whens, range(n)))
    keep = [i for i in range(n) if i % 3 != 0]
    for i in range(0, n, 3):
        assert h.cancel(i)
    assert h.cancelled_count == len(range(0, n, 3))
    live = h.compact()
    assert live == len(keep)
    assert len(h) == len(keep)
    assert h.cancelled_count == 0
    assert [t for _, _, t in h.entries()] == sorted(keep, key=lambda i: whens[i])
    # a cancelled handle must not be cancellable twice
    assert not h.cancel(0)


def test_purge_preserves_relative_order_among_live_handles():
    """Compaction is a stable filter, so it cannot reorder two live timers."""
    n = 64
    whens = _distinct_deadlines(n, seed=9)
    h = ma.heap_from_pairs(zip(whens, range(n)))
    for i in range(n):
        if i % 2:
            h.cancel(i)
    before = [t for _, _, t in h.entries()]
    h.compact()
    assert [t for _, _, t in h.entries()] == [t for t in before if t % 2 == 0]


def test_heapify_restores_invariant_after_manual_reordering():
    """`_run_once` heapifies a freshly built list; the same has to hold here.

    All three parallel arrays are permuted together, so the (deadline, token)
    pairing survives the scribble and only the heap invariant is broken.
    """
    rng = np.random.default_rng(3)
    n = 257
    h = ma.heap_from_pairs(zip(rng.permutation(n).astype(np.float64), range(n)))
    p = rng.permutation(n)
    h.when[: h.n] = h.when[: h.n][p]
    h.seq[: h.n] = h.seq[: h.n][p]
    h.tok[: h.n] = h.tok[: h.n][p]
    scribbled_when = h.when[: h.n].copy()
    scribbled_tok = h.tok[: h.n].copy()
    h.heapify()
    assert [t for _, _, t in h.entries()] == ma.heapq_reference(
        [float(x) for x in scribbled_when], [int(x) for x in scribbled_tok]
    )


def test_purge_on_empty_and_all_cancelled():
    h = ma.TimerHeap()
    assert h.compact() == 0
    h.extend([1.0, 2.0, 3.0], [1, 2, 3])
    for i in (1, 2, 3):
        h.cancel(i)
    assert h.compact() == 0
    assert h.peek() is None


def test_growth_preserves_pending_entries():
    """The arrays are reallocated as the heap grows; a stale copy would drop
    already-pending timers."""
    h = ma.TimerHeap(capacity=1)
    rng = random.Random(4)
    whens = rng.sample(range(5000), 733)
    for i, w in enumerate(whens):
        h.push(float(w), i)
    assert len(h) == 733
    assert [t for _, _, t in h.entries()] == ma.heapq_reference(
        [float(w) for w in whens], list(range(733))
    )
    assert h.peek() == float(min(whens))


def test_push_rejects_mismatched_lengths():
    h = ma.TimerHeap()
    with pytest.raises(ValueError):
        h.extend([1.0, 2.0], [1])
    with pytest.raises(ValueError):
        h.extend([1.0, 2.0], [1, 2], [0])
