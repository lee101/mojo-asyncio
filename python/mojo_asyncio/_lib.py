"""ctypes bridge to the compiled Mojo timer kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so address argtypes must stay `c_int64`; `c_int` truncates them and
segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-asyncio.so"

_I = ctypes.c_int64
_F = ctypes.c_double


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.asz_sift_up.restype = _I
    lib.asz_sift_up.argtypes = [_I, _I, _I, _I, _I]
    lib.asz_sift_down.restype = _I
    lib.asz_sift_down.argtypes = [_I, _I, _I, _I, _I]
    lib.asz_heapify.restype = _I
    lib.asz_heapify.argtypes = [_I, _I, _I, _I]
    lib.asz_count_due.restype = _I
    lib.asz_count_due.argtypes = [_I, _I, _F]
    lib.asz_purge_cancelled.restype = _I
    lib.asz_purge_cancelled.argtypes = [_I, _I, _I, _I, _I, _I, _I, _I]
    lib.asz_schedule_many.restype = _I
    lib.asz_schedule_many.argtypes = [_I, _I, _I, _I, _I, _I, _I, _I, _I, _I]
    lib.asz_should_compact.restype = _I
    lib.asz_should_compact.argtypes = [_I, _I, _I, _F]
    lib.asz_timer_tick.restype = _I
    lib.asz_timer_tick.argtypes = [
        _I, _I, _I, _I, _I, _I, _F, _F, _F, _I, _I, _I, _I, _I,
    ]
    lib.asz_pop.restype = _I
    lib.asz_pop.argtypes = [_I, _I, _I, _I, _I, _I, _I]
    lib.asz_min_live.restype = _F
    lib.asz_min_live.argtypes = [_I, _I, _I]
    return lib


lib = _load()


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def sift_up(when, seq, tok, n, i):
    return int(lib.asz_sift_up(addr(when), addr(seq), addr(tok), int(n), int(i)))


def sift_down(when, seq, tok, n, i):
    return int(lib.asz_sift_down(addr(when), addr(seq), addr(tok), int(n), int(i)))


def heapify(when, seq, tok, n):
    return int(lib.asz_heapify(addr(when), addr(seq), addr(tok), int(n)))


def count_due(when, n, end_time):
    return int(lib.asz_count_due(addr(when), int(n), float(end_time)))


def purge_cancelled(when, seq, tok, canc, n):
    """Return `(live_when, live_seq, live_tok)` after dropping cancelled handles."""
    n = int(n)
    ow = np.empty(n, dtype=np.float64)
    os_ = np.empty(n, dtype=np.int64)
    ot = np.empty(n, dtype=np.int64)
    live = int(
        lib.asz_purge_cancelled(
            addr(when), addr(seq), addr(tok), addr(canc), n,
            addr(ow), addr(os_), addr(ot),
        )
    )
    return ow[:live].copy(), os_[:live].copy(), ot[:live].copy()


def schedule_many(n, when, seq, tok, canc, in_when, in_seq, in_tok, in_canc):
    """Append a batch of timers to a heap that currently holds `n` entries.

    Returns ``(new_size, when, seq, tok, canc)``; the arrays may be reallocated
    because the heap has to grow, and the caller owns them.
    """
    count = in_when.size
    if count == 0:
        return n, when, seq, tok, canc
    when = _grow(when, int(n) + count)
    seq = _grow(seq, int(n) + count)
    tok = _grow(tok, int(n) + count)
    canc = _grow(canc, int(n) + count)
    size = int(
        lib.asz_schedule_many(
            addr(when), addr(seq), addr(tok), addr(canc), int(n),
            addr(in_when), addr(in_seq), addr(in_tok), addr(in_canc), count,
        )
    )
    return size, when, seq, tok, canc


def _grow(a: np.ndarray, size: int) -> np.ndarray:
    if a.size >= size:
        return a
    out = np.zeros(size, dtype=a.dtype)
    out[: a.size] = a
    return out


def pop(when, seq, tok, canc, n, of, oi):
    """Pop the minimum key in place. Returns the new heap size."""
    return int(lib.asz_pop(addr(when), addr(seq), addr(tok), addr(canc),
                           int(n), addr(of), addr(oi)))


def min_live(when, canc, n):
    """Smallest deadline among non-cancelled entries, or -1.0 when there is none."""
    return float(lib.asz_min_live(addr(when), addr(canc), int(n)))


def should_compact(n, cancelled, min_sched, min_frac):
    return bool(
        lib.asz_should_compact(int(n), int(cancelled), int(min_sched), float(min_frac))
    )


def timer_tick(when, seq, tok, canc, n, cancelled_count, now, clock_res, max_timeout):
    """Run one `_run_once` timer step. Returns a dict of the kernel's outputs."""
    cap = max(int(n), 1)
    ow = np.empty(cap, dtype=np.float64)
    os_ = np.empty(cap, dtype=np.int64)
    ot = np.empty(cap, dtype=np.int64)
    si = np.empty(4, dtype=np.int64)
    sf = np.empty(2, dtype=np.float64)
    pops = int(
        lib.asz_timer_tick(
            addr(when), addr(seq), addr(tok), addr(canc), int(n),
            int(cancelled_count), float(now), float(clock_res), float(max_timeout),
            addr(ow), addr(ot), addr(os_), addr(si), addr(sf),
        )
    )
    return {
        "pops": pops,
        "size": int(si[1]),
        "cancelled_count": int(si[2]),
        "head_tok": int(si[3]),
        "timeout": float(sf[0]),
        "head_when": float(sf[1]),
        "popped_when": ow[:pops].copy(),
        "popped_seq": os_[:pops].copy(),
        "popped_tok": ot[:pops].copy(),
    }
