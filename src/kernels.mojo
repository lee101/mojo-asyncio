"""Timer-scheduler kernels for the mojo-asyncio port.

`asyncio` is an IO scheduler: there is no arithmetic-heavy numeric core. The one
place it runs a real loop over an array is the *scheduled timer heap*
(`BaseEventLoop._scheduled`, ordered by `TimerHandle.__lt__`, i.e. by `_when`)
and the per-iteration bookkeeping `_run_once` performs on it:

  * dropping cancelled handles off the head of the heap,
  * the `scheduled[0]._when - time()` select-timeout clamp to
    `[0, MAXIMUM_SELECT_TIMEOUT]`,
  * draining every handle whose `_when < now + _clock_resolution`,
  * the `cancelled_count / sched_count > 0.5` compaction heuristic and the
    O(n) filter + heapify that follows it.

That bookkeeping is what this file implements. Buffers cross the C ABI as
64-bit addresses and the pointer is rebuilt inside the body, because `@export`
rejects parametric functions and an inferred pointer origin makes a symbol
parametric.

Key order is `(when, seq)`. `asyncio` compares on `_when` alone and documents
ties as undefined; the monotonic `seq` insertion counter is a total-order
refinement that makes the pop order deterministic without ever reordering two
handles that CPython would keep ordered.
"""

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = Pointer[Int8, AnyOrigin[mut=True]]


def fptr(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def iptr(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def bptr(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def _less(w: FPtr, s: IPtr, a: Int, b: Int) -> Bool:
    var wa = w[unsafe_offset=a]
    var wb = w[unsafe_offset=b]
    if wa < wb:
        return True
    if wa > wb:
        return False
    return s[unsafe_offset=a] < s[unsafe_offset=b]


def _swap3(w: FPtr, s: IPtr, t: IPtr, a: Int, b: Int):
    var tw = w[unsafe_offset=a]
    w[unsafe_offset=a] = w[unsafe_offset=b]
    w[unsafe_offset=b] = tw
    var ts = s[unsafe_offset=a]
    s[unsafe_offset=a] = s[unsafe_offset=b]
    s[unsafe_offset=b] = ts
    var tt = t[unsafe_offset=a]
    t[unsafe_offset=a] = t[unsafe_offset=b]
    t[unsafe_offset=b] = tt


def _sift_down(w: FPtr, s: IPtr, t: IPtr, n: Int, i0: Int):
    var i = i0
    while True:
        var child = 2 * i + 1
        if child >= n:
            return
        if child + 1 < n and _less(w, s, child + 1, child):
            child += 1
        if _less(w, s, child, i):
            _swap3(w, s, t, i, child)
            i = child
        else:
            return


def _sift_up(w: FPtr, s: IPtr, t: IPtr, i0: Int):
    var i = i0
    while i > 0:
        var parent = (i - 1) >> 1
        if _less(w, s, i, parent):
            _swap3(w, s, t, i, parent)
            i = parent
        else:
            return


@export("asz_sift_up")
def asz_sift_up(when_addr: Int, seq_addr: Int, tok_addr: Int, n: Int,
                i: Int) abi("C") -> Int:
    """Restore the heap property upwards from index `i`. Returns the final index."""
    var w = fptr(when_addr)
    var s = iptr(seq_addr)
    var t = iptr(tok_addr)
    var idx = i
    while idx > 0:
        var parent = (idx - 1) >> 1
        if _less(w, s, idx, parent):
            _swap3(w, s, t, idx, parent)
            idx = parent
        else:
            break
    return idx


@export("asz_sift_down")
def asz_sift_down(when_addr: Int, seq_addr: Int, tok_addr: Int, n: Int,
                  i: Int) abi("C") -> Int:
    """Restore the heap property downwards from index `i`. Returns the final index."""
    var w = fptr(when_addr)
    var s = iptr(seq_addr)
    var t = iptr(tok_addr)
    _sift_down(w, s, t, n, i)
    return i


@export("asz_heapify")
def asz_heapify(when_addr: Int, seq_addr: Int, tok_addr: Int,
                n: Int) abi("C") -> Int:
    """Bottom-up heapify, as `heapq.heapify` does: last parent down to the root."""
    var w = fptr(when_addr)
    var s = iptr(seq_addr)
    var t = iptr(tok_addr)
    var i = n >> 1
    while i > 0:
        i -= 1
        _sift_down(w, s, t, n, i)
    return 0


@export("asz_count_due")
def asz_count_due(when_addr: Int, n: Int, end_time: Float64) abi("C") -> Int:
    """Number of heap entries with `when < end_time`.

    The heap's minimum is at index 0 and the subtree at index i only contains
    keys no smaller than its root, so the due entries are exactly the leftmost
    prefix of the array: a plain binary search finds its end in O(log n).
    """
    var w = fptr(when_addr)
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) >> 1
        if w[unsafe_offset=mid] < end_time:
            lo = mid + 1
        else:
            hi = mid
    return lo


@export("asz_purge_cancelled")
def asz_purge_cancelled(when_addr: Int, seq_addr: Int, tok_addr: Int,
                        canc_addr: Int, n: Int, out_when_addr: Int,
                        out_seq_addr: Int, out_tok_addr: Int) abi("C") -> Int:
    """The `_run_once` compaction pass: drop cancelled handles, then heapify.

    `asyncio` builds a fresh Python list and calls `heapq.heapify` on it. This
    does the same thing in one compiled pass: stable forward copy of the live
    entries into the output arrays, then a bottom-up heapify over them. Returns
    the number of live entries.
    """
    var w = fptr(when_addr)
    var s = iptr(seq_addr)
    var t = iptr(tok_addr)
    var c = bptr(canc_addr)
    var ow = fptr(out_when_addr)
    var os = iptr(out_seq_addr)
    var ot = iptr(out_tok_addr)
    var live = 0
    for i in range(n):
        if c[unsafe_offset=i] == 0:
            ow[unsafe_offset=live] = w[unsafe_offset=i]
            os[unsafe_offset=live] = s[unsafe_offset=i]
            ot[unsafe_offset=live] = t[unsafe_offset=i]
            live += 1
    var i2 = live >> 1
    while i2 > 0:
        i2 -= 1
        _sift_down(ow, os, ot, live, i2)
    return live


@export("asz_schedule_many")
def asz_schedule_many(when_addr: Int, seq_addr: Int, tok_addr: Int,
                      canc_addr: Int, n: Int, in_when_addr: Int,
                      in_seq_addr: Int, in_tok_addr: Int,
                      in_canc_addr: Int, count: Int) abi("C") -> Int:
    """Append `count` timers to a heap of size `n` and sift each one up.

    This is `call_at` in a loop: the Python-level per-timer cost of a heap push
    is the dominant cost of registering a batch of timers, so the whole batch is
    done in one crossing. Returns the new heap size.
    """
    var w = fptr(when_addr)
    var s = iptr(seq_addr)
    var t = iptr(tok_addr)
    var c = bptr(canc_addr)
    var iw = fptr(in_when_addr)
    var iseq = iptr(in_seq_addr)
    var it = iptr(in_tok_addr)
    var ic = bptr(in_canc_addr)
    var size = n
    for k in range(count):
        w[unsafe_offset=size] = iw[unsafe_offset=k]
        s[unsafe_offset=size] = iseq[unsafe_offset=k]
        t[unsafe_offset=size] = it[unsafe_offset=k]
        c[unsafe_offset=size] = ic[unsafe_offset=k]
        size += 1
        _sift_up(w, s, t, size - 1)
    return size


@export("asz_should_compact")
def asz_should_compact(n: Int, cancelled: Int, min_sched: Int,
                       min_frac: Float64) abi("C") -> Int:
    """The `_run_once` compaction predicate, verbatim.

    `sched_count > _MIN_SCHEDULED_TIMER_HANDLES and
     _timer_cancelled_count / sched_count > _MIN_CANCELLED_TIMER_HANDLES_FRACTION`
    """
    if n <= min_sched:
        return 0
    if Float64(cancelled) / Float64(n) > min_frac:
        return 1
    return 0


@export("asz_timer_tick")
def asz_timer_tick(when_addr: Int, seq_addr: Int, tok_addr: Int,
                   canc_addr: Int, n: Int, cancelled_count: Int, now: Float64,
                   clock_res: Float64, max_timeout: Float64, out_when_addr: Int,
                   out_tok_addr: Int, out_seq_addr: Int, stats_i_addr: Int,
                   stats_f_addr: Int) abi("C") -> Int:
    """One `_run_once` timer step, in a single pass over the heap.

    Reproduces, in order:

    1. `while self._scheduled and self._scheduled[0]._cancelled:` pop and
       un-schedule, decrementing the cancelled counter.
    2. the select timeout `scheduled[0]._when - time()`, clamped with
       `> MAXIMUM_SELECT_TIMEOUT` first and `< 0` second, so the result is in
       `[0, max_timeout]`. When the heap is empty the timeout is `None`, encoded
       as a negative sentinel.
    3. `end_time = time() + _clock_resolution` and the drain of every handle
       with `handle._when < end_time`.

    `stats_i` receives `[pops, new_size, new_cancelled_count, head_tok]` and
    `stats_f` receives `[timeout, head_when]`. Returns the number of handles
    drained.
    """
    var w = fptr(when_addr)
    var s = iptr(seq_addr)
    var t = iptr(tok_addr)
    var c = bptr(canc_addr)
    var ow = fptr(out_when_addr)
    var ot = iptr(out_tok_addr)
    var os = iptr(out_seq_addr)
    var si = iptr(stats_i_addr)
    var sf = fptr(stats_f_addr)

    var size = n
    var cc = cancelled_count
    while size > 0 and c[unsafe_offset=0] != 0:
        c[unsafe_offset=0] = 0
        cc -= 1
        var last = size - 1
        if last > 0:
            w[unsafe_offset=0] = w[unsafe_offset=last]
            s[unsafe_offset=0] = s[unsafe_offset=last]
            t[unsafe_offset=0] = t[unsafe_offset=last]
            c[unsafe_offset=0] = c[unsafe_offset=last]
            _sift_down(w, s, t, size - 1, 0)
        size -= 1

    var timeout = Float64(-1.0)
    if size > 0:
        timeout = w[unsafe_offset=0] - now
        if timeout > max_timeout:
            timeout = max_timeout
        elif timeout < 0.0:
            timeout = 0.0

    var end_time = now + clock_res
    var pops = 0
    while size > 0 and w[unsafe_offset=0] < end_time:
        ow[unsafe_offset=pops] = w[unsafe_offset=0]
        os[unsafe_offset=pops] = s[unsafe_offset=0]
        ot[unsafe_offset=pops] = t[unsafe_offset=0]
        c[unsafe_offset=0] = 0
        var last = size - 1
        if last > 0:
            w[unsafe_offset=0] = w[unsafe_offset=last]
            s[unsafe_offset=0] = s[unsafe_offset=last]
            t[unsafe_offset=0] = t[unsafe_offset=last]
            c[unsafe_offset=0] = c[unsafe_offset=last]
            _sift_down(w, s, t, size - 1, 0)
        size -= 1
        pops += 1

    var head_when = Float64(0.0)
    var head_tok = Int64(-1)
    if size > 0:
        head_when = w[unsafe_offset=0]
        head_tok = t[unsafe_offset=0]

    si[unsafe_offset=0] = Int64(pops)
    si[unsafe_offset=1] = Int64(size)
    si[unsafe_offset=2] = Int64(cc)
    si[unsafe_offset=3] = head_tok
    sf[unsafe_offset=0] = timeout
    sf[unsafe_offset=1] = head_when
    return pops


@export("asz_pop")
def asz_pop(when_addr: Int, seq_addr: Int, tok_addr: Int, canc_addr: Int,
            n: Int, out_f_addr: Int, out_i_addr: Int) abi("C") -> Int:
    """Pop the minimum key, the way `heapq.heappop` does.

    Move the last element to the root, shrink by one, sift down. `out_f[0]`
    receives the popped deadline and `out_i` receives `[seq, token]`. Returns
    the new heap size.
    """
    var w = fptr(when_addr)
    var s = iptr(seq_addr)
    var t = iptr(tok_addr)
    var c = bptr(canc_addr)
    var of = fptr(out_f_addr)
    var oi = iptr(out_i_addr)
    if n <= 0:
        return 0
    of[unsafe_offset=0] = w[unsafe_offset=0]
    oi[unsafe_offset=0] = s[unsafe_offset=0]
    oi[unsafe_offset=1] = t[unsafe_offset=0]
    var size = n - 1
    if size > 0:
        w[unsafe_offset=0] = w[unsafe_offset=size]
        s[unsafe_offset=0] = s[unsafe_offset=size]
        t[unsafe_offset=0] = t[unsafe_offset=size]
        c[unsafe_offset=0] = c[unsafe_offset=size]
        _sift_down(w, s, t, size, 0)
    return size


@export("asz_min_live")
def asz_min_live(when_addr: Int, canc_addr: Int, n: Int) abi("C") -> Float64:
    """Smallest deadline among the non-cancelled handles.

    `_run_once` gets this handle by popping cancelled entries off the head, so
    the effective head is the minimum live key. Returns -1.0 when every handle
    is cancelled, which is the "empty heap" case for the timeout computation.
    """
    var w = fptr(when_addr)
    var c = bptr(canc_addr)
    var best = Float64(-1.0)
    for i in range(n):
        if c[unsafe_offset=i] == 0:
            var v = w[unsafe_offset=i]
            if best < 0.0 or v < best:
                best = v
    return best
