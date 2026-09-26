"""Parity of the select-timeout clamp and the drain rule with
`asyncio.base_events.BaseEventLoop._run_once`, checked against the real
constants and the real code path.
"""

import asyncio
import time
from asyncio import base_events

import pytest
import mojo_asyncio as ma


def test_constants_come_from_the_real_loop():
    assert ma.MAXIMUM_SELECT_TIMEOUT == base_events.MAXIMUM_SELECT_TIMEOUT == 86400
    assert ma.MIN_SCHEDULED_TIMER_HANDLES == base_events._MIN_SCHEDULED_TIMER_HANDLES
    assert (
        ma.MIN_CANCELLED_TIMER_HANDLES_FRACTION
        == base_events._MIN_CANCELLED_TIMER_HANDLES_FRACTION
    )
    assert ma.CLOCK_RESOLUTION == time.get_clock_info("monotonic").resolution


def test_clamp_matches_the_loop_formula():
    """`if timeout > MAXIMUM_SELECT_TIMEOUT: MAXIMUM_SELECT_TIMEOUT; elif timeout < 0: 0`"""
    for when, now in [
        (100.0, 90.0), (10.0, 90.0), (1e9, 0.0), (-5.0, 0.0),
        (86400.0, 0.0), (86400.5, 0.0), (-0.0, 0.0), (1.0, 1.0),
    ]:
        expected = when - now
        if expected > base_events.MAXIMUM_SELECT_TIMEOUT:
            expected = float(base_events.MAXIMUM_SELECT_TIMEOUT)
        elif expected < 0:
            expected = 0.0
        assert ma.clamp_timeout(when, now) == expected


def test_compaction_predicate_boundaries():
    """The predicate is `count > 100 and cancelled/count > 0.5`, both strict."""
    cases = [
        (100, 100, False),   # not strictly greater than the minimum
        (101, 50, False),    # 50/101 < 0.5
        (101, 51, True),
        (102, 51, False),    # exactly 0.5 is not greater than 0.5
        (102, 52, True),
        (1000, 500, False),
        (1000, 501, True),
        (0, 0, False),
    ]
    for n, cancelled, expected in cases:
        assert ma._lib.should_compact(
            n, cancelled,
            base_events._MIN_SCHEDULED_TIMER_HANDLES,
            base_events._MIN_CANCELLED_TIMER_HANDLES_FRACTION,
        ) is expected, (n, cancelled)


def test_should_compact_on_the_heap():
    h = ma.TimerHeap()
    h.extend([float(i) for i in range(200)], list(range(200)))
    assert not h.should_compact()
    for i in range(0, 100):
        h.cancel(i)
    assert not h.should_compact()          # 100/200 == 0.5, not greater
    h.cancel(100)
    assert h.should_compact()              # 101/200 > 0.5
    assert h.compact() == 99


def test_tick_drains_strictly_before_end_time():
    """`while self._scheduled[0]._when >= end_time: break` -- the clock
    resolution window is exclusive, so an off-by-one here is a timer that fires
    one iteration early. `0.25` is used rather than the real 1e-09 so that
    `now + res` is exact in binary and the test measures the comparison, not
    rounding."""
    res = 0.25
    h = ma.TimerHeap()
    h.extend([9.5, 10.0, 10.25, 10.5], [1, 2, 3, 4])
    out = h.tick(9.75, clock_res=res)
    assert [t for _, _, t in out["drained"]] == [1]
    assert out["head_tok"] == 2
    out = h.tick(10.0, clock_res=res)
    assert [t for _, _, t in out["drained"]] == [2]
    assert out["head_tok"] == 3
    out = h.tick(10.25, clock_res=res)
    assert [t for _, _, t in out["drained"]] == [3]
    # the timeout comes from the head *before* the drain, so a timer that is
    # due right now yields 0 rather than the next timer's remaining delay
    assert out["timeout"] == 0.0
    assert out["head_tok"] == 4
    out = h.tick(10.5, clock_res=res)
    assert [t for _, _, t in out["drained"]] == [4]
    assert out["size"] == 0
    assert out["head_tok"] == -1
    assert h.tick(10.5, clock_res=res)["timeout"] is None


def test_tick_timeout_is_none_for_an_empty_heap_and_zero_when_late():
    h = ma.TimerHeap()
    assert h.tick(100.0)["timeout"] is None
    h.extend([5.0], [7])
    out = h.tick(100.0)
    # 5.0 - 100.0 is negative, so the clamp makes the selector timeout 0
    assert out["timeout"] == 0.0
    assert [t for _, _, t in out["drained"]] == [7]
    assert out["size"] == 0
    assert out["head_tok"] == -1


def test_tick_leaves_a_future_deadline_pending():
    h = ma.TimerHeap()
    h.extend([105.0], [7])
    out = h.tick(100.0)
    assert out["timeout"] == pytest.approx(5.0)
    assert out["drained"] == []
    assert out["head_tok"] == 7
    assert out["size"] == 1


def test_tick_clamps_a_distant_deadline():
    h = ma.TimerHeap()
    h.extend([1e9], [1])
    out = h.tick(0.0)
    assert out["timeout"] == float(base_events.MAXIMUM_SELECT_TIMEOUT)


def test_tick_drops_cancelled_heads_and_keeps_the_counter():
    h = ma.TimerHeap()
    h.extend([1.0, 2.0, 3.0], [10, 11, 12])
    h.cancel(10)
    assert h.cancelled_count == 1
    out = h.tick(0.0)
    # the cancelled head is discarded without ever being reported as drained
    assert out["drained"] == []
    assert out["size"] == 2
    assert out["cancelled_count"] == 0
    assert out["head_tok"] == 11
    assert out["timeout"] == 2.0


def test_tick_matches_a_hand_written_reference():
    import random

    rng = random.Random(21)
    whens = [float(w) for w in rng.sample(range(2000), 400)]
    h = ma.TimerHeap()
    h.extend(whens, list(range(400)))
    now = 500.0
    res = 0.125
    end = now + res
    # the timeout is computed from the head *before* the drain, as in _run_once
    expected_timeout = min(max(0.0, min(whens) - now),
                           float(base_events.MAXIMUM_SELECT_TIMEOUT))
    pops = [w for w in sorted(whens) if w < end]
    left = [w for w in sorted(whens) if w >= end]
    out = h.tick(now, clock_res=res)
    assert [w for w, _, _ in out["drained"]] == pops
    assert out["timeout"] == expected_timeout
    assert out["size"] == len(left)
    assert out["head_when"] == (left[0] if left else None)


def test_select_timeout_on_a_real_loop_with_no_timers():
    loop = asyncio.new_event_loop()
    try:
        assert ma.select_timeout(loop) is None
        loop.call_soon(lambda: None)
        assert ma.select_timeout(loop) == 0.0
    finally:
        loop.close()
