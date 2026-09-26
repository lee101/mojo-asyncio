"""Parity against a *running* `asyncio` event loop.

The loop is instrumented, not reimplemented: `BaseEventLoop._run_once` is left
alone and every timeout it hands to its selector is recorded. The Mojo port
then has to predict each of those values, from the loop's own `_scheduled`
heap, and the order in which the loop actually fires its timers has to match
the order the port drains them.
"""

import asyncio
import random

import pytest

import mojo_asyncio as ma


def _instrumented_loop():
    loop = asyncio.new_event_loop()
    seen = []
    real_select = loop._selector.select

    def spy(timeout, _real=real_select, _loop=loop):
        seen.append((timeout, ma.select_timeout(_loop)))
        return _real(timeout)

    loop._selector.select = spy
    return loop, seen


# `_run_once` reads `self.time()` and the spy reads it again a few microseconds
# later, so the two can only agree to within the instrumentation overhead. A
# wrong clamp, a cancelled handle that should have been skipped, or the
# `self._ready` short-circuit would each move the answer by milliseconds or by
# 86400 seconds, not by microseconds.
CLOCK_SLACK = 1e-3


def test_select_timeouts_match_a_running_loop():
    loop, seen = _instrumented_loop()
    try:
        fired = []
        delays = [round(0.001 + 0.0015 * i, 6) for i in range(12)]
        for i, d in enumerate(delays):
            loop.call_later(d, fired.append, i)
        loop.call_later(0.4, loop.stop)
        loop.run_forever()
        assert fired == sorted(range(12), key=lambda i: delays[i])
        assert len(seen) > 3
        mismatched = [
            (a, b) for a, b in seen
            if not (a == b or (a is not None and b is not None
                               and abs(a - b) < CLOCK_SLACK))
        ]
        assert not mismatched, mismatched
        assert any(t is not None and t > 0.0 for t, _ in seen)
    finally:
        loop.close()


def test_select_timeouts_match_with_cancelled_timers():
    """A cancelled handle is popped off the head before the timeout is computed,
    so predicting it means ignoring cancelled entries, not just reading index 0."""
    loop, seen = _instrumented_loop()
    try:
        handles = [loop.call_later(0.05, lambda: None) for _ in range(6)]
        for hd in handles[::2]:
            hd.cancel()
        loop.call_later(0.2, loop.stop)
        loop.run_forever()
        mismatched = [
            (a, b) for a, b in seen
            if not (a == b or (a is not None and b is not None
                               and abs(a - b) < CLOCK_SLACK))
        ]
        assert not mismatched, mismatched
        assert any(
            t is not None and abs(t - 0.05) < 1e-3 for t, _ in seen
        ), "expected the surviving 0.05s timers to drive the selector timeout"
    finally:
        loop.close()


def test_drain_order_matches_the_order_the_loop_fires_timers():
    rng = random.Random(17)
    delays = sorted(round(rng.uniform(0.001, 0.05), 6) for _ in range(20))
    loop = asyncio.new_event_loop()
    try:
        fired = []
        for i, d in enumerate(delays):
            loop.call_later(d, fired.append, i)
        loop.call_later(0.3, loop.stop)
        loop.run_forever()

        # Rebuild the same timer set from the loop's own handle list and let the
        # Mojo port drain it at the same `now` the loop used.
        port = ma.heap_from_pairs(
            (loop.time() - 0.3 + d, i) for i, d in enumerate(delays)
        )
        drained = []
        while len(port):
            drained.append(port.pop()[1])
        assert drained == fired
        assert fired == sorted(fired)
    finally:
        loop.close()


def test_wait_for_deadline_ordering_matches_the_port():
    """`asyncio.wait_for` arms one `call_later` timeout per wait; the port must
    order a batch of those deadlines the same way the loop drains them."""
    loop = asyncio.new_event_loop()
    try:
        order = []

        async def case(delay, value):
            try:
                await asyncio.wait_for(asyncio.sleep(10), delay)
            except asyncio.TimeoutError:
                order.append(value)

        tasks = [loop.create_task(case(d, i)) for i, d in
                 enumerate([0.05, 0.01, 0.03, 0.02, 0.04])]
        loop.run_until_complete(asyncio.gather(*tasks))
        assert order == [1, 3, 2, 4, 0]

        port = ma.heap_from_pairs([(d, i) for i, d in
                                   enumerate([0.05, 0.01, 0.03, 0.02, 0.04])])
        assert [port.pop()[1] for _ in range(5)] == order
    finally:
        loop.close()


def test_select_timeout_agrees_with_a_hand_computed_value():
    loop = asyncio.new_event_loop()
    try:
        now = loop.time()
        loop.call_later(2.5, lambda: None)
        got = ma.select_timeout(loop, now=now)
        assert got == pytest.approx(2.5, abs=CLOCK_SLACK)
        assert ma.min_live(loop) == pytest.approx(now + 2.5, abs=CLOCK_SLACK)
    finally:
        loop.close()
