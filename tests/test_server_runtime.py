from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from wemeet.sw.server.runtime import InferenceRuntime, RuntimeUnavailable


def test_single_slot_stays_busy_until_worker_actually_finishes():
    runtime = InferenceRuntime(ready=True)
    started, release = Event(), Event()

    def work():
        started.set()
        assert release.wait(5)
        return {"ok": True}

    with ThreadPoolExecutor() as pool:
        future = pool.submit(runtime.execute, work, key=("s", "b"), fingerprint="one")
        assert started.wait(5)
        try:
            assert runtime.busy
            with pytest.raises(RuntimeUnavailable) as error:
                runtime.execute(lambda: {}, key=None)
            assert error.value.status_code == 429
            # Cancelling the caller cannot stop a running inference thread.
            assert not future.cancel()
            assert runtime.busy
        finally:
            release.set()
        assert future.result() == {"ok": True}
    assert not runtime.busy


def test_error_releases_slot_and_result_cache_is_bounded():
    runtime = InferenceRuntime(ready=True)

    def broken():
        raise ValueError("failed")

    with pytest.raises(ValueError):
        runtime.execute(broken)
    assert not runtime.busy
    runtime.cache_limit_bytes = 100
    for index in range(10):
        runtime.execute(lambda: {"ok": True}, key=index, fingerprint="same")
    assert len(runtime._cache) <= 4
    assert sum(item[3] for item in runtime._cache.values()) <= 100


def test_expired_cache_does_not_confirm_a_later_request(monkeypatch):
    runtime = InferenceRuntime(ready=True)
    runtime.cache_ttl_s = -1
    runtime.execute(lambda: {"value": 1}, key="key", fingerprint="one")
    assert runtime.execute(lambda: {"value": 2}, key="key", fingerprint="one") == {"value": 2}
