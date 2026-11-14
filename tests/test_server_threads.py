"""컨테이너 CPU 할당량에 스레드 수 맞추기 — 할당량이 없거나 이상하면 아무것도 하지 않는다."""

import pytest

from wemeet.sw.server import threads


@pytest.fixture(autouse=True)
def many_visible_cpus(monkeypatch):
    # CHEETAH 처럼 컨테이너 안에서는 서버 전체 CPU 가 보인다
    monkeypatch.setattr(threads, "_visible_cpus", lambda: 128)


def write(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"cpu.max": "300000 100000\n"}, 3),  # cgroup v2
        ({"cpu.max": "max 100000\n"}, None),  # v2 제한 없음
        ({"cpu/cpu.cfs_quota_us": "300000\n", "cpu/cpu.cfs_period_us": "100000\n"}, 3),  # v1
        ({"cpu,cpuacct/cpu.cfs_quota_us": "200000", "cpu,cpuacct/cpu.cfs_period_us": "100000"}, 2),
        (
            {"cpu/cpu.cfs_quota_us": "-1\n", "cpu/cpu.cfs_period_us": "100000\n"},
            None,
        ),  # v1 제한 없음
        ({"cpu.max": "50000 100000\n"}, 1),  # 0.5개분이어도 최소 1개
        ({"cpu.max": "250000 100000\n"}, 2),  # 넘치지 않게 내림
        ({"cpu.max": "garbage\n"}, None),  # 깨진 값
        ({"cpu/cpu.cfs_quota_us": "abc", "cpu/cpu.cfs_period_us": "100000"}, None),
        ({"cpu.max": "300000 0\n"}, None),
        ({}, None),  # 파일 없음 (Windows·macOS)
    ],
)
def test_cgroup_quota_becomes_thread_count(tmp_path, files, expected):
    assert threads.decide_cpu_threads({}, write(tmp_path, files)) == expected


def test_never_more_threads_than_visible_cpus(tmp_path, monkeypatch):
    monkeypatch.setattr(threads, "_visible_cpus", lambda: 2)
    assert threads.decide_cpu_threads({}, write(tmp_path, {"cpu.max": "800000 100000"})) == 2


def test_environment_variable_overrides_and_can_turn_it_off(tmp_path):
    root = write(tmp_path, {"cpu.max": "300000 100000"})
    assert threads.decide_cpu_threads({threads.ENV: "5"}, root) == 5
    assert threads.decide_cpu_threads({threads.ENV: "0"}, root) is None
    assert threads.decide_cpu_threads({threads.ENV: "많이"}, root) == 3  # 잘못된 값은 무시하고 자동


def test_limit_sets_library_variables_without_overriding_the_user(tmp_path, monkeypatch):
    monkeypatch.setattr(threads, "cpu_threads", None)
    env = {"MKL_NUM_THREADS": "1"}
    assert threads.limit_cpu_threads(env, write(tmp_path, {"cpu.max": "300000 100000"})) == 3
    assert env["OMP_NUM_THREADS"] == "3" and env["OPENBLAS_NUM_THREADS"] == "3"
    assert env["MKL_NUM_THREADS"] == "1"  # 사용자가 정한 값은 그대로
    assert threads.cpu_threads == 3


def test_nothing_changes_without_a_quota(tmp_path, monkeypatch):
    monkeypatch.setattr(threads, "cpu_threads", 7)
    env = {}
    assert threads.limit_cpu_threads(env, tmp_path) is None
    assert env == {} and threads.cpu_threads is None
    threads.apply_library_threads()  # 손대지 않음 — torch 도 부르지 않는다


def test_health_shows_the_thread_count(monkeypatch):
    from fastapi.testclient import TestClient

    from wemeet.sw import server

    monkeypatch.setattr(threads, "cpu_threads", 3)
    assert TestClient(server.app).get("/api/health").json()["cpu_threads"] == 3
