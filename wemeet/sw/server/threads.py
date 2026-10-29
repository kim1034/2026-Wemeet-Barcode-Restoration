"""컨테이너가 준 CPU 개수에 계산 스레드 수를 맞춘다.

CHEETAH 같은 컨테이너는 cgroup 으로 CPU 를 3개분만 주는데 `os.cpu_count()` 는 서버 전체(128)를
보여 준다. torch·OpenCV·numpy 가 그 숫자를 믿고 스레드를 수십 개 만들면 0.1초 주기의 할당량을
몇 ms 만에 다 쓰고, 남은 시간 동안 프로세스 전체가 멈춘다(스로틀링). CHEETAH 실측(같은 GPU·사진,
스레드 64 → 3): 기하 추정 287 → 4.3ms, TPS 보정 209 → 10.8ms.

할당량을 못 읽거나 제한이 없으면 아무것도 하지 않는다 — 내 PC·맥북에서는 지금과 같다.
`WEMEET_CPU_THREADS` 로 직접 정할 수 있고, `0` 이면 자동 설정을 끈다.

이 모듈은 표준 라이브러리만 쓴다. numpy·torch 보다 **먼저** 불러야 환경변수가 그 라이브러리에
닿기 때문이다 (`wemeet/sw/server/__init__.py` 맨 위).
"""

import logging
import os
from pathlib import Path

ENV = "WEMEET_CPU_THREADS"
CGROUP_ROOT = Path("/sys/fs/cgroup")
# numpy(OpenBLAS)·torch(OpenMP)·MKL 이 프로세스 시작 때 읽는 값. 사용자가 이미 정했으면 그대로 둔다.
_LIBRARY_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

log = logging.getLogger(__name__)

#: 서버가 맞춘 스레드 수. None 이면 손대지 않았다는 뜻이다 (`/api/health` 에 그대로 보인다).
cpu_threads: int | None = None


def cgroup_cpu_limit(root: Path = CGROUP_ROOT) -> int | None:
    """cgroup 할당량을 CPU 개수로 바꾼다. 제한이 없거나 읽지 못하면 None."""
    try:
        quota, period = (root / "cpu.max").read_text().split()[:2]  # cgroup v2: "300000 100000"
    except (OSError, ValueError):
        quota = period = None
        for folder in (root / "cpu", root / "cpu,cpuacct", root):  # cgroup v1
            try:
                quota = (folder / "cpu.cfs_quota_us").read_text().strip()
                period = (folder / "cpu.cfs_period_us").read_text().strip()
                break
            except OSError:
                continue
    if quota is None or quota == "max":
        return None
    try:
        quota, period = int(quota), int(period)
    except ValueError:
        return None
    if quota <= 0 or period <= 0:  # v1 의 -1 = 제한 없음
        return None
    return max(1, quota // period)


def _visible_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # Windows·macOS
        return os.cpu_count() or 1


def decide_cpu_threads(env=os.environ, root: Path = CGROUP_ROOT) -> int | None:
    """맞출 스레드 수. None 이면 손대지 않는다."""
    raw = env.get(ENV, "").strip()
    if raw:
        try:
            chosen = int(raw)
        except ValueError:
            log.warning("%s=%r 는 숫자가 아니라 무시하고 자동으로 정합니다", ENV, raw)
        else:
            return chosen if chosen > 0 else None  # 0 이하 = 끄기
    limit = cgroup_cpu_limit(root)
    if limit is None:
        return None
    return max(1, min(limit, _visible_cpus()))


def limit_cpu_threads(env=os.environ, root: Path = CGROUP_ROOT) -> int | None:
    """서버 시작 때 한 번 — numpy·torch 를 불러오기 전에 부른다."""
    global cpu_threads
    cpu_threads = decide_cpu_threads(env, root)
    if cpu_threads is not None:
        for name in _LIBRARY_ENV:
            env.setdefault(name, str(cpu_threads))
        log.info("CPU 할당량에 맞춰 계산 스레드를 %d개로 제한합니다", cpu_threads)
    return cpu_threads


def apply_library_threads() -> None:
    """이미 불러온 torch·OpenCV 에도 같은 수를 건다 (모델 준비 직전)."""
    if cpu_threads is None:
        return
    import cv2
    import torch

    torch.set_num_threads(cpu_threads)
    cv2.setNumThreads(cpu_threads)
