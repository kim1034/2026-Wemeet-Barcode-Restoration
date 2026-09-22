#!/usr/bin/env python3
"""
바코드 OBB 탐지 학습 — 치타(CHEETAH) GPU 컨테이너용

컨테이너 환경에서 자주 터지는 것들을 미리 처리한다
  1. data.yaml 의 path: 를 현재 머신 경로로 자동 수정 (윈도우에서 만든 절대경로가 그대로 박혀 있음)
  2. libGL 없는 컨테이너에서 opencv 가 죽는 문제 → opencv-python-headless 사용을 안내
  3. GPU 개수/VRAM 에 맞춘 배치 크기 자동 계산 (단일 GPU는 AutoBatch, 멀티 GPU는 VRAM 기준)
  4. vCPU 수에 맞춘 workers (컨테이너는 코어가 적은 경우가 많아 기본 8이면 오히려 느려짐)
  5. 비율(5:3 / 6:4)을 깨는 증강 차단: shear / perspective / 회전 / 상하좌우 반전 off
  6. 세션이 끊겨도 로그가 남게 즉시 flush, 결과는 영구 스토리지 경로에 저장

v2 변경점
  - 단일 클래스 학습 (--single-cls, 기본 켬): 5:3 / 6:4 는 예측 박스의 가로÷세로로 후처리에서 판정
    (두 클래스는 막대 무늬가 같고 비율만 10% 달라서, 클래스 출력으로 구분하면 헷갈림)
  - cuDNN 충돌 자동 해결: pip 로 설치한 nvidia-cudnn 경로를 잡아 스스로 재실행
  - 학습이 끝나면 result_bundle.tar.gz 를 자동으로 만들고 내려받기 명령을 출력

사용 순서
  pip install ultralytics "opencv-python-headless==4.11.0.86" nvidia-cudnn-cu12==8.9.2.26
  python train_cheetah.py --data dataset --check          # 환경·데이터만 점검
  python train_cheetah.py --data dataset --quick           # 시험 학습 (10%, 5 epoch)
  nohup python train_cheetah.py --data dataset -u > train.log 2>&1 &   # 본 학습 (끊겨도 계속)
  python train_cheetah.py --data dataset --resume          # 이어서
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# 비율 유지를 위해 반드시 꺼야 하는 증강 (박스가 찌그러지면 5:3 / 6:4 가 깨짐)
FIXED_AUG = dict(degrees=0.0, shear=0.0, perspective=0.0, fliplr=0.0, flipud=0.0)
# 데이터에 이미 회전·블러·노이즈·손상이 들어 있으므로 색상 증강만 가볍게 유지
COLOR_AUG = dict(hsv_h=0.015, hsv_s=0.4, hsv_v=0.4, mosaic=1.0, close_mosaic=10, scale=0.5, translate=0.1)


def ensure_cudnn():
    """시스템 cuDNN 과 torch 가 충돌하는 컨테이너 대응.
    pip 로 깐 nvidia-cudnn 경로를 LD_LIBRARY_PATH 앞에 붙이고 스스로 한 번 재실행한다
    (LD_LIBRARY_PATH 는 프로세스 시작 전에 정해져야 적용되므로)."""
    if os.environ.get("_CUDNN_FIXED") or os.name == "nt":
        return  # 윈도우는 torch 가 cuDNN 을 자체 포함하므로 손댈 필요 없음
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia.cudnn")
        if spec is None or not spec.submodule_search_locations:
            return
        lib = os.path.join(list(spec.submodule_search_locations)[0], "lib")
    except Exception:
        return
    if os.path.isdir(lib) and lib not in os.environ.get("LD_LIBRARY_PATH", ""):
        env = dict(os.environ, _CUDNN_FIXED="1",
                   LD_LIBRARY_PATH=lib + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        print(f"🔧 cuDNN 경로 자동 설정: {lib}", flush=True)
        os.execve(sys.executable, [sys.executable] + sys.argv, env)


class Tee:
    """화면과 로그 파일에 동시에 기록 (PowerShell 파이프 없이 로그 남기기)."""
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def __getattr__(self, k):
        return getattr(self.stream, k)


def setup_console(log_path=None):
    # 윈도우 콘솔/파이프(cp949)에서 이모지·특수문자 때문에 죽지 않도록
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(errors="replace")
        except Exception:
            pass
    if log_path and not isinstance(sys.stdout, Tee):
        fh = open(log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout, sys.stderr = Tee(sys.stdout, fh), Tee(sys.stderr, fh)


def log(*a):
    print(*a, flush=True)  # nohup 로그가 바로 보이도록


def check_opencv():
    try:
        import cv2  # noqa: F401  # 임포트 자체가 목적 (libGL 누락 감지)
        return True
    except ImportError as e:
        if "libGL" in str(e) or "libgthread" in str(e):
            log("❌ opencv 가 libGL 을 못 찾습니다 (컨테이너에 GUI 라이브러리가 없음).")
            log("   해결: pip uninstall -y opencv-python && pip install opencv-python-headless")
        else:
            log(f"❌ opencv import 실패: {e}")
        return False


def env_report():
    log("🖥️  환경")
    log(f"   python {sys.version.split()[0]} · cpu {os.cpu_count()} core")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        for line in out.stdout.strip().splitlines():
            log(f"   {line}")
    except Exception:
        log("   nvidia-smi 실행 불가 (GPU 미할당 컨테이너일 수 있음)")
    try:
        import torch
        log(f"   torch {torch.__version__} · CUDA {torch.version.cuda} · 사용 가능 {torch.cuda.is_available()}")
        return torch
    except Exception as e:
        log(f"   ⚠️ torch 를 불러올 수 없음: {type(e).__name__}: {e}")
        return None


def fix_data_yaml(data_dir: Path) -> Path:
    yaml_path = data_dir / "data.yaml"
    if not yaml_path.exists():
        sys.exit(f"❌ {yaml_path} 없음. --data 에 dataset 폴더를 지정하세요.")
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    want = f"path: {data_dir.resolve().as_posix()}"
    new, changed = [], False
    for line in lines:
        if line.strip().startswith("path:") and line.strip() != want:
            line, changed = want, True
        new.append(line)
    if changed:
        shutil.copy(yaml_path, yaml_path.with_suffix(".yaml.bak"))
        yaml_path.write_text("\n".join(new) + "\n", encoding="utf-8")
        log(f"🔧 data.yaml path → {data_dir.resolve()} (원본 data.yaml.bak)")
    log("📁 데이터")
    total = 0
    for split in ("train", "val"):
        n_img = len(list((data_dir / "images" / split).glob("*.jpg")))
        n_lbl = len(list((data_dir / "labels" / split).glob("*.txt")))
        total += n_img
        log(f"   {split}: 이미지 {n_img:,} / 라벨 {n_lbl:,}")
        if n_img == 0 or n_img != n_lbl:
            sys.exit(f"❌ {split} 의 이미지/라벨 개수 불일치 — 업로드가 덜 끝났는지 확인하세요.")
    free = shutil.disk_usage(data_dir).free / 1e9
    log(f"   총 {total:,}장 · 남은 디스크 {free:.1f} GB" + ("  ⚠️ 20GB 이상 확보 권장" if free < 20 else ""))
    return yaml_path


def plan_device(torch, requested):
    if requested:
        n = max(1, len([c for c in requested.split(",") if c.strip().isdigit()]))
        return requested, n, 24.0
    if torch is None or not torch.cuda.is_available():
        log("   ⚠️ GPU 미인식 → CPU 학습은 현실적으로 불가능합니다. 컨테이너 GPU 할당을 확인하세요.")
        return "cpu", 1, 0.0
    n = torch.cuda.device_count()
    vram = min(torch.cuda.get_device_properties(i).total_memory for i in range(n)) / 1e9
    return ",".join(str(i) for i in range(n)), n, vram


def main():
    ap = argparse.ArgumentParser(description="바코드 OBB 학습 (치타 GPU)")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--model", default="yolo11s-obb.pt", help="yolo11n/s/m/l/x-obb.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", default="auto")
    ap.add_argument("--single-cls", dest="single_cls", action="store_true", default=True,
                    help="단일 클래스 학습 (기본). 비율은 후처리에서 박스 모양으로 판정")
    ap.add_argument("--two-cls", dest="single_cls", action="store_false", help="예전처럼 2클래스로 학습")
    ap.add_argument("--device", default=None, help="비우면 자동 (GPU 전부). 예: 0 / 0,1,2,3")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--project", default="runs/obb", help="결과 저장 위치 (영구 스토리지 경로 권장)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--quick", action="store_true", help="데이터 10%%, 5 epoch 시험 학습")
    ap.add_argument("--resume", nargs="?", const="auto", default=None,
                    help="중단된 학습 이어서. 경로를 주면 그 last.pt 에서, 안 주면 자동으로 찾음")
    ap.add_argument("--check", action="store_true", help="환경/데이터만 점검하고 종료")
    ap.add_argument("--cooldown", type=int, default=0,
                    help="에폭이 끝날 때마다 N초 쉬기 (노트북 발열 관리, 예: 60)")
    ap.add_argument("--log", default=None, help="화면 출력을 이 파일에도 저장 (윈도우에서 Tee-Object 대신)")
    args = ap.parse_args()

    setup_console()          # 인코딩 안전장치 먼저 (아래 출력에서 죽지 않도록)
    ensure_cudnn()
    setup_console(args.log)
    if not check_opencv():
        sys.exit(1)
    torch = env_report()
    yaml_path = fix_data_yaml(Path(args.data))
    device, n_gpu, vram = plan_device(torch, args.device)

    if args.batch == "auto":
        if device == "cpu":
            batch = 4
        elif n_gpu == 1:
            batch = -1                                   # Ultralytics AutoBatch: 실제 VRAM 측정해서 결정
        else:
            per_gpu = max(2, int(vram / 8 * (1280 / args.imgsz) ** 2))
            batch = min(per_gpu, 16) * n_gpu             # 멀티 GPU는 AutoBatch 미지원 → 직접 계산
    else:
        batch = int(args.batch)
    # 윈도우는 워커 프로세스 생성 비용이 커서 과하게 늘리면 오히려 느려짐
    cap = 8 if os.name != "nt" else 4
    workers = args.workers if args.workers is not None else max(2, min(cap, (os.cpu_count() or 4) // max(1, n_gpu)))

    cfg = dict(
        data=str(yaml_path.resolve()), imgsz=args.imgsz,
        epochs=5 if args.quick else args.epochs, fraction=0.1 if args.quick else 1.0,
        batch=batch, device=device, workers=workers, single_cls=args.single_cls,
        patience=20, seed=2026, cos_lr=True, amp=True, val=True, plots=True,
        project=args.project, name=args.name or ("smoke_test" if args.quick else "barcode_v1"),
        exist_ok=True, **FIXED_AUG, **COLOR_AUG,
    )
    log("⚙️  학습 설정")
    for k in ("model", "imgsz", "epochs", "fraction", "batch", "device", "workers", "single_cls"):
        log(f"   {k} = {args.model if k == 'model' else cfg[k]}")
    log(f"   비율 유지용 off = {FIXED_AUG}")
    log(f"   결과 저장 = {Path(args.project).resolve() / cfg['name']}")
    if args.check:
        log("✅ 점검 완료 (학습은 하지 않음)")
        return

    from ultralytics import YOLO

    def cooldown(trainer):
        if args.cooldown > 0:
            log(f"   🧊 {args.cooldown}초 쉬는 중 (발열 관리) — epoch {trainer.epoch + 1} 끝")
            time.sleep(args.cooldown)

    t0 = time.perf_counter()
    if args.resume:
        last = Path(args.resume) if args.resume != "auto" else Path(args.project) / cfg["name"] / "weights" / "last.pt"
        if not last.exists():
            # 예상 경로에 없으면 현재 폴더 아래에서 가장 최근 last.pt 를 찾는다 (이름이 맞는 것 우선)
            cands = sorted(Path(".").rglob("last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            named = [c for c in cands if cfg["name"] in c.parts]
            if not cands:
                sys.exit(f"❌ 이어서 할 체크포인트(last.pt)를 찾지 못했습니다: {last}")
            last = (named or cands)[0]
            log(f"🔎 last.pt 자동 탐색 → {last}")
        log(f"▶️  {last} 에서 이어서 학습")
        model = YOLO(str(last))
        model.add_callback("on_train_epoch_end", cooldown)
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.add_callback("on_train_epoch_end", cooldown)
        results = model.train(**cfg)
    log(f"⏱️  학습 시간 {(time.perf_counter() - t0) / 3600:.2f} 시간")

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    log("\n📊 best.pt 검증")
    m = YOLO(str(best)).val(data=cfg["data"], imgsz=args.imgsz, single_cls=args.single_cls, device=device.split(",")[0] if device != "cpu" else "cpu")
    log(f"   mAP50 = {m.box.map50:.4f} · mAP50-95 = {m.box.map:.4f}")
    for i, name in enumerate(m.names.values()):
        if i < len(m.box.ap50):
            log(f"   {name}: mAP50 = {m.box.ap50[i]:.4f} · mAP50-95 = {m.box.ap[i]:.4f}")
    shutil.copy(best, Path(args.data) / "best.pt")

    # 반납 전에 내려받을 파일을 한 덩어리로 (결과를 잃지 않도록)
    import tarfile
    bundle = Path.home() / "result_bundle.tar.gz"
    items = [save_dir / "weights", save_dir / "results.csv", save_dir / "args.yaml",
             *save_dir.glob("*.png"), *save_dir.glob("val_batch*.jpg")]
    with tarfile.open(bundle, "w:gz") as tar:
        for p in items:
            if p.exists():
                tar.add(p, arcname=str(p.relative_to(save_dir)))
        log_path = Path(args.log) if args.log else Path.home() / "train.log"
        if log_path.exists():
            tar.add(log_path, arcname="train.log")
    log(f"\n✅ 완료\n   가중치 {best}\n   그래프/예측 {save_dir}")
    log(f"\n📦 결과 묶음 {bundle} ({bundle.stat().st_size / 1e6:.1f} MB)")
    log("   ⚠️  컨테이너 반납 전에 반드시 내 PC(PowerShell)에서 내려받으세요:")
    log(f"   scp -P <포트> -i .\\private.pem jovyan@<IP>:{bundle} .")


if __name__ == "__main__":
    main()
