#!/usr/bin/env python3
"""
바코드 비전 학습용 데이터셋 자동 생성기 (단일 파일)

생성 조건
  1. 해상도      : 모든 이미지 1920x1080 (FHD)
  2. 객체 수     : 이미지당 바코드 딱 1개
  3. 라벨 제거   : 흰 종이/테두리 없이, 투명 배경 바코드(막대 + 숫자)만 배경 위에 직접 합성
  4. 카메라 효과 : 30fps 카메라 영상의 한 프레임처럼 모션 블러, 초점 흐림, 센서 노이즈,
                  조명 변화, 비네팅, JPEG 압축을 랜덤하게 적용
  5. 반복 생성   : for 루프로 N장을 순차 저장 (중단 후 재실행하면 이어서 생성)

추가로 지키는 규칙 (이전 합의)
  - 막대 영역(검은 세로 막대들이 모인 직사각형)의 가로:세로 = 정확히 5:3 또는 6:4
    짝수 번호 → 5:3 (클래스 0), 홀수 번호 → 6:4 (클래스 1) → 1만 장이면 정확히 5,000장씩
  - YOLO OBB 라벨: 막대 영역 4꼭짓점 (0~1 정규화, 좌상→우상→우하→좌하)
  - 10% 패딩 박스까지 화면 안에 들어오게 배치 → 추론 후 패딩 크롭이 잘리지 않음

설치 & 실행
  pip install numpy opencv-python pillow python-barcode
  python make_barcode_dataset.py --preview 12                     # 미리보기 12장 (그리드 이미지)
  python make_barcode_dataset.py --out dataset --num 10000        # 본 생성
  python make_barcode_dataset.py --out dataset --num 10000 --bg-dir backgrounds/   # 실사 배경 사용 (권장)

  병렬로 빠르게: 터미널 여러 개에서 구간을 나눠 실행 (이미지마다 시드가 고정이라 결과가 같음)
  python make_barcode_dataset.py --out dataset --start 0    --end 2500
  python make_barcode_dataset.py --out dataset --start 2500 --end 5000   ...
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from barcode import EAN13, Code39, Code128
from PIL import Image, ImageDraw, ImageFont

# ═════════════════════════════════════════════
# 설정 (여기 숫자만 바꾸면 분포가 바뀜)
# ═════════════════════════════════════════════
CFG = {
    "W": 1920, "H": 1080,
    "classes": {0: ("barcode_5x3", (5, 3)), 1: ("barcode_6x4", (3, 2))},  # 6:4 = 3:2
    "symbology": {"code128": 75, "ean13": 15, "code39": 10},
    "bars_render_px": (400, 900),     # 원본 렌더 시 막대 영역 가로
    "bars_screen_px": (120, 820),     # 화면 속 막대 영역 가로 (로그 균등)
    "digits_prob": 0.8,               # 막대 아래 숫자 표시 확률
    "rot_small_prob": 0.6,            # 60%는 ±15° 이내, 나머지는 360° 아무 방향
    "rot_small_deg": 15,
    # 손상 종류 비율 (50장마다 한 바퀴, 클래스 5:3 / 6:4 각각 같은 분포)
    "damage": {"normal": 12, "crumple": 6, "wave": 4, "twist": 3, "fold": 3,
               "tear": 4, "wear": 3, "glare": 8, "mixed": 7},
    "glare_cover": (0.20, 0.60),      # 반사로 하얗게 날아가는 막대 영역 비율
    "tear_box_full": True,            # True: 찢겨 없어진 부분까지 박스에 포함 / False: 남아 있는 부분만
    "pad": 0.10,                      # 패딩 (각 변 10%)
    "edge_margin": 30,                # 패딩 박스와 화면 가장자리 사이 최소 px
    "min_bg_luma": 0.38,              # 막대 아래 배경 최소 밝기 (너무 어두우면 검은 막대가 안 보임)
    "photo_ratio": 0.7,               # --bg-dir 이 있을 때 실사 배경 비율
    "clutter": (10, 28),              # 주변 방해물 개수
    # 카메라 (30fps: 노출 최대 1/30초)
    "motion_small_prob": 0.6, "motion_small_px": (0, 3), "motion_large_px": (3, 12),
    "defocus": [(0.55, (0.0, 0.4)), (0.35, (0.5, 1.5)), (0.10, (1.5, 3.0))],  # (확률, 반경 px)
    "noise_sigma": (1.5, 9.0),        # 8비트 기준 가우시안 노이즈 표준편차
    "exposure": (0.65, 1.30),
    "jpeg_quality": (70, 95),
    "val_every": 10,                  # 10쌍 중 1쌍은 val → 9,000 / 1,000 (클래스 반반)
}

WORDS = ["SHIP", "TO", "FROM", "ORDER", "LOT", "QTY", "EXP", "SKU", "ZONE", "BIN", "FRAGILE", "KEEP DRY",
         "HANDLE WITH CARE", "EXPRESS", "RETURN", "PRIORITY", "MADE IN KOREA", "NET WT", "BOX", "PALLET"]
FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


# ═════════════════════════════════════════════
# 공용 유틸
# ═════════════════════════════════════════════
def chance(rng, p):
    return rng.random() < p


def pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def weighted(rng, table):
    keys = list(table)
    p = np.array([table[k] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def log_uniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def smooth_noise(h, w, rng, cells=6):
    """부드러운 저주파 노이즈 (0~1)."""
    g = rng.random((max(2, cells * h // max(h, w)), max(2, cells * w // max(h, w))), dtype=np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)


class Fonts:
    def __init__(self, user_font=None):
        paths = ([user_font] if user_font else []) + FONT_CANDIDATES
        self.path = next((p for p in paths if p and Path(p).exists()), None)
        if self.path is None:
            print("ℹ️  시스템 폰트를 못 찾아 PIL 기본 폰트를 씁니다 (--font 로 .ttf 지정 가능)")

    def get(self, size):
        size = max(8, int(size))
        return ImageFont.truetype(self.path, size) if self.path else ImageFont.load_default(size)

    def draw_fit(self, draw, text, box, fill=255):
        """box(x0,y0,x1,y1) 안에 들어가는 최대 크기로 가운데 정렬해 그린다."""
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        if bw < 8 or bh < 8:
            return
        l, t, r, b = self.get(100).getbbox(text)
        size = 100 * min(bh / max(b - t, 1), bw / max(r - l, 1))
        for _ in range(10):
            f = self.get(size)
            l, t, r, b = f.getbbox(text)
            if r - l <= bw and b - t <= bh:
                break
            size *= 0.92
        draw.text((x0 + (bw - (r - l)) / 2 - l, y0 + (bh - (b - t)) / 2 - t), text, font=f, fill=fill)


# ═════════════════════════════════════════════
# 1) 투명 배경 바코드 (막대 영역 비율 정확히 고정)
# ═════════════════════════════════════════════
def digits(rng, n):
    return "".join(str(d) for d in rng.integers(0, 10, n))


def letters(rng, n):
    return "".join(pick(rng, "ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(n))


def make_data(sym, rng):
    if sym == "ean13":
        return "880" + digits(rng, 9)
    if sym == "code39":
        return letters(rng, 2) + digits(rng, int(rng.integers(3, 6)))
    if chance(rng, 0.5):
        return digits(rng, int(rng.integers(10, 15)))
    return letters(rng, 2) + digits(rng, int(rng.integers(8, 12)))


def encode(sym, data):
    """→ (모듈 문자열: '1' 막대 / '0' 공백, 사람이 읽는 숫자)"""
    if sym == "code128":
        mod, text = Code128(data).build()[0], data
    elif sym == "code39":
        mod, text = Code39(data, add_checksum=False).build()[0], data
    else:
        bc = EAN13(data)
        mod, text = bc.build()[0], bc.get_fullcode()
    return mod.replace("G", "1").strip("0"), text


def make_barcode(rng, ratio, fonts):
    """투명 배경 바코드 알파맵 생성.
    반환: alpha(HxW float, 1=잉크), bars_rect(x0,y0,x1,y1), 메타
    막대 영역 가로(모듈 수 x 모듈 두께)가 비율 단위의 배수가 되도록 두께를 골라 세로가 정수로 딱 떨어지게 한다."""
    uw, uh = ratio
    lo, hi = CFG["bars_render_px"]
    for _ in range(100):
        sym = weighted(rng, CFG["symbology"])
        data = make_data(sym, rng)
        mods, text = encode(sym, data)
        n = len(mods)
        cands = [mp for mp in range(2, hi // n + 1) if n * mp >= lo and (n * mp) % uw == 0]
        if cands:
            break
    else:
        raise RuntimeError("비율에 맞는 모듈 두께를 찾지 못함")
    mp = int(pick(rng, cands))
    bw, bh = n * mp, n * mp * uh // uw

    margin = 16  # 회전·블러 여유
    show_digits = chance(rng, CFG["digits_prob"])
    gap, dh = int(bh * rng.uniform(0.02, 0.05)), int(bh * rng.uniform(0.11, 0.17))
    Wc = bw + 2 * margin
    Hc = bh + 2 * margin + (gap + dh if show_digits else 0)

    img = Image.new("L", (Wc, Hc), 0)
    d = ImageDraw.Draw(img)
    i = 0
    while i < n:  # 같은 길이의 막대를 연속 구간 단위로 그림
        if mods[i] == "0":
            i += 1
            continue
        j = i
        while j < n and mods[j] == "1":
            j += 1
        d.rectangle([margin + i * mp, margin, margin + j * mp - 1, margin + bh - 1], fill=255)
        i = j
    if show_digits:
        fonts.draw_fit(d, text, (margin, margin + bh + gap, margin + bw, margin + bh + gap + dh))

    alpha = np.asarray(img, np.float32) / 255
    # 인쇄 느낌: 잉크 번짐 + 농도 얼룩
    alpha = cv2.GaussianBlur(alpha, (0, 0), rng.uniform(0.3, 0.8))
    density = 1 - rng.uniform(0.0, 0.25) * smooth_noise(Hc, Wc, rng, cells=8)
    alpha = alpha * density * rng.uniform(0.88, 0.99)
    meta = {"symbology": sym, "data": text, "module_px": mp, "bars_render_px": [bw, bh], "digits": show_digits}
    return alpha, (margin, margin, margin + bw, margin + bh), meta


# ═════════════════════════════════════════════
# 1-b) 손상: 구김 / 물결 / 비틀림 / 각진 접힘 / 찢김 / 마모 / 빛반사
#      - 변형은 바코드 자체 평면에서 remap 으로 적용
#      - 정답 박스는 '원래 막대 영역이 변형된 발자국(footprint)' 전체를 감싸므로
#        찢겨 없어진 부분도 박스 안에 포함된다 (손가락에 가려진 경우와 같은 규칙)
# ═════════════════════════════════════════════
FORCE_KIND = None  # 미리보기에서 --kind 로 특정 손상만 볼 때 사용


def damage_plan(idx, rng):
    """이미지 번호로 손상 종류를 정해 비율을 정확히 맞춘다."""
    if FORCE_KIND:
        kind = FORCE_KIND
        if kind == "mixed":
            pool = ["crumple", "wave", "twist", "fold", "tear", "wear", "glare"]
            return kind, list(rng.choice(pool, size=int(rng.integers(2, 4)), replace=False))
        return kind, [] if kind == "normal" else [kind]
    pattern = [k for k, n in CFG["damage"].items() for _ in range(n)]
    np.random.default_rng(777).shuffle(pattern)          # 고정 순서 (실행마다 동일)
    kind = pattern[(idx // 2) % len(pattern)]            # 짝수=5:3, 홀수=6:4 → 두 클래스 분포 동일
    if kind == "normal":
        return kind, []
    if kind == "mixed":
        pool = ["crumple", "wave", "twist", "fold", "tear", "wear", "glare"]
        return kind, list(rng.choice(pool, size=int(rng.integers(2, 4)), replace=False))
    return kind, [kind]


def tear_mask(shape, bars, rng):
    """가장자리에서 들쭉날쭉하게 찢겨 나간 영역 (1 = 떨어져 나감)."""
    H, W = shape
    x0, y0, x1, y1 = bars
    bw, bh = x1 - x0, y1 - y0
    side = pick(rng, ["l", "r", "t", "b"])
    frac = rng.uniform(0.08, 0.30)
    n = int(rng.integers(6, 14))
    if side in ("l", "r"):
        base = x0 + frac * bw if side == "l" else x1 - frac * bw
        ys = np.linspace(-H, 2 * H, n)
        xs = base + rng.normal(0, 0.05 * bw, n)
        pts = np.stack([xs, ys], 1)
        far = [[-W, 2 * H], [-W, -H]] if side == "l" else [[2 * W, 2 * H], [2 * W, -H]]
    else:
        base = y0 + frac * bh if side == "t" else y1 - frac * bh
        xs = np.linspace(-W, 2 * W, n)
        ys = base + rng.normal(0, 0.05 * bh, n)
        pts = np.stack([xs, ys], 1)
        far = [[2 * W, -H], [-W, -H]] if side == "t" else [[2 * W, 2 * H], [-W, 2 * H]]
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.int32(np.vstack([pts, far]))], 255)
    return cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 1.0)


def apply_damage(alpha, bars, kinds, rng):
    """→ (변형된 잉크 알파, 발자국 마스크, 표면 음영, 메타)"""
    x0, y0, x1, y1 = bars
    bw, bh = x1 - x0, y1 - y0
    short = min(bw, bh)
    pad = int(0.20 * short) + 8                      # 변형으로 밀려나갈 여유
    alpha = cv2.copyMakeBorder(alpha, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    H, W = alpha.shape
    x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    foot = np.zeros((H, W), np.float32)
    foot[y0:y1, x0:x1] = 1                           # 원래 막대 영역 = 정답 박스의 기준
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mx, my = xx.copy(), yy.copy()
    height = np.zeros((H, W), np.float32)
    info = {}

    if "twist" in kinds:                              # 행주 짜듯 비틀림
        tmax = np.deg2rad(rng.uniform(6, 20)) * pick(rng, [-1, 1])
        th = tmax * (xx - cx) / (bw / 2)
        px, py = xx - cx, yy - cy
        mx, my = cx + px * np.cos(th) - py * np.sin(th), cy + px * np.sin(th) + py * np.cos(th)
        height += rng.uniform(0.02, 0.05) * short * np.sin(th * 2)
        info["twist_deg"] = round(float(np.degrees(tmax)), 1)

    if "fold" in kinds:                               # 각지게 접힘: 한쪽 면이 눌리고 밝기가 뚝 끊김
        th = rng.uniform(0, np.pi)
        nx, ny = np.cos(th), np.sin(th)
        px, py = rng.uniform(x0 + 0.25 * bw, x1 - 0.25 * bw), rng.uniform(y0 + 0.25 * bh, y1 - 0.25 * bh)
        d = (xx - px) * nx + (yy - py) * ny
        comp = rng.uniform(0.55, 0.85)
        k = (1 / comp - 1) * (d > 0)
        mx, my = mx + d * k * nx, my + d * k * ny
        height += np.where(d > 0, -rng.uniform(0.04, 0.12) * d, 0)
        info["fold_compress"] = round(float(comp), 2)

    if "crumple" in kinds:                             # 여러 방향으로 접힌 주름
        n = int(rng.integers(3, 8))
        for _ in range(n):
            px, py = rng.uniform(x0, x1), rng.uniform(y0, y1)
            th = rng.uniform(0, np.pi)
            nx, ny = np.cos(th), np.sin(th)
            d = (xx - px) * nx + (yy - py) * ny
            t = -(xx - px) * ny + (yy - py) * nx
            width = max(2.0, rng.uniform(0.03, 0.20) * short)
            length = rng.uniform(0.5, 1.5) * max(bw, bh)
            height += pick(rng, [-1, 1]) * rng.uniform(0.15, 0.5) * width \
                * np.clip(1 - np.abs(d) / width, 0, 1) * np.exp(-0.5 * (t / (length / 2)) ** 2)
        info["creases"] = n

    if "wave" in kinds:                                # 구불구불한 물결
        lam = rng.uniform(0.25, 0.8) * bw
        th = rng.uniform(0, np.pi)
        height += rng.uniform(0.06, 0.16) * short * np.sin(
            2 * np.pi * (xx * np.cos(th) + yy * np.sin(th)) / lam + rng.uniform(0, 6.28))
        info["wave_lambda_px"] = round(float(lam), 1)

    if np.any(height):                                 # 높이 → 시차(막대가 꺾여 보임)
        hn = height / (np.abs(height).max() + 1e-6)
        k = rng.uniform(-1, 1, 2) * rng.uniform(0.03, 0.09) * short
        mx, my = mx - k[0] * hn, my - k[1] * hn

    mx, my = mx.astype(np.float32), my.astype(np.float32)
    rm = lambda a, bv=0.0: cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bv)
    alpha, foot, height = rm(alpha), rm(foot), rm(height)

    # 표면 음영 (접힌 면끼리 밝기가 달라 보이게) — 바코드 주변만 부드럽게 적용
    shade = np.ones((H, W), np.float32)
    if np.any(height):
        gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8
        nrm = np.sqrt(gx ** 2 + gy ** 2 + 1)
        light = rng.uniform(-1, 1, 2)
        light = np.append(light / (np.linalg.norm(light) + 1e-6) * 0.7, 0.72).astype(np.float32)
        diff = (-gx / nrm) * light[0] + (-gy / nrm) * light[1] + (1 / nrm) * light[2]
        # 접힌 자국이 새까맣게 보이지 않도록 어두워지는 정도를 제한
        feather = cv2.GaussianBlur(np.clip(cv2.dilate(foot, np.ones((9, 9), np.uint8)), 0, 1), (0, 0), 0.05 * short + 2)
        shade = 1 + (np.clip(diff / light[2], 0.62, 1.25) - 1) * feather

    if "tear" in kinds:                                # 찢겨 나감 (발자국은 그대로 → 박스는 원래 영역 유지)
        torn = tear_mask((H, W), (x0, y0, x1, y1), rng)
        alpha = alpha * (1 - torn)
        if not CFG["tear_box_full"]:
            foot = foot * (1 - torn)
        shade = shade * (1 - 0.35 * cv2.GaussianBlur(torn, (0, 0), 3) * (1 - torn))
        info["torn_ratio"] = round(float((torn * foot).sum() / max(foot.sum(), 1)), 3)

    if "wear" in kinds:                                # 마모: 긁힘 + 잉크 벗겨짐 + 얼룩
        sc = np.zeros((H, W), np.uint8)
        for _ in range(int(rng.integers(6, 25))):
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            L, th = rng.uniform(0.1, 0.7) * bw, rng.uniform(0, np.pi)
            cv2.line(sc, (int(x), int(y)), (int(x + L * np.cos(th)), int(y + L * np.sin(th))), 255,
                     int(rng.integers(1, max(2, int(0.02 * short)))), cv2.LINE_AA)
        scratch = cv2.GaussianBlur(sc.astype(np.float32) / 255, (0, 0), 1.0)
        patch = np.clip((smooth_noise(H, W, rng, 7) - rng.uniform(0.45, 0.65)) * 6, 0, 1)
        alpha = alpha * (1 - rng.uniform(0.4, 0.8) * scratch) * (1 - rng.uniform(0.2, 0.6) * patch)
        smudge = np.clip((smooth_noise(H, W, rng, 9) - 0.7) * 5, 0, 1) * foot
        alpha = np.clip(alpha + rng.uniform(0.1, 0.35) * smudge, 0, 1)
        info["wear"] = True

    return np.clip(alpha, 0, 1), foot, shade, (x0, y0, x1, y1), info


def glare_on_bars(img, foot_scene, rng):
    """막대 영역의 일부가 하얗게 날아가도록 반사광을 씌운다 (목표 비율까지 이진 탐색)."""
    H, W = img.shape[:2]
    d = 4
    small = lambda a: cv2.resize(a, (W // d, H // d), interpolation=cv2.INTER_AREA)
    foot_s = small(foot_scene) > 0.5
    if foot_s.sum() < 20:
        return 0.0
    target = rng.uniform(*CFG["glare_cover"])
    ys, xs = np.nonzero(foot_s)
    gx, gy = rng.uniform(*np.quantile(xs, [0.2, 0.8])), rng.uniform(*np.quantile(ys, [0.2, 0.8]))
    L = np.sqrt(foot_s.sum())
    sx = L * rng.uniform(0.35, 0.8)
    sy = sx / (rng.uniform(3, 8) if chance(rng, 0.35) else rng.uniform(1, 2.2))  # 줄무늬형 / 덩어리형
    th, gain = rng.uniform(0, np.pi), rng.uniform(1.4, 2.4)
    YY, XX = np.mgrid[0:H // d, 0:W // d].astype(np.float32)

    def field(scale, XX=XX, YY=YY):
        u = (XX - gx) * np.cos(th) + (YY - gy) * np.sin(th)
        v = -(XX - gx) * np.sin(th) + (YY - gy) * np.cos(th)
        return np.exp(-0.5 * ((u / (sx * scale)) ** 2 + (v / (sy * scale)) ** 2)) * gain

    lo, hi = 0.05, 6.0
    for _ in range(12):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if float((np.clip(field(mid), 0, 1)[foot_s] > 0.75).mean()) < target else (lo, mid)
    a = cv2.resize(np.clip(field((lo + hi) / 2), 0, 1), (W, H), interpolation=cv2.INTER_CUBIC)
    a = np.clip(a + cv2.GaussianBlur(a, (0, 0), 15) * 0.3, 0, 1)[..., None]
    tint = np.array([rng.uniform(0.92, 1.0), rng.uniform(0.96, 1.0), 1.0], np.float32)  # BGR, 살짝 따뜻하게
    img[:] = 1 - (1 - np.clip(img, 0, 1)) * (1 - a * rng.uniform(0.9, 1.0) * tint)
    return round(float((a[..., 0][foot_scene > 0.5] > 0.75).mean()), 3)


# ═════════════════════════════════════════════
# 2) 배경 (실사진 or 절차적 텍스처) + 오염
# ═════════════════════════════════════════════
def list_images(d):
    if not d or not Path(d).exists():
        return []
    return sorted(str(p) for p in Path(d).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def photo_background(path, rng, W, H):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    cw = min(w, h * W / H) * rng.uniform(0.5, 1.0)
    ch = cw * H / W
    x, y = int(rng.uniform(0, w - cw + 1)), int(rng.uniform(0, h - ch + 1))
    crop = img[y:y + int(ch), x:x + int(cw)]
    crop = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA if crop.shape[1] > W else cv2.INTER_LINEAR)
    if chance(rng, 0.5):
        crop = crop[:, ::-1]
    return crop.astype(np.float32) / 255


def procedural_background(rng, W, H):
    kind = weighted(rng, {"cardboard": 35, "concrete": 20, "plastic": 15, "wood": 15, "metal": 15})
    fine = cv2.resize(rng.random((H // 2, W // 2), dtype=np.float32), (W, H))
    if kind == "cardboard":
        base = rng.uniform(0.60, 0.85) * np.array([0.52, 0.72, 1.0], np.float32)  # BGR
        fib = cv2.resize(rng.random((H, W // 10), dtype=np.float32), (W, H))
        tex = 1 + 0.18 * (smooth_noise(H, W, rng, 5) - 0.5) + 0.08 * (fine - 0.5) + 0.06 * (fib - 0.5)
    elif kind == "concrete":
        base = np.full(3, rng.uniform(0.50, 0.78), np.float32)
        tex = 1 + 0.28 * (smooth_noise(H, W, rng, 8) - 0.5) + 0.15 * (fine - 0.5)
    elif kind == "plastic":
        base = rng.uniform(0.70, 0.92) * np.array(pick(rng, [(1, 1, 1), (0.85, 0.95, 1.0), (1.0, 0.9, 0.8)]), np.float32)
        tex = 1 + 0.08 * (smooth_noise(H, W, rng, 4) - 0.5) + 0.04 * (fine - 0.5)
    elif kind == "wood":
        base = rng.uniform(0.62, 0.85) * np.array([0.6, 0.78, 1.0], np.float32)
        grain = cv2.resize(rng.random((H // 3, 40), dtype=np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        tex = 1 + 0.22 * (grain - 0.5) + 0.05 * (fine - 0.5)
    else:
        base = np.full(3, rng.uniform(0.55, 0.75), np.float32)
        brush = cv2.resize(rng.random((H, 30), dtype=np.float32), (W, H))
        tex = 1 + 0.12 * (brush - 0.5) + 0.05 * (fine - 0.5)
    return (base * tex[..., None]).astype(np.float32), kind


def grunge(img, rng):
    """얼룩, 긁힘, 먼지 점."""
    H, W = img.shape[:2]
    for _ in range(int(rng.integers(1, 4))):
        m = np.clip((smooth_noise(H // 8, W // 8, rng, 10) - rng.uniform(0.55, 0.75)) * rng.uniform(3, 8), 0, 1)
        m = cv2.resize(m, (W, H))[..., None] * rng.uniform(0.15, 0.45)
        img = img * (1 - m * rng.uniform(0.3, 0.7))
    lay = np.zeros((H, W), np.uint8)
    for _ in range(int(rng.integers(20, 100))):
        x, y, L, th = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(15, 220), rng.uniform(0, np.pi)
        cv2.line(lay, (int(x), int(y)), (int(x + L * np.cos(th)), int(y + L * np.sin(th))), 255, 1, cv2.LINE_AA)
    a = (lay.astype(np.float32) / 255 * rng.uniform(0.15, 0.4))[..., None]
    img = img * (1 - a) + (0.95 if chance(rng, 0.5) else 0.15) * a
    n = int(rng.integers(300, 2500))
    img[rng.integers(0, H, n), rng.integers(0, W, n)] *= rng.uniform(0.2, 0.7, (n, 1)).astype(np.float32)
    return img


# ═════════════════════════════════════════════
# 3) 배치 (막대 영역 패딩 박스 + 숫자까지 화면 안, 배경이 너무 어둡지 않은 곳)
# ═════════════════════════════════════════════
def transform(M, pts):
    pts = np.asarray(pts, np.float64)
    return (pts @ M[:, :2].T + M[:, 2]).astype(np.float32)


def rect(x0, y0, x1, y1):
    return np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def ratio_box(pts, axis_rad, ratio):
    """pts를 모두 감싸면서 방향 = 막대 가로축, 가로/세로 = ratio 인 최소 사각형 → 4꼭짓점."""
    c, s = np.cos(axis_rad), np.sin(axis_rad)
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    u, v = p[:, 0] * c + p[:, 1] * s, -p[:, 0] * s + p[:, 1] * c
    w0, h0 = np.ptp(u) + 1, np.ptp(v) + 1
    w, h = (h0 * ratio, h0) if w0 / h0 < ratio else (w0, w0 / ratio)
    uc, vc = (u.max() + u.min()) / 2, (v.max() + v.min()) / 2
    cen = np.array([uc * c - vc * s, uc * s + vc * c])
    ex, ey = np.array([c, s]) * w / 2, np.array([-s, c]) * h / 2
    return np.float32([cen - ex - ey, cen + ex - ey, cen + ex + ey, cen - ex + ey])


def luma_under(bg, poly):
    small = cv2.resize(bg, (bg.shape[1] // 4, bg.shape[0] // 4), interpolation=cv2.INTER_AREA)
    m = np.zeros(small.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.round(poly / 4).astype(np.int32), 1)
    if m.sum() == 0:
        return 0.0
    y = small[..., 0] * 0.114 + small[..., 1] * 0.587 + small[..., 2] * 0.299
    return float(y[m > 0].mean())


def place(rng, alpha_shape, bars, foot_pts, ratio, bg):
    """foot_pts: 변형된 막대 영역의 외곽 점들 (원본 캔버스 좌표)."""
    W, H, e = CFG["W"], CFG["H"], CFG["edge_margin"]
    Hc, Wc = alpha_shape
    x0, y0, x1, y1 = bars
    scale_pad = 1 + 2 * CFG["pad"]
    for _ in range(60):
        s = log_uniform(rng, *CFG["bars_screen_px"]) / (x1 - x0)
        ang = rng.uniform(-CFG["rot_small_deg"], CFG["rot_small_deg"]) if chance(rng, CFG["rot_small_prob"]) \
            else rng.uniform(-180, 180)
        M = cv2.getRotationMatrix2D((Wc / 2, Hc / 2), ang, s)  # 회전 + 확대만 → 비율 그대로
        M[:, 2] -= (Wc / 2, Hc / 2)                            # 캔버스 중심을 원점으로
        edge = transform(M, rect(x0, y0, x1, y1))              # 변형 전 가로축 방향 (박스 방향의 기준)
        axis = float(np.arctan2(edge[1][1] - edge[0][1], edge[1][0] - edge[0][0]))
        box = ratio_box(transform(M, foot_pts), axis, ratio)
        c = box.mean(0)
        padded = (box - c) * scale_pad + c
        pts = np.vstack([padded, transform(M, rect(0, 0, Wc, Hc))])
        lo, hi = pts.min(0), pts.max(0)
        if (hi - lo)[0] > W - 2 * e or (hi - lo)[1] > H - 2 * e:
            continue
        shift = np.float32([rng.uniform(e - lo[0], W - e - hi[0]), rng.uniform(e - lo[1], H - e - hi[1])])
        M[:, 2] += shift
        box, padded = box + shift, padded + shift
        if luma_under(bg, padded) < CFG["min_bg_luma"]:
            continue
        return M, box, padded, float(ang), float(s)
    return None


# ═════════════════════════════════════════════
# 4) 주변 방해물 (바코드 영역은 피해서 배치, 바코드는 절대 넣지 않음)
# ═════════════════════════════════════════════
def patch_paper(rng, fonts, size):
    w, h = int(size * rng.uniform(0.6, 1.5)), int(size * rng.uniform(0.35, 1.0))
    img = Image.new("RGB", (w, h), tuple(int(v) for v in pick(rng, [(248, 248, 245), (160, 240, 255), (215, 205, 255), (255, 230, 200)])))
    a = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    lh = max(10, int(h * rng.uniform(0.1, 0.18)))
    y = int(h * 0.08)
    while y + lh < h * 0.95:
        t = " ".join(pick(rng, WORDS) if chance(rng, 0.6) else digits(rng, int(rng.integers(3, 9)))
                     for _ in range(int(rng.integers(1, 4))))
        fonts.draw_fit(d, t, (int(w * 0.06), y, int(w * rng.uniform(0.5, 0.94)), y + lh), fill=(30, 30, 30))
        y += int(lh * rng.uniform(1.15, 1.5))
    return np.asarray(img, np.float32) / 255, np.asarray(a, np.float32) / 255


def patch_shape(rng, fonts, size):
    s = int(size)
    m = np.zeros((s, s), np.uint8)
    k = pick(rng, ["poly", "circle", "rect"])
    if k == "poly":
        cv2.fillPoly(m, [rng.uniform(0, s, (int(rng.integers(3, 7)), 2)).astype(np.int32)], 255, cv2.LINE_AA)
    elif k == "circle":
        cv2.circle(m, (s // 2, s // 2), s // 2 - 1, 255, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(m, (0, int(s * 0.25)), (s - 1, int(s * 0.75)), 255, -1)
    color = rng.uniform(0.05, 0.95, 3).astype(np.float32)
    return np.ones((s, s, 3), np.float32) * color, m.astype(np.float32) / 255


def patch_print(rng, fonts, size):
    """박스 표면에 직접 인쇄된 글씨."""
    w, h = int(size * rng.uniform(1.2, 2.5)), int(size * rng.uniform(0.3, 0.5))
    a = Image.new("L", (w, h), 0)
    fonts.draw_fit(ImageDraw.Draw(a), pick(rng, WORDS), (0, 0, w, h))
    color = np.array(pick(rng, [(20, 20, 20), (160, 60, 30), (40, 40, 180), (30, 120, 40)]), np.float32) / 255
    return np.ones((h, w, 3), np.float32) * color, np.asarray(a, np.float32) / 255 * rng.uniform(0.5, 0.9)


def patch_grid(rng, fonts, size):
    """격자/그물망 (규칙적인 선이라 바코드와 구분됨)."""
    s = int(size)
    m = np.zeros((s, s), np.uint8)
    p, t = int(rng.integers(10, 35)), int(rng.integers(1, 3))
    for k in range(0, s, p):
        cv2.line(m, (k, 0), (k, s), 255, t)
        cv2.line(m, (0, k), (s, k), 255, t)
    return np.full((s, s, 3), rng.uniform(0.1, 0.8), np.float32), m.astype(np.float32) / 255 * rng.uniform(0.5, 0.9)


def patch_scribble(rng, fonts, size):
    s = int(size)
    m = np.zeros((s, s), np.uint8)
    for _ in range(int(rng.integers(1, 4))):
        pts = np.clip(np.cumsum(rng.normal(0, s * 0.12, (int(rng.integers(4, 12)), 2)), 0) + s / 2, 0, s - 1)
        cv2.polylines(m, [pts.astype(np.int32)], False, 255, max(2, int(s * 0.02)), cv2.LINE_AA)
    color = np.array(pick(rng, [(20, 20, 20), (180, 40, 30), (30, 30, 170)]), np.float32) / 255
    return np.ones((s, s, 3), np.float32) * color, m.astype(np.float32) / 255 * 0.85


PATCHES = {"paper": (patch_paper, 30), "shape": (patch_shape, 20), "print": (patch_print, 20),
           "grid": (patch_grid, 12), "scribble": (patch_scribble, 18)}


def add_clutter(img, rng, fonts, keepout):
    """keepout(바코드 패딩 박스를 조금 더 키운 사각형)과 겹치지 않는 방해물만 그린다."""
    H, W = img.shape[:2]
    keep = np.zeros((H // 4, W // 4), np.uint8)
    cv2.fillConvexPoly(keep, np.round(keepout / 4).astype(np.int32), 1)
    n_ok = 0
    for _ in range(int(rng.integers(*CFG["clutter"])) * 2):  # 겹쳐서 버리는 몫까지 넉넉히 시도
        if n_ok >= CFG["clutter"][1]:
            break
        kind = weighted(rng, {**{k: v[1] for k, v in PATCHES.items()}, "tape": 10, "cable": 10})
        if kind in ("tape", "cable"):
            layer = np.zeros((H, W), np.uint8)
            if kind == "tape":
                c, th, bw = rng.uniform(0, [W, H]), rng.uniform(0, np.pi), rng.uniform(40, 160)
                dvec, along = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
                poly = np.array([c + along * 3000 + dvec * bw / 2, c - along * 3000 + dvec * bw / 2,
                                 c - along * 3000 - dvec * bw / 2, c + along * 3000 - dvec * bw / 2])
                cv2.fillConvexPoly(layer, poly.astype(np.int32), 255, cv2.LINE_AA)
                color, amax = np.array([0.35, 0.55, 0.72], np.float32) * rng.uniform(0.8, 1.1), rng.uniform(0.35, 0.8)
            else:
                t = np.linspace(0, 1, 50)[:, None]
                p = rng.uniform(-100, [W + 100, H + 100], (4, 2))
                curve = (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]
                cv2.polylines(layer, [curve.astype(np.int32)], False, 255, int(rng.integers(3, 16)), cv2.LINE_AA)
                color, amax = rng.uniform(0.02, 0.9, 3).astype(np.float32), 1.0
            if (cv2.resize(layer, (W // 4, H // 4), interpolation=cv2.INTER_AREA)[keep > 0] > 0).any():
                continue
            x, y, bw_, bh_ = cv2.boundingRect(layer)
            if bw_ == 0:
                continue
            a = (layer[y:y + bh_, x:x + bw_].astype(np.float32) / 255 * amax)[..., None]
            region = img[y:y + bh_, x:x + bw_]
            region[:] = region * (1 - a) + color * a
        else:
            rgb, a = PATCHES[kind][0](rng, fonts, log_uniform(rng, 50, 600))
            h, w = a.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(0, 360), 1.0)
            M[:, 2] += rng.uniform(-100, [W + 100, H + 100]) - (w / 2, h / 2)
            quad = transform(M, rect(0, 0, w, h))
            area, _ = cv2.intersectConvexConvex(quad, keepout)
            if area > 0:
                continue
            lo = np.maximum(np.floor(quad.min(0)).astype(int), 0)
            hi = np.minimum(np.ceil(quad.max(0)).astype(int), [W, H])
            if (hi - lo).min() < 2:
                continue
            T = M.copy()
            T[:, 2] -= lo
            size = (int(hi[0] - lo[0]), int(hi[1] - lo[1]))
            aw = cv2.warpAffine(a, T, size)[..., None]
            rw = cv2.warpAffine(rgb[:, :, ::-1].copy(), T, size)  # RGB→BGR
            region = img[lo[1]:hi[1], lo[0]:hi[0]]
            region[:] = region * (1 - aw) + rw * aw
        n_ok += 1
    return n_ok


# ═════════════════════════════════════════════
# 5) 조명 + 카메라 촬영 효과
# ═════════════════════════════════════════════
def lighting(img, rng):
    """조명 방향 그라데이션, 스포트 조명, 그림자 띠, 색온도, 비네팅."""
    H, W = img.shape[:2]
    h, w = H // 8, W // 8
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    th = rng.uniform(0, 2 * np.pi)
    proj = (xx / w - 0.5) * np.cos(th) + (yy / h - 0.5) * np.sin(th)
    field = 1 - rng.uniform(0, 0.45) * (proj - proj.min()) / (np.ptp(proj) + 1e-6)
    cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.3, 1.0) * w
    field += rng.uniform(0, 0.35) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    if chance(rng, 0.3):  # 블라인드/선반 그림자
        p = (xx * np.cos(th + 1) + yy * np.sin(th + 1)) % rng.uniform(8, 30)
        field *= 1 - rng.uniform(0.1, 0.3) * cv2.GaussianBlur((p < 4).astype(np.float32), (0, 0), 1.5)
    vig = 1 - rng.uniform(0, 0.35) * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 2
    field = cv2.resize(field * vig, (W, H), interpolation=cv2.INTER_CUBIC)
    temp = rng.uniform(-1, 1)  # -1 차가운 빛, +1 따뜻한 빛
    wb = np.array([1 - 0.12 * temp, 1.0, 1 + 0.12 * temp], np.float32)  # BGR
    img = img * field[..., None] * wb
    if chance(rng, 0.15):  # 광택면 반사광
        sx, sy = rng.uniform(80, 350), rng.uniform(30, 200)
        gx, gy = rng.uniform(0, W), rng.uniform(0, H)
        YY, XX = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32) * 4
        g = np.exp(-0.5 * (((XX - gx) / sx) ** 2 + ((YY - gy) / sy) ** 2)) * rng.uniform(0.3, 0.9)
        g = cv2.resize(g, (W, H))[..., None]
        img = 1 - (1 - np.clip(img, 0, 1)) * (1 - g)
    return img


def motion_kernel(length, angle):
    size = int(np.ceil(length)) // 2 * 2 + 3
    k = np.zeros((size * 4, size * 4), np.float32)
    c, dx, dy = size * 2, np.cos(np.deg2rad(angle)) * length * 2, np.sin(np.deg2rad(angle)) * length * 2
    cv2.line(k, (int(c - dx), int(c - dy)), (int(c + dx), int(c + dy)), 1.0, 4, cv2.LINE_AA)
    k = cv2.resize(k, (size, size), interpolation=cv2.INTER_AREA)
    return k / k.sum()


def disk_kernel(r):
    R = int(np.ceil(r)) + 1
    n = (2 * R + 1) * 4
    yy, xx = (np.mgrid[0:n, 0:n].astype(np.float32) + 0.5) / 4 - (R + 0.5)
    k = cv2.resize((np.hypot(xx, yy) <= r).astype(np.float32), (2 * R + 1, 2 * R + 1), interpolation=cv2.INTER_AREA)
    return k / k.sum()


TO_LIN = (np.linspace(0, 1, 4096) ** 2.2).astype(np.float32)
TO_SRGB = (np.linspace(0, 1, 4096) ** (1 / 2.2)).astype(np.float32)


def lut(img, table):
    return table[(np.clip(img, 0, 1) * 4095).astype(np.uint16)]


def camera(img, rng):
    """30fps 카메라 프레임: 노출 → 초점 흐림 → 모션 블러 → 센서 노이즈 → 감마 → JPEG."""
    H, W = img.shape[:2]
    lin = lut(img, TO_LIN) * rng.uniform(*CFG["exposure"])  # 선형광에서 노출·블러 (밝은 곳이 자연스럽게 번짐)

    u, acc, defocus = rng.random(), 0.0, 0.0
    for p, (lo, hi) in CFG["defocus"]:
        acc += p
        if u < acc:
            defocus = rng.uniform(lo, hi)
            break
    motion = rng.uniform(*(CFG["motion_small_px"] if chance(rng, CFG["motion_small_prob"]) else CFG["motion_large_px"]))
    m_ang = rng.uniform(0, 180)
    kernel = None
    if defocus >= 0.5:
        kernel = disk_kernel(defocus)
    if motion >= 0.7:
        mk = motion_kernel(motion, m_ang)
        if kernel is None:
            kernel = mk
        else:  # 두 흐림을 커널 하나로 합쳐 한 번만 필터링 (속도 2배)
            s = kernel.shape[0] + mk.shape[0] - 1
            big = np.zeros((s, s), np.float32)
            o = (s - kernel.shape[0]) // 2
            big[o:o + kernel.shape[0], o:o + kernel.shape[0]] = kernel
            kernel = cv2.filter2D(big, -1, mk, borderType=cv2.BORDER_CONSTANT)
            kernel /= kernel.sum()
    if kernel is not None:
        lin = cv2.filter2D(lin, -1, kernel, borderType=cv2.BORDER_REFLECT101)

    out = lut(lin, TO_SRGB)
    sigma = rng.uniform(*CFG["noise_sigma"]) / 255
    luma_noise = rng.standard_normal((H, W, 1), dtype=np.float32)
    chroma_noise = cv2.resize(rng.standard_normal((H // 2, W // 2, 3), dtype=np.float32), (W, H))
    shot = np.sqrt(np.clip(out, 0, 1))  # 밝은 곳일수록 샷 노이즈 ↑
    out = out + sigma * (0.8 * luma_noise + 0.5 * chroma_noise) * (0.5 + shot)
    out8 = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
    if chance(rng, 0.5):  # 카메라 ISP 샤프닝
        amt = rng.uniform(0.2, 0.8)
        out8 = cv2.addWeighted(out8, 1 + amt, cv2.GaussianBlur(out8, (0, 0), 1.0), -amt, 0)

    q = int(rng.integers(CFG["jpeg_quality"][0], CFG["jpeg_quality"][1] + 1))
    ok, buf = cv2.imencode(".jpg", out8, [cv2.IMWRITE_JPEG_QUALITY, q])
    params = {"defocus_px": round(defocus, 2), "motion_px": round(motion, 2), "motion_deg": round(m_ang, 1),
              "noise_sigma": round(sigma * 255, 2), "jpeg_quality": q}
    return buf.tobytes(), params


# ═════════════════════════════════════════════
# 한 장 생성
# ═════════════════════════════════════════════
def generate_one(idx, seed, fonts, bg_paths):
    rng = np.random.default_rng([seed, idx])  # 이미지 번호마다 고정 시드 → 재실행/병렬 실행해도 결과 동일
    W, H = CFG["W"], CFG["H"]
    cls = idx % 2
    name, (uw, uh) = CFG["classes"][cls]
    ratio = uw / uh
    alpha, bars, bmeta = make_barcode(rng, (uw, uh), fonts)

    # 1) 손상: 변형/찢김/마모는 바코드 평면에서, 빛반사는 합성 뒤 장면에서
    kind, kinds = damage_plan(idx, rng)
    alpha, foot, shade, bars, dmeta = apply_damage(alpha, bars, [k for k in kinds if k != "glare"], rng)
    cnts, _ = cv2.findContours((foot > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    foot_pts = np.vstack(cnts).reshape(-1, 2).astype(np.float32)[::3]  # 발자국 외곽 (정답 박스의 기준)

    # 2) 배경 + 배치 (막대 아래가 충분히 밝아야 검은 막대가 보임)
    for _ in range(30):
        if bg_paths and chance(rng, CFG["photo_ratio"]):
            path = pick(rng, bg_paths)
            bg, bg_kind = photo_background(path, rng, W, H), f"photo:{Path(path).name}"
            if bg is None:
                continue
        else:
            bg, bg_kind = procedural_background(rng, W, H)
        if chance(rng, 0.85):
            bg = grunge(bg, rng)
        placed = place(rng, alpha.shape, bars, foot_pts, ratio, bg)
        if placed:
            break
    else:
        raise RuntimeError(f"{idx}: 배치 가능한 배경을 찾지 못함")
    M, box, padded, angle, scale = placed

    # 3) 주변 노이즈 (바코드 패딩 박스를 1.15배 키운 영역은 비워둠)
    c = padded.mean(0)
    Hc, Wc = alpha.shape
    keepout = cv2.convexHull(np.vstack([(padded - c) * 1.15 + c, transform(M, rect(0, 0, Wc, Hc))])).reshape(-1, 2)
    n_clutter = add_clutter(bg, rng, fonts, keepout.astype(np.float32))

    # 4) 바코드 합성: 축소 시 계단/모아레 방지용 저역 필터 → 회전·확대 워프 → 잉크색 + 표면 음영
    blur = (lambda a: cv2.GaussianBlur(a, (0, 0), 0.45 / scale)) if scale < 0.9 else (lambda a: a)
    canvas_pts = transform(M, rect(0, 0, Wc, Hc))
    lo = np.maximum(np.floor(canvas_pts.min(0)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(canvas_pts.max(0)).astype(int) + 2, [W, H])
    T = M.copy()
    T[:, 2] -= lo
    size = (int(hi[0] - lo[0]), int(hi[1] - lo[1]))
    aw = cv2.warpAffine(blur(alpha), T, size, flags=cv2.INTER_LINEAR)[..., None]
    sw = cv2.warpAffine(shade, T, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)[..., None]
    ink = np.array([rng.uniform(0.02, 0.18)] * 3, np.float32) * rng.uniform(0.9, 1.1, 3).astype(np.float32)
    region = bg[lo[1]:hi[1], lo[0]:hi[0]]
    region[:] = (region * (1 - aw) + ink * aw) * sw

    # 5) 빛반사 (막대 영역 일부가 하얗게 날아감)
    glare_cover = 0.0
    if "glare" in kinds:
        foot_scene = np.zeros((H, W), np.float32)
        foot_scene[lo[1]:hi[1], lo[0]:hi[0]] = cv2.warpAffine(foot, T, size, flags=cv2.INTER_LINEAR)
        glare_cover = glare_on_bars(bg, foot_scene, rng)

    img = lighting(bg, rng)
    jpeg, cam = camera(img, rng)

    label = f"{cls} " + " ".join(f"{v:.6f}" for v in (box / np.float32([W, H])).reshape(-1))
    side_w, side_h = np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[3] - box[0])
    meta = {"index": idx, "class": name, **bmeta, "damage": kind, "damage_parts": "+".join(kinds) or "-",
            "glare_cover": glare_cover, **{k: v for k, v in dmeta.items()},
            "bars_screen_px": [round(float(side_w), 1), round(float(side_h), 1)],
            "angle_deg": round(angle, 2), "background": bg_kind, "clutter": n_clutter, **cam}
    return jpeg, label, meta, box, padded


# ═════════════════════════════════════════════
# 실행
# ═════════════════════════════════════════════
def crop_padded(img, box, pad):
    """막대 영역 박스를 가로로 펴서 10% 패딩 크롭 (추론 후처리와 같은 방식)."""
    c = box.mean(0)
    w, h = np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[3] - box[0])
    ang = np.degrees(np.arctan2(box[1][1] - box[0][1], box[1][0] - box[0][0]))
    Wc = int(round(w * (1 + 2 * pad)))
    Hc = int(round(Wc * h / w))
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, 1.0)
    M[:, 2] += (Wc / 2 - c[0], Hc / 2 - c[1])
    return cv2.warpAffine(img, M, (Wc, Hc), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def run_preview(args, fonts, bg_paths):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes, crops = [], []
    t0 = time.perf_counter()
    for i in range(args.start, args.start + args.preview):
        jpeg, label, meta, box, padded = generate_one(i, args.seed, fonts, bg_paths)
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        crops.append(cv2.resize(crop_padded(img, box, CFG["pad"]), (300, 180 if meta["class"] == "barcode_5x3" else 200)))
        ov = img.copy()
        cv2.polylines(ov, [box.astype(np.int32)], True, (255, 255, 0), 3, cv2.LINE_AA)
        cv2.polylines(ov, [padded.astype(np.int32)], True, (0, 220, 255), 2, cv2.LINE_AA)
        th = cv2.resize(ov, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(th, f"#{i} {meta['class']} [{meta['damage_parts']}] mb{meta['motion_px']:.0f} df{meta['defocus_px']:.1f}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        scenes.append(th)
    dt = (time.perf_counter() - t0) / args.preview

    def grid(cells, cols):
        hmax = max(c.shape[0] for c in cells)
        cells = [cv2.copyMakeBorder(c, 0, hmax - c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for c in cells]
        cells += [np.full_like(cells[0], 40)] * (-len(cells) % cols)
        return np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])

    cv2.imwrite(str(out / "preview_scenes.jpg"), grid(scenes, 4))
    cv2.imwrite(str(out / "preview_crops.jpg"), grid(crops, 6))
    print(f"✅ 미리보기 {args.preview}장 · 평균 {dt:.2f}초/장 → 1만 장 예상 {dt * 10000 / 3600:.1f}시간 (프로세스 1개 기준)")
    print(f"💾 {out / 'preview_scenes.jpg'} (하늘색 = 막대 영역 박스, 노랑 = 10% 패딩)")
    print(f"💾 {out / 'preview_crops.jpg'} (10% 패딩 크롭)")


def main():
    ap = argparse.ArgumentParser(description="바코드 비전 데이터셋 생성기")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--num", type=int, default=10000, help="전체 장수")
    ap.add_argument("--start", type=int, default=0, help="이 번호부터 (병렬 분할용)")
    ap.add_argument("--end", type=int, default=None, help="이 번호 전까지 (기본: --num)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bg-dir", default=None, help="실사 배경 사진 폴더 (없으면 절차적 배경)")
    ap.add_argument("--font", default=None, help="숫자용 .ttf 경로 (선택)")
    ap.add_argument("--preview", type=int, default=0, help="N장만 만들어 그리드로 확인")
    ap.add_argument("--kind", default=None, choices=list(CFG["damage"]), help="미리보기에서 특정 손상만")
    args = ap.parse_args()

    global FORCE_KIND
    FORCE_KIND = args.kind
    fonts = Fonts(args.font)
    bg_paths = list_images(args.bg_dir)
    print(f"배경: 실사 {len(bg_paths)}장" + (" (없음 → 절차적 배경 100%)" if not bg_paths else ""))
    if args.preview:
        return run_preview(args, fonts, bg_paths)

    out = Path(args.out).resolve()
    for sub in ("images", "labels"):
        for split in ("train", "val"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {k}: {v[0]}\n" for k, v in CFG["classes"].items()), encoding="utf-8")

    end = args.end if args.end is not None else args.num
    meta_path = out / f"meta_{args.start:06d}_{end:06d}.csv"
    fields = ["index", "split", "class", "symbology", "data", "module_px", "bars_render_px", "digits",
              "damage", "damage_parts", "glare_cover", "torn_ratio", "creases", "wave_lambda_px", "twist_deg",
              "fold_compress", "wear", "bars_screen_px", "angle_deg", "background", "clutter",
              "defocus_px", "motion_px", "motion_deg", "noise_sigma", "jpeg_quality"]
    new_file = not meta_path.exists()
    t0, made = time.perf_counter(), 0

    with open(meta_path, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for idx in range(args.start, end):  # ← 순차 생성 루프
            # 짝(5:3, 6:4) 단위로 나눠야 val에도 두 클래스가 반반 → (idx // 2) 기준
            split = "val" if (idx // 2) % CFG["val_every"] == CFG["val_every"] - 1 else "train"
            img_path = out / "images" / split / f"{idx:06d}.jpg"
            lbl_path = out / "labels" / split / f"{idx:06d}.txt"
            if img_path.exists() and lbl_path.exists():  # 이미 만든 건 건너뜀 (이어서 생성)
                continue
            jpeg, label, meta, _, _ = generate_one(idx, args.seed, fonts, bg_paths)
            img_path.write_bytes(jpeg)                   # 카메라 단계 JPEG 그대로 저장 (이중 압축 없음)
            lbl_path.write_text(label + "\n")
            writer.writerow({"split": split, **{k: v for k, v in meta.items() if k in fields}})
            made += 1
            if made % 50 == 0:
                fcsv.flush()
                el = time.perf_counter() - t0
                left = (end - idx - 1) * el / made
                print(f"  {idx + 1:,}/{end:,} · {el / 60:.1f}분 경과 · 남은 시간 약 {left / 60:.0f}분")

    print(f"✅ 완료: 새로 {made:,}장 생성 → {out}")


if __name__ == "__main__":
    main()
