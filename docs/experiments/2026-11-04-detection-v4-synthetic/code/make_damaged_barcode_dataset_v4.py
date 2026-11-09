#!/usr/bin/env python3
"""
바코드 검출 학습용 합성 데이터 v4 — 실카메라와의 차이를 줄인다.

**1280×720 으로 바로 그린다** (학습 imgsz 1280). 축소 단계가 없다.
v3 의 픽셀 단위 설정(바코드 크기·여백·블러·방해물 크기)은 이 화면 크기에 맞춰 다시 잡았다.
`--height 1080` 이면 1920×1080 으로 그린다 (학습 imgsz 1920, 11/04). 픽셀 값에 1.5 를 곱하면 v3 원래 값이 된다.

v3(`make_damaged_barcode_dataset_v3.py`)의 막대·손상·카메라 코드를 그대로 쓰고
아래만 바꾼다. 실행하려면 두 파일이 같은 폴더에 있어야 한다.

  1. 흰 라벨 종이 (80%) : 실제 택배 송장처럼 흰/미색 종이 위에 막대가 인쇄돼 있다.
                          글자 줄·구분선·칸·큰 숫자가 같이 있고, 종이는 막대와 함께
                          구겨지고 찢긴다 (같은 remap 을 공유). 나머지 20%는 v3 처럼
                          상자에 직접 인쇄된 막대.
  2. 어두운 바닥 허용    : 종이가 있으면 막대 밑 밝기는 종이가 보장하므로 검은 컨베이어·
                          어두운 상자 위에도 놓는다 (v3 는 밝은 배경에만 배치).
  3. 종이가 화면 밖으로   : 정답 박스 + 10% 여백만 화면 안이면 된다. 종이 끝은 잘려도 된다.
  4. 실카메라 열화        : 웹캠/IP 카메라 스트림처럼 저해상도 업스케일, 강한 재압축
                          (스트림 → 프레임 저장), 하이라이트 포화를 확률적으로 섞는다.

정답(YOLO-OBB, 막대 영역 4꼭짓점)과 클래스 규칙(짝수 5:3 / 홀수 6:4)은 v3 와 같다.
val 은 10쌍 중 1쌍 (v3 이전 dataset_720 과 같은 규칙) → 4만 장이면 36,000 / 4,000.

  python make_damaged_barcode_dataset_v4.py --preview 12 --out preview_v4
  python make_damaged_barcode_dataset_v4.py --out dataset_v4 --num 40000 --workers 3
"""

import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import make_damaged_barcode_dataset_v3 as v3
from make_damaged_barcode_dataset_v3 import (
    CFG,
    WORDS,
    Fonts,
    chance,
    digits,
    letters,
    pick,
    rect,
    smooth_noise,
    transform,
)

# 화면 픽셀 단위 값은 1280x720 기준으로 적고 PX(= 화면 높이 / 720)를 곱해 쓴다.
# --height 1080 이면 PX = 1.5 → v3 원래 값(1920x1080 기준)과 같아진다.
PX = 1.0


def set_height(height):
    """화면 크기를 정한다. v3 의 CFG 객체를 직접 고쳐서 v3 함수들도 이 값을 쓴다.
    Pool 워커는 모듈을 새로 읽으므로 워커 시작 때(_init)도 다시 불러야 한다."""
    global PX
    PX = height / 720
    CFG.update({
        "W": round(1280 * PX), "H": height,
        "bars_screen_px": (80 * PX, 547 * PX),      # 화면 속 막대 영역 가로 (로그 균등)
        "bars_big_px": (267 * PX, 547 * PX),        # 큰 바코드 보강 구간
        "edge_margin": round(20 * PX),
        "motion_small_px": (0, 2 * PX), "motion_large_px": (2 * PX, 8 * PX),
        "defocus": [(0.55, (0.0, 0.27 * PX)), (0.35, (0.33 * PX, 1.0 * PX)), (0.10, (1.0 * PX, 2.0 * PX))],
    })


set_height(720)

CFG4 = {
    "paper_prob": 0.80,
    "paper_w": (1.12, 1.9),      # 종이 가로 = 막대 가로 x 이 배수
    "paper_h": (1.7, 3.4),       # 종이 세로 = 막대 세로 x 이 배수
    "paper_luma": (0.82, 0.98),
    "lowres_prob": 0.30, "lowres_scale": (0.5, 0.9),     # 360p~650p 카메라를 720 으로 키운 느낌
    "recompress_prob": 0.30, "recompress_q": (25, 60),   # 스트림 압축 블록
    "clip_prob": 0.15,                                   # 자동노출 실패로 하이라이트 포화
    "val_every": 10,
}


# ═════════════════════════════════════════════
# 1) 라벨 종이 (막대 캔버스를 감싸는 큰 캔버스)
# ═════════════════════════════════════════════
def make_label(rng, alpha, bars, fonts):
    """막대 알파(v3.make_barcode 결과)를 종이 캔버스 안에 앉힌다.

    반환: ink(잉크 알파, 막대+글자), paper(종이 마스크), bars(새 캔버스 기준 막대 사각형)
    """
    x0, y0, x1, y1 = bars
    bw, bh = x1 - x0, y1 - y0
    ah, aw = alpha.shape
    pw = int(max(aw + 8, bw * rng.uniform(*CFG4["paper_w"])))
    ph = int(max(ah + 8, bh * rng.uniform(*CFG4["paper_h"])))
    # 막대는 종이 아래쪽에 두는 경우가 많다 (송장: 위에 주소·글자, 아래 바코드)
    ox = int(rng.uniform(0, pw - aw))
    oy = int((ph - ah) * (rng.uniform(0.55, 1.0) if chance(rng, 0.7) else rng.uniform(0.0, 0.45)))
    edge = int(0.04 * min(pw, ph)) + 4  # 종이 테두리 바깥 여유 (변형·그림자)
    W, H = pw + 2 * edge, ph + 2 * edge

    paper = np.zeros((H, W), np.float32)
    corner = int(rng.uniform(0, 0.04) * min(pw, ph))
    cv2.rectangle(paper, (edge + corner, edge), (edge + pw - 1 - corner, edge + ph - 1), 1.0, -1)
    cv2.rectangle(paper, (edge, edge + corner), (edge + pw - 1, edge + ph - 1 - corner), 1.0, -1)
    paper = cv2.GaussianBlur(paper, (0, 0), 0.7)

    text = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(text)
    bx0, by0 = edge + ox, edge + oy                       # 막대 캔버스 위치
    keep = (bx0 + x0 - 0.12 * bw, by0 + y0 - 0.15 * bh,   # 막대 + 조용한 여백은 비워 둔다
            bx0 + x1 + 0.12 * bw, by0 + ah + 0.04 * bh)

    def free(box):
        a0, b0, a1, b1 = box
        return a1 < keep[0] or a0 > keep[2] or b1 < keep[1] or b0 > keep[3]

    ink_w = max(1, int(0.006 * pw))
    y = edge + int(rng.uniform(0.03, 0.08) * ph)
    while y < edge + ph * 0.96:
        lh = int(ph * rng.uniform(0.035, 0.09))
        line = (edge + int(0.05 * pw), y, edge + int(pw * rng.uniform(0.45, 0.95)), y + lh)
        if free(line):
            r = rng.random()
            if r < 0.6:
                t = " ".join(pick(rng, WORDS) if chance(rng, 0.5) else digits(rng, int(rng.integers(3, 10)))
                             for _ in range(int(rng.integers(1, 4))))
                fonts.draw_fit(d, t, line)
            elif r < 0.75:  # 구분선
                d.line([(edge, y + lh // 2), (edge + pw, y + lh // 2)], fill=255, width=ink_w)
            elif r < 0.9:   # 칸 + 큰 숫자/문자 (분류 코드)
                d.rectangle(line, outline=255, width=ink_w)
                fonts.draw_fit(d, letters(rng, 1) + digits(rng, int(rng.integers(1, 4))),
                               (line[0] + 4, line[1] + 2, line[2] - 4, line[3] - 2))
        y += int(lh * rng.uniform(1.1, 1.6))
    if chance(rng, 0.6):  # 종이 테두리 인쇄선
        d.rectangle((edge + 3, edge + 3, edge + pw - 4, edge + ph - 4), outline=255, width=ink_w)

    ink = np.asarray(text, np.float32) / 255 * rng.uniform(0.8, 0.98)
    ink[by0:by0 + ah, bx0:bx0 + aw] = np.maximum(ink[by0:by0 + ah, bx0:bx0 + aw], alpha)
    ink *= paper > 0.5
    return ink, paper, (bx0 + x0, by0 + y0, bx0 + x1, by0 + y1)


# ═════════════════════════════════════════════
# 2) 손상 — v3.apply_damage 와 같은 변형을 종이에도 적용
# ═════════════════════════════════════════════
def apply_damage(alpha, paper, bars, kinds, rng):
    """v3.apply_damage 와 같은 규칙. 종이 마스크도 같은 remap 으로 휘고 같이 찢긴다.

    → (잉크 알파, 종이 마스크, 발자국, 음영, 막대 사각형, 메타)
    """
    x0, y0, x1, y1 = bars
    bw, bh = x1 - x0, y1 - y0
    short = min(bw, bh)
    pad = int(0.20 * short) + 8
    border = lambda a: cv2.copyMakeBorder(a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    alpha, paper = border(alpha), border(paper)
    H, W = alpha.shape
    x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    foot = np.zeros((H, W), np.float32)
    foot[y0:y1, x0:x1] = 1
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    mx, my = xx.copy(), yy.copy()
    height = np.zeros((H, W), np.float32)
    info = {}

    if "twist" in kinds:
        tmax = np.deg2rad(rng.uniform(6, 20)) * pick(rng, [-1, 1])
        th = tmax * (xx - cx) / (bw / 2)
        px, py = xx - cx, yy - cy
        mx, my = cx + px * np.cos(th) - py * np.sin(th), cy + px * np.sin(th) + py * np.cos(th)
        height += rng.uniform(0.02, 0.05) * short * np.sin(th * 2)
        info["twist_deg"] = round(float(np.degrees(tmax)), 1)
    if "fold" in kinds:
        th = rng.uniform(0, np.pi)
        nx, ny = np.cos(th), np.sin(th)
        px, py = rng.uniform(x0 + 0.25 * bw, x1 - 0.25 * bw), rng.uniform(y0 + 0.25 * bh, y1 - 0.25 * bh)
        d = (xx - px) * nx + (yy - py) * ny
        comp = rng.uniform(0.55, 0.85)
        k = (1 / comp - 1) * (d > 0)
        mx, my = mx + d * k * nx, my + d * k * ny
        height += np.where(d > 0, -rng.uniform(0.04, 0.12) * d, 0)
        info["fold_compress"] = round(float(comp), 2)
    if "crumple" in kinds:
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
    if "wave" in kinds:
        lam = rng.uniform(0.25, 0.8) * bw
        th = rng.uniform(0, np.pi)
        height += rng.uniform(0.06, 0.16) * short * np.sin(
            2 * np.pi * (xx * np.cos(th) + yy * np.sin(th)) / lam + rng.uniform(0, 6.28))
        info["wave_lambda_px"] = round(float(lam), 1)
    if np.any(height):
        hn = height / (np.abs(height).max() + 1e-6)
        k = rng.uniform(-1, 1, 2) * rng.uniform(0.03, 0.09) * short
        mx, my = mx - k[0] * hn, my - k[1] * hn

    mx, my = mx.astype(np.float32), my.astype(np.float32)
    rm = lambda a: cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    alpha, paper, foot, height = rm(alpha), rm(paper), rm(foot), rm(height)

    shade = np.ones((H, W), np.float32)
    if np.any(height):
        gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8
        nrm = np.sqrt(gx ** 2 + gy ** 2 + 1)
        light = rng.uniform(-1, 1, 2)
        light = np.append(light / (np.linalg.norm(light) + 1e-6) * 0.7, 0.72).astype(np.float32)
        diff = (-gx / nrm) * light[0] + (-gy / nrm) * light[1] + (1 / nrm) * light[2]
        # 종이까지 같이 음영이 지도록 범위를 종이로 넓힌다
        area = np.maximum(foot, paper)
        feather = cv2.GaussianBlur(np.clip(cv2.dilate(area, np.ones((9, 9), np.uint8)), 0, 1), (0, 0), 0.05 * short + 2)
        shade = 1 + (np.clip(diff / light[2], 0.62, 1.25) - 1) * feather

    if "tear" in kinds:  # 종이째 찢겨 나가 바닥이 드러난다. 박스는 원래 영역 유지 (v3 와 같음)
        torn = v3.tear_mask((H, W), (x0, y0, x1, y1), rng)
        alpha, paper = alpha * (1 - torn), paper * (1 - torn)
        shade = shade * (1 - 0.35 * cv2.GaussianBlur(torn, (0, 0), 3) * (1 - torn))
        info["torn_ratio"] = round(float((torn * foot).sum() / max(foot.sum(), 1)), 3)
    if "wear" in kinds:
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

    return np.clip(alpha, 0, 1), np.clip(paper, 0, 1), foot, shade, (x0, y0, x1, y1), info


# ═════════════════════════════════════════════
# 3) 배치 — 정답 박스 + 여백만 화면 안. 종이가 있으면 바닥 밝기 조건 없음
# ═════════════════════════════════════════════
def place(rng, alpha_shape, bars, foot_pts, ratio, bg, on_paper):
    W, H, e = CFG["W"], CFG["H"], CFG["edge_margin"]
    Hc, Wc = alpha_shape
    x0, y0, x1, y1 = bars
    scale_pad = 1 + 2 * CFG["pad"]
    for _ in range(60):
        target_px = rng.uniform(*CFG["bars_big_px"]) if chance(rng, 0.5) \
            else v3.log_uniform(rng, *CFG["bars_screen_px"])
        s = target_px / (x1 - x0)
        ang = rng.uniform(-CFG["rot_small_deg"], CFG["rot_small_deg"]) if chance(rng, CFG["rot_small_prob"]) \
            else rng.uniform(-180, 180)
        M = cv2.getRotationMatrix2D((Wc / 2, Hc / 2), ang, s)
        M[:, 2] -= (Wc / 2, Hc / 2)
        edge = transform(M, rect(x0, y0, x1, y1))
        axis = float(np.arctan2(edge[1][1] - edge[0][1], edge[1][0] - edge[0][0]))
        box = v3.ratio_box(transform(M, foot_pts), axis, ratio)
        c = box.mean(0)
        padded = (box - c) * scale_pad + c
        # 종이가 있으면 종이는 화면 밖으로 나가도 된다. 막대 박스 + 여백만 들어오면 된다
        pts = padded if on_paper else np.vstack([padded, transform(M, rect(0, 0, Wc, Hc))])
        lo, hi = pts.min(0), pts.max(0)
        if (hi - lo)[0] > W - 2 * e or (hi - lo)[1] > H - 2 * e:
            continue
        shift = np.float32([rng.uniform(e - lo[0], W - e - hi[0]), rng.uniform(e - lo[1], H - e - hi[1])])
        M[:, 2] += shift
        box, padded = box + shift, padded + shift
        if not on_paper and v3.luma_under(bg, padded) < CFG["min_bg_luma"]:
            continue
        return M, box, padded, float(ang), float(s)
    return None


def dark_background(rng, W, H):
    """검은 컨베이어 벨트·어두운 상자. v3 배경은 전부 밝은 편이라 종이 장면에 섞는다."""
    base = np.full(3, rng.uniform(0.06, 0.30), np.float32) * rng.uniform(0.9, 1.1, 3).astype(np.float32)
    fine = cv2.resize(rng.random((H // 2, W // 2), dtype=np.float32), (W, H))
    belt = cv2.resize(rng.random((4, W // 6), dtype=np.float32), (W, H))  # 벨트 진행 방향 줄무늬
    tex = 1 + 0.25 * (smooth_noise(H, W, rng, 6) - 0.5) + 0.15 * (fine - 0.5) + 0.12 * (belt - 0.5)
    return (base * tex[..., None]).astype(np.float32), "dark_belt"


# ═════════════════════════════════════════════
# 3-b) v3 에서 픽셀 크기가 코드에 박혀 있던 함수들 — 1280x720 기준 값으로 옮겼다
# ═════════════════════════════════════════════
def grunge(img, rng):
    """얼룩, 긁힘, 먼지 점."""
    H, W = img.shape[:2]
    for _ in range(int(rng.integers(1, 4))):
        m = np.clip((smooth_noise(H // 8, W // 8, rng, 10) - rng.uniform(0.55, 0.75)) * rng.uniform(3, 8), 0, 1)
        m = cv2.resize(m, (W, H))[..., None] * rng.uniform(0.15, 0.45)
        img = img * (1 - m * rng.uniform(0.3, 0.7))
    lay = np.zeros((H, W), np.uint8)
    for _ in range(int(rng.integers(20, 100))):
        x, y, L, th = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(10 * PX, 147 * PX), rng.uniform(0, np.pi)
        cv2.line(lay, (int(x), int(y)), (int(x + L * np.cos(th)), int(y + L * np.sin(th))), 255, 1, cv2.LINE_AA)
    a = (lay.astype(np.float32) / 255 * rng.uniform(0.15, 0.4))[..., None]
    img = img * (1 - a) + (0.95 if chance(rng, 0.5) else 0.15) * a
    n = int(rng.integers(round(133 * PX ** 2), round(1111 * PX ** 2)))  # 먼지 개수는 면적에 비례
    img[rng.integers(0, H, n), rng.integers(0, W, n)] *= rng.uniform(0.2, 0.7, (n, 1)).astype(np.float32)
    return img


def add_clutter(img, rng, fonts, keepout):
    """keepout(바코드 패딩 박스·종이를 감싼 영역)과 겹치지 않는 방해물만 그린다."""
    H, W = img.shape[:2]
    keep = np.zeros((H // 4, W // 4), np.uint8)
    cv2.fillConvexPoly(keep, np.round(keepout / 4).astype(np.int32), 1)
    n_ok = 0
    for _ in range(int(rng.integers(*CFG["clutter"])) * 2):
        if n_ok >= CFG["clutter"][1]:
            break
        kind = v3.weighted(rng, {**{k: v[1] for k, v in v3.PATCHES.items()}, "tape": 10, "cable": 10})
        if kind in ("tape", "cable"):
            layer = np.zeros((H, W), np.uint8)
            if kind == "tape":
                c, th, bw = rng.uniform(0, [W, H]), rng.uniform(0, np.pi), rng.uniform(27 * PX, 107 * PX)
                dvec, along = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
                poly = np.array([c + along * 2000 + dvec * bw / 2, c - along * 2000 + dvec * bw / 2,
                                 c - along * 2000 - dvec * bw / 2, c + along * 2000 - dvec * bw / 2])
                cv2.fillConvexPoly(layer, poly.astype(np.int32), 255, cv2.LINE_AA)
                color, amax = np.array([0.35, 0.55, 0.72], np.float32) * rng.uniform(0.8, 1.1), rng.uniform(0.35, 0.8)
            else:
                t = np.linspace(0, 1, 50)[:, None]
                p = rng.uniform(-67 * PX, [W + 67 * PX, H + 67 * PX], (4, 2))
                curve = (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]
                cv2.polylines(layer, [curve.astype(np.int32)], False, 255, round(int(rng.integers(2, 11)) * PX), cv2.LINE_AA)
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
            rgb, a = v3.PATCHES[kind][0](rng, fonts, v3.log_uniform(rng, 33 * PX, 400 * PX))
            h, w = a.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(0, 360), 1.0)
            M[:, 2] += rng.uniform(-67 * PX, [W + 67 * PX, H + 67 * PX]) - (w / 2, h / 2)
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


def lighting(img, rng):
    """v3.lighting 과 같다. 광택면 반사광 크기만 1280x720 기준."""
    H, W = img.shape[:2]
    h, w = H // 8, W // 8
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    th = rng.uniform(0, 2 * np.pi)
    proj = (xx / w - 0.5) * np.cos(th) + (yy / h - 0.5) * np.sin(th)
    field = 1 - rng.uniform(0, 0.45) * (proj - proj.min()) / (np.ptp(proj) + 1e-6)
    cx, cy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.3, 1.0) * w
    field += rng.uniform(0, 0.35) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    if chance(rng, 0.3):
        p = (xx * np.cos(th + 1) + yy * np.sin(th + 1)) % rng.uniform(8, 30)
        field *= 1 - rng.uniform(0.1, 0.3) * cv2.GaussianBlur((p < 4).astype(np.float32), (0, 0), 1.5)
    vig = 1 - rng.uniform(0, 0.35) * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 2
    field = cv2.resize(field * vig, (W, H), interpolation=cv2.INTER_CUBIC)
    temp = rng.uniform(-1, 1)
    wb = np.array([1 - 0.12 * temp, 1.0, 1 + 0.12 * temp], np.float32)
    img = img * field[..., None] * wb
    if chance(rng, 0.15):
        sx, sy = rng.uniform(53 * PX, 233 * PX), rng.uniform(20 * PX, 133 * PX)
        gx, gy = rng.uniform(0, W), rng.uniform(0, H)
        YY, XX = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32) * 4
        g = np.exp(-0.5 * (((XX - gx) / sx) ** 2 + ((YY - gy) / sy) ** 2)) * rng.uniform(0.3, 0.9)
        g = cv2.resize(g, (W, H))[..., None]
        img = 1 - (1 - np.clip(img, 0, 1)) * (1 - g)
    return img


# ═════════════════════════════════════════════
# 4) 실카메라 열화 (v3.camera 뒤에 붙인다)
# ═════════════════════════════════════════════
def degrade(jpeg, rng):
    """저해상도 업스케일 · 하이라이트 포화 · 스트림 재압축. 적용한 게 없으면 원본 바이트 그대로."""
    lowres, clip, recompress = (chance(rng, CFG4[k]) for k in ("lowres_prob", "clip_prob", "recompress_prob"))
    params = {"lowres": 0.0, "clip": clip, "recompress_q": 0}
    if not (lowres or clip or recompress):
        return jpeg, params
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    if lowres:
        s = rng.uniform(*CFG4["lowres_scale"])
        small = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (W, H), interpolation=pick(rng, [cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
        params["lowres"] = round(float(s), 2)
    if clip:
        img = cv2.convertScaleAbs(img, alpha=rng.uniform(1.15, 1.5), beta=rng.uniform(0, 25))
    q = int(rng.integers(*CFG4["recompress_q"])) if recompress else 92
    params["recompress_q"] = q if recompress else 0
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes(), params


# ═════════════════════════════════════════════
# 한 장 생성
# ═════════════════════════════════════════════
def generate_one(idx, seed, fonts, bg_paths):
    rng = np.random.default_rng([seed, 4, idx])  # v3 와 다른 시드 공간 (같은 번호라도 다른 장면)
    W, H = CFG["W"], CFG["H"]
    cls = idx % 2
    name, (uw, uh) = CFG["classes"][cls]
    ratio = uw / uh
    alpha, bars, bmeta = v3.make_barcode(rng, (uw, uh), fonts)

    on_paper = chance(rng, CFG4["paper_prob"])
    if on_paper:
        alpha, paper, bars = make_label(rng, alpha, bars, fonts)
    else:
        paper = np.zeros_like(alpha)

    kind, kinds = v3.damage_plan(idx, rng)
    alpha, paper, foot, shade, bars, dmeta = apply_damage(alpha, paper, bars, [k for k in kinds if k != "glare"], rng)
    cnts, _ = cv2.findContours((foot > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 볼록 껍질 — [::3] 솎기는 직사각형 발자국(꼭짓점 4개)에서 윗변만 남겨 박스를 밀었다 (v3 와 같은 수정)
    foot_pts = cv2.convexHull(np.vstack(cnts).reshape(-1, 2)).reshape(-1, 2).astype(np.float32)

    for _ in range(30):
        if on_paper and chance(rng, 0.35):
            bg, bg_kind = dark_background(rng, W, H)
        elif bg_paths and chance(rng, CFG["photo_ratio"]):
            path = pick(rng, bg_paths)
            bg, bg_kind = v3.photo_background(path, rng, W, H), f"photo:{Path(path).name}"
            if bg is None:
                continue
        else:
            bg, bg_kind = v3.procedural_background(rng, W, H)
        if chance(rng, 0.85):
            bg = grunge(bg, rng)
        placed = place(rng, alpha.shape, bars, foot_pts, ratio, bg, on_paper)
        if placed:
            break
    else:
        raise RuntimeError(f"{idx}: 배치 가능한 배경을 찾지 못함")
    M, box, padded, angle, scale = placed

    c = padded.mean(0)
    Hc, Wc = alpha.shape
    canvas = transform(M, rect(0, 0, Wc, Hc))
    # 방해물은 종이 전체(화면 안 부분)와 겹치지 않게 — 종이 위에 다른 물건이 얹힌 장면은 v3 범위 밖
    keepout = cv2.convexHull(np.vstack([(padded - c) * 1.15 + c, canvas])).reshape(-1, 2)
    n_clutter = add_clutter(bg, rng, fonts, keepout.astype(np.float32))

    blur = (lambda a: cv2.GaussianBlur(a, (0, 0), 0.45 / scale)) if scale < 0.9 else (lambda a: a)
    lo = np.clip(np.floor(canvas.min(0)).astype(int) - 2, 0, [W, H])
    hi = np.clip(np.ceil(canvas.max(0)).astype(int) + 2, 0, [W, H])
    T = M.copy()
    T[:, 2] -= lo
    size = (int(hi[0] - lo[0]), int(hi[1] - lo[1]))
    warp = lambda a, bv=0.0: cv2.warpAffine(a, T, size, flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT, borderValue=bv)
    aw, sw = warp(blur(alpha))[..., None], warp(shade, 1.0)[..., None]
    region = bg[lo[1]:hi[1], lo[0]:hi[0]]
    if on_paper:
        pw = warp(paper)[..., None]
        # 종이 그림자: 바닥 위에 살짝 떠 있는 느낌
        off = (np.array([rng.uniform(-1, 1), rng.uniform(0.3, 1)]) * rng.uniform(2 * PX, 10 * PX)).astype(np.float32)
        shadow = cv2.GaussianBlur(cv2.warpAffine(pw[..., 0], np.float32([[1, 0, off[0]], [0, 1, off[1]]]), size),
                                  (0, 0), rng.uniform(3 * PX, 10 * PX))[..., None]
        region[:] = region * (1 - rng.uniform(0.25, 0.55) * shadow)
        luma = rng.uniform(*CFG4["paper_luma"])
        tint = np.array([1.0, 1.0, 1.0]) if chance(rng, 0.7) else np.array([0.88, 0.96, 1.0])  # 감열지 미색
        tex = 1 + 0.04 * (smooth_noise(size[1], size[0], rng, 12) - 0.5)
        paper_rgb = (luma * tint).astype(np.float32) * tex[..., None]
        region[:] = region * (1 - pw) + paper_rgb * pw
    ink = np.array([rng.uniform(0.02, 0.18)] * 3, np.float32) * rng.uniform(0.9, 1.1, 3).astype(np.float32)
    region[:] = (region * (1 - aw) + ink * aw) * sw

    foot_scene = np.zeros((H, W), np.float32)
    foot_scene[lo[1]:hi[1], lo[0]:hi[0]] = warp(foot)
    glare_cover = 0.0
    if "glare" in kinds:
        glare_cover = v3.glare_on_bars(bg, foot_scene, rng)
    # 정답 박스 검사값 — 발자국이 박스 안에 들어간 비율 / 박스를 채운 비율 (11/04 라벨 밀림 버그 재발 감시)
    fmask = foot_scene > 0.5
    bmask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(bmask, np.round(box).astype(np.int32), 1)
    label_cover = round(float(bmask[fmask].mean()), 4) if fmask.any() else 0.0
    box_fill = round(float(fmask.sum() / max(int(bmask.sum()), 1)), 4)

    img = lighting(bg, rng)
    jpeg, cam = v3.camera(img, rng)
    jpeg, deg = degrade(jpeg, rng)

    label = f"{cls} " + " ".join(f"{v:.6f}" for v in (box / np.float32([W, H])).reshape(-1))
    side_w, side_h = np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[3] - box[0])
    meta = {"index": idx, "class": name, **bmeta, "paper": on_paper, "damage": kind,
            "damage_parts": "+".join(kinds) or "-", "glare_cover": glare_cover, **dmeta,
            "label_cover": label_cover, "box_fill": box_fill,
            "bars_screen_px": [round(float(side_w), 1), round(float(side_h), 1)],
            "angle_deg": round(angle, 2), "background": bg_kind, "clutter": n_clutter, **cam, **deg}
    return jpeg, label, meta, box, padded


# ═════════════════════════════════════════════
# 실행
# ═════════════════════════════════════════════
FIELDS = ["index", "split", "class", "symbology", "data", "module_px", "bars_render_px", "digits", "paper",
          "damage", "damage_parts", "glare_cover", "torn_ratio", "creases", "wave_lambda_px", "twist_deg",
          "fold_compress", "wear", "bars_screen_px", "angle_deg", "background", "clutter",
          "defocus_px", "motion_px", "motion_deg", "noise_sigma", "jpeg_quality", "lowres", "clip", "recompress_q",
          "label_cover", "box_fill"]
_WORKER = {}


def _init(seed, font, bg_dir, out, height):
    set_height(height)
    _WORKER.update(seed=seed, fonts=Fonts(font), bg=v3.list_images(bg_dir), out=Path(out))


def split_of(idx):
    # 짝(5:3, 6:4) 단위로 나눠야 val 에도 두 클래스가 반반 → (idx // 2) 기준
    return "val" if (idx // 2) % CFG4["val_every"] == CFG4["val_every"] - 1 else "train"


def _make(idx):
    out, split = _WORKER["out"], split_of(idx)
    img_path = out / "images" / split / f"{idx:06d}.jpg"
    lbl_path = out / "labels" / split / f"{idx:06d}.txt"
    if img_path.exists() and lbl_path.exists():  # 이어서 생성
        return None
    jpeg, label, meta, _, _ = generate_one(idx, _WORKER["seed"], _WORKER["fonts"], _WORKER["bg"])
    img_path.write_bytes(jpeg)
    lbl_path.write_text(label + "\n")
    return {"split": split, **{k: v for k, v in meta.items() if k in FIELDS}}


def run_preview(args, fonts, bg_paths):
    # v3 미리보기를 그대로 쓰되 장면 생성만 v4 로 바꾼다
    v3.generate_one = lambda i, seed, f, b: generate_one(i, seed, f, b)
    v3.run_preview(args, fonts, bg_paths)


def main():
    ap = argparse.ArgumentParser(description="바코드 검출 합성 데이터 v4 (라벨 종이 + 실카메라 열화)")
    ap.add_argument("--out", default="dataset_v4")
    ap.add_argument("--num", type=int, default=40000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bg-dir", default=None, help="실사 배경 사진 폴더 (바코드 없는 사진만)")
    ap.add_argument("--font", default=None)
    ap.add_argument("--workers", type=int, default=1, help="병렬 프로세스 수 (CPU 코어 수 정도)")
    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--kind", default=None, choices=list(CFG["damage"]))
    ap.add_argument("--height", type=int, default=720, choices=[720, 1080],
                    help="화면 높이. 1080 이면 1920x1080 (학습 imgsz 1920 용)")
    args = ap.parse_args()

    set_height(args.height)
    print(f"화면: {CFG['W']}x{CFG['H']} (픽셀 배율 {PX:g})")
    v3.FORCE_KIND = args.kind
    fonts = Fonts(args.font)
    bg_paths = v3.list_images(args.bg_dir)
    print(f"배경: 실사 {len(bg_paths)}장" + (" (없음 → 절차적·어두운 벨트 배경)" if not bg_paths else ""))
    if args.preview:
        return run_preview(args, fonts, bg_paths)

    out = Path(args.out).resolve()
    for sub in ("images", "labels"):
        for split in ("train", "val"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {k}: {v[0]}\n" for k, v in CFG["classes"].items()), encoding="utf-8")

    end = args.end if args.end is not None else args.num
    meta_path = out / f"meta_{args.start:06d}_{end:06d}.csv"
    new_file = not meta_path.exists()
    t0, made, todo = time.perf_counter(), 0, end - args.start
    with open(meta_path, "a", newline="", encoding="utf-8") as fcsv, \
            Pool(args.workers, initializer=_init, initargs=(args.seed, args.font, args.bg_dir, out, args.height)) as pool:
        writer = csv.DictWriter(fcsv, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, row in enumerate(pool.imap(_make, range(args.start, end), chunksize=8), 1):
            if row:
                writer.writerow(row)
                made += 1
            if i % 200 == 0:
                fcsv.flush()
                el = time.perf_counter() - t0
                print(f"  {i:,}/{todo:,} · {el / 60:.1f}분 경과 · 남은 시간 약 {(todo - i) * el / i / 60:.0f}분", flush=True)
    print(f"✅ 완료: 새로 {made:,}장 생성 → {out}")


if __name__ == "__main__":
    main()
