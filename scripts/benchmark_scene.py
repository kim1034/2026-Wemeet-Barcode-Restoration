"""벤치마크 장면 — 컨베이어 벨트 위 택배 상자·비닐 봉투의 송장 라벨에 바코드를 붙인다.

    벨트 (고무 결 · 이음매 · 위아래 금속 레일)
      └ 포장 (골판지 상자 + 테이프 | 비닐 봉투 + 주름 · 광택) + 그림자
          └ 송장 라벨 (흰 종이 · 글자 줄 · 칸 선)
              └ 바코드 크롭 — **마지막에 원본 픽셀 그대로 붙인다**

크롭은 리사이즈·회전하지 않고, 조명·광택도 크롭 위에는 얹지 않는다. 구간 라벨의
「정답 제어점으로 펴면 읽힘」은 크롭 원본에서 판정한 것이라, 장면에서 크롭을
바꾸면 그 판정과 정답 제어점이 둘 다 어긋난다. 그래서 포장도 회전하지 않는다.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw

W, H = 1920, 1080


def _noise(rng, h, w, cells):
    """부드러운 저주파 노이즈 (0~1)."""
    g = rng.random((max(2, cells * h // max(h, w)), max(2, cells * w // max(h, w))))
    return cv2.resize(g.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def _belt(rng):
    rail = int(rng.integers(70, 120))
    img = np.empty((H, W, 3), np.float32)
    # 고무 벨트: 어둡고, 진행 방향(가로)으로 긴 결이 있다
    base = rng.uniform(0.10, 0.22)
    streak = cv2.resize(rng.random((H // 2, 6), dtype=np.float32), (W, H))
    fine = rng.random((H, W), dtype=np.float32)
    belt = base * (1 + 0.25 * (streak - 0.5) + 0.20 * (fine - 0.5))
    img[:] = belt[..., None] * np.array([1.0, 1.0, rng.uniform(0.95, 1.08)], np.float32)
    # 이음매: 진행 방향에 수직인 어두운 줄
    period, phase = rng.uniform(180, 420), rng.uniform(0, 420)
    for x in np.arange(phase % period, W, period):
        img[:, int(x) : int(x) + 3] *= 0.55
    # 위아래 금속 레일 (가로로 긁힌 결 + 볼트)
    for y0 in (0, H - rail):
        brushed = cv2.resize(rng.random((rail, 12), dtype=np.float32), (W, rail))
        img[y0 : y0 + rail] = (rng.uniform(0.50, 0.68) * (1 + 0.12 * (brushed - 0.5)))[..., None]
        edge = rail - 4 if y0 == 0 else 0
        img[y0 + edge : y0 + edge + 4] = 0.05
        for x in np.arange(rng.uniform(30, 90), W, rng.uniform(150, 260)):
            cv2.circle(img, (int(x), y0 + rail // 2), 7, (0.30, 0.30, 0.30), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x) - 2, y0 + rail // 2 - 2), 2, (0.85, 0.85, 0.85), -1)
    return img, rail


def _cardboard(rng, h, w):
    base = rng.uniform(0.55, 0.78) * np.array([0.52, 0.70, 0.95], np.float32)  # BGR
    fib = cv2.resize(rng.random((h, max(2, w // 10)), dtype=np.float32), (w, h))
    tex = 1 + 0.12 * (_noise(rng, h, w, 5) - 0.5) + 0.06 * (fib - 0.5)
    pkg = base * tex[..., None]
    # 테이프: 가로나 세로로 한 줄, 약간 반짝인다
    tw = int(rng.uniform(0.12, 0.22) * min(h, w))
    tape = (
        np.array([0.55, 0.72, 0.85], np.float32) if rng.random() < 0.5 else pkg.mean((0, 1)) * 1.12
    )
    if rng.random() < 0.5:
        c = int(rng.uniform(0.3, 0.7) * h)
        pkg[c - tw // 2 : c + tw // 2] = pkg[c - tw // 2 : c + tw // 2] * 0.4 + tape * 0.6 + 0.04
    else:
        c = int(rng.uniform(0.3, 0.7) * w)
        pkg[:, c - tw // 2 : c + tw // 2] = (
            pkg[:, c - tw // 2 : c + tw // 2] * 0.4 + tape * 0.6 + 0.04
        )
    return pkg, "box"


def _vinyl(rng, h, w):
    # BGR: 회색 · 흰색 · 검정 · 남색 택배 봉투
    color = [(0.70, 0.70, 0.70), (0.90, 0.90, 0.90), (0.14, 0.14, 0.15), (0.32, 0.20, 0.14)]
    base = np.array(color[int(rng.integers(len(color)))], np.float32)
    # 주름: 봉투 길이 방향으로 길게 늘어진 음영 (가로로 늘린 노이즈)
    along = rng.random() < 0.5
    small = rng.random((3, int(rng.integers(10, 20))) if along else (int(rng.integers(10, 20)), 3))
    folds = cv2.resize(small.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    shade = 1 + 0.30 * (folds - 0.5) + 0.06 * (_noise(rng, h, w, 12) - 0.5)
    pkg = base * shade[..., None]
    # 광택: 주름 능선 위로 부드러운 띠 + 접힌 선 몇 개
    gloss = np.clip((folds - rng.uniform(0.6, 0.75)) * 4, 0, 1)
    lines = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(2, 6))):
        p0 = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        ang, length = rng.uniform(0, np.pi), rng.uniform(0.2, 0.6) * max(h, w)
        p1 = (int(p0[0] + length * np.cos(ang)), int(p0[1] + length * np.sin(ang)))
        cv2.line(lines, p0, p1, 1.0, int(rng.integers(2, 5)), cv2.LINE_AA)
    lines = cv2.GaussianBlur(lines, (0, 0), 2.0)
    pkg = pkg + (gloss * rng.uniform(0.15, 0.35) + lines * 0.25)[..., None]
    # 한쪽 끝 봉합선
    s = int(rng.uniform(0.03, 0.06) * w)
    pkg[:, :s] *= 0.85
    return np.clip(pkg, 0, 1), "vinyl"


def _label(rng, fonts, words, crop, lh, lw, cx, cy):
    """흰 송장 라벨. 종이 색은 크롭 테두리에 맞춰 경계가 튀지 않게 한다."""
    ch, cw = crop.shape
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    paper = int(np.median(border))
    img = Image.new("L", (lw, lh), paper)
    d = ImageDraw.Draw(img)
    ink = int(rng.integers(20, 60))
    # 칸 선: 라벨 테두리 + 위쪽 글자 영역을 가로로 나눈다
    d.rectangle([4, 4, lw - 5, lh - 5], outline=ink, width=3)
    top = cy - 8
    # 줄 높이 40~60px 정도가 되게 줄 수를 정한다 (글자가 과하게 커지지 않게)
    rows = int(np.clip((top - 10) // rng.uniform(40, 60), 2, 8))
    for i in range(rows):
        y0, y1 = 10 + i * (top - 10) // rows, 10 + (i + 1) * (top - 10) // rows
        d.line([8, y1, lw - 9, y1], fill=ink, width=2)
        n_words = int(rng.integers(1, 4))
        text = " ".join(
            str(words[int(rng.integers(len(words)))])
            if rng.random() < 0.6
            else "".join(str(v) for v in rng.integers(0, 10, int(rng.integers(3, 10))))
            for _ in range(n_words)
        )
        x1 = int(lw * rng.uniform(0.45, 0.95))
        fonts.draw_fit(d, text, (16, y0 + 6, x1, y1 - 6), fill=ink)
    # 바코드 아래 여백에 짧은 글자 한 줄
    below = cy + ch + 8
    if lh - below > 24:
        num = "".join(str(v) for v in rng.integers(0, 10, 12))
        fonts.draw_fit(d, num, (cx, below, cx + cw, min(lh - 10, below + 40)), fill=ink)
    lab = np.asarray(img, np.float32) / 255
    lab = lab * (1 + 0.03 * (rng.random((lh, lw), dtype=np.float32) - 0.5))
    return np.repeat(lab[..., None], 3, axis=2)


def compose(crop: np.ndarray, rng: np.random.Generator, fonts, words):
    """crop(흑백) 을 붙인 장면(BGR uint8), 메타, 크롭 위치 [x, y, w, h]."""
    ch, cw = crop.shape
    img, rail = _belt(rng)
    belt_h = H - 2 * rail

    # 라벨 크기: 크롭 + 좌우 여백 + 위 글자 영역 + 아래 여백
    mx = int(rng.uniform(0.06, 0.15) * cw) + 20
    top = rng.uniform(0.5, 0.9) * ch + 40
    bot = rng.uniform(0.25, 0.45) * ch + 20
    room = belt_h - 60 - ch
    if top + bot > room:
        top, bot = top * room / (top + bot), bot * room / (top + bot)
    top, bot = int(top), int(bot)
    lw, lh = cw + 2 * mx, ch + top + bot

    # 포장: 라벨보다 크고, 화면 밖으로 나가도 된다 (벨트 영역 안으로만 자른다)
    pw = lw + int(rng.uniform(80, 500))
    ph = min(belt_h - 20, lh + int(rng.uniform(40, 300)))
    px = int(rng.uniform(-0.15 * pw, W - 0.85 * pw))
    py = int(rng.uniform(rail + 10, H - rail - 10 - ph + 1))
    pkg, kind = (_cardboard if rng.random() < 0.5 else _vinyl)(rng, ph, pw)

    # 그림자 → 포장
    sx, sy = int(rng.uniform(8, 25)), int(rng.uniform(8, 25))
    shadow = np.zeros((H, W), np.float32)
    cv2.rectangle(shadow, (px + sx, py + sy), (px + pw + sx, py + ph + sy), 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 12)
    shadow[:rail] = shadow[H - rail :] = 0
    img *= (1 - 0.55 * shadow)[..., None]
    x0, x1 = max(px, 0), min(px + pw, W)
    img[py : py + ph, x0:x1] = pkg[:, x0 - px : x1 - px]

    # 라벨: 포장 안, 화면 안
    lo, hi = max(px, 0) + 10, min(px + pw, W) - lw - 10
    lx = int(rng.uniform(lo, hi + 1)) if hi >= lo else int(np.clip(lo, 10, W - lw - 10))
    ly = int(rng.uniform(py + 10, py + ph - lh - 10 + 1))
    img[ly : ly + lh, lx : lx + lw] = _label(rng, fonts, words, crop, lh, lw, mx, top)

    # 조명: 한쪽이 밝은 완만한 기울기 + 센서 노이즈
    gx = np.linspace(-1, 1, W, dtype=np.float32)[None, :] * rng.uniform(-0.15, 0.15)
    gy = np.linspace(-1, 1, H, dtype=np.float32)[:, None] * rng.uniform(-0.10, 0.10)
    img = img * (1 + gx + gy)[..., None] * rng.uniform(0.85, 1.1)
    img += rng.normal(0, rng.uniform(0.004, 0.015), img.shape).astype(np.float32)
    scene = np.clip(img * 255, 0, 255).astype(np.uint8)

    x, y = lx + mx, ly + top
    scene[y : y + ch, x : x + cw] = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    return scene, kind, [x, y, cw, ch]
