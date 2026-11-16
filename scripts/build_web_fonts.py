"""web/fonts/ 의 글꼴(Pretendard)을 다시 만든다.

실행: uv run --with "fonttools[woff]" --with brotli python -m scripts.build_web_fonts
(fonttools 와 brotli 는 글꼴을 깎을 때만 쓴다 — 프로젝트 의존성에는 넣지 않았다.)

작업자 화면은 글꼴을 CDN 에서 받지 않는다. 망이 끊긴 현장에서도 글자가 그대로여야
하므로 우리 서버가 직접 내준다(GET /fonts/{name}). 그러자면 2MB 짜리 전체 글꼴을
그대로 둘 수는 없어서, **화면에 실제로 나오는 글자만 남긴 부분집합**을 만든다.

한글은 넣은 글자만 들어간다. 화면에 새 낱말이 생기면 그 글자가 빠져 있을 수 있다.
빠져도 깨지지는 않고 그 글자만 시스템 글꼴로 떨어지는데, 굵기와 자간이 달라 눈에
띈다. 그래서 web/ 의 화면 문구(index.html 과 .mjs 모듈)나 서버의 한글 응답 문구를
고친 뒤에는 이 스크립트를 한 번 돌린다.

받아오는 곳은 jsdelivr 이다. 만들 때만 망이 필요하고, 돌릴 때는 필요 없다.
"""

from __future__ import annotations

import string
import subprocess
import sys
import tempfile
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
FONTS = WEB / "fonts"

PRETENDARD_URL = (
    "https://cdn.jsdelivr.net/npm/pretendard@1.3.9/dist/web/variable/woff2/PretendardVariable.woff2"
)

# 화면 CSS 가 쓰는 굵기는 400~800 이다. 축을 이 범위로 좁히면 크기가 30% 가까이 준다.
WEIGHT_RANGE = (400, 800)

# 바코드 번호는 서버가 읽어 낸 값이라 어떤 영문·숫자·기호든 올 수 있다 — ASCII 는 통째로 넣는다.
# 서버가 아직 내보내지 않은 말이 들어올 수 있어 흔한 낱말도 미리 얹어 둔다.
# 넉넉히 넣어도 한 글자에 수백 바이트라 값이 싸다.
EXTRA_TEXT = (
    string.printable.strip()
    + "가나다라마바사아자차카타파하"
    + "확인실패성공판독검출보정기하추정디코딩원본이미지카메라영상바코드"
    + "번호시간처리전송상태연동필요없음대기중시작중지저장삭제취소닫기"
    + "건개장회초분시일월년오류경고주의"
)

# 남이 만든 큰 파일 — 우리 화면 문구가 들어 있지 않다
SKIP = {"three.module.min.js"}


def used_characters() -> str:
    """화면과 서버가 쓰는 글자를 모은다.

    index.html 과 .mjs 모듈에는 화면에 뜨는 문구가 글자 그대로 들어 있고, 서버 쪽
    파이썬에는 실패 사유 같은 응답 문구가 들어 있다. 셋이면 화면에 나올 말은 거의 다다.
    """
    chars: set[str] = set(EXTRA_TEXT)
    sources = [
        *sorted(WEB.glob("*.html")),
        *sorted(p for p in WEB.glob("*.mjs") if p.name not in SKIP),
        *sorted((ROOT / "wemeet").rglob("*.py")),
    ]
    for path in sources:
        chars |= set(path.read_text(encoding="utf-8"))
    keep = sorted(
        c for c in chars if c.isprintable() and not unicodedata.category(c).startswith("C")
    )
    return "".join(keep)


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:  # noqa: S310 - 고정된 https 주소
        return response.read()


def build_pretendard(text: str) -> Path:
    """전체 가변 글꼴을 받아 글자와 굵기를 깎는다."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    out = FONTS / "Pretendard.woff2"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        source = tmp_dir / "PretendardVariable.woff2"
        source.write_bytes(_download(PRETENDARD_URL))
        chars_file = tmp_dir / "chars.txt"
        chars_file.write_text(text, encoding="utf-8")
        subset = tmp_dir / "subset.woff2"
        subprocess.run(  # noqa: S603 - 인자는 전부 이 파일 안에서 만든다
            [
                sys.executable,
                "-m",
                "fontTools.subset",
                str(source),
                f"--text-file={chars_file}",
                "--layout-features=*",
                "--flavor=woff2",
                f"--output-file={subset}",
            ],
            check=True,
        )
        font = TTFont(subset)
        instancer.instantiateVariableFont(font, {"wght": WEIGHT_RANGE}, inplace=True)
        font.flavor = "woff2"
        font.save(out)
    return out


def main() -> None:
    FONTS.mkdir(parents=True, exist_ok=True)
    text = used_characters()
    hangul = sum(1 for c in text if "가" <= c <= "힣")
    print(f"글자 {len(text)}자 (한글 {hangul}자)")
    out = build_pretendard(text)
    print(f"  {out.relative_to(ROOT)}  {out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
