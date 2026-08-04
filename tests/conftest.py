"""테스트용 바코드 이미지를 코드로 만든다.

downloads/ 는 git 제외 대상이라 GitHub 러너에 존재하지 않는다.
테스트가 이미지 파일을 읽으면 CI에서만 깨지므로, 전부 코드로 생성한다.

렌더링할 때 넣은 번호가 곧 정답이다. 디코더가 붙는 2단계부터
판독 성공/실패를 CI에서 자동 판정할 수 있다.
"""

import io
from collections.abc import Callable

import numpy as np
import pytest
from barcode import Code128
from barcode.writer import ImageWriter
from PIL import Image

# import 순서 주의: ruff의 isort 규칙은 같은 그룹 안에서 `import x` 를 전부 먼저 놓고
# 그다음 `from x import y` 를 알파벳순으로 놓는다. 이 순서를 바꾸면 I001 이 뜬다.


@pytest.fixture
def render_code128() -> Callable[[str], np.ndarray]:
    """번호를 주면 그 번호가 찍힌 Code128 이미지(BGR uint8)를 만드는 함수."""

    def _render(text: str) -> np.ndarray:
        buffer = io.BytesIO()
        Code128(text, writer=ImageWriter()).write(buffer)
        buffer.seek(0)
        rgb = np.asarray(Image.open(buffer).convert("RGB"))
        # OpenCV 계열은 BGR 이다. 계약(crop_bgr_uint8)도 BGR 이므로 여기서 뒤집는다
        return np.ascontiguousarray(rgb[:, :, ::-1])

    return _render


@pytest.fixture
def clean_barcode_bgr(render_code128: Callable[[str], np.ndarray]) -> np.ndarray:
    """훼손 없는 정상 Code128 이미지. 정답 번호는 WEMEET0001 이다."""
    return render_code128("WEMEET0001")


# 아직 만들지 않은 픽스처 2개 (설계 문서 §11):
#
#   warped_barcode_bgr   곡률 왜곡을 준 바코드
#   glared_barcode_bgr   반사 효과를 준 바코드
#
# 둘 다 wemeet.data.synthesis 를 쓴다. 그 파일은 2단계 산출물이므로
# synthesis.py 가 생기는 시점에 여기에 추가한다. 지금 만들면 ImportError 로
# 전체 테스트가 죽는다.
