import importlib
from unittest.mock import patch

import numpy as np
import pytest

from scripts.mobilenet_followup.core import plain, target_from_gray
from scripts.mobilenet_followup.native_bridge import decoder_mode
from wemeet.sw import decoding, pipeline


@pytest.mark.parametrize("enhance", [False, True])
@pytest.mark.parametrize("reload_decoder", [False, True])
def test_native_imported_decoder_gets_identical_clahe_setting(enhance, reload_decoder):
    if reload_decoder:
        importlib.reload(decoding)
    image = plain(target_from_gray(np.zeros((10, 20), dtype=np.uint8)))
    original = pipeline.decode
    original_decoder = decoding.decode
    with patch.object(decoding, "_read", return_value=None) as read:
        with decoder_mode(enhance):
            assert pipeline.decode is decoding.decode
            pipeline.decode(image)
            assert read.call_count == (2 if enhance else 1)
        assert pipeline.decode is original
        assert decoding.decode is original_decoder
