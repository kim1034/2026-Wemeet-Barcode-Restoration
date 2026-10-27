"""Separate amendment: use the same decoder setting in the native parity check.

The primary evaluation/latency algorithms are unchanged. Upstream pipeline imports
decode by value, so patching decoding.decode alone does not change its default.
"""

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from scripts.controlled_geometry.common import digest, read_json
from scripts.mobilenet_followup import controller, core, worker
from wemeet.sw import decoding, pipeline


@contextmanager
def decoder_mode(enhance):
    with core.decoder_mode(enhance), patch.object(pipeline, "decode", decoding.decode):
        yield


def main():
    mode = sys.argv.pop(1)
    root = Path(sys.argv[sys.argv.index("--root") + 1])
    amendment = read_json(root / "native_bridge_amendment/seal.json")
    for filename, expected in amendment["files"].items():
        assert digest(filename) == expected, filename
    if mode == "worker":
        with patch.object(worker, "decoder_mode", decoder_mode):
            worker.main()
    elif mode == "controller":
        original = subprocess.Popen

        def popen(command, *args, **kwargs):
            if isinstance(command, list) and command[1:3] == [
                "-m",
                "scripts.mobilenet_followup.worker",
            ]:
                command = [
                    command[0],
                    "-m",
                    "scripts.mobilenet_followup.native_bridge",
                    "worker",
                    *command[3:],
                ]
            return original(command, *args, **kwargs)

        with patch.object(subprocess, "Popen", popen):
            controller.main()
    else:
        raise ValueError(mode)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(78)
