#!/usr/bin/env bash
set -euo pipefail
cd /home/jnuadmin/endtoend/kangmin/etc/2026-Wemeet-Barcode-Restoration
source ../barcode_setup/env.sh
export CUDA_VISIBLE_DEVICES=''
exec "$UV_PROJECT_ENVIRONMENT/bin/python" -u -m scripts.controlled_durable_20261026.watch \
  --campaign runs/stage2-controlled-v2/campaign-gpu4567-20261026-resume-json
