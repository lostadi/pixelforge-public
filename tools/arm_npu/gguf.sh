#!/usr/bin/env bash
set -euo pipefail
PIXELFORGE_GGUF_KIT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PIXELFORGE_GGUF_REPO=$(CDPATH= cd -- "$PIXELFORGE_GGUF_KIT/../.." && pwd)
PIXELFORGE_GGUF_PYTHON=${PIXELFORGE_PYTHON:-$PIXELFORGE_GGUF_REPO/custom-models/gguf-runtime/venv/bin/python}
if [ ! -x "$PIXELFORGE_GGUF_PYTHON" ] && [ -z "${PIXELFORGE_PYTHON:-}" ]; then
  PIXELFORGE_GGUF_PYTHON=$PIXELFORGE_GGUF_REPO/custom-models/minicpm5/venv/bin/python
fi
if [ ! -x "$PIXELFORGE_GGUF_PYTHON" ]; then
  PIXELFORGE_GGUF_PYTHON=python3
fi
PIXELFORGE_GGUF_ACTION=${1:-help}
case "$PIXELFORGE_GGUF_ACTION" in
  inspect) PIXELFORGE_GGUF_SCRIPT=gguf_inspect.py ;;
  compile) PIXELFORGE_GGUF_SCRIPT=compile_gguf.py ;;
  run) PIXELFORGE_GGUF_SCRIPT=run_gguf.py ;;
  *) printf '%s\n' 'Usage: bash tools/arm_npu/gguf.sh inspect|compile|run [options]'; exit 2 ;;
esac
shift
exec "$PIXELFORGE_GGUF_PYTHON" "$PIXELFORGE_GGUF_KIT/$PIXELFORGE_GGUF_SCRIPT" "$@"
