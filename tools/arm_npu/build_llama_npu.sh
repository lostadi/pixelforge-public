#!/usr/bin/env bash
# Build an isolated pinned llama.cpp tree; never patch the running service.
set -euo pipefail
PIXELFORGE_LLAMA_KIT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PIXELFORGE_LLAMA_REVISION=254b177307b14d055bbe4c050c9c441a1e06d8a4
PIXELFORGE_LLAMA_ORIGINAL=${PIXELFORGE_LLAMA_SOURCE:-/data/data/com.termux/files/home/ostadix-voice/runtime/llama-cpp}
PIXELFORGE_LLAMA_OFFLINE=${PIXELFORGE_LLAMA_ARCHIVE:-"$PIXELFORGE_LLAMA_KIT/../../custom-models/gguf-runtime/source/llama-source.tar.gz"}
PIXELFORGE_LLAMA_WORK=${1:-/tmp/pixelforge-llama-npu}
PIXELFORGE_LLAMA_TREE=$PIXELFORGE_LLAMA_WORK/source
PIXELFORGE_LLAMA_BUILD=$PIXELFORGE_LLAMA_WORK/build
mkdir -p -- "$PIXELFORGE_LLAMA_WORK"
if [ ! -d "$PIXELFORGE_LLAMA_TREE" ]; then
  PIXELFORGE_LLAMA_STAGING=$(mktemp -d "$PIXELFORGE_LLAMA_WORK/.source-XXXXXX")
  trap 'rm -rf -- "$PIXELFORGE_LLAMA_STAGING"' EXIT
  if [ -z "${PIXELFORGE_LLAMA_ARCHIVE:-}" ] && command -v git >/dev/null && \
    git -c "safe.directory=$PIXELFORGE_LLAMA_ORIGINAL" -C "$PIXELFORGE_LLAMA_ORIGINAL" \
      cat-file -e "$PIXELFORGE_LLAMA_REVISION^{commit}" 2>/dev/null; then
    git -c "safe.directory=$PIXELFORGE_LLAMA_ORIGINAL" -C "$PIXELFORGE_LLAMA_ORIGINAL" \
      archive "$PIXELFORGE_LLAMA_REVISION" | tar -xf - -C "$PIXELFORGE_LLAMA_STAGING"
  else
    python3 -I - "$PIXELFORGE_LLAMA_OFFLINE" "$PIXELFORGE_LLAMA_KIT/LLAMA_ENGINE.json" \
      "$PIXELFORGE_LLAMA_REVISION" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import sys

archive, metadata, revision = sys.argv[1:]
pins = json.loads(Path(metadata).read_text())
expected = pins.get('source_archive_sha256', '')
if pins.get('commit') != revision or not re.fullmatch(r'[0-9a-f]{64}', expected):
    raise SystemExit('Offline llama source needs a valid SHA256 pin for the expected commit')
digest = hashlib.sha256()
with Path(archive).open('rb') as source:
    for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
        digest.update(block)
if digest.hexdigest() != expected:
    raise SystemExit('Offline llama source archive SHA256 mismatch')
PY
    tar -xzf "$PIXELFORGE_LLAMA_OFFLINE" -C "$PIXELFORGE_LLAMA_STAGING"
  fi
  printf '%s\n' "$PIXELFORGE_LLAMA_REVISION" > "$PIXELFORGE_LLAMA_STAGING/.pixelforge-source-owner"
  mv -- "$PIXELFORGE_LLAMA_STAGING" "$PIXELFORGE_LLAMA_TREE"
  trap - EXIT
elif [ "$(cat "$PIXELFORGE_LLAMA_TREE/.pixelforge-source-owner" 2>/dev/null)" != "$PIXELFORGE_LLAMA_REVISION" ]; then
  echo 'Refusing to modify a source tree not created by this isolated build.' >&2
  exit 2
fi
if ! rg -q 'ggml_npu_bridge.h' "$PIXELFORGE_LLAMA_TREE/ggml/src/ggml-cpu/ggml-cpu.c"; then
  patch -d "$PIXELFORGE_LLAMA_TREE" -p1 < "$PIXELFORGE_LLAMA_KIT/llama_npu_bridge.patch"
fi
cp -- "$PIXELFORGE_LLAMA_KIT/ggml_npu_bridge.cc" "$PIXELFORGE_LLAMA_KIT/ggml_npu_bridge.h" \
  "$PIXELFORGE_LLAMA_TREE/ggml/src/ggml-cpu/"
test -f "$PIXELFORGE_LLAMA_KIT/gguf_generate.cc"
cat > "$PIXELFORGE_LLAMA_TREE/pixelforge-target.cmake" <<'CMAKE'
add_executable(pixelforge-gguf "${PIXELFORGE_KIT_SOURCE}/gguf_generate.cc")
target_include_directories(pixelforge-gguf PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
target_compile_features(pixelforge-gguf PRIVATE cxx_std_17)
target_link_libraries(pixelforge-gguf PRIVATE llama)
CMAKE
if ! rg -q 'pixelforge-target.cmake' "$PIXELFORGE_LLAMA_TREE/CMakeLists.txt"; then
  printf '\ninclude("${CMAKE_CURRENT_SOURCE_DIR}/pixelforge-target.cmake")\n' >> "$PIXELFORGE_LLAMA_TREE/CMakeLists.txt"
fi
cmake -S "$PIXELFORGE_LLAMA_TREE" -B "$PIXELFORGE_LLAMA_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DPIXELFORGE_KIT_SOURCE="$PIXELFORGE_LLAMA_KIT" \
  -DGGML_CPU_REPACK=OFF -DGGML_CPU_KLEIDIAI=OFF -DGGML_OPENMP=OFF \
  -DGGML_NATIVE=OFF -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_HTML=OFF -DLLAMA_BUILD_IS_DEV=OFF
cmake --build "$PIXELFORGE_LLAMA_BUILD" --target pixelforge-gguf -j 2
printf '%s\n' "$PIXELFORGE_LLAMA_BUILD/bin/pixelforge-gguf"
