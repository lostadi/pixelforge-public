#!/data/data/com.termux/files/usr/bin/bash
# Compile the Android AArch64 helper on the phone itself.
set -eu
PIXELFORGE_SOURCE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PIXELFORGE_NATIVE_CXX=${PIXELFORGE_ARM_CXX:-/data/data/com.termux/files/usr/bin/clang++}
PIXELFORGE_NATIVE_DRIVER=()
if [ ! -x "$PIXELFORGE_NATIVE_CXX" ] && [ -z "${PIXELFORGE_ARM_CXX:-}" ]; then
  PIXELFORGE_NATIVE_CXX=/data/data/com.termux/files/usr/bin/clang
  PIXELFORGE_NATIVE_DRIVER=(--driver-mode=g++)
fi
PIXELFORGE_NATIVE_OUTPUT=${1:-/data/local/tmp/pixelforge-npu/npu_pipe}
mkdir -p -- "$(dirname -- "$PIXELFORGE_NATIVE_OUTPUT")"
PIXELFORGE_NATIVE_TEMP=$(mktemp "${PIXELFORGE_NATIVE_OUTPUT}.XXXXXX")
trap 'rm -f -- "$PIXELFORGE_NATIVE_TEMP"' EXIT
"$PIXELFORGE_NATIVE_CXX" "${PIXELFORGE_NATIVE_DRIVER[@]}" -D__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__ \
  -O2 -Wall -Wextra -std=c++17 "$PIXELFORGE_SOURCE_DIR/npu_pipe.cc" \
  -ldl -o "$PIXELFORGE_NATIVE_TEMP"
chmod 755 "$PIXELFORGE_NATIVE_TEMP"
mv -f -- "$PIXELFORGE_NATIVE_TEMP" "$PIXELFORGE_NATIVE_OUTPUT"
