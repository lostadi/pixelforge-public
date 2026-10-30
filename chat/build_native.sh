#!/usr/bin/env bash
# Rebuild the pinned architecture engine and chat bridge from source.
set -euo pipefail
PIXELFORGE_CHAT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PIXELFORGE_CHAT_BUILD=${PIXELFORGE_CHAT_BUILD_DIR:-"$PIXELFORGE_CHAT_ROOT/build"}
PIXELFORGE_CHAT_NATIVE=${PIXELFORGE_CHAT_NATIVE:-OFF}
case "$PIXELFORGE_CHAT_BUILD" in
  /*) ;;
  *) printf '%s\n' 'PIXELFORGE_CHAT_BUILD_DIR must be an absolute path' >&2; exit 2 ;;
esac
case "$PIXELFORGE_CHAT_NATIVE" in
  ON|OFF) ;;
  *) printf '%s\n' 'PIXELFORGE_CHAT_NATIVE must be ON or OFF' >&2; exit 2 ;;
esac
PIXELFORGE_CHAT_SOURCE_KIND=${1:-}
PIXELFORGE_CHAT_INPUT=${2:-}
case "$PIXELFORGE_CHAT_SOURCE_KIND" in
  --archive|--source) test -n "$PIXELFORGE_CHAT_INPUT" ;;
  --fetch) ;;
  *) printf '%s\n' 'Usage: chat/build_native.sh --archive PINNED_TAR_GZ | --source LLAMA_GIT_CHECKOUT | --fetch' >&2; exit 2 ;;
esac
PIXELFORGE_CHAT_REVISION=254b177307b14d055bbe4c050c9c441a1e06d8a4
PIXELFORGE_CHAT_KIT="$PIXELFORGE_CHAT_ROOT/../tools/arm_npu"
mkdir -p -- "$PIXELFORGE_CHAT_BUILD"
PIXELFORGE_CHAT_TREE="$PIXELFORGE_CHAT_BUILD/source"
if [ ! -d "$PIXELFORGE_CHAT_TREE" ]; then
  PIXELFORGE_CHAT_STAGING=$(mktemp -d "$PIXELFORGE_CHAT_BUILD/.source-XXXXXX")
  trap 'rm -rf -- "$PIXELFORGE_CHAT_STAGING"' EXIT
  if [ "$PIXELFORGE_CHAT_SOURCE_KIND" = --archive ]; then
    python3 - "$PIXELFORGE_CHAT_INPUT" "$PIXELFORGE_CHAT_KIT/LLAMA_ENGINE.json" <<'PY'
import hashlib,json,sys,tarfile
from pathlib import Path
path=Path(sys.argv[1]);meta=json.loads(Path(sys.argv[2]).read_text())
if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['source_archive_sha256']:
    raise SystemExit('Pinned source archive SHA-256 mismatch')
with tarfile.open(path) as archive:
    for item in archive:
        if item.name.startswith('/') or '..' in Path(item.name).parts or item.issym() or item.islnk():
            raise SystemExit('Unsafe source archive member')
PY
    tar -xzf "$PIXELFORGE_CHAT_INPUT" -C "$PIXELFORGE_CHAT_STAGING"
  else
    if [ "$PIXELFORGE_CHAT_SOURCE_KIND" = --fetch ]; then
      PIXELFORGE_CHAT_INPUT="$PIXELFORGE_CHAT_BUILD/upstream.git"
      git init --bare "$PIXELFORGE_CHAT_INPUT"
      git -C "$PIXELFORGE_CHAT_INPUT" fetch --depth=1 https://github.com/ggml-org/llama.cpp "$PIXELFORGE_CHAT_REVISION"
    fi
    git -C "$PIXELFORGE_CHAT_INPUT" archive "$PIXELFORGE_CHAT_REVISION" | tar -xf - -C "$PIXELFORGE_CHAT_STAGING"
  fi
  printf '%s\n' "$PIXELFORGE_CHAT_REVISION" > "$PIXELFORGE_CHAT_STAGING/.pixelforge-chat-source-owner"
  mv -- "$PIXELFORGE_CHAT_STAGING" "$PIXELFORGE_CHAT_TREE"
  trap - EXIT
fi
test "$(cat "$PIXELFORGE_CHAT_TREE/.pixelforge-chat-source-owner")" = "$PIXELFORGE_CHAT_REVISION"
python3 - "$PIXELFORGE_CHAT_TREE" "$PIXELFORGE_CHAT_KIT/llama_npu_bridge.patch" <<'PY'
import hashlib,subprocess,sys
from pathlib import Path
tree,patch = map(Path,sys.argv[1:])
marker = tree / '.pixelforge-chat-patch-sha256'
expected = hashlib.sha256(patch.read_bytes()).hexdigest()
patched = 'ggml_npu_bridge.h' in (tree / 'ggml/src/ggml-cpu/ggml-cpu.c').read_text()
if marker.exists() or marker.is_symlink() or patched:
    if (tree.is_symlink() or marker.is_symlink() or not marker.is_file() or not patched
            or marker.read_text().strip() != expected):
        raise SystemExit('Cached llama source has no matching PixelForge patch SHA-256. '
                         'Move ' + str(tree) + ' aside, then regenerate source by rerunning '
                         'chat/build_native.sh with the pinned --archive, --source, or --fetch input.')
else:
    if tree.is_symlink():
        raise SystemExit('Refusing to patch a symlinked cached source directory: ' + str(tree))
    subprocess.run(['patch','--fuzz=0','-d',str(tree),'-p1','-i',str(patch)],check=True)
    marker.write_text(expected + '\n')
PY
cp -- "$PIXELFORGE_CHAT_ROOT/native/ggml_npu_bridge.cc" "$PIXELFORGE_CHAT_ROOT/native/ggml_npu_bridge.h" "$PIXELFORGE_CHAT_TREE/ggml/src/ggml-cpu/"
cat > "$PIXELFORGE_CHAT_TREE/pixelforge-chat.cmake" <<'CMAKE'
add_executable(pixelforge-chat "${PIXELFORGE_CHAT_SOURCE}/chat_runner.cc")
target_include_directories(pixelforge-chat PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/vendor" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${PIXELFORGE_CHAT_SOURCE}")
target_compile_features(pixelforge-chat PRIVATE cxx_std_17)
target_link_libraries(pixelforge-chat PRIVATE llama)
CMAKE
if ! rg -q 'pixelforge-chat.cmake' "$PIXELFORGE_CHAT_TREE/CMakeLists.txt"; then
  printf '\ninclude("${CMAKE_CURRENT_SOURCE_DIR}/pixelforge-chat.cmake")\n' >> "$PIXELFORGE_CHAT_TREE/CMakeLists.txt"
fi
cmake -S "$PIXELFORGE_CHAT_TREE" -B "$PIXELFORGE_CHAT_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DPIXELFORGE_CHAT_SOURCE="$PIXELFORGE_CHAT_ROOT/native" \
  -DGGML_CPU_REPACK=OFF -DGGML_CPU_KLEIDIAI=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE="$PIXELFORGE_CHAT_NATIVE" -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_HTML=OFF -DLLAMA_BUILD_IS_DEV=OFF
cmake --build "$PIXELFORGE_CHAT_BUILD" --target pixelforge-chat -j "${PIXELFORGE_CHAT_JOBS:-2}"
python3 - "$PIXELFORGE_CHAT_ROOT" "$PIXELFORGE_CHAT_BUILD" "$PIXELFORGE_CHAT_NATIVE" <<'PY'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]);build=Path(sys.argv[2]);files=[*sorted((root/'native').glob('*')),root/'build_native.sh']
pins={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
pins['bin/pixelforge-chat']=hashlib.sha256((build/'bin/pixelforge-chat').read_bytes()).hexdigest()
manifest={'upstream_commit':'254b177307b14d055bbe4c050c9c441a1e06d8a4','native_cpu':sys.argv[3]=='ON','files':pins}
(build/'NATIVE_BUILD.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
printf '%s\n' "$PIXELFORGE_CHAT_BUILD/bin/pixelforge-chat"
