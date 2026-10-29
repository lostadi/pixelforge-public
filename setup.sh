#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python_bin=python3
if ! command -v "$python_bin" >/dev/null 2>&1; then python_bin=python; fi
action=${1:-help}
case "$action" in
    verify)
        shift
        exec "$python_bin" "$root/scripts/check_public_release.py" --root "$root" "$@"
        ;;
    pack-source)
        shift
        exec "$python_bin" "$root/scripts/release_source.py" --root "$root" "$@"
        ;;
    chat)
        shift
        exec "$python_bin" "$root/chat/install.py" "$@"
        ;;
    help|-h|--help)
        printf '%s\n' 'PixelForge source release' \
          '  sh setup.sh verify' \
          '  sh setup.sh pack-source --output /fresh/path/pixelforge-source.tar.gz' \
          '  sh setup.sh chat --help' \
          'See README.md for compiler, voice and supported-device setup.'
        ;;
    *)
        printf '%s\n' "Unknown public setup action: $action" >&2
        exit 2
        ;;
esac
