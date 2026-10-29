#!/bin/sh
# Compatibility installer name. Installed commands are compiled executables.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
if [ "$#" -ne 1 ] || [ -z "$1" ] || [ "$(basename -- "$1")" != o ]; then
    printf 'usage: %s BIN_DIRECTORY/o\n' "${0##*/}" >&2
    exit 2
fi
exec python3 "$ROOT/scripts/install_native_binaries.py" \
    --repo-root "$ROOT" --bin-dir "$(dirname -- "$1")" --front-door-only
