#!/bin/sh
# Compatibility entrypoint; command grammar and dispatch belong to native o-cli.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OCLI_BIN=${O_LANG_OCLI_BIN:-"$ROOT/target/release/o-cli"}
if [ ! -x "$OCLI_BIN" ]; then
    printf 'error: native Ostadix front door is missing: %s; run setup.sh --minimal\n' "$OCLI_BIN" >&2
    exit 1
fi
export O_LANG_ROOT="${O_LANG_ROOT:-$ROOT}"
exec "$OCLI_BIN" "$@"
