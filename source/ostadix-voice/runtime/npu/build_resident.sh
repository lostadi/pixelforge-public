#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
exec "${CC:-clang}" -O2 -Wall -Wextra \
  -I c_api-0.1.0/include resident.c -L c_api-0.1.0/lib/android_arm64 \
  -Wl,-rpath,'$ORIGIN/c_api-0.1.0/lib/android_arm64' \
  -llitert-lm -o litert_npu_resident
