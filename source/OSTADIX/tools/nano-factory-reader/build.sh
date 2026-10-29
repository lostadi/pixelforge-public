#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
reader_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
android_jar=${ANDROID_JAR:-/data/data/com.termux/files/home/android-sdk/platforms/android-34/android.jar}
mkdir -p "$reader_root/build/classes" "$reader_root/build/dex"
javac --release 8 -cp "$android_jar" -d "$reader_root/build/classes" \
    "$reader_root/src/org/ostadix/nano/FactoryReader.java"
jar cf "$reader_root/build/classes.jar" -C "$reader_root/build/classes" .
d8 --min-api 28 --lib "$android_jar" --output "$reader_root/build/dex" \
    "$reader_root/build/classes.jar"
jar cf "$reader_root/build/factory-reader.jar" -C "$reader_root/build/dex" classes.dex
sha256sum "$reader_root/build/factory-reader.jar"
