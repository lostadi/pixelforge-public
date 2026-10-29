#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$APP_ROOT/../.." && pwd)
TERMINAL_ROOT="$REPO_ROOT/apps/android-terminal"
BUILD_ROOT="$APP_ROOT/build"
INTERMEDIATES="$BUILD_ROOT/intermediates"
OUTPUT_DIR="$BUILD_ROOT/outputs/apk/debug"
ANDROID_SDK_ROOT=${ANDROID_SDK_ROOT:-$HOME/android-sdk}
ANDROID_JAR=${ANDROID_JAR:-$ANDROID_SDK_ROOT/platforms/android-34/android.jar}
LIBXPOSED_API_AAR=${LIBXPOSED_API_AAR:-/data/data/com.termux/files/usr/tmp/libxposed-api-102.0.0.aar}
DEBUG_KEYSTORE=${DEBUG_KEYSTORE:-$HOME/.android/debug.keystore}
EXPECTED_API_SHA256=423484a6e1807e7a423c4b88fcd8176d104318259d91791877fed88fe91479d0
NATIVE_HASH_MANIFEST="$APP_ROOT/app/src/main/resources/META-INF/ostadix/native-sha256.txt"
MIN_SDK=31
TARGET_SDK=34
VERSION_CODE=21
VERSION_NAME=0.21.0-gemini-coexistence

for tool in aapt2 apksigner d8 jar javac javap keytool readelf sed sha256sum unzip; do
    command -v "$tool" >/dev/null 2>&1 || { echo "Missing build tool: $tool" >&2; exit 1; }
done
[[ -f "$ANDROID_JAR" ]] || { echo "Android platform jar missing: $ANDROID_JAR" >&2; exit 1; }
[[ -f "$LIBXPOSED_API_AAR" ]] || { echo "Pinned libxposed API AAR missing" >&2; exit 1; }
actual_api=$(sha256sum "$LIBXPOSED_API_AAR"); actual_api=${actual_api%% *}
[[ "$actual_api" == "$EXPECTED_API_SHA256" ]] || { echo "libxposed API hash mismatch" >&2; exit 1; }

TERMINAL_NATIVE="$TERMINAL_ROOT/build/intermediates/package/lib/arm64-v8a"
while read -r expected_hash native_name; do
    [[ -n "$expected_hash" && -n "$native_name" ]] || continue
    native_path="$TERMINAL_NATIVE/$native_name"
    [[ -f "$native_path" ]] || {
        echo "Build apps/android-terminal first; missing $native_name" >&2; exit 1;
    }
    actual_hash=$(sha256sum "$native_path"); actual_hash=${actual_hash%% *}
    [[ "$actual_hash" == "$expected_hash" ]] || {
        echo "Native closure hash changed for $native_name; inspect and update this experiment" >&2
        exit 1
    }
done <"$NATIVE_HASH_MANIFEST"

if [[ -d "$INTERMEDIATES" ]]; then
    find "$INTERMEDIATES" -mindepth 1 -delete
fi
mkdir -p "$INTERMEDIATES/compiled-res" "$INTERMEDIATES/generated" \
    "$INTERMEDIATES/classes" "$INTERMEDIATES/test-classes" \
    "$INTERMEDIATES/dex" "$INTERMEDIATES/lib" \
    "$INTERMEDIATES/package/lib/arm64-v8a" "$OUTPUT_DIR"

echo "[1/6] Extracting pinned compile-only libxposed API 102"
unzip -jo "$LIBXPOSED_API_AAR" classes.jar -d "$INTERMEDIATES/lib" >/dev/null
LIBXPOSED_CLASSES="$INTERMEDIATES/lib/classes.jar"

echo "[2/6] Compiling resources"
aapt2 compile --dir "$APP_ROOT/app/src/main/res" \
    -o "$INTERMEDIATES/compiled-res/resources.zip"
aapt2 link -o "$INTERMEDIATES/base-unsigned.apk" -I "$ANDROID_JAR" \
    --manifest "$APP_ROOT/app/src/main/AndroidManifest.xml" \
    --java "$INTERMEDIATES/generated" --min-sdk-version "$MIN_SDK" \
    --target-sdk-version "$TARGET_SDK" --version-code "$VERSION_CODE" \
    --version-name "$VERSION_NAME" -R "$INTERMEDIATES/compiled-res/resources.zip" \
    --auto-add-overlay

echo "[3/6] Compiling Java and DEX"
IDENTITY_DIR="$INTERMEDIATES/generated/org/ostadix/aicore/extension"
mkdir -p "$IDENTITY_DIR"
SOURCES_HASH=$(find "$APP_ROOT/app/src/main/java" -type f -name '*.java' -print0 | \
    sort -z | xargs -0 sha256sum | sha256sum)
SOURCES_HASH=${SOURCES_HASH%% *}
cat >"$IDENTITY_DIR/OwnedBuildIdentity.java" <<IDENTITY
package org.ostadix.aicore.extension;
final class OwnedBuildIdentity {
    static final int VERSION = $VERSION_CODE;
    static final String SOURCES_SHA256 = "$SOURCES_HASH";
}
IDENTITY
mapfile -t JAVA_SOURCES < <(find "$INTERMEDIATES/generated" \
    "$APP_ROOT/app/src/main/java" -type f -name '*.java' -print | sort)
JAVA_SOURCES+=("$TERMINAL_ROOT/app/src/main/java/org/ostadix/terminal/OstadixRuntime.java")
JAVA_SOURCES+=("$TERMINAL_ROOT/app/src/main/java/org/ostadix/terminal/HostMcpClient.java")
# Compile Java 8 lambdas against the JDK's LambdaMetafactory signatures;
# the platform stubs omit that compiler-only method. D8 desugars them below.
javac -encoding UTF-8 --release 8 \
    -classpath "$ANDROID_JAR:$LIBXPOSED_CLASSES" -d "$INTERMEDIATES/classes" \
    "${JAVA_SOURCES[@]}"
mapfile -t TEST_SOURCES < <(find "$APP_ROOT/app/src/test/java" \
    -type f -name '*.java' -print | sort)
javac -encoding UTF-8 --release 8 \
    -classpath "$ANDROID_JAR:$LIBXPOSED_CLASSES:$INTERMEDIATES/classes" \
    -d "$INTERMEDIATES/test-classes" "${TEST_SOURCES[@]}"
java -classpath "$INTERMEDIATES/test-classes:$INTERMEDIATES/classes:$ANDROID_JAR:$LIBXPOSED_CLASSES" \
    org.ostadix.aicore.extension.ResultReplacementSelfTest
java -classpath "$INTERMEDIATES/test-classes:$INTERMEDIATES/classes:$ANDROID_JAR:$LIBXPOSED_CLASSES" \
    org.ostadix.aicore.extension.AsiDelegationSelfTest
java -classpath "$INTERMEDIATES/test-classes:$INTERMEDIATES/classes:$ANDROID_JAR:$LIBXPOSED_CLASSES" \
    org.ostadix.aicore.extension.NanoSourcePreparationSelfTest
java -classpath "$INTERMEDIATES/test-classes:$INTERMEDIATES/classes:$ANDROID_JAR:$LIBXPOSED_CLASSES" \
    org.ostadix.aicore.extension.GeminiLocalRouteSelfTest
java -classpath "$INTERMEDIATES/test-classes:$INTERMEDIATES/classes:$ANDROID_JAR:$LIBXPOSED_CLASSES" \
    org.ostadix.aicore.extension.GeminiPlainTextInputSelfTest
jar cf "$INTERMEDIATES/module-classes.jar" -C "$INTERMEDIATES/classes" .
d8 --min-api "$MIN_SDK" --lib "$ANDROID_JAR" --classpath "$LIBXPOSED_CLASSES" \
    --output "$INTERMEDIATES/dex" "$INTERMEDIATES/module-classes.jar"

echo "[4/6] Packaging disabled module and exact native closure"
for native_name in libostadix_runtime.so libostadix_cli.so libostadix_bash.so \
        libandroid-support.so libiconv.so libreadline_8.so libncursesw_6.so; do
    cp "$TERMINAL_NATIVE/$native_name" \
        "$INTERMEDIATES/package/lib/arm64-v8a/$native_name"
done
UNSIGNED="$INTERMEDIATES/OstadixAicoreExtension-unsigned.apk"
cp "$INTERMEDIATES/base-unsigned.apk" "$UNSIGNED"
jar uf "$UNSIGNED" -C "$INTERMEDIATES/dex" classes.dex \
    -C "$APP_ROOT/app/src/main/resources" META-INF \
    -C "$INTERMEDIATES/package" lib

echo "[5/6] Signing"
[[ -f "$DEBUG_KEYSTORE" ]] || {
    mkdir -p "$(dirname "$DEBUG_KEYSTORE")"
    keytool -genkeypair -noprompt -keystore "$DEBUG_KEYSTORE" -storepass android \
        -alias androiddebugkey -keypass android -dname 'CN=Android Debug,O=Android,C=US' \
        -keyalg RSA -keysize 2048 -validity 10000
}
OUTPUT_APK="$OUTPUT_DIR/OstadixAicoreExtension-debug.apk"
apksigner sign --ks "$DEBUG_KEYSTORE" --ks-key-alias androiddebugkey \
    --ks-pass pass:android --key-pass pass:android --min-sdk-version "$MIN_SDK" \
    --v1-signing-enabled false --v2-signing-enabled true --v3-signing-enabled true \
    --v4-signing-enabled false --alignment-preserved false --lib-page-alignment 16384 \
    --out "$OUTPUT_APK" "$UNSIGNED"

echo "[6/6] Verifying fail-closed package"
"$APP_ROOT/verify-source-policy.sh"
apksigner verify --verbose --print-certs "$OUTPUT_APK"
APK_ENTRIES="$INTERMEDIATES/apk-entries.txt"
jar tf "$OUTPUT_APK" >"$APK_ENTRIES"
for entry in java_init.list module.prop scope.list; do
    grep -Fx "META-INF/xposed/$entry" "$APK_ENTRIES" >/dev/null || exit 1
done
grep -Fx 'META-INF/ostadix/native-sha256.txt' "$APK_ENTRIES" >/dev/null || exit 1
[[ $(unzip -p "$OUTPUT_APK" META-INF/ostadix/native-sha256.txt) == \
        "$(cat "$NATIVE_HASH_MANIFEST")" ]] || exit 1
EXPECTED_SCOPE=$(printf 'com.google.android.as.oss\ncom.google.android.as\ncom.google.android.googlequicksearchbox\ncom.google.android.aicore')
[[ $(unzip -p "$OUTPUT_APK" META-INF/xposed/scope.list) == "$EXPECTED_SCOPE" ]] || exit 1
for native_name in libostadix_runtime.so libostadix_cli.so libostadix_bash.so \
        libandroid-support.so libiconv.so libreadline_8.so libncursesw_6.so; do
    grep -Fx "lib/arm64-v8a/$native_name" "$APK_ENTRIES" >/dev/null || exit 1
done
grep -Fq 'ostadix_aicore_extension_token' \
    "$APP_ROOT/app/src/main/java/org/ostadix/aicore/extension/ExtensionGate.java"
grep -Fq ':asoss-smart-reply-result-v1:' \
    "$APP_ROOT/app/src/main/java/org/ostadix/aicore/extension/ExtensionGate.java"
grep -Fq 'chain.proceed(new Object[] {outcome.replacement})' \
    "$APP_ROOT/app/src/main/java/org/ostadix/aicore/extension/AicoreHooks.java"
grep -Fq 'new OstadixResultBridge(getModuleApplicationInfo())' \
    "$APP_ROOT/app/src/main/java/org/ostadix/aicore/extension/AicoreOstadixModule.java"
MODULE_SOURCE="$APP_ROOT/app/src/main/java/org/ostadix/aicore/extension/AicoreOstadixModule.java"
grep -Fq 'Application.class.getDeclaredMethod("attach", Context.class)' "$MODULE_SOURCE"
for runtime_class in flv flw fna flo; do
    grep -Fq "Class.forName(\"$runtime_class\", false, loader)" "$MODULE_SOURCE"
done
for runtime_class in jht ksf isj ffg; do
    grep -Fq "Class.forName(\"$runtime_class\", false, loader)" "$MODULE_SOURCE"
done
grep -Fq 'setId("ostadix-asi/jht-fill-response-v1")' "$MODULE_SOURCE"
grep -Fq 'setId("ostadix-gemini/schema-function-inventory-v1")' "$MODULE_SOURCE"
grep -Fq 'SchemaFunctionInventory_Impl' "$MODULE_SOURCE"
grep -Fq 'event=schema_inventory_injected' \
    "$APP_ROOT/app/src/main/java/org/ostadix/aicore/extension/GeminiAppFunctionSchemaHooks.java"
grep -Fq 'chain.proceed(arguments)' \
    "$APP_ROOT/app/src/main/java/org/ostadix/aicore/extension/AsiHooks.java"
[[ $(grep -Fc 'ExtensionGate.isAsiAutofillExperimentEnabled(context)' \
    "$APP_ROOT/app/src/main/java/org/ostadix/aicore/extension/AsiHooks.java") -ge 2 ]] || exit 1
grep -Fq 'ExtensionGate.thermalPolicyAllows(context)' \
    "$APP_ROOT/app/src/main/java/org/ostadix/aicore/extension/AsiHooks.java"
if grep -Fq 'Class.forName("defpackage.' "$MODULE_SOURCE"; then
    echo 'Runtime class lookup must use installed default-package DEX names' >&2
    exit 1
fi
for invariant in 'bootstrapHandle' 'installedHooks.add' '.unhook()' 'bridge.close()' \
        'smokeRuntime()' 'smoke.selectedSourceIndex != 1' \
        'smoke.selectedScoreMilli != 950'; do
    grep -Fq "$invariant" "$MODULE_SOURCE"
done
grep -Fq 'runtime.postprocessAicoreSmartReplies(' \
    "$APP_ROOT/app/src/main/java/org/ostadix/aicore/extension/OstadixResultBridge.java"
grep -Fq 'nativeVersion();' \
    "$REPO_ROOT/apps/android-terminal/app/src/main/java/org/ostadix/terminal/OstadixRuntime.java"
RUNTIME_API="$INTERMEDIATES/ostadix-runtime-api.txt"
javap -classpath "$INTERMEDIATES/module-classes.jar" -p \
    org.ostadix.terminal.OstadixRuntime >"$RUNTIME_API"
for method in postprocessAicoreReplies postprocessAicoreSmartReplies; do
    grep -Fq "$method(" "$RUNTIME_API"
done
DYNAMIC_SYMBOLS="$INTERMEDIATES/ostadix-runtime-dynamic-symbols.txt"
readelf --dyn-syms --wide \
    "$INTERMEDIATES/package/lib/arm64-v8a/libostadix_runtime.so" \
    >"$DYNAMIC_SYMBOLS"
for symbol in \
        Java_org_ostadix_terminal_OstadixRuntime_nativePostprocessAicoreReplies \
        Java_org_ostadix_terminal_OstadixRuntime_nativePostprocessAicoreSmartReplies; do
    grep -Fq "$symbol" "$DYNAMIC_SYMBOLS"
done
for event in request_enter request_dispatched ostadix_selected result_forwarded \
        result_fallback inference_failure cancellation_forwarded; do
    grep -Fq "event=$event" \
        "$APP_ROOT/app/src/main/java/org/ostadix/aicore/extension/AicoreHooks.java"
done
[[ $(grep -Fc 'ExtensionGate.isSmartReplyExperimentEnabled(context)' \
        "$APP_ROOT/app/src/main/java/org/ostadix/aicore/extension/AicoreHooks.java") -ge 3 ]] || exit 1
aapt2 dump badging "$OUTPUT_APK" | sed -n '1,8p'
sha256sum "$OUTPUT_APK"
echo "$OUTPUT_APK"
