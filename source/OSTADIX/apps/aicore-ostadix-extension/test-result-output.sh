#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ANDROID_JAR=${ANDROID_JAR:-$HOME/android-sdk/platforms/android-34/android.jar}
JSON_JAR=${JSON_JAR:-${TMPDIR:-/tmp}/ostadix-json-20240303.jar}
EXPECTED=3cf6cd6892e32e2b4c1c39e0f52f5248a2f5b37646fdfbb79a66b46b618414ed
if [[ ! -f "$JSON_JAR" ]]; then
    curl --fail --location 'https://repo.maven.apache.org/maven2/org/json/json/20240303/json-20240303.jar' -o "$JSON_JAR"
fi
ACTUAL=$(sha256sum "$JSON_JAR"); ACTUAL=${ACTUAL%% *}
[[ "$ACTUAL" == "$EXPECTED" ]] || { echo 'org.json identity mismatch' >&2; exit 1; }
CLASSES="$APP_ROOT/build/intermediates"
REPORT=$(mktemp)
trap 'rm -f "$REPORT"' EXIT
java -classpath "$JSON_JAR:$CLASSES/test-classes:$CLASSES/classes:$ANDROID_JAR" \
    org.ostadix.aicore.extension.NanoToolOutputSelfTest "$REPORT"
cat "$REPORT"
java -classpath "$JSON_JAR:$CLASSES/test-classes:$CLASSES/classes:$ANDROID_JAR" \
    org.ostadix.aicore.extension.ResultHistorySelfTest
java -classpath "$JSON_JAR:$CLASSES/test-classes:$CLASSES/classes:$ANDROID_JAR" \
    org.ostadix.aicore.extension.NanoGenerationBudgetSelfTest
java -classpath "$JSON_JAR:$CLASSES/test-classes:$CLASSES/classes:$ANDROID_JAR" \
    org.ostadix.aicore.extension.NanoProgramPromptSelfTest
java -classpath "$JSON_JAR:$CLASSES/test-classes:$CLASSES/classes:$ANDROID_JAR" \
    org.ostadix.aicore.extension.NanoCandidateSelfTest
