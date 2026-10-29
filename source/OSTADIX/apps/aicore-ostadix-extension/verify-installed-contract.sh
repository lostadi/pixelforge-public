#!/data/data/com.termux/files/usr/bin/bash
# Read-only seam verification for the exact installed AS.OSS/AICore snapshot.
set -euo pipefail

APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$APP_ROOT/../.." && pwd)
AUDIT_ROOT="$REPO_ROOT/audits/pixel-ai-20260915"
JADX_ROOT=${ASOSS_JADX_ROOT:-/data/local/tmp/ostadix-ai-audit-20260915/asoss-jadx/sources}
ASI_JADX_ROOT=${ASI_JADX_ROOT:-/data/local/tmp/ostadix-ai-audit-20260915/asi-jadx/sources}

"$AUDIT_ROOT/check-extension-compatibility.sh"

require_line() {
    local file=$1 text=$2
    [[ -f "$file" ]] || { echo "contract=false missing=$file"; exit 4; }
    grep -Fq "$text" "$file" || {
        echo "contract=false file=$file missing_shape=$text"
        exit 4
    }
}

FNA="$JADX_ROOT/defpackage/fna.java"
FLW="$JADX_ROOT/defpackage/flw.java"
FLO="$JADX_ROOT/defpackage/flo.java"
RESULT="$JADX_ROOT/com/google/android/apps/aicore/aidl/SmartReplyResult.java"
REPLY="$JADX_ROOT/com/google/android/apps/aicore/aidl/SmartReplyReplyEntry.java"
SAFETY="$JADX_ROOT/com/google/android/apps/aicore/aidl/SafetyClassificationResult.java"

require_line "$FNA" 'public final class fna extends ISmartReplyResultCallback.Stub'
require_line "$FNA" 'public final void onSmartReplyInferenceFailure(int i)'
require_line "$FNA" 'public final void onSmartReplyInferenceSuccess(SmartReplyResult smartReplyResult)'
require_line "$FNA" 'this.a.b(smartReplyResult);'
require_line "$FLW" 'public final flp c(SmartReplyRequest smartReplyRequest, flv flvVar)'
require_line "$FLW" 'new flo((ICancellationCallback)'
require_line "$FLO" 'public final void a()'
require_line "$FLO" 'iCancellationCallback.cancel();'
require_line "$RESULT" 'public SmartReplyResult(List list, InferenceEventTraceResult inferenceEventTraceResult)'
require_line "$RESULT" 'public jya getResults()'
require_line "$REPLY" 'public float getScore()'
require_line "$REPLY" 'public int getSafetyClassificationResult()'
require_line "$REPLY" 'public String getText()'
require_line "$SAFETY" 'public static final int NONE = 0;'
require_line "$SAFETY" 'public static final int NOT_SAFE = 1;'

TVR="$ASI_JADX_ROOT/defpackage/tvr.java"
JFK="$ASI_JADX_ROOT/defpackage/jfk.java"
JHT="$ASI_JADX_ROOT/defpackage/jht.java"
ISH="$ASI_JADX_ROOT/defpackage/ish.java"
require_line "$TVR" 'SMART_REPLY(12)'
require_line "$TVR" 'AICORE_SMART_REPLY(25)'
require_line "$JFK" 'public final class jfk implements isu'
require_line "$JFK" 'tvr tvrVar = tvr.SMART_REPLY;'
require_line "$JHT" 'public final Optional d(final ksf ksfVar, isj isjVar, final ffg ffgVar, final ffg ffgVar2, List list)'
require_line "$JHT" 'new FillResponse.Builder()'
require_line "$JHT" '.setInlineSuggestions(arrayList).build()'
require_line "$ISH" 'public final isg b;'

# JADX presents installed default-package classes under a synthetic
# `defpackage` source folder. Verify the raw installed DEX descriptors used by
# Class.forName so source decompilation cannot mask a runtime name mismatch.
DEX_AUDIT=$(mktemp -d)
trap 'find "$DEX_AUDIT" -mindepth 1 -delete; rmdir "$DEX_AUDIT"' EXIT
ASOSS_APK=$(su -c 'pm path com.google.android.as.oss' | sed -n 's/^package://p' | head -1)
[[ -n "$ASOSS_APK" ]] || { echo 'contract=false missing_asoss_apk'; exit 4; }
su -c "cat '$ASOSS_APK'" > "$DEX_AUDIT/asoss.apk"
unzip -Z1 "$DEX_AUDIT/asoss.apk" | grep -E '^classes[0-9]*\.dex$' > "$DEX_AUDIT/dex.list"
while IFS= read -r dex; do
    unzip -p "$DEX_AUDIT/asoss.apk" "$dex"
done < "$DEX_AUDIT/dex.list" | strings > "$DEX_AUDIT/dex.strings"
for descriptor in 'Lflv;' 'Lflw;' 'Lfna;' 'Lflo;'; do
    grep -Fxq "$descriptor" "$DEX_AUDIT/dex.strings" || {
        echo "contract=false missing_dex_descriptor=$descriptor"
        exit 4
    }
done

ASI_APK=$(su -c 'pm path com.google.android.as' | sed -n 's/^package://p' | head -1)
[[ -n "$ASI_APK" ]] || { echo 'contract=false missing_asi_apk'; exit 4; }
su -c "cat '$ASI_APK'" > "$DEX_AUDIT/asi.apk"
unzip -Z1 "$DEX_AUDIT/asi.apk" | grep -E '^classes[0-9]*\.dex$' > "$DEX_AUDIT/asi-dex.list"
while IFS= read -r dex; do
    unzip -p "$DEX_AUDIT/asi.apk" "$dex"
done < "$DEX_AUDIT/asi-dex.list" | strings > "$DEX_AUDIT/asi-dex.strings"
for descriptor in 'Ljht;' 'Lksf;' 'Lisj;' 'Lffg;' 'Lish;' 'Lisg;'; do
    grep -Fxq "$descriptor" "$DEX_AUDIT/asi-dex.strings" || {
        echo "contract=false missing_asi_dex_descriptor=$descriptor"
        exit 4
    }
done

echo 'contract=true asoss_seam=fna.onSmartReplyInferenceSuccess asi_seam=jht.d replacement=SmartReplyResult/FillResponse caller_capture=flw.c cancellation=flo.a'
