#!/data/data/com.termux/files/usr/bin/bash
# Ensure the in-process fail-closed gate matches the shared live policy.
set -euo pipefail

APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$APP_ROOT/../.." && pwd)
# shellcheck source=../../audits/pixel-ai-20260915/extension-compatibility-policy.sh
source "$REPO_ROOT/audits/pixel-ai-20260915/extension-compatibility-policy.sh"
GATE="$APP_ROOT/app/src/main/java/org/ostadix/aicore/extension/ExtensionGate.java"

for expected in \
        "$EXPECTED_AICORE_VERSION_CODE" "$EXPECTED_AICORE_VERSION_NAME" \
        "$EXPECTED_AICORE_APK_SHA256" "$EXPECTED_AICORE_SIGNER_SHA256" \
        "$REVIEWED_AICORE_RC13_VERSION_CODE" "$REVIEWED_AICORE_RC13_VERSION_NAME" \
        "$REVIEWED_AICORE_RC13_APK_SHA256" \
        "$EXPECTED_ASOSS_VERSION_CODE" "$EXPECTED_ASOSS_VERSION_NAME" \
        "$EXPECTED_ASOSS_APK_SHA256" "$EXPECTED_ASOSS_SIGNER_SHA256"; do
    grep -Fq "$expected" "$GATE" || {
        echo "source_policy_match=false missing=$expected"
        exit 3
    }
done

for expected in \
        '16934935L' 'C.6.playstore.pixel11.961955194' \
        '16d2b265fbea8c8abc46b537b6191628f7a855d7e3fd760b4a60bb6ea9c93e68' \
        '3af39ab967aaa5d279e49b5f769cb66e40799838bc8799343ee57ae435d2455b'; do
    grep -Fq "$expected" "$GATE" || {
        echo "source_policy_match=false missing_asi=$expected"
        exit 3
    }
done

for expected in \
        '301803623L' '17.56.15.sa.arm64' \
        'c227beb9468f1740c395e457d5f06fb780f288a89156d8487ef069953c1c359a' \
        '301806951L' '17.58.16.sa.arm64' \
        '711714ac6df264bf5d319358f9dc8bae867f8a89ee13e7fc5d0e4d3df2611ff9' \
        '7ce83c1b71f3d572fed04c8d40c5cb10ff75e6d87d9df6fbd53f0468c2905053'; do
    grep -Fq "$expected" "$GATE" || {
        echo "source_policy_match=false missing_gsa=$expected"
        exit 3
    }
done

echo 'source_policy_match=true fields=22'
