package org.ostadix.aicore.extension;

import com.google.android.apps.aicore.aidl.InferenceEventTraceResult;
import com.google.android.apps.aicore.aidl.SmartReplyReplyEntry;
import com.google.android.apps.aicore.aidl.SmartReplyResult;

import java.util.Arrays;
import java.lang.reflect.Proxy;

import io.github.libxposed.api.XposedInterface;

public final class ResultReplacementSelfTest {
    private ResultReplacementSelfTest() {}

    public static void main(String[] arguments) throws Throwable {
        verifyExperimentIsolation();
        verifyDisabledForward();
        Object first = new Object();
        Object selected = new Object();
        InferenceEventTraceResult trace = new InferenceEventTraceResult();
        SmartReplyResult original = new SmartReplyResult(
                Arrays.asList(first, selected), trace);
        SmartReplyResult replacement = (SmartReplyResult)
                OstadixResultBridge.rebuildSelectedSmartReplyResult(
                original, original.getResults(), 1);

        if (replacement == original || replacement.getResults().size() != 1
                || replacement.getResults().get(0) != selected
                || replacement.getInferenceEventTraceResult() != trace) {
            throw new AssertionError(
                    "SmartReplyResult replacement did not preserve its trace metadata");
        }
        try {
            OstadixResultBridge.rebuildSelectedSmartReplyResult(
                    original, original.getResults(), 2);
            throw new AssertionError("out-of-range selection did not fail closed");
        } catch (IllegalArgumentException expected) {
            // Expected.
        }

        OstadixResultBridge.ReplyProjection projection =
                OstadixResultBridge.projectSmartReplies(Arrays.asList(
                        new SmartReplyReplyEntry("safe", 0, 0.6f),
                        new SmartReplyReplyEntry("", 0, 0.95f),
                        new SmartReplyReplyEntry("unsafe", 2, 0.99f)));
        if (projection.scores[0] != 600 || projection.scores[1] != 950
                || projection.scores[2] != 990
                || projection.hasText[0] != 1 || projection.hasText[1] != 0
                || projection.safetyClassifications[0] != 0
                || projection.safetyClassifications[2] != 2) {
            throw new AssertionError("Smart Reply scalar projection is incorrect");
        }

        RequestCorrelations correlations = new RequestCorrelations();
        Object callback = new Object();
        Object cancellation = new Object();
        RequestCorrelations.State state = correlations.begin(
                callback, new RequestIdentity(1000, 1));
        correlations.registerCancellation(cancellation, state);
        if (correlations.claimCallback(callback) != state
                || correlations.cancel(cancellation) != state
                || !correlations.isCancelled(state)
                || correlations.commitReplacement(state)
                || correlations.callbackCountForTest() != 0
                || correlations.cancellationCountForTest() != 0) {
            throw new AssertionError("cancellation was not latched before replacement delivery");
        }

        Object completedCallback = new Object();
        Object terminalCancellation = new Object();
        Object lateCancellation = new Object();
        RequestCorrelations.State completed = correlations.begin(
                completedCallback, new RequestIdentity(1000, 2));
        correlations.registerCancellation(terminalCancellation, completed);
        if (correlations.claimCallback(completedCallback) != completed
                || !correlations.commitReplacement(completed)
                || correlations.cancellationCountForTest() != 0) {
            throw new AssertionError("valid replacement delivery could not be committed");
        }
        correlations.registerCancellation(lateCancellation, completed);
        if (correlations.cancel(lateCancellation) != null
                || correlations.callbackCountForTest() != 0
                || correlations.cancellationCountForTest() != 0) {
            throw new AssertionError("terminal correlation accepted a late cancellation handle");
        }
        System.out.println("SmartReplyResult replacement self-test passed");
    }

    private static void verifyExperimentIsolation() {
        String[] packages = {ExtensionGate.ASOSS_PACKAGE, ExtensionGate.ASI_PACKAGE,
                ExtensionGate.GSA_PACKAGE, ExtensionGate.AICORE_PACKAGE, "unknown", null};
        String[] settings = {ExtensionGate.SMART_REPLY_EXPERIMENT_SETTING,
                ExtensionGate.ASI_AUTOFILL_EXPERIMENT_SETTING, "unknown", null};
        for (String hostPackage : packages) {
            for (String setting : settings) {
                for (String value : new String[] {null, "", "true", "1", "disabled"}) {
                    assertExperiment(false, hostPackage, true, setting, value);
                }
                assertExperiment(false, hostPackage, false, setting, "enabled-v1");
            }
        }
        assertExperiment(true, ExtensionGate.ASOSS_PACKAGE, true,
                ExtensionGate.SMART_REPLY_EXPERIMENT_SETTING, "enabled-v1");
        assertExperiment(true, ExtensionGate.ASI_PACKAGE, true,
                ExtensionGate.ASI_AUTOFILL_EXPERIMENT_SETTING, "enabled-v1");
        assertExperiment(false, ExtensionGate.ASOSS_PACKAGE, true,
                ExtensionGate.ASI_AUTOFILL_EXPERIMENT_SETTING, "enabled-v1");
        assertExperiment(false, ExtensionGate.ASI_PACKAGE, true,
                ExtensionGate.SMART_REPLY_EXPERIMENT_SETTING, "enabled-v1");
        for (String hostPackage : new String[] {ExtensionGate.GSA_PACKAGE,
                ExtensionGate.AICORE_PACKAGE, "unknown", null}) {
            for (String setting : settings) {
                assertExperiment(false, hostPackage, true, setting, "enabled-v1");
            }
        }
        if (ExtensionGate.isSmartReplyExperimentEnabled(null)
                || ExtensionGate.isAsiAutofillExperimentEnabled(null)) {
            throw new AssertionError("missing context enabled a candidate experiment");
        }
        System.out.println("Candidate experiment isolation: default-off, activation and package separation passed");
    }

    private static void assertExperiment(boolean expected, String hostPackage,
            boolean activated, String setting, String value) {
        if (ExtensionGate.candidateExperimentAllows(hostPackage, activated, setting, value)
                != expected) {
            throw new AssertionError("candidate experiment leaked across activation/host/setting");
        }
    }

    private static void verifyDisabledForward() throws Throwable {
        AicoreHooks hooks = new AicoreHooks(null, null, Callback.class);
        for (boolean originalThrows : new boolean[] {false, true}) {
            int[] calls = {0};
            Object expected = new Object();
            Throwable failure = new IllegalStateException("original failed");
            XposedInterface.Chain chain = (XposedInterface.Chain) Proxy.newProxyInstance(
                    XposedInterface.Chain.class.getClassLoader(),
                    new Class<?>[] {XposedInterface.Chain.class}, (proxy, method, args) -> {
                        if (!"proceed".equals(method.getName()) || method.getParameterCount() != 0) {
                            throw new AssertionError("disabled Smart Reply hook inspected or changed request");
                        }
                        calls[0]++;
                        if (originalThrows) throw failure;
                        return expected;
                    });
            try {
                if (hooks.interceptForward(chain) != expected || originalThrows) {
                    throw new AssertionError("disabled Smart Reply hook changed original outcome");
                }
            } catch (Throwable error) {
                if (!originalThrows || error != failure) throw error;
            }
            if (calls[0] != 1) throw new AssertionError("disabled Smart Reply hook repeated original");
        }
        System.out.println("Disabled Smart Reply hook: original result/error preserved with no request inspection");
    }

    private static final class Callback {
        Object a;
    }
}
