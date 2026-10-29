package org.ostadix.aicore.extension;

import android.content.Context;
import android.os.Process;
import android.util.Log;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import io.github.libxposed.api.XposedInterface;
import org.ostadix.terminal.OstadixRuntime;

final class AsiHooks {
    private static final String TAG = "OstadixAicoreExperiment";
    private static final AtomicLong NEXT_REQUEST = new AtomicLong();

    private final OstadixResultBridge bridge;
    private final Context context;

    AsiHooks(Context context, OstadixResultBridge bridge) {
        this.context = context;
        this.bridge = bridge;
    }

    Object interceptFillResponse(final XposedInterface.Chain chain) throws Throwable {
        if (!ExtensionGate.isAsiAutofillExperimentEnabled(context)) {
            return chain.proceed();
        }
        final String[] selectedRequest = new String[1];
        return AsiDelegation.invoke(new AsiDelegation.Preparation() {
            public Object[] prepare() throws Throwable {
                return prepareArguments(chain, selectedRequest);
            }
        }, new AsiDelegation.Original() {
            public Object invoke(Object[] arguments) throws Throwable {
                Object response = arguments == null ? chain.proceed() : chain.proceed(arguments);
                if (arguments != null) {
                    Log.i(TAG, "event=asi_result_forwarded request_id=" + selectedRequest[0]);
                }
                return response;
            }
        }, new AsiDelegation.Failure() {
            public void record(Throwable error) {
                Log.w(TAG, "event=asi_result_fallback error_class=" + error.getClass().getName());
            }
        });
    }

    private Object[] prepareArguments(XposedInterface.Chain chain, String[] selectedRequest)
            throws Throwable {
        Object value = chain.getArg(4);
        if (!(value instanceof List)) {
            return null;
        }
        List<?> candidates = (List<?>) value;
        if (candidates.isEmpty() || !ExtensionGate.isAsiAutofillExperimentEnabled(context)
                || !ExtensionGate.thermalPolicyAllows(context)) {
            return null;
        }
        RequestIdentity identity = new RequestIdentity("asi-autofill", "asi-process",
                Process.myUid(), NEXT_REQUEST.incrementAndGet());
        Log.i(TAG, "event=asi_candidates_enter request_id=" + identity.requestId
                + " candidate_count=" + candidates.size()
                + " caller_provenance=asi_process_local original_client_uid=unobserved");
        try {
            OstadixResultBridge.AsiSelectionOutcome outcome =
                    bridge.selectAsiCandidates(candidates, identity);
            OstadixRuntime.Evaluation evaluation = outcome.evaluation;
            Log.i(TAG, "event=asi_ostadix_selected request_id=" + identity.requestId
                    + " candidate_count=" + candidates.size()
                    + " source_index=" + evaluation.selectedSourceIndex
                    + " score_milli=" + evaluation.selectedScoreMilli
                    + " elapsed_ms=" + evaluation.elapsedMs
                    + " intent_sha256=" + evaluation.executionIntentSha256
                    + " result_identity=" + evaluation.resultContentIdentity);
            if (!ExtensionGate.isAsiAutofillExperimentEnabled(context)) {
                Log.w(TAG, "event=asi_result_fallback request_id=" + identity.requestId
                        + " reason=activation_gate");
                return null;
            }
            Object[] arguments = new Object[5];
            for (int index = 0; index < 4; index++) {
                arguments[index] = chain.getArg(index);
            }
            arguments[4] = outcome.selectedCandidates;
            selectedRequest[0] = identity.requestId;
            Log.i(TAG, "event=asi_result_selected_for_forwarding request_id=" + identity.requestId
                    + " selected_source_index=" + evaluation.selectedSourceIndex);
            return arguments;
        } catch (Throwable error) {
            ExtensionGate.rethrowIfVmFatal(error);
            Log.w(TAG, "event=asi_result_fallback request_id=" + identity.requestId
                    + " error_class=" + error.getClass().getName());
            return null;
        }
    }
}
