package org.ostadix.aicore.extension;

import android.content.Context;
import android.os.Binder;
import android.util.Log;

import java.lang.reflect.Field;
import java.util.concurrent.atomic.AtomicLong;

import io.github.libxposed.api.XposedInterface;
import org.ostadix.terminal.OstadixRuntime;

final class AicoreHooks {
    private static final String TAG = "OstadixAicoreExperiment";
    private static final AtomicLong NEXT_REQUEST = new AtomicLong();
    private static final RequestCorrelations CORRELATIONS = new RequestCorrelations();

    private final Context context;
    private final OstadixResultBridge bridge;
    private final Field callbackField;

    AicoreHooks(Context context, OstadixResultBridge bridge, Class<?> callbackClass)
            throws Exception {
        this.context = context;
        this.bridge = bridge;
        callbackField = callbackClass.getDeclaredField("a");
        callbackField.setAccessible(true);
    }

    Object interceptForward(XposedInterface.Chain chain) throws Throwable {
        if (!ExtensionGate.isSmartReplyExperimentEnabled(context)) {
            return chain.proceed();
        }
        Object callback = chain.getArg(1);
        RequestIdentity identity = new RequestIdentity(
                Binder.getCallingUid(), NEXT_REQUEST.incrementAndGet());
        RequestCorrelations.State state = CORRELATIONS.begin(callback, identity);
        Log.i(TAG, "event=request_enter request_id=" + identity.requestId
                + " caller_uid=" + identity.uid);
        Object cancellation;
        try {
            cancellation = chain.proceed();
        } catch (Throwable error) {
            CORRELATIONS.complete(state);
            Log.w(TAG, "event=request_dispatch_error request_id=" + identity.requestId
                    + " error_class=" + error.getClass().getName());
            throw error;
        }
        CORRELATIONS.registerCancellation(cancellation, state);
        Log.i(TAG, "event=request_dispatched request_id=" + identity.requestId
                + " cancellation_handle=" + (cancellation != null));
        return cancellation;
    }

    Object interceptSuccess(XposedInterface.Chain chain) throws Throwable {
        Object callback = callbackField.get(chain.getThisObject());
        RequestCorrelations.State state = CORRELATIONS.claimCallback(callback);
        if (state == null) {
            return chain.proceed();
        }
        RequestIdentity identity = state.identity;
        if (!ExtensionGate.isSmartReplyExperimentEnabled(context)) {
            CORRELATIONS.complete(state);
            Log.w(TAG, "event=result_fallback request_id=" + identity.requestId
                    + " reason=activation_gate");
            return chain.proceed();
        }
        if (!ExtensionGate.thermalPolicyAllows(context)) {
            CORRELATIONS.complete(state);
            Log.w(TAG, "event=result_fallback request_id=" + identity.requestId
                    + " reason=thermal_gate");
            return chain.proceed();
        }
        if (CORRELATIONS.isCancelled(state)) {
            CORRELATIONS.complete(state);
            Log.w(TAG, "event=result_fallback request_id=" + identity.requestId
                    + " reason=cancellation");
            return chain.proceed();
        }
        OstadixResultBridge.SelectionOutcome outcome;
        try {
            outcome = bridge.select(chain.getArg(0), identity);
        } catch (Throwable error) {
            ExtensionGate.rethrowIfVmFatal(error);
            Log.w(TAG, "event=result_fallback request_id=" + identity.requestId
                    + " reason=transformation_error error_class="
                    + error.getClass().getName());
            CORRELATIONS.complete(state);
            return chain.proceed();
        }
        OstadixRuntime.Evaluation evaluation = outcome.evaluation;
        Log.i(TAG, "event=ostadix_selected request_id=" + identity.requestId
                + " reply_count=" + outcome.replyCount
                + " source_index=" + evaluation.selectedSourceIndex
                + " score_milli=" + evaluation.selectedScoreMilli
                + " elapsed_ms=" + evaluation.elapsedMs
                + " intent_sha256=" + safe(evaluation.executionIntentSha256)
                + " scope_identity=" + safe(evaluation.requestScopeContentIdentity)
                + " admission_sha256=" + safe(evaluation.admissionSha256)
                + " result_identity=" + safe(evaluation.resultContentIdentity));
        if (!ExtensionGate.isSmartReplyExperimentEnabled(context)) {
            CORRELATIONS.complete(state);
            Log.w(TAG, "event=result_fallback request_id=" + identity.requestId
                    + " reason=activation_gate");
            return chain.proceed();
        }
        if (!CORRELATIONS.commitReplacement(state)) {
            Log.w(TAG, "event=result_fallback request_id=" + identity.requestId
                    + " reason=cancellation");
            return chain.proceed();
        }
        // Delivery sits outside the transformation catch: if the existing
        // downstream callback throws, propagate it and never invoke it twice.
        try {
            Object delivered = chain.proceed(new Object[] {outcome.replacement});
            Log.i(TAG, "event=result_forwarded request_id=" + identity.requestId
                    + " selected_source_index=" + evaluation.selectedSourceIndex);
            return delivered;
        } catch (Throwable error) {
            Log.e(TAG, "event=result_delivery_error request_id=" + identity.requestId
                    + " error_class=" + error.getClass().getName());
            throw error;
        }
    }

    Object interceptFailure(XposedInterface.Chain chain) throws Throwable {
        Object callback = callbackField.get(chain.getThisObject());
        RequestCorrelations.State state = CORRELATIONS.claimCallback(callback);
        if (state != null) {
            CORRELATIONS.complete(state);
            RequestIdentity identity = state.identity;
            Object value = chain.getArg(0);
            int errorCode = value instanceof Number ? ((Number) value).intValue() : -1;
            Log.w(TAG, "event=inference_failure request_id=" + identity.requestId
                    + " error_code=" + errorCode);
        }
        return chain.proceed();
    }

    Object interceptCancellation(XposedInterface.Chain chain) throws Throwable {
        RequestCorrelations.State state = CORRELATIONS.cancel(chain.getThisObject());
        if (state != null) {
            RequestIdentity identity = state.identity;
            boolean propagated = false;
            try {
                propagated = bridge.cancel(identity);
            } catch (Throwable error) {
                ExtensionGate.rethrowIfVmFatal(error);
                Log.w(TAG, "event=cancellation_error request_id=" + identity.requestId
                        + " error_class=" + error.getClass().getName());
            }
            Log.i(TAG, "event=cancellation_forwarded request_id=" + identity.requestId
                    + " ostadix_active_request=" + propagated);
        }
        return chain.proceed();
    }

    private static String safe(String value) {
        if (value == null || value.isEmpty()) {
            return "none";
        }
        return value;
    }
}
