package com.google.android.apps.aicore.aidl;

import java.util.List;

public final class SmartReplyResult {
    private final List<?> results;
    private final InferenceEventTraceResult trace;

    public SmartReplyResult(List<?> results, InferenceEventTraceResult trace) {
        this.results = results;
        this.trace = trace;
    }

    public List<?> getResults() { return results; }
    public InferenceEventTraceResult getInferenceEventTraceResult() { return trace; }
}
