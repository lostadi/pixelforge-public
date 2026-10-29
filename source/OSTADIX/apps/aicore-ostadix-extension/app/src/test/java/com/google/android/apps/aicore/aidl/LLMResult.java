package com.google.android.apps.aicore.aidl;

import java.util.List;

public final class LLMResult {
    private final List<?> results;
    private final InferenceEventTraceResult trace;
    private final LegionResultMetadata legion;
    private final List<?> thought;

    public LLMResult(List<?> results, InferenceEventTraceResult trace,
            LegionResultMetadata legion, List<?> thought) {
        this.results = results;
        this.trace = trace;
        this.legion = legion;
        this.thought = thought;
    }

    public List<?> getResults() { return results; }
    public InferenceEventTraceResult getInferenceEventTraceResult() { return trace; }
    public LegionResultMetadata getLegionResultMetadata() { return legion; }
    public List<?> getThoughtProcess() { return thought; }
}
