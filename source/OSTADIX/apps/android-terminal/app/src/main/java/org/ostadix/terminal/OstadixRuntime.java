package org.ostadix.terminal;

import org.json.JSONException;
import org.json.JSONObject;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/** In-process JNI wrapper around the stable Ostadix embedding API. */
public final class OstadixRuntime implements AutoCloseable {
    static {
        try {
            // An embedding host can load the package-owned library by absolute
            // path before this class initializes. Probe the stable JNI symbol
            // first because an injected class loader may not expose that same
            // extracted directory to System.loadLibrary's name lookup.
            nativeVersion();
        } catch (UnsatisfiedLinkError notLoaded) {
            System.loadLibrary("ostadix_runtime");
        }
    }

    public static final class Evaluation {
        public final boolean ok;
        public final String stage;
        public final String type;
        public final String output;
        public final String message;
        public final String callerId;
        public final String requestId;
        public final String executionIntentSha256;
        public final String requestScopeContentIdentity;
        public final String admissionSha256;
        public final String resultContentIdentity;
        public final long elapsedMs;
        public final int selectedSourceIndex;
        public final int selectedScoreMilli;

        private Evaluation(boolean ok, String stage, String type, String output, String message,
                String callerId, String requestId, String executionIntentSha256,
                String requestScopeContentIdentity, String admissionSha256,
                String resultContentIdentity, long elapsedMs, int selectedSourceIndex,
                int selectedScoreMilli) {
            this.ok = ok;
            this.stage = stage;
            this.type = type;
            this.output = output;
            this.message = message;
            this.callerId = callerId;
            this.requestId = requestId;
            this.executionIntentSha256 = executionIntentSha256;
            this.requestScopeContentIdentity = requestScopeContentIdentity;
            this.admissionSha256 = admissionSha256;
            this.resultContentIdentity = resultContentIdentity;
            this.elapsedMs = elapsedMs;
            this.selectedSourceIndex = selectedSourceIndex;
            this.selectedScoreMilli = selectedScoreMilli;
        }

        public String terminalText() {
            if (ok) {
                return "[" + type + "] " + output;
            }
            return "error (" + stage + "): " + message;
        }
    }

    private long handle;
    private final AtomicLong nextRequestId = new AtomicLong();
    private final ReentrantReadWriteLock lifecycle = new ReentrantReadWriteLock();
    private static final long DEFAULT_TIMEOUT_MS = 30_000;

    public OstadixRuntime(String shimDirectory, String runtimeExecutable, String bashExecutable) {
        handle = nativeCreate(shimDirectory, runtimeExecutable, bashExecutable);
        if (handle == 0) {
            throw new IllegalStateException("Unable to initialize the Ostadix runtime");
        }
    }

    public Evaluation evaluate(String source) {
        try {
            String requestId = "terminal-" + nextRequestId.incrementAndGet();
            JSONObject result = new JSONObject(evaluateBoundedJson(source, "{}",
                    "org.ostadix.terminal", requestId, DEFAULT_TIMEOUT_MS));
            boolean ok = result.optBoolean("ok", false);
            return new Evaluation(
                    ok,
                    result.optString("stage", ok ? "complete" : "runtime"),
                    result.optString("type", "value"),
                    result.optString("output", ""),
                    result.optString("message", "Unknown runtime error"),
                    result.optString("callerId", "org.ostadix.terminal"),
                    result.optString("requestId", requestId),
                    result.optString("executionIntentSha256", ""),
                    result.optString("requestScopeContentIdentity", ""),
                    result.optString("admissionSha256", ""),
                    result.optString("resultContentIdentity", ""),
                    result.optLong("elapsedMs", 0),
                    result.optInt("selectedSourceIndex", -1),
                    result.optInt("selectedScoreMilli", 0));
        } catch (JSONException error) {
            return new Evaluation(false, "bridge", "", "", error.getMessage(),
                    "org.ostadix.terminal", "", "", "", "", "", 0, -1, 0);
        }
    }

    /**
     * Execute one complete O document with ordinary JSON bindings and retain the
     * canonical RuntimeRequest JSON envelope for agent and service embeddings.
     */
    public String evaluateBoundedJson(String source, String bindingsJson,
            String callerId, String requestId, long timeoutMs) {
        if (source == null || bindingsJson == null || callerId == null || requestId == null) {
            throw new IllegalArgumentException("bounded evaluation arguments must be non-null");
        }
        Lock lock = lifecycle.readLock();
        lock.lock();
        try {
            if (handle == 0) {
                throw new IllegalStateException("Ostadix runtime is closed");
            }
            return nativeEvaluateBoundedWithBindings(handle, source, bindingsJson,
                    callerId, requestId, timeoutMs);
        } finally {
            lock.unlock();
        }
    }

    /** Run the fixed AICore reply selector without exposing generated text to Ostadix. */
    public Evaluation postprocessAicoreReplies(int[] scoresMilli,
            int[] stopReasons, int[] maxPolicyScoresMilli, int policyLimit,
            String callerId, String requestId, long timeoutMs) {
        Lock lock = lifecycle.readLock();
        lock.lock();
        try {
            if (handle == 0) {
                throw new IllegalStateException("Ostadix runtime is closed");
            }
            JSONObject result = new JSONObject(nativePostprocessAicoreReplies(handle,
                    scoresMilli, stopReasons, maxPolicyScoresMilli, policyLimit,
                    callerId, requestId, timeoutMs));
            boolean ok = result.optBoolean("ok", false);
            return new Evaluation(ok,
                    result.optString("stage", ok ? "complete" : "runtime"),
                    result.optString("type", "value"), result.optString("output", ""),
                    result.optString("message", "Unknown runtime error"),
                    result.optString("callerId", callerId),
                    result.optString("requestId", requestId),
                    result.optString("executionIntentSha256", ""),
                    result.optString("requestScopeContentIdentity", ""),
                    result.optString("admissionSha256", ""),
                    result.optString("resultContentIdentity", ""),
                    result.optLong("elapsedMs", 0),
                    result.optInt("selectedSourceIndex", -1),
                    result.optInt("selectedScoreMilli", 0));
        } catch (JSONException error) {
            return new Evaluation(false, "bridge", "", "", error.getMessage(), callerId,
                    requestId, "", "", "", "", 0, -1, 0);
        } finally {
            lock.unlock();
        }
    }

    /** Select one safe, nonempty Smart Reply without exposing generated text to Ostadix. */
    public Evaluation postprocessAicoreSmartReplies(int[] scoresMilli,
            int[] hasText, int[] safetyClassifications,
            String callerId, String requestId, long timeoutMs) {
        Lock lock = lifecycle.readLock();
        lock.lock();
        try {
            if (handle == 0) {
                throw new IllegalStateException("Ostadix runtime is closed");
            }
            JSONObject result = new JSONObject(nativePostprocessAicoreSmartReplies(handle,
                    scoresMilli, hasText, safetyClassifications,
                    callerId, requestId, timeoutMs));
            boolean ok = result.optBoolean("ok", false);
            return new Evaluation(ok,
                    result.optString("stage", ok ? "complete" : "runtime"),
                    result.optString("type", "value"), result.optString("output", ""),
                    result.optString("message", "Unknown runtime error"),
                    result.optString("callerId", callerId),
                    result.optString("requestId", requestId),
                    result.optString("executionIntentSha256", ""),
                    result.optString("requestScopeContentIdentity", ""),
                    result.optString("admissionSha256", ""),
                    result.optString("resultContentIdentity", ""),
                    result.optLong("elapsedMs", 0),
                    result.optInt("selectedSourceIndex", -1),
                    result.optInt("selectedScoreMilli", 0));
        } catch (JSONException error) {
            return new Evaluation(false, "bridge", "", "", error.getMessage(), callerId,
                    requestId, "", "", "", "", 0, -1, 0);
        } finally {
            lock.unlock();
        }
    }

    /** Cancel one active request without waiting for its evaluation call to return. */
    public boolean cancelRequest(String callerId, String requestId) {
        Lock lock = lifecycle.readLock();
        lock.lock();
        try {
            return handle != 0 && nativeCancelRequest(handle, callerId, requestId);
        } finally {
            lock.unlock();
        }
    }

    public static String version() {
        return nativeVersion();
    }

    @Override
    public void close() {
        Lock lock = lifecycle.writeLock();
        lock.lock();
        try {
            if (handle != 0) {
                nativeDestroy(handle);
                handle = 0;
            }
        } finally {
            lock.unlock();
        }
    }

    private static native long nativeCreate(String shimDirectory, String runtimeExecutable,
            String bashExecutable);
    private static native String nativeEvaluate(long handle, String source);
    private static native String nativeEvaluateBounded(long handle, String source,
            String callerId, String requestId, long timeoutMs);
    private static native String nativeEvaluateBoundedWithBindings(long handle, String source,
            String bindingsJson, String callerId, String requestId, long timeoutMs);
    private static native String nativePostprocessAicoreReplies(long handle, int[] scoresMilli,
            int[] stopReasons, int[] maxPolicyScoresMilli, int policyLimit,
            String callerId, String requestId, long timeoutMs);
    private static native String nativePostprocessAicoreSmartReplies(long handle,
            int[] scoresMilli, int[] hasText, int[] safetyClassifications,
            String callerId, String requestId, long timeoutMs);
    private static native boolean nativeCancelRequest(long handle, String callerId,
            String requestId);
    private static native String nativeVersion();
    private static native void nativeDestroy(long handle);
}
