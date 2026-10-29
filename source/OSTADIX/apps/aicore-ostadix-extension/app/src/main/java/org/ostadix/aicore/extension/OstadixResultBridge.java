package org.ostadix.aicore.extension;

import android.content.pm.ApplicationInfo;

import org.ostadix.terminal.OstadixRuntime;

import java.io.File;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Collections;
import java.util.List;

final class OstadixResultBridge implements AutoCloseable {
    private static final int MAX_REPLIES = 3;
    private static final long TIMEOUT_MS = 500L;

    private final OstadixRuntime runtime;

    OstadixResultBridge(ApplicationInfo moduleApplicationInfo) throws Exception {
        if (moduleApplicationInfo == null || moduleApplicationInfo.nativeLibraryDir == null) {
            throw new IllegalStateException("module native library directory unavailable");
        }
        File nativeDirectory = new File(moduleApplicationInfo.nativeLibraryDir);
        File runtimeLibrary = new File(nativeDirectory, "libostadix_runtime.so");
        if (!runtimeLibrary.isFile()) {
            throw new IllegalStateException("packaged OSTADIX JNI runtime unavailable");
        }
        System.load(runtimeLibrary.getAbsolutePath());
        File o = requireExecutable(nativeDirectory, "libostadix_cli.so");
        File bash = requireExecutable(nativeDirectory, "libostadix_bash.so");
        // Bash is a native Rust backend and uses no legacy shim. The immutable
        // module native directory is therefore also a valid empty shim root;
        // initialization writes nothing into AS.OSS private storage.
        runtime = new OstadixRuntime(nativeDirectory.getAbsolutePath(),
                o.getAbsolutePath(), bash.getAbsolutePath());
    }

    static final class SelectionOutcome {
        final Object replacement;
        final OstadixRuntime.Evaluation evaluation;
        final int replyCount;

        SelectionOutcome(Object replacement, OstadixRuntime.Evaluation evaluation, int replyCount) {
            this.replacement = replacement;
            this.evaluation = evaluation;
            this.replyCount = replyCount;
        }
    }

    static final class ReplyProjection {
        final int[] scores;
        final int[] hasText;
        final int[] safetyClassifications;

        ReplyProjection(int[] scores, int[] hasText, int[] safetyClassifications) {
            this.scores = scores;
            this.hasText = hasText;
            this.safetyClassifications = safetyClassifications;
        }
    }

    static final class AsiSelectionOutcome {
        final List<?> selectedCandidates;
        final OstadixRuntime.Evaluation evaluation;

        AsiSelectionOutcome(List<?> selectedCandidates, OstadixRuntime.Evaluation evaluation) {
            this.selectedCandidates = selectedCandidates;
            this.evaluation = evaluation;
        }
    }

    AsiSelectionOutcome selectAsiCandidates(List<?> candidates, RequestIdentity identity)
            throws Exception {
        if (candidates.isEmpty() || candidates.size() > MAX_REPLIES) {
            throw new IllegalArgumentException("unsupported ASI candidate count");
        }
        int count = candidates.size();
        int[] scores = new int[count];
        int[] hasText = new int[count];
        int[] safety = new int[count];
        for (int index = 0; index < count; index++) {
            Object candidate = candidates.get(index);
            Field entityField = candidate.getClass().getDeclaredField("b");
            entityField.setAccessible(true);
            Object entity = entityField.get(candidate);
            Field textField = entity.getClass().getDeclaredField("a");
            textField.setAccessible(true);
            Object candidateText = textField.get(entity);
            hasText[index] = candidateText instanceof CharSequence
                    && ((CharSequence) candidateText).length() > 0 ? 1 : 0;
            // ASI has already applied provider safety and ranking before this seam.
            // Preserve that order as descending scalar metadata for OSTADIX.
            scores[index] = (count - index) * 1000;
            safety[index] = 0;
        }
        OstadixRuntime.Evaluation result = runtime.postprocessAicoreSmartReplies(
                scores, hasText, safety, identity.callerId(), identity.requestId, TIMEOUT_MS);
        if (!result.ok || result.selectedSourceIndex < 0
                || result.selectedSourceIndex >= candidates.size()) {
            throw new IllegalStateException("OSTADIX ASI candidate selection failed closed");
        }
        return new AsiSelectionOutcome(Collections.singletonList(
                candidates.get(result.selectedSourceIndex)), result);
    }

    SelectionOutcome select(Object originalResult, RequestIdentity identity) throws Exception {
        Method getResults = originalResult.getClass().getMethod("getResults");
        List<?> replies = (List<?>) getResults.invoke(originalResult);
        if (replies == null || replies.isEmpty() || replies.size() > MAX_REPLIES) {
            throw new IllegalArgumentException("unsupported Smart Reply count");
        }
        int count = replies.size();
        ReplyProjection projection = projectSmartReplies(replies);

        OstadixRuntime.Evaluation result = runtime.postprocessAicoreSmartReplies(
                projection.scores, projection.hasText, projection.safetyClassifications,
                identity.callerId(), identity.requestId, TIMEOUT_MS);
        if (!result.ok || result.selectedSourceIndex < 0
                || result.selectedSourceIndex >= replies.size()) {
            throw new IllegalStateException("OSTADIX result selection failed closed");
        }

        return new SelectionOutcome(
                rebuildSelectedSmartReplyResult(
                        originalResult, replies, result.selectedSourceIndex),
                result, count);
    }

    static ReplyProjection projectSmartReplies(List<?> replies) throws Exception {
        int count = replies.size();
        int[] scores = new int[count];
        int[] hasText = new int[count];
        int[] safetyClassifications = new int[count];
        for (int index = 0; index < count; index++) {
            Object reply = replies.get(index);
            scores[index] = scaledMilli((Number) reply.getClass()
                    .getMethod("getScore").invoke(reply));
            // A SmartReplyResult is delivered only after generation completes,
            // so a nonempty entry is a finished candidate. Only the presence
            // bit crosses JNI; generated text stays in the AS.OSS result object.
            Object text = reply.getClass().getMethod("getText").invoke(reply);
            hasText[index] = text instanceof CharSequence
                    && ((CharSequence) text).length() > 0 ? 1 : 0;
            // SafetyClassificationResult NONE is 0; every other or future
            // value is conservatively ineligible.
            safetyClassifications[index] = ((Number) reply.getClass()
                    .getMethod("getSafetyClassificationResult").invoke(reply)).intValue();
        }
        return new ReplyProjection(scores, hasText, safetyClassifications);
    }

    static Object rebuildSelectedSmartReplyResult(
            Object originalResult, List<?> replies, int selectedIndex) throws Exception {
        if (selectedIndex < 0 || selectedIndex >= replies.size()) {
            throw new IllegalArgumentException("selected reply index is out of range");
        }
        Object selected = replies.get(selectedIndex);
        Method trace = originalResult.getClass().getMethod("getInferenceEventTraceResult");
        ClassLoader loader = originalResult.getClass().getClassLoader();
        Class<?> traceClass = Class.forName(
                "com.google.android.apps.aicore.aidl.InferenceEventTraceResult", false, loader);
        Constructor<?> constructor = originalResult.getClass().getConstructor(
                List.class, traceClass);
        return constructor.newInstance(Collections.singletonList(selected),
                trace.invoke(originalResult));
    }

    boolean cancel(RequestIdentity identity) {
        return runtime.cancelRequest(identity.callerId(), identity.requestId);
    }

    OstadixRuntime.Evaluation smokeRuntime() {
        return runtime.postprocessAicoreSmartReplies(
                new int[] {600, 950}, new int[] {1, 1}, new int[] {0, 0},
                "asoss-startup-smoke", "smart-reply-smoke", TIMEOUT_MS);
    }

    @Override
    public void close() {
        runtime.close();
    }

    private static File requireExecutable(File directory, String name) {
        File executable = new File(directory, name);
        if (!executable.isFile() || !executable.canExecute()) {
            throw new IllegalStateException("packaged executable unavailable: " + name);
        }
        return executable;
    }

    private static int scaledMilli(Number value) {
        double scaled = value.doubleValue() * 1000.0d;
        if (!Double.isFinite(scaled) || scaled > Integer.MAX_VALUE
                || scaled < Integer.MIN_VALUE) {
            throw new IllegalArgumentException("non-finite or out-of-range reply score");
        }
        return (int) Math.round(scaled);
    }

}
