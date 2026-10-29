package org.ostadix.terminal;

import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.Method;

/** Host-side JNI smoke test; invokes the JSON-returning native edge directly. */
public final class RuntimeJniSmoke {
    private RuntimeJniSmoke() {
    }

    private static String jsonStringField(String json, String name) {
        String prefix = "\"" + name + "\":\"";
        int start = json.indexOf(prefix);
        if (start < 0) {
            throw new AssertionError("missing JSON field " + name + ": " + json);
        }
        start += prefix.length();
        int end = json.indexOf('"', start);
        if (end < 0) {
            throw new AssertionError("unterminated JSON field " + name + ": " + json);
        }
        return json.substring(start, end);
    }

    public static void main(String[] arguments) throws Exception {
        if (arguments.length != 3) {
            throw new IllegalArgumentException("expected backend, O, and Bash paths");
        }
        OstadixRuntime runtime = new OstadixRuntime(arguments[0], arguments[1], arguments[2]);
        try {
            Field handleField = OstadixRuntime.class.getDeclaredField("handle");
            handleField.setAccessible(true);
            long handle = handleField.getLong(runtime);

            Method evaluate = OstadixRuntime.class.getDeclaredMethod(
                    "nativeEvaluate", long.class, String.class);
            evaluate.setAccessible(true);
            String response = (String) evaluate.invoke(
                    null,
                    handle,
                    "text^(Hello from JNI)_text");
            if (!response.contains("\"ok\":true")
                    || !response.contains("Hello from JNI")) {
                throw new AssertionError("unexpected JNI response: " + response);
            }
            System.out.println(response);

            Method bounded = OstadixRuntime.class.getDeclaredMethod(
                    "nativeEvaluateBounded", long.class, String.class,
                    String.class, String.class, long.class);
            bounded.setAccessible(true);
            String boundedResponse = (String) bounded.invoke(
                    null,
                    handle,
                    "$input",
                    "jni-smoke/uid-test",
                    "jni-request-1",
                    5_000L);
            // No bindings are supplied at this JNI surface, so use a literal
            // for the successful graph request after proving a missing input
            // is a typed evaluation error.
            if (!boundedResponse.contains("\"ok\":false")
                    || !boundedResponse.contains("input")) {
                throw new AssertionError("unexpected bounded error: " + boundedResponse);
            }
            boundedResponse = (String) bounded.invoke(
                    null,
                    handle,
                    "text^(Bounded JNI)_text",
                    "jni-smoke/uid-test",
                    "jni-request-2",
                    5_000L);
            if (!boundedResponse.contains("\"ok\":true")
                    || !boundedResponse.contains("\"callerId\":\"jni-smoke/uid-test\"")
                    || !boundedResponse.contains("\"requestId\":\"jni-request-2\"")
                    || !boundedResponse.contains("\"planNodes\"")
                    || !boundedResponse.contains("\"executionIntentSha256\"")
                    || !boundedResponse.contains("\"requestScopeContentIdentity\"")
                    || !boundedResponse.contains("\"admissionSha256\"")
                    || !boundedResponse.contains("\"resultContentIdentity\"")
                    || !boundedResponse.contains("Bounded JNI")) {
                throw new AssertionError("unexpected bounded response: " + boundedResponse);
            }
            System.out.println(boundedResponse);

            Method boundedWithBindings = OstadixRuntime.class.getDeclaredMethod(
                    "nativeEvaluateBoundedWithBindings", long.class, String.class,
                    String.class, String.class, String.class, long.class);
            boundedWithBindings.setAccessible(true);
            String boundResponse = (String) boundedWithBindings.invoke(
                    null,
                    handle,
                    "$answer",
                    "{\"answer\":42}",
                    "jni-smoke/uid-test",
                    "jni-request-bound",
                    5_000L);
            if (!boundResponse.contains("\"ok\":true")
                    || !boundResponse.contains("\"output\":\"42\"")
                    || !boundResponse.contains("\"typedValue\"")
                    || !boundResponse.contains("\"oirSha256\"")
                    || !boundResponse.contains("\"planSha256\"")
                    || !boundResponse.contains("\"analyzedGraphSha256\"")) {
                throw new AssertionError("unexpected bound response: " + boundResponse);
            }
            String invalidBindings = (String) boundedWithBindings.invoke(
                    null,
                    handle,
                    "$answer",
                    "[]",
                    "jni-smoke/uid-test",
                    "jni-request-invalid-bindings",
                    5_000L);
            if (!invalidBindings.contains("\"ok\":false")
                    || !invalidBindings.contains("bindings JSON must be an object")) {
                throw new AssertionError("invalid bindings were accepted: " + invalidBindings);
            }
            System.out.println(boundResponse);
            System.out.println(invalidBindings);

            Method cancel = OstadixRuntime.class.getDeclaredMethod(
                    "nativeCancelRequest", long.class, String.class, String.class);
            cancel.setAccessible(true);
            File marker = File.createTempFile("ostadix-jni-dispatch-", ".marker");
            if (!marker.delete()) {
                throw new AssertionError("could not reserve cancellation marker path");
            }
            String escapedMarker = marker.getAbsolutePath().replace("'", "'\\''");
            String slowSource = "bash^(\nprintf started > '" + escapedMarker
                    + "'\nsleep 10\nprintf done\n)_bash";
            final String[] cancelledResponse = new String[1];
            final Throwable[] cancellationFailure = new Throwable[1];
            long cancellationStarted = System.nanoTime();
            Thread slowRequest = new Thread(new Runnable() {
                @Override
                public void run() {
                    try {
                        cancelledResponse[0] = (String) bounded.invoke(null, handle, slowSource,
                                "asoss/uid-1000", "llm-result-cancelled", 30_000L);
                    } catch (Throwable error) {
                        cancellationFailure[0] = error;
                    }
                }
            }, "ostadix-jni-cancellation-smoke");
            slowRequest.start();
            long markerDeadline = System.nanoTime() + 5_000_000_000L;
            while (!marker.isFile() && System.nanoTime() < markerDeadline) {
                Thread.sleep(10L);
            }
            if (!marker.isFile()) {
                throw new AssertionError("Bash request did not reach dispatched code");
            }
            boolean cancellationAccepted = (Boolean) cancel.invoke(null, handle,
                    "asoss/uid-1000", "llm-result-cancelled");
            slowRequest.join(3_000L);
            if (!cancellationAccepted || slowRequest.isAlive()
                    || cancellationFailure[0] != null
                    || cancelledResponse[0] == null
                    || !cancelledResponse[0].contains("cancellation")) {
                throw new AssertionError("JNI cancellation failed: " + cancelledResponse[0],
                        cancellationFailure[0]);
            }
            long cancellationMs = (System.nanoTime() - cancellationStarted) / 1_000_000L;
            if (cancellationMs >= 3_000L) {
                throw new AssertionError("JNI cancellation was not bounded: " + cancellationMs);
            }
            if (!marker.delete()) {
                throw new AssertionError("could not remove cancellation marker");
            }
            System.out.println("JNI cancellation elapsedMs=" + cancellationMs + " "
                    + cancelledResponse[0]);

            Method postprocess = OstadixRuntime.class.getDeclaredMethod(
                    "nativePostprocessAicoreReplies", long.class, int[].class, int[].class,
                    int[].class, int.class, String.class, String.class, long.class);
            postprocess.setAccessible(true);
            String selected = (String) postprocess.invoke(null, handle,
                    new int[] {600, 950, 990}, new int[] {0, 0, 1},
                    new int[] {20, 900, 10}, 500,
                    "asoss/uid-1000", "llm-result-1", 5_000L);
            if (!selected.contains("\"ok\":true")
                    || !selected.contains("\"selectedSourceIndex\":0")
                    || !selected.contains("\"selectedScoreMilli\":600")) {
                throw new AssertionError("unexpected AICore selection: " + selected);
            }
            String changed = (String) postprocess.invoke(null, handle,
                    new int[] {600, 950, 990}, new int[] {0, 0, 1},
                    new int[] {20, 100, 10}, 500,
                    "asoss/uid-1000", "llm-result-2", 5_000L);
            if (!changed.contains("\"ok\":true")
                    || !changed.contains("\"selectedSourceIndex\":1")
                    || !changed.contains("\"selectedScoreMilli\":950")
                    || !jsonStringField(selected, "executionIntentSha256").equals(
                            jsonStringField(changed, "executionIntentSha256"))
                    || jsonStringField(selected, "requestScopeContentIdentity").equals(
                            jsonStringField(changed, "requestScopeContentIdentity"))) {
                throw new AssertionError("AICore result did not depend on Ostadix: " + changed);
            }
            String rejected = (String) postprocess.invoke(null, handle,
                    new int[] {999}, new int[] {0}, new int[] {501}, 500,
                    "asoss/uid-1000", "llm-result-rejected", 5_000L);
            if (!rejected.contains("\"ok\":false")
                    || !rejected.contains("no policy-safe finished reply")) {
                throw new AssertionError("unsafe reply was not rejected: " + rejected);
            }
            System.out.println(selected);
            System.out.println(changed);
            System.out.println(rejected);

            Method smartReplies = OstadixRuntime.class.getDeclaredMethod(
                    "nativePostprocessAicoreSmartReplies", long.class, int[].class,
                    int[].class, int[].class, String.class, String.class, long.class);
            smartReplies.setAccessible(true);
            String smartSelected = (String) smartReplies.invoke(null, handle,
                    new int[] {600, 950, 990}, new int[] {1, 0, 1},
                    new int[] {0, 0, 1},
                    "asoss/uid-1000", "smart-reply-1", 5_000L);
            if (!smartSelected.contains("\"ok\":true")
                    || !smartSelected.contains("\"selectedSourceIndex\":0")
                    || !smartSelected.contains("\"selectedScoreMilli\":600")) {
                throw new AssertionError(
                        "unexpected Smart Reply selection: " + smartSelected);
            }
            String smartChanged = (String) smartReplies.invoke(null, handle,
                    new int[] {600, 950, 990}, new int[] {1, 1, 1},
                    new int[] {0, 0, 1},
                    "asoss/uid-1000", "smart-reply-2", 5_000L);
            if (!smartChanged.contains("\"ok\":true")
                    || !smartChanged.contains("\"selectedSourceIndex\":1")
                    || !smartChanged.contains("\"selectedScoreMilli\":950")
                    || !jsonStringField(smartSelected, "executionIntentSha256").equals(
                            jsonStringField(smartChanged, "executionIntentSha256"))
                    || jsonStringField(smartSelected, "requestScopeContentIdentity").equals(
                            jsonStringField(smartChanged, "requestScopeContentIdentity"))
                    || jsonStringField(selected, "executionIntentSha256").equals(
                            jsonStringField(smartSelected, "executionIntentSha256"))) {
                throw new AssertionError(
                        "Smart Reply result did not use its dedicated O contract: "
                                + smartChanged);
            }
            String smartRejected = (String) smartReplies.invoke(null, handle,
                    new int[] {999}, new int[] {1}, new int[] {2},
                    "asoss/uid-1000", "smart-reply-rejected", 5_000L);
            if (!smartRejected.contains("\"ok\":false")
                    || !smartRejected.contains("no safe nonempty reply")) {
                throw new AssertionError(
                        "unsafe Smart Reply was not rejected: " + smartRejected);
            }
            String invalidPresence = (String) smartReplies.invoke(null, handle,
                    new int[] {999}, new int[] {2}, new int[] {0},
                    "asoss/uid-1000", "smart-reply-invalid-presence", 5_000L);
            if (!invalidPresence.contains("\"ok\":false")
                    || !invalidPresence.contains("has_text values must be 0 or 1")) {
                throw new AssertionError(
                        "invalid Smart Reply presence was not rejected: " + invalidPresence);
            }
            System.out.println(smartSelected);
            System.out.println(smartChanged);
            System.out.println(smartRejected);
            System.out.println(invalidPresence);
        } finally {
            runtime.close();
        }
    }
}
