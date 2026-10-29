package org.ostadix.aicore.extension;

import org.json.JSONObject;

/** Runs on the JVM with a real org.json implementation, not the Android SDK stubs. */
public final class NanoToolOutputSelfTest {
    public static void main(String[] args) throws Exception {
        JSONObject number = new JSONObject("{\"value\":{\"t\":\"number\",\"v\":{\"kind\":\"int\",\"v\":\"276\"}},"
                + "\"execution_evidence\":{\"result_content_identity\":\"synthetic-result-id\"}}");
        String actual = NanoToolOutput.completed(number);
        require(actual.startsWith("Ostadix output:\n276\n") && actual.contains("synthetic-result-id"),
                "actual number or identity missing");
        StringBuilder longText = new StringBuilder();
        for (int i = 0; i < 4096; i++) { longText.append('x'); }
        number.put("value", new JSONObject().put("t", "string").put("v", longText.toString()));
        require(NanoToolOutput.completed(number).contains(longText), "actual output truncated");
        number.put("value", new JSONObject().put("t", "text").put("v",
                new JSONObject().put("encoding", "utf-8").put("utf8", "actual UTF-8 text")));
        require(NanoToolOutput.completed(number).startsWith("Ostadix output:\nactual UTF-8 text\n"), "text value not decoded");
        JSONObject failed = new JSONObject().put("result", new JSONObject().put("error", "backend failed"))
                .put("stdout", new JSONObject().put("text", "partial output").put("complete", true))
                .put("stderr", new JSONObject().put("text", "compiler error").put("complete", true));
        String failure = NanoToolOutput.failed(failed);
        require(failure.contains("partial output") && failure.contains("compiler error")
                && failure.contains("not retried") && !failure.contains("Nothing was executed"),
                "execution failure hid output or implied rollback");
        String traceback = "[python[*ephemeral*]]: backend recv_step failed: Traceback (most recent call last):\n"
                + "  File \"/private/backend.py\", line 100, in handle_exec\n"
                + "  File \"<O-python>\", line 3, in <module>\n"
                + "TypeError: 'int' object is not iterable\n\nGenerated Python source: /private/generated.py";
        failed.getJSONObject("result").put("error", traceback);
        failed.getJSONObject("stderr").put("text", traceback);
        String saved = failed.toString();
        failure = NanoToolOutput.failed(failed);
        require(failure.startsWith("Ostadix execution failed: TypeError: 'int' object is not iterable")
                && !failure.contains("Traceback") && !failure.contains("/private/")
                && failure.contains("Show program and execution details")
                && failure.contains("partial output") && failure.contains("not rolled back"),
                "traceback summary lost the actual error, output or effect warning");
        require(saved.equals(failed.toString()), "formatting modified the full saved evidence");
        failed.getJSONObject("result").put("error", "compiler: " + longText);
        failed.getJSONObject("stdout").put("text", longText.toString());
        failed.getJSONObject("stderr").put("text", "diagnostic: " + longText);
        failure = NanoToolOutput.failed(failed);
        require(failure.length() < 2500 && failure.contains("Excerpt"), "large diagnostics were not explicitly bounded");
        StringBuilder emoji = new StringBuilder();
        for (int i = 0; i < 1000; i++) { emoji.append("\ud83d\ude80"); }
        failed.getJSONObject("result").put("error", emoji.toString());
        failure = NanoToolOutput.failed(failed);
        require(!failure.contains("\ud83d\n"), "Unicode diagnostic excerpt split a surrogate pair");
        String report = "Actual tool output: number, identity, long text, failure streams, concise diagnostics "
                + "and preserved raw evidence passed\n";
        try (java.io.FileOutputStream out = new java.io.FileOutputStream(args[0])) {
            out.write(report.getBytes(java.nio.charset.StandardCharsets.UTF_8)); out.getFD().sync();
        }
    }
    private static void require(boolean condition, String message) {
        if (!condition) { throw new AssertionError(message); }
    }
}
