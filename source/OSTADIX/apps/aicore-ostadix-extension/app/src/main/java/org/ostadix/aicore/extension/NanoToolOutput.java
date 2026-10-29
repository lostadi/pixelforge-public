package org.ostadix.aicore.extension;

import org.json.JSONObject;

/** Formats actual tool output independently of any model interpretation. */
final class NanoToolOutput {
    static String completed(JSONObject result) throws Exception {
        return "Ostadix output:\n" + value(result.get("value")) + "\n\nExecuted locally with Ostadix. "
                + "Result identity: " + result.getJSONObject("execution_evidence").getString("result_content_identity");
    }

    private static String value(Object item) throws Exception {
        if (!(item instanceof JSONObject)) { return String.valueOf(item); }
        JSONObject tagged = (JSONObject) item;
        String type = tagged.optString("t"); Object v = tagged.opt("v");
        if ("number".equals(type) && v instanceof JSONObject) {
            JSONObject n = (JSONObject) v;
            if ("int".equals(n.optString("kind")) || "float".equals(n.optString("kind"))) {
                return n.get("v").toString();
            }
        }
        if ("text".equals(type) && v instanceof JSONObject && ((JSONObject) v).has("utf8")) {
            return ((JSONObject) v).getString("utf8");
        }
        if ("string".equals(type) || "text".equals(type) || "bool".equals(type)) {
            return String.valueOf(v);
        }
        // Keep all values and their types for collections or less common O types.
        return tagged.toString(2);
    }

    static String failed(JSONObject result) {
        JSONObject nativeResult = result.optJSONObject("result");
        String error = nativeResult == null ? result.optString("error", "No result was returned")
                : nativeResult.optString("error", result.optString("error", "Execution failed"));
        String summary = diagnosticSummary(error);
        StringBuilder text = new StringBuilder("Ostadix execution failed: ").append(summary);
        for (String name : new String[]{"stdout", "stderr"}) {
            JSONObject stream = result.optJSONObject(name);
            String output = stream == null ? "" : stream.optString("text").trim();
            // The raw MCP response is already saved as evidence. Error streams often
            // repeat its traceback or serialize the entire native result again.
            if (!output.isEmpty() && !error.contains(output) && !output.contains(error)
                    && !("stderr".equals(name) && output.contains(summary))) {
                text.append("\n\n").append(name).append(":\n").append(excerpt(output, 600));
                if (!stream.optBoolean("complete")) { text.append("\n[Output excerpt.]"); }
            }
        }
        text.append("\n\nOpen Show program and execution details in Ostadix Results for the saved error and output.");
        text.append("\nExecution was not retried; completed effects were not rolled back.");
        return text.toString();
    }

    private static String diagnosticSummary(String error) {
        // Keep the actual terminal Python exception, not backend paths and frames.
        // For other diagnostics retain the beginning verbatim and mark excerpts.
        String terminalException = null;
        for (String line : error.split("\\r?\\n")) {
            String trimmed = line.trim();
            if (trimmed.matches("(?:[A-Za-z_][A-Za-z0-9_.]*)?(?:Error|Exception|Interrupt|Exit)(?::.*)?")) {
                terminalException = trimmed;
            }
        }
        return excerpt(terminalException == null ? error.trim() : terminalException, 600);
    }

    private static String excerpt(String text, int limit) {
        if (text.codePointCount(0, text.length()) <= limit) { return text; }
        return text.substring(0, text.offsetByCodePoints(0, limit)) + "\n[Excerpt; see execution details.]";
    }

    private NanoToolOutput() {}
}
