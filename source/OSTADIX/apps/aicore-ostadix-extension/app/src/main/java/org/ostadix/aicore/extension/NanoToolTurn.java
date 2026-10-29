package org.ostadix.aicore.extension;

import android.content.Context;
import android.os.CancellationSignal;
import android.os.SystemClock;
import android.util.Log;
import org.json.JSONArray;
import org.json.JSONObject;
import org.ostadix.terminal.HostMcpClient;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Locale;

/** One host-owned tool interaction, using the existing source-first MCP adapter. */
final class NanoToolTurn {
    private final Context context;
    private final String id;
    private final String conversation;
    private final CancellationSignal cancellation;
    private final JSONObject evidence = new JSONObject();
    private String phase = "not_dispatched";

    NanoToolTurn(Context context, String id, String conversation, CancellationSignal cancellation) {
        this.context = context; this.id = id; this.conversation = conversation; this.cancellation = cancellation;
    }

    String run(String userText) throws Exception {
        long start = SystemClock.elapsedRealtime();
        final JSONArray history = NanoConversationContext.load(context.getFilesDir(), conversation, id);
        evidence.put("schema", "ostadix.gemini-local-tool-turn/v2").put("request_id", id)
                .put("ordinary_assistant_request", true).put("caller_uid", android.os.Process.myUid())
                .put("caller_pid", android.os.Process.myPid()).put("user_text", userText)
                .put("conversation_key", conversation).put("context_turns", history)
                .put("stock_callback_delegations", 0).put("mcp_dispatch_attempts", 0)
                .put("execution_dispatch_attempts", 0).put("source_correction_attempts", 0)
                .put("source_checks", new JSONArray());
        try {
            check();
            phase = "source_generation"; save();
            JSONObject generated = NanoActionClient.generate(context,
                    NanoProgramPrompt.generation(userText, history), cancellation);
            evidence.put("source_generation", generated);
            String initialSource = extractSource(candidate(generated, true));
            evidence.put("initial_source", initialSource); save();
            NanoSourcePreparation.Prepared prepared = NanoSourcePreparation.prepare(initialSource,
                    new NanoSourcePreparation.Operations() {
                        public void checkActive() { check(); }
                        public String validate(String source) throws Exception { return validateSource(source); }
                        public String correct(String source, String error) throws Exception {
                            check(); phase = "source_correction";
                            evidence.put("source_correction_attempts", 1); save();
                            JSONObject correction = NanoActionClient.generate(context,
                                    NanoProgramPrompt.correction(userText, history, source, error), cancellation);
                            evidence.put("source_correction", correction); save();
                            return extractSource(candidate(correction, true));
                        }
                    });
            String source = prepared.source;
            evidence.put("mcp_request", new JSONObject().put("method", "tools/call").put("params",
                    new JSONObject().put("name", "o_execute").put("arguments",
                            new JSONObject().put("source", source).put("timeout_secs", 120))))
                    .put("source_sha256", digest(source));
            check(); phase = "mcp_dispatch";
            evidence.put("mcp_dispatch_attempts", evidence.getInt("mcp_dispatch_attempts") + 1)
                    .put("execution_dispatch_attempts", 1); save();
            JSONObject result;
            try (HostMcpClient host = new HostMcpClient()) {
                result = new JSONObject(host.execute(context, source, "{}", 120000, id, cancellation));
            }
            evidence.put("mcp_result", result); phase = "mcp_returned"; save();
            if (!"completed".equals(result.optString("state")) || result.optInt("exit_code", -1) != 0
                    || !result.has("result") || !result.getJSONObject("result").optBoolean("ok")) {
                throw new IllegalStateException(NanoToolOutput.failed(result));
            }
            NanoTurnStore.publish(context, id, userText, NanoToolOutput.completed(result.getJSONObject("result"))
                    + "\n\nNano is preparing an explanation. The execution result above is already saved.", null, false);
            check(); phase = "result_consumption"; save();
            JSONObject nativeResult = result.getJSONObject("result");
            JSONObject interpretationInput = new JSONObject().put("ok", nativeResult.getBoolean("ok"))
                    .put("type", nativeResult.getString("type")).put("value", nativeResult.get("value"));
            JSONObject consumed = NanoActionClient.generate(context, frame(
                    "Report the actual Ostadix result below in at most two short sentences. "
                    + "The value uses tagged OValue JSON (t=type,v=value). "
                    + "Do not reproduce JSON, source code, hashes, or execution evidence. "
                    + "The host adds execution evidence separately. Report what the tool actually returned. "
                    + "Do not recalculate or substitute an expected answer.\nUSER REQUEST:\n" + userText
                    + "\nACTUAL TOOL RESULT:\n" + interpretationInput.toString()), cancellation);
            evidence.put("result_consumption", consumed);
            String interpretation = candidate(consumed, false);
            check();
            String answer = NanoToolOutput.completed(nativeResult)
                    + "\n\nNano's explanation (not independently verified):\n" + interpretation;
            if (prepared.corrected) { answer += "\n\nNano corrected the generated program before execution."; }
            evidence.put("answer", answer).put("elapsed_ms", SystemClock.elapsedRealtime() - start);
            phase = "answer_ready"; save();
            return answer;
        } catch (Exception failure) {
            String failedAt = phase;
            evidence.put("failure_phase", phase).put("error", failure.toString());
            phase = "failed"; save();
            if ("result_consumption".equals(failedAt)) {
                JSONObject executed = evidence.getJSONObject("mcp_result").getJSONObject("result");
                check();
                String answer = NanoToolOutput.completed(executed)
                        + "\n\nNano could not explain this result: " + failure.getMessage();
                evidence.put("answer", answer).put("elapsed_ms", SystemClock.elapsedRealtime() - start);
                phase = "answer_ready_without_interpretation"; save();
                return answer;
            }
            throw failure;
        }
    }

    private String validateSource(String source) throws Exception {
        check(); phase = "source_validation";
        JSONArray checks = evidence.getJSONArray("source_checks");
        JSONObject entry = new JSONObject().put("source_sha256", digest(source))
                .put("request", new JSONObject().put("method", "tools/call").put("params",
                        new JSONObject().put("name", "o_execute").put("arguments", new JSONObject()
                                .put("source", source).put("action", "check").put("timeout_secs", 15))));
        checks.put(entry);
        evidence.put("mcp_dispatch_attempts", evidence.getInt("mcp_dispatch_attempts") + 1); save();
        JSONObject result;
        try (HostMcpClient host = new HostMcpClient()) {
            result = new JSONObject(host.check(context, source, 15000, id + "-check-" + checks.length(), cancellation));
        }
        entry.put("result", result); save(); check();
        JSONObject nativeResult = result.optJSONObject("result");
        if ("check".equals(result.optString("action")) && nativeResult != null
                && "parse".equals(nativeResult.optString("stage"))) {
            if ("completed".equals(result.optString("state")) && result.optInt("exit_code", -1) == 0
                    && nativeResult.optBoolean("ok")) {
                JSONObject structure = nativeResult.optJSONObject("source_structure");
                if (structure == null || !"ostadix.source-structure/v1".equals(structure.optString("schema"))) {
                    throw new IllegalStateException("Native runtime lacks the source preflight contract. "
                            + "Rebuild/install current Ostadix. Nothing was executed.");
                }
                JSONArray bindings = structure.getJSONArray("required_initial_bindings");
                String[] missing = new String[bindings.length()];
                for (int i = 0; i < missing.length; i++) { missing[i] = bindings.getString(i); }
                String contractError = NanoSourcePreparation.executableContractError(
                        structure.getBoolean("top_level_literal_text"), missing,
                        structure.getJSONArray("languages").length());
                if (contractError == null) { contractError = pythonSyntaxContractError(structure); }
                if (contractError == null) { contractError = pythonResultContractError(structure); }
                if (contractError != null) { entry.put("source_contract_rejection", contractError); save(); }
                return contractError;
            }
            if ("failed".equals(result.optString("state")) && result.optInt("exit_code", 0) != 0
                    && nativeResult.has("ok") && !nativeResult.getBoolean("ok")
                    && nativeResult.has("error")) { return nativeResult.getString("error"); }
        }
        throw new IllegalStateException("Could not validate Nano's program. Nothing was executed. "
                + (nativeResult == null ? result.optString("error", "Unexpected validation response")
                        : nativeResult.optString("error", "Unexpected validation response")));
    }

    private void check() {
        cancellation.throwIfCanceled();
        if (!GeminiNanoActionHooks.enabled(context)) {
            throw new IllegalStateException("local assistant activation revoked; no retry");
        }
    }

    private void save() throws Exception {
        evidence.put("phase", phase);
        try (FileOutputStream out = context.openFileOutput(id + ".json", Context.MODE_PRIVATE)) {
            out.write((evidence.toString() + "\n").getBytes(StandardCharsets.UTF_8)); out.getFD().sync();
        }
        Log.i("OstadixGeminiNano", "event=turn_phase request_id=" + id + " phase=" + phase);
    }

    private static String frame(String text) { return "\n<ctrl99>user\n" + text + "<ctrl100>\n<ctrl99>model\n"; }
    static String pythonSyntaxContractError(JSONObject structure) {
        JSONArray diagnostics = structure.optJSONArray("backend_syntax_checks");
        if (diagnostics == null) { return null; }
        for (int i = 0; i < diagnostics.length(); i++) {
            JSONObject diagnostic = diagnostics.optJSONObject(i);
            if (diagnostic == null || !"python".equals(diagnostic.optString("language"))
                    || !"invalid".equals(diagnostic.optString("state"))) { continue; }
            String location = "Python syntax error in block " + (i + 1);
            if (diagnostic.optInt("line", -1) > 0) { location += ", line " + diagnostic.optInt("line"); }
            if (diagnostic.optInt("column", -1) > 0) { location += ", column " + diagnostic.optInt("column"); }
            return location + ": " + diagnostic.optString("message", "invalid Python syntax")
                    + ". Correct the complete source. Nothing was executed.";
        }
        return null;
    }

    static String pythonResultContractError(JSONObject structure) {
        JSONArray languages = structure.optJSONArray("languages");
        JSONArray diagnostics = structure.optJSONArray("backend_syntax_checks");
        if (structure.optInt("plan_nodes", -1) != 2
                || languages == null || languages.length() != 1 || !"python".equals(languages.optString(0))
                || diagnostics == null || diagnostics.length() != 1) { return null; }
        JSONObject diagnostic = diagnostics.optJSONObject(0);
        if (diagnostic == null || !"python".equals(diagnostic.optString("language"))) { return null; }
        return NanoSourcePreparation.pythonResultContractError(
                diagnostic.optString("state"), diagnostic.optString("result_capture"));
    }

    static String candidate(JSONObject result, boolean beforeExecution) throws Exception {
        String status = beforeExecution ? " Nothing was executed." : " The Ostadix result is already saved.";
        if (!result.optBoolean("ok")) { throw new IllegalStateException("Local Nano failed: " + result.opt("error") + status); }
        JSONArray candidates = result.getJSONObject("result").getJSONArray("candidates");
        if (candidates.length() != 1) {
            throw new IllegalStateException("Nano returned " + candidates.length()
                    + " candidates; one complete response is required." + status);
        }
        JSONObject first = candidates.getJSONObject(0);
        if (first.optBoolean("text_truncated_in_event") || first.optInt("finish_reason_enum") != 1) {
            throw new IllegalStateException("Nano returned incomplete text (finish reason "
                    + first.optInt("finish_reason_enum", -1)
                    + (first.optBoolean("text_truncated_in_event") ? ", truncated" : "")
                    + ")." + status);
        }
        String text = first.getString("text");
        if (text.trim().isEmpty()) { throw new IllegalStateException("Nano returned an empty response." + status); }
        return text;
    }
    private static String extractSource(String text) {
        String s = text.trim();
        if (s.startsWith("```")) {
            int newline = s.indexOf('\n');
            if (newline < 0 || !s.endsWith("```")) { throw new IllegalArgumentException("incomplete source fence"); }
            s = s.substring(newline + 1, s.length() - 3);
        }
        if (s.contains("```")) { throw new IllegalArgumentException("multiple source fences; not dispatched"); }
        return s;
    }
    static String digest(String text) throws Exception {
        StringBuilder out = new StringBuilder();
        for (byte b : MessageDigest.getInstance("SHA-256").digest(text.getBytes(StandardCharsets.UTF_8))) {
            out.append(String.format(Locale.ROOT, "%02x", b & 255));
        }
        return out.toString();
    }
}
