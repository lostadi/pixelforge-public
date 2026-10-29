package org.ostadix.aicore.extension;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Comparator;
import org.json.JSONArray;
import org.json.JSONObject;

/** Private, bounded context from the same pinned GSA chat identity only. */
final class NanoConversationContext {
    static JSONArray load(File directory, String conversation, String currentId) {
        JSONArray turns = new JSONArray();
        if (conversation == null || !conversation.matches("[a-f0-9]{64}")) { return turns; }
        File[] files = directory.listFiles((dir, name) -> name.startsWith("ostadix-turn-")
                && name.endsWith(".json") && !name.equals(currentId + ".json"));
        if (files == null) { return turns; }
        Arrays.sort(files, Comparator.comparingLong(File::lastModified).reversed());
        int bytes = 2; // Array delimiters; accepted entries also account for commas.
        for (File file : files) {
            if (turns.length() == 3) { break; }
            if (file.length() > 512 * 1024) { continue; }
            try {
                JSONObject record = new JSONObject(new String(Files.readAllBytes(file.toPath()), StandardCharsets.UTF_8));
                if (!conversation.equals(record.optString("conversation_key"))) { continue; }
                JSONObject turn = new JSONObject().put("request_id", record.getString("request_id"))
                        .put("user_request", record.getString("user_text"))
                        .put("phase", record.optString("phase"))
                        .put("execution_dispatch_attempts", record.optInt("execution_dispatch_attempts"));
                if (record.has("error")) { turn.put("error", record.getString("error")); }
                JSONObject request = record.optJSONObject("mcp_request");
                if (request != null) {
                    turn.put("source", request.getJSONObject("params").getJSONObject("arguments").getString("source"));
                }
                JSONObject mcp = record.optJSONObject("mcp_result");
                if (mcp != null && mcp.optJSONObject("result") != null) {
                    JSONObject result = mcp.getJSONObject("result");
                    if (result.has("value")) { turn.put("actual_value", result.get("value")); }
                }
                int size = turn.toString().getBytes(StandardCharsets.UTF_8).length;
                // Preserve a long task's request even when its generated program exceeds the context budget.
                if (bytes + size + 1 > 6000) {
                    turn.remove("source"); turn.remove("actual_value");
                    turn.put("details_omitted", true);
                    size = turn.toString().getBytes(StandardCharsets.UTF_8).length;
                }
                if (bytes + size + 1 > 6000) {
                    if (turns.length() == 0) {
                        turns.put(new JSONObject().put("request_id", record.getString("request_id"))
                                .put("context_unavailable", "Latest request exceeds context budget. "
                                        + "Ask the user to restate it; do not substitute older requests."));
                    }
                    break;
                }
                turns.put(turn); bytes += size + 1;
            } catch (Exception incompleteRecord) { /* Atomicity is not assumed while another record is saved. */ }
        }
        return turns;
    }
    private NanoConversationContext() {}
}
