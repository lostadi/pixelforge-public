package org.ostadix.aicore.extension;

import android.content.Context;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import java.io.File;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;

final class NanoTurnStore {
    static String publish(Context context, String id, String text, String answer, Throwable failure, boolean terminal) {
        try {
            File file = new File(context.getFilesDir(), ResultHistory.checkedId(id) + ".json");
            JSONObject record = file.isFile() ? ResultHistory.read(file) : new JSONObject()
                    .put("request_id", id).put("user_text", text).put("phase", "not_dispatched");
            if (failure != null) { record.put("delivery_error", GeminiNanoActionHooks.displayFailure(failure)); }
            // A completed native result remains visible even when explanation or UI delivery is cancelled.
            if (answer == null && terminal) {
                JSONObject mcp = record.optJSONObject("mcp_result");
                JSONObject executed = mcp == null ? null : mcp.optJSONObject("result");
                if (!record.optString("answer").isEmpty()) {
                    answer = record.getString("answer");
                } else if (executed != null && executed.optBoolean("ok")) {
                    answer = NanoToolOutput.completed(executed)
                            + "\n\nThe assistant turn stopped after execution. Execution was not retried.";
                } else {
                    answer = "Local Ostadix request stopped: " + (failure == null ? "No final answer" :
                            GeminiNanoActionHooks.displayFailure(failure)) + "\n\nExecution was not retried.";
                }
            }
            if (answer != null) { record.put("display_text", answer); }
            record.put("history_updated_at", System.currentTimeMillis()).put("history_terminal", terminal);
            byte[] bytes = (record.toString() + "\n").getBytes(StandardCharsets.UTF_8);
            if (bytes.length > ResultHistory.MAX_BYTES) { throw new IllegalStateException("History record exceeds 16 MiB"); }
            Uri uri = Uri.parse("content://" + ResultHistory.AUTHORITY + "/" + id);
            try (OutputStream output = context.getContentResolver().openOutputStream(uri, "w")) {
                if (output == null) { throw new IllegalStateException("History provider unavailable"); }
                output.write(bytes);
            }
            Bundle result = context.getContentResolver().call(uri, "commit", id, null);
            if (result == null || !result.getBoolean("saved")) { throw new IllegalStateException("History save was not confirmed"); }
            Log.i("OstadixGeminiNano", "event=history_saved request_id=" + id + " terminal=" + terminal);
            return answer;
        } catch (Throwable error) {
            ExtensionGate.rethrowIfVmFatal(error);
            Log.e("OstadixGeminiNano", "event=history_save_failed request_id=" + id, error);
            return answer == null ? null : answer + "\n\nCould not save this answer in Ostadix Results: "
                    + GeminiNanoActionHooks.displayFailure(error);
        }
    }
}
