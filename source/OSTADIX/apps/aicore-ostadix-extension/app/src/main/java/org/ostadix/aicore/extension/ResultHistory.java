package org.ostadix.aicore.extension;

import android.content.Context;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;

/** Private durable records; neither Android backup nor an external read API is enabled. */
final class ResultHistory {
    static final String AUTHORITY = "org.ostadix.aicore.extension.results";
    static final int MAX_BYTES = 16 * 1024 * 1024;

    static String checkedId(String id) {
        if (id == null || !id.matches("ostadix-turn-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")) {
            throw new IllegalArgumentException("Invalid Ostadix request identity");
        }
        return id;
    }

    static File directory(Context context) throws IOException {
        File directory = new File(context.getFilesDir(), "history");
        if (!directory.isDirectory() && !directory.mkdirs()) { throw new IOException("Cannot create result history"); }
        return directory;
    }

    static JSONObject read(File file) throws Exception {
        try (FileInputStream in = new FileInputStream(file);
                ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[16384]; int n;
            while ((n = in.read(buffer)) != -1) {
                if (out.size() + n > MAX_BYTES) { throw new IOException("Result record exceeds 16 MiB"); }
                out.write(buffer, 0, n);
            }
            return new JSONObject(new String(out.toByteArray(), StandardCharsets.UTF_8));
        }
    }

    static String display(JSONObject record) throws Exception {
        String answer = record.optString("display_text", record.optString("answer", ""));
        if (!answer.isEmpty()) { return answer; }
        JSONObject mcp = record.optJSONObject("mcp_result");
        JSONObject result = mcp == null ? null : mcp.optJSONObject("result");
        if (result != null && result.optBoolean("ok")) {
            return NanoToolOutput.completed(result) + "\n\nNano's explanation is not available.";
        }
        String error = record.optString("error", "");
        if (!error.isEmpty()) { return "Local request failed:\n" + error + "\n\nExecution was not retried."; }
        return "Last recorded stage: " + record.optString("phase", "unknown")
                + "\nNo completed result is recorded yet. This screen does not restart execution.";
    }
}
