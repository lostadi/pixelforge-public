package org.ostadix.aicore.extension;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import org.json.JSONObject;

public final class ResultHistorySelfTest {
    public static void main(String[] args) throws Exception {
        String id = "ostadix-turn-2a14e81b-d89f-414f-a5aa-a25c71f3c1ff";
        ResultHistory.checkedId(id);
        for (String invalid : new String[]{"../escape", id + "/extra", "", null}) {
            try { ResultHistory.checkedId(invalid); throw new AssertionError("unsafe path accepted"); }
            catch (IllegalArgumentException expected) { }
        }
        JSONObject record = new JSONObject().put("request_id", id).put("display_text", "Actual output 276\nSecond line");
        File file = File.createTempFile("ostadix-history-", ".json");
        try {
            Files.write(file.toPath(), record.toString().getBytes(StandardCharsets.UTF_8));
            require(ResultHistory.display(ResultHistory.read(file)).equals("Actual output 276\nSecond line"), "saved answer changed");
            record.remove("display_text");
            record.put("mcp_result", new JSONObject().put("result", new JSONObject()
                    .put("ok", true).put("value", new JSONObject().put("t", "string").put("v", "actual output"))
                    .put("execution_evidence", new JSONObject().put("result_content_identity", "id"))));
            require(ResultHistory.display(record).contains("actual output"), "native result lost without explanation");
            record.remove("mcp_result"); record.put("error", "parser rejected generated source");
            require(ResultHistory.display(record).contains("parser rejected"), "error lost");
            Files.write(file.toPath(), new byte[ResultHistory.MAX_BYTES + 1]);
            try { ResultHistory.read(file); throw new AssertionError("unbounded record accepted"); }
            catch (java.io.IOException expected) { }
        } finally { Files.deleteIfExists(file.toPath()); }
        System.out.println("History: saved answer, native result without explanation, error, path and record bounds passed");
    }
    private static void require(boolean condition, String message) { if (!condition) { throw new AssertionError(message); } }
}
