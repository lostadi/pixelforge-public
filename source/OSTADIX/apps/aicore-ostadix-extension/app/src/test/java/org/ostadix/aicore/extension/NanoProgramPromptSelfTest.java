package org.ostadix.aicore.extension;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;
import org.json.JSONArray;
import org.json.JSONObject;

/** JVM regression checks for complete prompts and private, bounded chat context. */
public final class NanoProgramPromptSelfTest {
    private static final String CHAT = repeat("a", 64);
    private static final String OTHER_CHAT = repeat("b", 64);

    public static void main(String[] args) throws Exception {
        String request = "REQUEST BEGIN\n" + repeat("\ud83d\ude80", 200) + "\nREQUEST END";
        JSONArray history = new JSONArray().put(new JSONObject().put("user_request", "prior task"));
        String prompt = NanoProgramPrompt.generation(request, history);
        require(prompt.contains(request), "current Unicode request changed");
        requirePythonResultContract(prompt);
        require(prompt.contains("Default to ONE Python block")
                && prompt.contains("python^(\n# Import required modules here.")
                && prompt.contains("__oval_result__ = result\n)_python")
                && prompt.contains("Put EVERY Python import, assignment and function inside")
                && !prompt.contains("ADDITIONAL LANGUAGE SYNTAX"),
                "ordinary task lost its self-contained Python layout or gained irrelevant backend examples");
        require(prompt.getBytes(StandardCharsets.UTF_8).length <= 16384, "prompt exceeds host byte limit");
        JSONArray tooLarge = new JSONArray().put(new JSONObject()
                .put("user_request", "UNIQUE_OLD_CONTEXT" + repeat("\ud83d\ude80", 5000)));
        String omitted = NanoProgramPrompt.generation(request, tooLarge);
        require(omitted.contains(request) && !omitted.contains("UNIQUE_OLD_CONTEXT")
                && omitted.contains("context omitted"), "oversize context silently changed current request");
        require(omitted.getBytes(StandardCharsets.UTF_8).length <= 16384, "Unicode context limit used characters");
        String source = "python^(\n__oval_result__ = '" + repeat("x", 10000) + "'\n)_python";
        String error = "An explicit syntax failure \ud83d\ude80";
        String corrected = NanoProgramPrompt.correction(request, tooLarge, source, error);
        requirePythonResultContract(corrected);
        require(corrected.contains("Keep the candidate's language(s)")
                && corrected.contains("Repair Python wrapper/import/result errors inside the same Python block")
                && corrected.contains("Return O source, never Markdown fences"),
                "correction may translate a wrapper failure into another language or Markdown");
        require(corrected.contains(request) && corrected.contains(source) && corrected.contains(error),
                "correction silently truncated request, source or error");
        rejectPrompt(() -> NanoProgramPrompt.generation(repeat("\ud83d\ude80", 5000), history));
        rejectPrompt(() -> NanoProgramPrompt.correction(request, history, repeat("x", 20000), error));
        String polyglot = NanoProgramPrompt.generation("Combine Rust and Bash work in parallel", new JSONArray());
        require(polyglot.contains("ADDITIONAL LANGUAGE SYNTAX") && polyglot.contains("rust^(")
                && polyglot.contains("bash^(") && polyglot.contains("autonomous(batch("),
                "requested polyglot capabilities lost their language guide");
        JSONArray earlierPolyglot = new JSONArray().put(new JSONObject()
                .put("user_request", "Use Rust for this task"));
        require(NanoProgramPrompt.generation("Continue the task", earlierPolyglot)
                .contains("ADDITIONAL LANGUAGE SYNTAX"), "follow-up lost an explicitly requested language");
        JSONArray mistakenCode = new JSONArray().put(new JSONObject().put("user_request", "Inspect the current process")
                .put("source", "rust^(fn main(){})_rust"));
        String simpleFollowUp = NanoProgramPrompt.generation("Explain the result", mistakenCode);
        require(!simpleFollowUp.contains("ADDITIONAL LANGUAGE SYNTAX")
                && simpleFollowUp.contains("rust^(fn main(){})_rust"),
                "failed generated language selected the guide or history was altered");
        String identity = NanoProgramPrompt.generation("Return the process user ID", new JSONArray());
        require(identity.contains("os.getuid() is the process user ID (UID)")
                && identity.contains("os.getpid() is the process ID (PID)"),
                "device identifier guidance confuses UID with PID");

        Path directory = Files.createTempDirectory("ostadix-private-context-");
        try {
            write(directory, "oldest", record("oldest", CHAT, "oldest same chat"), 1000);
            write(directory, "older", record("older", CHAT, "older same chat"), 2000);
            write(directory, "recent", record("recent", CHAT, "recent same chat"), 3000);
            write(directory, "newest", record("newest", CHAT, "newest same chat"), 4000);
            write(directory, "other", record("other", OTHER_CHAT, "PRIVATE_OTHER_CHAT"), 9000);
            write(directory, "current", record("current", CHAT, "CURRENT_NOT_HISTORY"), 10000);
            write(directory, "legacy", record("legacy", null, "PRIVATE_LEGACY_RECORD"), 11000);
            Path malformed = directory.resolve("ostadix-turn-malformed.json");
            Files.write(malformed, "{\"unfinished\":".getBytes(StandardCharsets.UTF_8));
            require(malformed.toFile().setLastModified(12000), "could not set test record timestamp");
            JSONArray selected = NanoConversationContext.load(directory.toFile(), CHAT, "ostadix-turn-current");
            require(selected.length() == 3, "same-chat context count incorrect");
            require(selected.getJSONObject(0).getString("request_id").equals("ostadix-turn-newest")
                    && selected.getJSONObject(1).getString("request_id").equals("ostadix-turn-recent")
                    && selected.getJSONObject(2).getString("request_id").equals("ostadix-turn-older"),
                    "newest eligible turns not selected in order");
            require(!selected.toString().contains("PRIVATE_")
                    && !selected.toString().contains("CURRENT_NOT_HISTORY"), "cross-chat/current context leak");
            require(NanoConversationContext.load(directory.toFile(), null, "unused").length() == 0,
                    "null chat identity admitted history");
            require(NanoConversationContext.load(directory.toFile(), "unavailable-test", "unused").length() == 0,
                    "unavailable identity admitted history");
            require(NanoConversationContext.load(directory.toFile(), "", "unused").length() == 0,
                    "empty identity matched legacy records");
        } finally { remove(directory); }

        Path bounded = Files.createTempDirectory("ostadix-context-byte-boundary-");
        try {
            // Size each projected record to 1,999 bytes. Their serialized array
            // is 6,001 bytes, exposing counts that omit JSON array punctuation.
            for (int i = 0; i < 3; i++) {
                String suffix = "boundary-" + i;
                JSONObject projected = new JSONObject().put("request_id", "ostadix-turn-" + suffix)
                        .put("user_request", "").put("phase", "answer_ready").put("execution_dispatch_attempts", 1);
                int padding = 1999 - projected.toString().getBytes(StandardCharsets.UTF_8).length;
                write(bounded, suffix, record(suffix, CHAT, repeat("x", padding)), 1000 + i);
            }
            JSONArray selected = NanoConversationContext.load(bounded.toFile(), CHAT, "unused");
            require(selected.toString().getBytes(StandardCharsets.UTF_8).length <= 6000,
                    "context byte limit omitted JSON array punctuation");
            require(selected.getJSONObject(0).getString("request_id").equals("ostadix-turn-boundary-2"),
                    "bounded selection lost newest context");
        } finally { remove(bounded); }
        System.out.println("Nano prompt/context: complete Unicode input, explicit oversize rejection, "
                + "private same-chat/latest selection, malformed record and byte budgets passed");
    }

    private interface PromptOperation { String run(); }

    private static void requirePythonResultContract(String prompt) {
        require(prompt.contains("Python MUST assign its actual answer to __oval_result__")
                && prompt.contains("__oval_result__ = result before )_python")
                && prompt.contains("A variable named result or answer is not returned automatically"),
                "generated/corrected Python may compute into an unreturned local variable");
    }

    private static void rejectPrompt(PromptOperation operation) {
        try { operation.run(); }
        catch (IllegalArgumentException expected) {
            require(expected.getMessage().contains("Nothing was executed")
                    && expected.getMessage().contains("not silently truncated"), "unclear oversized request error");
            return;
        }
        throw new AssertionError("oversized required prompt admitted");
    }

    private static JSONObject record(String suffix, String conversation, String request) throws Exception {
        JSONObject record = new JSONObject().put("request_id", "ostadix-turn-" + suffix)
                .put("user_text", request).put("phase", "answer_ready").put("execution_dispatch_attempts", 1);
        if (conversation != null) { record.put("conversation_key", conversation); }
        return record;
    }

    private static void write(Path directory, String suffix, JSONObject record, long modified) throws Exception {
        File file = directory.resolve("ostadix-turn-" + suffix + ".json").toFile();
        Files.write(file.toPath(), record.toString().getBytes(StandardCharsets.UTF_8));
        require(file.setLastModified(modified), "could not set test record timestamp");
    }

    private static void remove(Path directory) throws Exception {
        try (Stream<Path> paths = Files.walk(directory)) {
            for (Path path : (Iterable<Path>) paths.sorted(Comparator.reverseOrder())::iterator) {
                Files.deleteIfExists(path);
            }
        }
    }

    private static String repeat(String text, int count) {
        StringBuilder value = new StringBuilder();
        for (int i = 0; i < count; i++) { value.append(text); }
        return value.toString();
    }

    private static void require(boolean condition, String message) {
        if (!condition) { throw new AssertionError(message); }
    }
}
