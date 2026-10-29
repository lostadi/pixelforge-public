package org.ostadix.aicore.extension;

import org.json.JSONArray;
import org.json.JSONObject;

/** A candidate failure must describe whether execution has already occurred. */
public final class NanoCandidateSelfTest {
    public static void main(String[] args) throws Exception {
        JSONObject response = response("python^( __oval_result__ = 2 )_python", 1, false);
        require(NanoToolTurn.candidate(response, true).equals("python^( __oval_result__ = 2 )_python"),
                "complete candidate was changed");
        reject(response("partial", 2, false), true, "finish reason 2");
        reject(response("partial", 1, true), true, "truncated");
        reject(response("   ", 1, false), true, "empty");
        reject(response("partial explanation", 2, false), false, "finish reason 2");
        JSONObject zero = new JSONObject().put("ok", true)
                .put("result", new JSONObject().put("candidates", new JSONArray()));
        reject(zero, true, "0 candidates");
        JSONObject generationFailure = new JSONObject().put("ok", false).put("error", "generation timed out");
        reject(generationFailure, true, "generation timed out");
        reject(generationFailure, false, "generation timed out");
        JSONObject diagnostic = new JSONObject().put("language", "python").put("state", "valid")
                .put("result_capture", "none");
        JSONObject structure = new JSONObject().put("plan_nodes", 2).put("languages", new JSONArray().put("python"))
                .put("backend_syntax_checks", new JSONArray().put(diagnostic));
        require(NanoToolTurn.pythonResultContractError(structure) != null,
                "sole static Python block without a result was admitted");
        structure.put("plan_nodes", 4);
        require(NanoToolTurn.pythonResultContractError(structure) == null,
                "Python setup block with a separate O result rejected");
        structure.put("plan_nodes", 2);
        diagnostic.put("result_capture", "explicit_result");
        require(NanoToolTurn.pythonResultContractError(structure) == null, "explicit result rejected");
        diagnostic.remove("result_capture");
        require(NanoToolTurn.pythonResultContractError(structure) == null, "old metadata treated as no result");
        diagnostic.put("result_capture", "none").put("state", "skipped");
        require(NanoToolTurn.pythonResultContractError(structure) == null, "dynamic/skipped Python block rejected");
        diagnostic.put("state", "valid");
        structure.getJSONArray("languages").put("bash");
        require(NanoToolTurn.pythonResultContractError(structure) == null, "mixed runtime output rejected");
        structure.put("languages", new JSONArray().put("python"));
        structure.getJSONArray("backend_syntax_checks").put(diagnostic);
        require(NanoToolTurn.pythonResultContractError(structure) == null, "multiple Python blocks rejected");
        checkPythonSyntaxCorrection();
        System.out.println("Nano candidate diagnostics: finished, incomplete, truncated, empty, missing and failed "
                + "responses preserve execution status; sole static Python result contract and bounded "
                + "pre-execution Python syntax correction verified");
    }

    private static void checkPythonSyntaxCorrection() throws Exception {
        JSONObject diagnostic = new JSONObject().put("language", "python").put("state", "invalid")
                .put("message", "invalid syntax").put("line", 3).put("column", 8);
        final JSONObject invalid = new JSONObject().put("languages", new JSONArray().put("python").put("bash"))
                .put("backend_syntax_checks", new JSONArray().put(new JSONObject()
                        .put("language", "python").put("state", "valid")).put(diagnostic));
        String error = NanoToolTurn.pythonSyntaxContractError(invalid);
        require(error != null && error.contains("block 2, line 3, column 8: invalid syntax")
                && error.contains("Nothing was executed"), "confirmed Python syntax failure lost its location");
        require(NanoToolTurn.pythonSyntaxContractError(new JSONObject()) == null,
                "absent diagnostic treated as invalid syntax");
        for (String state : new String[]{"valid", "skipped", "unavailable", "unknown", ""}) {
            diagnostic.put("state", state);
            require(NanoToolTurn.pythonSyntaxContractError(invalid) == null,
                    "unconfirmed Python syntax failure rejected: " + state);
        }
        diagnostic.put("state", "invalid").put("language", "bash");
        require(NanoToolTurn.pythonSyntaxContractError(invalid) == null, "non-Python diagnostic reinterpreted");
        diagnostic.put("language", "python");
        final int[] attempts = new int[2];
        NanoSourcePreparation.Prepared corrected = NanoSourcePreparation.prepare("invalid", new NanoSourcePreparation.Operations() {
            public void checkActive() { }
            public String validate(String source) {
                attempts[0]++;
                return "invalid".equals(source) ? NanoToolTurn.pythonSyntaxContractError(invalid) : null;
            }
            public String correct(String source, String message) {
                attempts[1]++;
                require(message.contains("invalid syntax"), "syntax error did not reach correction prompt");
                return "corrected";
            }
        });
        require(corrected.corrected && "corrected".equals(corrected.source)
                && attempts[0] == 2 && attempts[1] == 1,
                "Python syntax rejection did not use exactly one validated correction");
    }

    private static JSONObject response(String text, int finishReason, boolean truncated) throws Exception {
        return new JSONObject().put("ok", true).put("result", new JSONObject().put("candidates",
                new JSONArray().put(new JSONObject().put("text", text).put("finish_reason_enum", finishReason)
                        .put("text_truncated_in_event", truncated))));
    }

    private static void reject(JSONObject response, boolean beforeExecution, String expected) throws Exception {
        try { NanoToolTurn.candidate(response, beforeExecution); }
        catch (IllegalStateException failure) {
            String message = failure.getMessage();
            require(message.contains(expected), "failure lost native completion reason: " + message);
            require(message.contains("Nothing was executed") == beforeExecution,
                    "failure misreported execution status: " + message);
            require(message.contains("already saved") != beforeExecution,
                    "explanation failure lost completed result: " + message);
            return;
        }
        throw new AssertionError("incomplete response accepted");
    }

    private static void require(boolean condition, String message) {
        if (!condition) { throw new AssertionError(message); }
    }
}
