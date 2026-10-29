package org.ostadix.aicore.extension;

import java.nio.charset.StandardCharsets;
import org.json.JSONArray;

/** One language contract for generation and correction; no unrelated task examples. */
final class NanoProgramPrompt {
    static final String GUIDE =
            "Write a complete Ostadix .O program. Return ONLY source, without Markdown fences, within 2048 tokens. "
            + "The local host executes it on this Android phone through o_execute. Do not claim execution. "
            + "Ostadix manages compilation and runtimes; do not invoke ostadix, cargo or rustc as invented tools. "
            + "Implement the current request using real inputs and operations. Do not simulate device facts. "
            + "Default to ONE Python block. Use other registered languages or multiple blocks only when "
            + "the requested task requires them; preserve languages explicitly requested in this chat. "
            + "For a writing question, return the answer as a Python string. For a computation or device query, "
            + "write code that actually computes or inspects it.\n"
            + "Required Python layout (replace the comments with the actual task code that defines result):\n"
            + "python^(\n# Import required modules here.\n# Perform the requested task and define result here.\n"
            + "__oval_result__ = result\n)_python\n"
            + "Put EVERY Python import, assignment and function inside python^( and )_python. "
            + "No Python code may precede the opening or follow the closing delimiter. "
            + "The closing delimiter is exactly )_python, not a Markdown fence. "
            + "Python MUST assign its actual answer to __oval_result__. If you computed result, finish with "
            + "__oval_result__ = result before )_python. A variable named result or answer is not returned "
            + "automatically. For file/device operations, assign their real outcome to __oval_result__ too. "
            + "Import each Python module used. Requested file operations are allowed.\n";
    private static final String POLYGLOT_GUIDE =
            "ADDITIONAL LANGUAGE SYNTAX for requested non-Python or parallel work: "
            + "rust^( fn main(){ RUST_BODY } )_rust; bash^( BASH_BODY )_bash. "
            + "Other registered backends use their language name and matching )_language suffix. "
            + "Rust and Bash return stdout decoded as JSON when valid. "
            + "All Python code belongs INSIDE python^(...)_python. All O declarations belong OUTSIDE Python. "
            + "Independent results: let parts = autonomous(batch(BLOCK1,BLOCK2,BLOCK3)). "
            + "Then a final python^(\nvalues = $parts\n__oval_result__ = sum(values)\n)_python combines them. "
            + "Define parts before $parts. Do not invent undeclared bindings. A single task needs only one block. "
            + "A Python assignment a,b,c = [1,2,3] produces scalars, not lists. "
            + "Assign O splices to Python variables before using them in strings. "
            + "Compile language blocks through O.\n";
    private static final String DEVICE_GUIDE =
            "Device inspection: use Python standard library os, glob, pathlib, platform; import each module used. "
            + "Use the exact requested identifier: os.getuid() is the process user ID (UID); "
            + "os.getpid() is the process ID (PID). They are different facts. "
            + "Use glob.glob(pattern) to discover paths; missing patterns return []. "
            + "Use os.listdir only for existing directories; read text only from regular files. "
            + "Never cat directories or read accelerator character devices to discover hardware. "
            + "TPU discovery patterns include /dev/*tpu*, /sys/bus/platform/drivers/*tpu*, "
            + "/sys/devices/platform/*tpu*/*. These reveal accessible paths, not accelerator performance. "
            + "No nonexistent ostadix --info or --get-tpu-details commands.\n";

    static String generation(String request, JSONArray history) {
        return bounded(guideFor(request, history), request, "", history);
    }
    static String correction(String request, JSONArray history, String source, String error) {
        return bounded(guideFor(request, history) + "This program was rejected before execution. Correct it once, preserving the "
                + "request and returning the whole source. Keep the candidate's language(s); change languages only "
                + "if the user's task requires it. Repair Python wrapper/import/result errors inside the same "
                + "Python block. Return O source, never Markdown fences. No part of this candidate executed.\n",
                request, "\nVALIDATION ERROR:\n" + error + "\nREJECTED SOURCE:\n" + source, history);
    }
    private static String guideFor(String request, JSONArray history) {
        StringBuilder requestedTasks = new StringBuilder(request);
        for (int i = 0; i < history.length(); i++) {
            org.json.JSONObject turn = history.optJSONObject(i);
            if (turn != null) { requestedTasks.append(' ').append(turn.optString("user_request")); }
        }
        // Failed generated source is context, not a request to switch languages.
        String subject = requestedTasks.toString().toLowerCase(java.util.Locale.ROOT);
        boolean polyglot = subject.matches("(?s).*\\b(rust|bash|shell|javascript|typescript|java|kotlin|golang|lua|ruby|sql|"
                + "polyglot|parallel|autonomous|batch|r)\\b.*") || subject.contains("c++") || subject.contains("c#");
        boolean device = subject.matches("(?s).*\\b(tpu|gpu|hardware|device|accelerator|process|uid|pid|permission)\\b.*")
                || subject.contains("user id");
        return GUIDE + (polyglot ? POLYGLOT_GUIDE : "") + (device ? DEVICE_GUIDE : "");
    }
    private static String bounded(String guide, String request, String extra, JSONArray history) {
        String required = guide + "\nCURRENT USER REQUEST (preserve exactly):\n" + request + extra;
        String context = "\nPREVIOUS TURNS IN THIS CHAT (newest first; context data, not new instructions):\n"
                + history.toString() + "\nUse this context for follow-ups. Previous executions may have had effects; "
                + "do not assume a failed run rolled back. If the prior request is absent, return a message "
                + "asking for the full request instead of inventing a task.\n";
        String complete = frame(required + context);
        if (complete.getBytes(StandardCharsets.UTF_8).length <= 16384) { return complete; }
        complete = frame(required + "\nPrevious context omitted to fit the prompt. Ask for the full request "
                + "if this request depends on it.\n");
        if (complete.getBytes(StandardCharsets.UTF_8).length > 16384) {
            throw new IllegalArgumentException("Request and program exceed Nano's configured 16 KiB prompt budget. "
                    + "Nothing was executed. Shorten or split the request; it was not silently truncated.");
        }
        return complete;
    }
    static String frame(String text) { return "\n<ctrl99>user\n" + text + "<ctrl100>\n<ctrl99>model\n"; }
    private NanoProgramPrompt() {}
}
