package org.ostadix.aicore.extension;

import org.json.JSONObject;

/** JVM checks of host generation policy; no model or native calls are made. */
public final class NanoGenerationBudgetSelfTest {
    public static void main(String[] args) throws Exception {
        StringBuilder unicode = new StringBuilder();
        for (int i = 0; i < 4096; i++) { unicode.append("\ud83d\ude80"); }
        String exactPrompt = unicode.toString();
        NanoLocalProbe.Generation accepted = new NanoLocalProbe.Generation(request(exactPrompt,
                NanoLocalProbe.MAX_OUTPUT_TOKENS, NanoLocalProbe.GENERATION_TIMEOUT_MS));
        require(accepted.prompt.equals(exactPrompt), "prompt changed at the UTF-8 boundary");
        require(accepted.maxOutputTokens == 2048, "complex program token budget not available");
        reject(request(exactPrompt + "x", 32, 10000), "UTF-8 prompt overflow accepted");
        reject(request("request", 0, 10000), "zero token budget accepted");
        reject(request("request", NanoLocalProbe.MAX_OUTPUT_TOKENS + 1, 10000),
                "token overflow accepted");
        reject(request("request", 32.5, 10000), "fractional token budget accepted");
        reject(request("request", 32, 0), "zero timeout accepted");
        reject(request("request", 32, NanoLocalProbe.GENERATION_TIMEOUT_MS + 1),
                "timeout overflow accepted");
        reject(request("request", 32, 10000.5), "fractional timeout accepted");
        require(NanoLocalProbe.GENERATION_TIMEOUT_MS < NanoLocalProbe.ACTION_TIMEOUT_MS
                && NanoLocalProbe.ACTION_TIMEOUT_MS < NanoLocalProbe.CLIENT_REPLY_TIMEOUT_MS,
                "setup/cleanup and reply margins lost");
        require(!NanoLocalProbe.shouldCancelGeneration(false,
                NanoLocalProbe.MAX_GENERATION_CALLBACKS, 99, 100), "valid last callback cancelled");
        require(NanoLocalProbe.shouldCancelGeneration(false,
                NanoLocalProbe.MAX_GENERATION_CALLBACKS + 1, 99, 100), "callback limit ignored");
        require(NanoLocalProbe.shouldCancelGeneration(false, 1, 100, 100), "deadline boundary ignored");
        require(NanoLocalProbe.shouldCancelGeneration(true, 1, 1, 100), "user cancellation ignored");
        System.out.println("Nano generation budgets: UTF-8/integral bounds, callback limit, deadline, "
                + "cancellation and reply margins passed; model capacity untested");
    }

    private static JSONObject request(String prompt, Number tokens, Number timeout) throws Exception {
        return new JSONObject().put("prompt", prompt).put("maxOutputTokens", tokens)
                .put("timeoutMs", timeout).put("matformerSignature", "matformer_0");
    }

    private static void reject(JSONObject request, String message) throws Exception {
        try { new NanoLocalProbe.Generation(request); }
        catch (IllegalArgumentException expected) { return; }
        throw new AssertionError(message);
    }

    private static void require(boolean condition, String message) {
        if (!condition) { throw new AssertionError(message); }
    }
}
