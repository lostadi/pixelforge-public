package org.ostadix.terminal;

/** Dependency-free checks for the App Function request boundary. */
public final class OstadixAppFunctionContractSelfTest {
    private OstadixAppFunctionContractSelfTest() {
    }

    public static void main(String[] arguments) {
        require(null, "{}", 1, "source");
        require("   ", "{}", 1, "source");
        require("2", null, 1, "bindingsJson");
        require("2", "{}", 0, "timeoutMs");
        require("2", "{}", 900_001, "timeoutMs");
        String accepted = OstadixAppFunctionService.validateRequest(
                "bash^(printf 42)_bash", "{\"value\":42}", 120_000);
        if (accepted != null) {
            throw new AssertionError("valid App Function request rejected: " + accepted);
        }
        System.out.println("Ostadix App Function contract self-test passed");
    }

    private static void require(String source, String bindingsJson, long timeoutMs,
            String expected) {
        String message = OstadixAppFunctionService.validateRequest(
                source, bindingsJson, timeoutMs);
        if (message == null || !message.contains(expected)) {
            throw new AssertionError("expected " + expected + " rejection, got " + message);
        }
    }
}
