package org.ostadix.aicore.extension;

/** Confirms that correction cannot become an unbounded or operational retry. */
public final class NanoSourcePreparationSelfTest {
    private static final class Probe implements NanoSourcePreparation.Operations {
        int validations, corrections, activeChecks;
        int failActiveAt;
        boolean transportFailure, repairStillInvalid;
        public void checkActive() {
            if (++activeChecks == failActiveAt) { throw new IllegalStateException("cancelled"); }
        }
        public String validate(String source) throws Exception {
            validations++;
            if (transportFailure) { throw new java.io.IOException("completion unknown"); }
            return "valid".equals(source) ? null : "missing )_python";
        }
        public String correct(String source, String error) {
            corrections++;
            if (!"invalid".equals(source) || !"missing )_python".equals(error)) {
                throw new AssertionError("correction input mismatch");
            }
            return repairStillInvalid ? "invalid" : "valid";
        }
    }

    public static void main(String[] args) throws Exception {
        require(NanoSourcePreparation.executableContractError(true, new String[0], 0) != null,
                "bare Python admitted as executable O");
        require(NanoSourcePreparation.executableContractError(false, new String[0], 1) == null,
                "O executable block rejected");
        require(NanoSourcePreparation.executableContractError(false, new String[]{"parts"}, 1) != null,
                "missing binding admitted");
        require(NanoSourcePreparation.executableContractError(true, new String[0], 1) != null,
                "literal script containing a block admitted");
        require(NanoSourcePreparation.executableContractError(false, new String[0], 0) != null,
                "nonexecuting document admitted");
        require(NanoSourcePreparation.pythonResultContractError("valid", "none") != null,
                "confirmed missing Python answer was admitted");
        for (String capture : new String[]{"explicit_result", "trailing_expression", "stdout", "unknown", ""}) {
            require(NanoSourcePreparation.pythonResultContractError("valid", capture) == null,
                    "valid/unknown Python return contract was rejected: " + capture);
        }
        for (String state : new String[]{"skipped", "unavailable", "invalid", ""}) {
            require(NanoSourcePreparation.pythonResultContractError(state, "none") == null,
                    "unconfirmed Python return contract was rejected: " + state);
        }
        Probe valid = new Probe();
        NanoSourcePreparation.Prepared first = NanoSourcePreparation.prepare("valid", valid);
        require(first.source.equals("valid") && !first.corrected && valid.validations == 1
                && valid.corrections == 0, "valid source changed");

        Probe repaired = new Probe();
        NanoSourcePreparation.Prepared second = NanoSourcePreparation.prepare("invalid", repaired);
        require(second.source.equals("valid") && second.corrected && repaired.validations == 2
                && repaired.corrections == 1, "repair was not validated exactly once");

        Probe invalid = new Probe(); invalid.repairStillInvalid = true;
        try { NanoSourcePreparation.prepare("invalid", invalid); throw new AssertionError("invalid admitted"); }
        catch (IllegalArgumentException expected) {
            require(expected.getMessage().contains("Nothing was executed"), "unclear failure");
        }
        require(invalid.validations == 2 && invalid.corrections == 1, "unbounded correction");

        Probe transport = new Probe(); transport.transportFailure = true;
        try { NanoSourcePreparation.prepare("invalid", transport); throw new AssertionError("transport swallowed"); }
        catch (java.io.IOException expected) { }
        require(transport.validations == 1 && transport.corrections == 0, "operational failure retried");

        for (int stopAt = 1; stopAt <= 3; stopAt++) {
            Probe cancelled = new Probe(); cancelled.failActiveAt = stopAt;
            try { NanoSourcePreparation.prepare("invalid", cancelled); throw new AssertionError("cancel ignored"); }
            catch (IllegalStateException expected) { }
            require(cancelled.validations == (stopAt == 1 ? 0 : 1)
                    && cancelled.corrections == (stopAt == 3 ? 1 : 0), "work continued after cancellation");
        }
        require("missing delimiter".equals(GeminiNanoActionHooks.displayFailure(
                new IllegalStateException("missing delimiter"))), "raw Java exception shown");
        require(!GeminiNanoActionHooks.selects("local-all-v1", "What is two plus two?")
                && !GeminiNanoActionHooks.selects("local-all-v1", "explain the last error")
                && !GeminiNanoActionHooks.selects("local-o-v1", "What is two plus two?")
                && GeminiNanoActionHooks.selects("local-o-v1", "Use Ostadix for this")
                && !GeminiNanoActionHooks.selects("off", "Use Ostadix for this")
                && !GeminiNanoActionHooks.selects("local-all-v1", null), "route selection failed");
        System.out.println("Nano source preparation: valid, corrected, rejected, transport and cancellation boundaries passed");
    }

    private static void require(boolean condition, String message) {
        if (!condition) { throw new AssertionError(message); }
    }
}
