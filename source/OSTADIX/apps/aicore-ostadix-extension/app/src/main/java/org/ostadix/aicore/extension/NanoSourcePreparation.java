package org.ostadix.aicore.extension;

/** At most one model correction, exclusively before any execution dispatch. */
final class NanoSourcePreparation {
    interface Operations {
        void checkActive() throws Exception;
        /** Null means valid. A string means a confirmed parse or source-contract rejection.
         * Transport, cancellation and other failures must throw. */
        String validate(String source) throws Exception;
        String correct(String source, String parseError) throws Exception;
    }

    static final class Prepared {
        final String source;
        final boolean corrected;
        Prepared(String source, boolean corrected) { this.source = source; this.corrected = corrected; }
    }

    static Prepared prepare(String source, Operations operations) throws Exception {
        operations.checkActive();
        String error = operations.validate(source);
        if (error == null) { return new Prepared(source, false); }
        operations.checkActive();
        String corrected = operations.correct(source, error);
        operations.checkActive();
        String remainingError = operations.validate(corrected);
        if (remainingError != null) {
            throw new IllegalArgumentException("Nano generated an invalid program, and its one correction "
                    + "attempt did not fix it. Nothing was executed. Validation: " + remainingError);
        }
        return new Prepared(corrected, true);
    }

    /** Enforces this adapter's empty-bindings executable-source contract using native plan metadata.
     * O itself still permits literal documents and callers that supply initial bindings. */
    static String executableContractError(boolean literalText, String[] requiredBindings, int languages) {
        if (literalText) {
            return "Bare top-level text/code is outside an O executable block. Put Python inside "
                    + "python^(...)_python and Bash inside bash^(...)_bash; O declarations belong outside.";
        }
        if (requiredBindings.length != 0) {
            return "Undefined O bindings: " + java.util.Arrays.toString(requiredBindings)
                    + ". This request has no initial bindings. Define every value with let before its $splice; "
                    + "do not refer to $parts unless an earlier O declaration created parts.";
        }
        if (languages == 0) {
            return "No parsed executable language block. Return a complete O program with "
                    + "python^(...)_python, rust^(...)_rust or bash^(...)_bash containing the actual task.";
        }
        return null;
    }

    /** Applies only to a sole static Python block; unknown/dynamic cases stay unknown. */
    static String pythonResultContractError(String state, String resultCapture) {
        if (!"valid".equals(state) || !"none".equals(resultCapture)) { return null; }
        return "This Python block does not publish an answer. After computing the actual requested result, "
                + "assign __oval_result__ = result, or end with the actual result expression. "
                + "Assigning result or answer alone returns null. Nothing was executed.";
    }

    private NanoSourcePreparation() {}
}
