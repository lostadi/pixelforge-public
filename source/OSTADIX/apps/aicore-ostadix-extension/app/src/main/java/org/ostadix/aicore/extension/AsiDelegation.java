package org.ostadix.aicore.extension;

/** Keeps transformation fallback outside the original method's exception boundary. */
final class AsiDelegation {
    interface Preparation { Object[] prepare() throws Throwable; }
    interface Original { Object invoke(Object[] arguments) throws Throwable; }
    interface Failure { void record(Throwable error); }

    static Object invoke(Preparation preparation, Original original, Failure failure)
            throws Throwable {
        Object[] arguments = null;
        try {
            arguments = preparation.prepare();
        } catch (Throwable error) {
            ExtensionGate.rethrowIfVmFatal(error);
            failure.record(error);
        }
        // Deliberately outside the fallback catch: an original-method exception
        // is returned to its caller and never causes another invocation.
        return original.invoke(arguments);
    }

    private AsiDelegation() {}
}
