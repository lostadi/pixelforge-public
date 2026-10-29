package org.ostadix.aicore.extension;

import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.UUID;

/** Read-only app_process probe against the installed Google APK, without installing hooks. */
public final class GeminiReflectiveContractProbe {
    public static void main(String[] args) throws Throwable {
        ClassLoader loader = GeminiReflectiveContractProbe.class.getClassLoader();
        // This test deliberately selects the exact APK under compatibility investigation.
        GeminiHostSymbols.configure(301806951L);
        GeminiLocalResponse.type(loader, "hdkj");
        String[] samples = {"actual local result", "Unicode \ud83d\ude80 / \u0646\u062a\u06cc\u062c\u06c1\nsecond line", "\"quotes\" \\ slash <>&"};
        for (int i = 0; i < samples.length; i++) {
            GeminiLocalResponse.text(loader, "ostadix-read-only-contract-" + i, samples[i]);
        }
        System.out.println("PASS production Gemini text protobuf roundtrip: 3 exact text/id samples");

        GeminiAppFunctionSchemaHooks schema = new GeminiAppFunctionSchemaHooks(loader);
        Field metadata = GeminiAppFunctionSchemaHooks.class.getDeclaredField("schemaMetadata");
        metadata.setAccessible(true);
        require(metadata.get(schema).getClass().getName().equals(
                "com.google.android.appfunctions.AppFunctionMetadata"), "schema metadata type");
        System.out.println("PASS production AppFunctions schema constructor and metadata type");

        String[] names = {"audl", "auja", "asgm", "asfw", "aygx", "asff", "ataa", "hdne", "hdkj"};
        Class<?>[] parameters = new Class<?>[names.length];
        for (int i = 0; i < names.length; i++) { parameters[i] = GeminiLocalResponse.type(loader, names[i]); }
        Class<?> sideOwner = GeminiLocalResponse.type(loader, "asvw");
        Method side = sideOwner.getDeclaredMethod("c", parameters);
        Class<?> callback = GeminiLocalResponse.type(loader, "asue");
        callback.getDeclaredMethod("invoke", Object.class, Object.class);
        require(callback.getDeclaredField("f").getType() == parameters[1], "callback captured input type");
        require(parameters[1].getDeclaredField("a").getType() == String.class, "input text field type");
        Class<?> store = sideOwner.getDeclaredField("c").getType();
        Class<?> controller = store.getDeclaredField("a").getType();
        require(controller.getDeclaredField("a").getType() == UUID.class, "private chat identity UUID field");
        System.out.println("PASS assistant entry/callback and private chat identity fields: " + side);

        Class<?> flowContract = Class.forName(
                "org.ostadix.aicore.extension.GeminiNanoActionHooks$FlowContract", false, loader);
        Constructor<?> flowConstructor = flowContract.getDeclaredConstructor(ClassLoader.class);
        flowConstructor.setAccessible(true);
        flowConstructor.newInstance(loader);
        System.out.println("PASS production coroutine FlowContract constructor");

        Method emptyStream = GeminiNanoActionHooks.class.getDeclaredMethod("emptySideStream", ClassLoader.class);
        emptyStream.setAccessible(true);
        Object closedStream = GeminiLocalResponse.call(emptyStream, null, loader);
        require(GeminiLocalResponse.type(loader, "heab").isInstance(closedStream), "closed stream Flow type");
        Class<?> sideStream = GeminiLocalResponse.type(loader, "asfw");
        require(sideStream.getMethod("l").getReturnType().isInstance(closedStream), "audio stream type");
        require(sideStream.getMethod("m").getReturnType().isInstance(closedStream), "operation stream type");
        sideStream.getMethod("n");
        System.out.println("PASS production empty side-stream construction/close and interface return types");
        System.out.println("PASS reflective compatibility only; no hooks, model requests or Ostadix execution dispatched");
    }

    private static void require(boolean condition, String description) {
        if (!condition) { throw new AssertionError(description); }
    }
}
