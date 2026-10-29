package org.ostadix.aicore.extension;

import android.content.Context;
import android.os.Process;
import android.os.SystemClock;
import android.provider.Settings;
import android.util.Log;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicLong;

import io.github.libxposed.api.XposedInterface;

/** Diagnostic only: observes exactly the host-selected synthetic input hash. */
final class GeminiRouteObservation {
    static final String SETTING = "ostadix_gemini_route_probe_sha256";
    private static final String TAG = "OstadixGeminiRoute";
    private static final AtomicLong SEQUENCE = new AtomicLong();

    static List<XposedInterface.HookHandle> install(XposedInterface module, Context context)
            throws ReflectiveOperationException {
        List<XposedInterface.HookHandle> handles = new ArrayList<>();
        String selected = Settings.Global.getString(context.getContentResolver(), SETTING);
        if (selected == null || !selected.matches("[0-9a-f]{64}")) { return handles; }
        ClassLoader loader = context.getClassLoader();
        try {
            add(module, context, loader, handles, "atma", "a", 1, null,
                    "com.google.android.apps.search.assistant.surfaces.voice.robin.data.ChatControllerId",
                    "auja", "asgm");
            add(module, context, loader, handles, "asws", "b", 0, "c", "auja", "asgm");
            add(module, context, loader, handles, "ayko", "b", 0, "a", "auja", "asgm");
            add(module, context, loader, handles, "ayld", "b", 0, "b", "auja", "asgm");
            add(module, context, loader, handles, "ayiz", "b", -1, null, "aygj", "ayij");
            for (String method : new String[] {"c", "d", "e"}) {
                add(module, context, loader, handles, "asuo", method, 0, "b",
                        "auja", "asew", "aygx", "asgm", "audl", "boolean", "boolean", "hdkj");
            }
            add(module, context, loader, handles, "asvw", "c", 1, "c",
                    "audl", "auja", "asgm", "asfw", "aygx", "asff", "ataa", "hdne", "hdkj");
            add(module, context, loader, handles, "asue", "invoke", -2, null,
                    "java.lang.Object", "java.lang.Object");
            Log.i(TAG, "event=observer_installed hooks=" + handles.size()
                    + " pid=" + Process.myPid() + " uid=" + Process.myUid());
            return handles;
        } catch (Throwable error) {
            for (XposedInterface.HookHandle handle : handles) { handle.unhook(); }
            if (error instanceof ReflectiveOperationException) {
                throw (ReflectiveOperationException) error;
            }
            throw error;
        }
    }

    private static void add(XposedInterface module, Context context, ClassLoader loader,
            List<XposedInterface.HookHandle> handles, String owner, String name,
            int inputIndex, String storeField, String... parameters)
            throws ReflectiveOperationException {
        Class<?>[] types = new Class<?>[parameters.length];
        for (int i = 0; i < types.length; i++) {
            types[i] = "boolean".equals(parameters[i]) ? boolean.class
                    : GeminiLocalResponse.type(loader, parameters[i]);
        }
        Method method = GeminiLocalResponse.type(loader, owner).getDeclaredMethod(name, types);
        method.setAccessible(true);
        handles.add(module.hook(method).setId("ostadix-gemini/observe-" + owner + "-" + name)
                .setExceptionMode(XposedInterface.ExceptionMode.PASSTHROUGH)
                .intercept(new XposedInterface.Hooker() {
                    public Object intercept(XposedInterface.Chain chain) throws Throwable {
                        Observation observation = prepare(context, chain, owner, name,
                                inputIndex, storeField);
                        // Never delegate inside observer error recovery. Original exceptions
                        // propagate once, including when event recording fails.
                        try {
                            Object result = chain.proceed();
                            finish(observation, "returned", result, null);
                            return result;
                        } catch (Throwable original) {
                            finish(observation, "threw", null, original);
                            throw original;
                        }
                    }
                }));
    }

    private static Observation prepare(Context context, XposedInterface.Chain chain,
            String owner, String name, int inputIndex, String storeField) {
        try {
            String selected = Settings.Global.getString(context.getContentResolver(), SETTING);
            if (selected == null || !selected.matches("[0-9a-f]{64}")) { return null; }
            Object receiver = chain.getThisObject();
            Object store = storeField == null ? null : field(receiver, storeField);
            Object input;
            String queryClass = "none";
            if (inputIndex == -2) {
                input = field(receiver, "f");
                store = field(field(receiver, "c"), "b");
                queryClass = "asue";
            } else if (inputIndex < 0) {
                Object query = chain.getArg(0);
                queryClass = query.getClass().getName();
                if (GeminiHostSymbols.name("aykm").equals(queryClass)) {
                    input = field(query, "a"); store = field(query, "l");
                } else if (GeminiHostSymbols.name("aylb").equals(queryClass)) {
                    input = field(query, "c"); store = field(query, "b");
                } else { return null; }
            } else { input = chain.getArg(inputIndex); }
            // Kotlin resumes suspend methods with null placeholders; their
            // original input lives in the continuation. This is not a new send.
            if (input == null) { return null; }
            Object text = field(input, "a");
            if (!(text instanceof String) || !selected.equals(sha256((String) text))) {
                return null;
            }
            String details = " call=" + SEQUENCE.incrementAndGet() + " stage=" + owner + "." + name
                    + " input_sha256=" + selected + " input_object=" + System.identityHashCode(input)
                    + " query_class=" + queryClass + " store_object=" + System.identityHashCode(store)
                    + " pid=" + Process.myPid() + " uid=" + Process.myUid();
            Observation observation = new Observation(details, store);
            Log.i(TAG, "event=enter" + details + state(store));
            return observation;
        } catch (Throwable error) {
            ExtensionGate.rethrowIfVmFatal(error);
            Log.w(TAG, "event=observer_error stage=" + owner + "." + name
                    + " error_class=" + error.getClass().getName());
            return null;
        }
    }

    private static void finish(Observation observation, String status, Object result,
            Throwable error) {
        if (observation == null) { return; }
        try {
            Log.i(TAG, "event=" + status + observation.details
                    + " elapsed_ms=" + (SystemClock.elapsedRealtime() - observation.started)
                    + " return_class=" + (result == null ? "null" : result.getClass().getName())
                    + " error_class=" + (error == null ? "none" : error.getClass().getName())
                    + state(observation.store));
        } catch (Throwable observationError) {
            ExtensionGate.rethrowIfVmFatal(observationError);
        }
    }

    private static String state(Object store) throws ReflectiveOperationException {
        if (store == null) { return " state=unobserved"; }
        Object flow = field(store, "n");
        Method get = flow.getClass().getMethod("b");
        get.setAccessible(true);
        Object snapshot = get.invoke(flow);
        return " complete_turns=" + ((List<?>) field(snapshot, "c")).size()
                + " pending=" + (field(snapshot, "d") != null)
                + " conversation_state=" + field(snapshot, "f");
    }

    private static Object field(Object object, String name) throws ReflectiveOperationException {
        Field field = object.getClass().getDeclaredField(name);
        field.setAccessible(true);
        return field.get(object);
    }

    private static String sha256(String text) throws Exception {
        byte[] bytes = MessageDigest.getInstance("SHA-256").digest(text.getBytes(StandardCharsets.UTF_8));
        StringBuilder result = new StringBuilder(64);
        for (byte value : bytes) { result.append(String.format(Locale.ROOT, "%02x", value & 255)); }
        return result.toString();
    }

    private static final class Observation {
        final String details;
        final Object store;
        final long started = SystemClock.elapsedRealtime();
        Observation(String details, Object store) { this.details = details; this.store = store; }
    }
}
