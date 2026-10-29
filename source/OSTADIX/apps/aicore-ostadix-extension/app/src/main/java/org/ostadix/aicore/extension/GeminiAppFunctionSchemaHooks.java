package org.ostadix.aicore.extension;

import android.util.Log;

import java.lang.reflect.Constructor;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import io.github.libxposed.api.XposedInterface;

/** Adds the Ostadix schema that the installed Gemini AppFunctions lister requires. */
final class GeminiAppFunctionSchemaHooks {
    private static final String TAG = "OstadixGeminiBridge";
    private static final String CATEGORY = "ostadix";
    private static final String SCHEMA_NAME = "executeO";
    private static final int SCHEMA_VERSION = 1;
    private static final int TYPE_LONG = 6;
    private static final int TYPE_STRING = 8;

    private final Object schemaKey;
    private final Object schemaMetadata;

    GeminiAppFunctionSchemaHooks(ClassLoader loader) throws ReflectiveOperationException {
        Class<?> keyClass = GeminiLocalResponse.type(loader, "vko");
        Constructor<?> keyConstructor = keyClass.getDeclaredConstructor(
                String.class, String.class, int.class);
        keyConstructor.setAccessible(true);
        schemaKey = keyConstructor.newInstance(CATEGORY, SCHEMA_NAME, SCHEMA_VERSION);

        Class<?> typeClass = Class.forName(
                "com.google.android.appfunctions.AppFunctionDataTypeMetadata", false, loader);
        Constructor<?> typeConstructor = typeClass.getDeclaredConstructor(
                String.class, String.class, int.class, boolean.class, boolean.class,
                String.class);
        typeConstructor.setAccessible(true);

        Class<?> parameterClass = Class.forName(
                "com.google.android.appfunctions.AppFunctionParameterMetadata", false, loader);
        Constructor<?> parameterConstructor = parameterClass.getDeclaredConstructor(
                String.class, String.class, String.class, String.class, typeClass);
        parameterConstructor.setAccessible(true);

        Object sourceType = typeConstructor.newInstance(
                "ostadix.schema.v1#executeO#source", CATEGORY,
                TYPE_STRING, false, false, null);
        Object bindingsType = typeConstructor.newInstance(
                "ostadix.schema.v1#executeO#bindingsJson", CATEGORY,
                TYPE_STRING, false, true, null);
        Object timeoutType = typeConstructor.newInstance(
                "ostadix.schema.v1#executeO#timeoutMs", CATEGORY,
                TYPE_LONG, false, true, null);
        Object returnType = typeConstructor.newInstance(
                "ostadix.schema.v1#executeO#return", CATEGORY,
                TYPE_STRING, false, false, null);

        Object source = parameterConstructor.newInstance(
                "ostadix.schema.v1#executeO#source", CATEGORY, "source",
                "One complete UTF-8 .O source document. Generate the program and pass its source "
                        + "directly, including its input values; do not pass a path, OIR, HGraph, "
                        + "or command line.",
                sourceType);
        Object bindings = parameterConstructor.newInstance(
                "ostadix.schema.v1#executeO#bindingsJson", CATEGORY, "bindingsJson",
                "Omit this parameter or pass the empty JSON object {}. Nonempty bindings are "
                        + "unsupported and rejected before execution; include input values in source.",
                bindingsType);
        Object timeout = parameterConstructor.newInstance(
                "ostadix.schema.v1#executeO#timeoutMs", CATEGORY, "timeoutMs",
                "Optional deadline from 1 through 900000 milliseconds, subject to the configured "
                        + "host limit. Default is 120000.", timeoutType);
        List<Object> parameters = Arrays.asList(source, bindings, timeout);

        Class<?> metadataClass = Class.forName(
                "com.google.android.appfunctions.AppFunctionMetadata", false, loader);
        Constructor<?> metadataConstructor = metadataClass.getDeclaredConstructor(
                String.class, String.class, boolean.class, String.class, List.class, typeClass);
        metadataConstructor.setAccessible(true);
        schemaMetadata = metadataConstructor.newInstance(
                "ostadix.schema.v1#executeO", CATEGORY, true,
                "Execute a complete Ostadix .O program locally and return the structured MCP "
                        + "result as JSON. Use this function whenever the user asks "
                        + "to use Ostadix, run .O source, perform computation through Ostadix, or "
                        + "combine supported runtimes. Generate a complete .O program, pass it in "
                        + "source, and read the execution state, actual result and execution evidence "
                        + "from the response. Report failures; do not automatically repeat execution.",
                parameters, returnType);
    }

    Object interceptInventory(XposedInterface.Chain chain) throws Throwable {
        Object original = chain.proceed();
        if (!(original instanceof Map)) {
            Log.e(TAG, "event=schema_inventory_failed reason=non_map_result");
            return original;
        }
        Map<?, ?> originalMap = (Map<?, ?>) original;
        LinkedHashMap<Object, Object> augmented = new LinkedHashMap<Object, Object>();
        augmented.putAll(originalMap);
        augmented.put(schemaKey, schemaMetadata);
        Log.i(TAG, "event=schema_inventory_injected original_count=" + originalMap.size()
                + " augmented_count=" + augmented.size());
        return Collections.unmodifiableMap(augmented);
    }
}
