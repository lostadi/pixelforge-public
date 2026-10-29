package org.ostadix.aicore.extension;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.ParcelFileDescriptor;
import android.os.PowerManager;
import android.os.Process;
import android.os.Bundle;
import android.os.ResultReceiver;
import android.os.SystemClock;
import android.provider.Settings;
import android.util.Base64;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

import org.json.JSONArray;
import org.json.JSONObject;

/** Explicit local Nano experiment in the original AICore process. */
public final class NanoLocalProbe {
    // Host policy, not measured model capacity. Cancellation remains cooperative
    // at native controller callbacks; the client has a separate bounded wait.
    static final int MAX_OUTPUT_TOKENS = 2048;
    static final int MAX_GENERATION_CALLBACKS = 16384;
    static final long GENERATION_TIMEOUT_MS = 180000L;
    static final long ACTION_TIMEOUT_MS = GENERATION_TIMEOUT_MS + 30000L;
    static final long CLIENT_REPLY_TIMEOUT_MS = ACTION_TIMEOUT_MS + 5000L;
    private static final String TAG = "OstadixNanoLocalProbe";
    private static final String ACTION = "org.ostadix.aicore.NANO_LOCAL_PROBE";
    private static final String SETTING = "ostadix_nano_local_probe_token";
    private static final String ACTIVATION = "local-factory-234-10745-v1";
    private static final AtomicBoolean REGISTERED = new AtomicBoolean();
    private static final AtomicBoolean BUSY = new AtomicBoolean();
    private static final AtomicReference<Probe> ACTIVE_ACTION = new AtomicReference<>();
    private static final Object EVENTS_LOCK = new Object();
    private static final ExecutorService WORKER = Executors.newSingleThreadExecutor(new ThreadFactory() {
        @Override public Thread newThread(Runnable runnable) {
            Thread thread = new Thread(runnable, "OstadixNanoLocalProbe");
            thread.setDaemon(true);
            return thread;
        }
    });

    private NanoLocalProbe() {}

    public static void register(final Context context) {
        if (!"com.google.android.aicore".equals(context.getPackageName())) {
            throw new IllegalArgumentException("AICore application context required");
        }
        if (!REGISTERED.compareAndSet(false, true)) return;
        try {
            context.registerReceiver(new BroadcastReceiver() {
                @Override public void onReceive(Context receiverContext, Intent intent) {
                    if (intent == null || !ACTION.equals(intent.getAction())) return;
                    String requested = intent.getStringExtra("request_id");
                    if (requested == null) requested = intent.getStringExtra("requestId");
                    final String requestId = requested;
                    if (requestId == null || !requestId.matches("[A-Za-z0-9._:-]{1,80}")) {
                        Log.w(TAG, "Rejected missing or invalid request_id");
                        return;
                    }
                    final Probe probe = new Probe(context, requestId,
                            intent.getLongExtra("deadline_elapsed_ms", 0),
                            intent.getBooleanExtra("allow_generation", false));
                    try {
                        probe.checkGates();
                    } catch (Throwable failure) {
                        probe.finish(false, failure);
                        rethrowFatal(failure);
                        return;
                    }
                    if (!BUSY.compareAndSet(false, true)) {
                        probe.finish(false, new IllegalStateException("another local probe is active"));
                        return;
                    }
                    try {
                        WORKER.execute(new Runnable() {
                            @Override public void run() {
                                try { probe.run(); }
                                finally { BUSY.set(false); }
                            }
                        });
                    } catch (RuntimeException failure) {
                        BUSY.set(false);
                        probe.finish(false, failure);
                    }
                }
            }, new IntentFilter(ACTION), "android.permission.DUMP", null, Context.RECEIVER_EXPORTED);
            NanoActionReceiver.register(context);
            Log.i(TAG, "Registered explicit local probe receiver uid=" + Process.myUid()
                    + " pid=" + Process.myPid());
        } catch (RuntimeException failure) {
            REGISTERED.set(false);
            throw failure;
        }
    }

    static void submitAction(Context context, String requestId, String prompt, ResultReceiver reply) {
        final Probe probe = new Probe(context, requestId,
                SystemClock.elapsedRealtime() + ACTION_TIMEOUT_MS, true);
        probe.reply = reply;
        try {
            // All generation policy remains owned here. The caller supplies text only.
            probe.requestGeneration = new Generation(new JSONObject().put("prompt", prompt)
                    .put("maxOutputTokens", MAX_OUTPUT_TOKENS).put("timeoutMs", GENERATION_TIMEOUT_MS)
                    .put("matformerSignature", "matformer_0"));
            probe.checkGates();
            if (!BUSY.compareAndSet(false, true)) {
                probe.finish(false, new IllegalStateException("local model busy; not dispatched; no retry"));
                return;
            }
            ACTIVE_ACTION.set(probe);
            try {
                WORKER.execute(new Runnable() {
                    @Override public void run() {
                        try { probe.run(); }
                        finally {
                            ACTIVE_ACTION.compareAndSet(probe, null);
                            BUSY.set(false);
                        }
                    }
                });
            } catch (RuntimeException failure) {
                ACTIVE_ACTION.compareAndSet(probe, null);
                BUSY.set(false);
                probe.finish(false, failure);
            }
        } catch (Throwable failure) {
            probe.finish(false, failure);
            rethrowFatal(failure);
        }
    }

    static void cancelAction(String requestId) {
        Probe probe = ACTIVE_ACTION.get();
        if (probe != null && probe.requestId.equals(requestId)) {
            probe.externalCancellation.set(true);
        }
    }

    static boolean shouldCancelGeneration(boolean externalCancellation, int callbacks,
                                          long now, long deadline) {
        return externalCancellation || callbacks > MAX_GENERATION_CALLBACKS || now >= deadline;
    }

    private static void checkEntryGates(Context context) {
        if (!ACTIVATION.equals(Settings.Global.getString(context.getContentResolver(), SETTING))) {
            throw new IllegalStateException("local probe activation is absent");
        }
        PowerManager power = (PowerManager) context.getSystemService(Context.POWER_SERVICE);
        if (power == null || power.getCurrentThermalStatus() >= PowerManager.THERMAL_STATUS_SEVERE) {
            throw new IllegalStateException("local probe thermal gate rejected");
        }
    }

    private static void rethrowFatal(Throwable failure) {
        if (failure instanceof VirtualMachineError) throw (VirtualMachineError) failure;
        if (failure instanceof ThreadDeath) throw (ThreadDeath) failure;
    }

    private static Throwable unwrap(Throwable failure) {
        return failure instanceof InvocationTargetException
                && ((InvocationTargetException) failure).getTargetException() != null
                ? ((InvocationTargetException) failure).getTargetException() : failure;
    }

    private static String sha256(byte[] bytes) throws Exception {
        StringBuilder value = new StringBuilder();
        for (byte part : MessageDigest.getInstance("SHA-256").digest(bytes)) {
            value.append(String.format(java.util.Locale.ROOT, "%02x", part & 255));
        }
        return value.toString();
    }

    private static final class Probe {
        private final Context context;
        private final String requestId;
        private final long deadlineElapsedMs;
        private final boolean allowGeneration;
        private final File directory;
        private final AtomicBoolean terminal = new AtomicBoolean();
        private Throwable evidenceFailure;
        private String phase = "request_enter";
        private boolean inferenceDispatched;
        private boolean inferenceReturned;
        private final AtomicBoolean externalCancellation = new AtomicBoolean();
        private Generation requestGeneration;
        private ResultReceiver reply;
        private JSONObject generationResult;

        Probe(Context context, String requestId, long deadlineElapsedMs, boolean allowGeneration) {
            this.context = context;
            this.requestId = requestId;
            this.deadlineElapsedMs = deadlineElapsedMs;
            this.allowGeneration = allowGeneration;
            this.directory = new File(context.getFilesDir(), "ostadix-nano-probe");
        }

        private void checkGates() {
            if (externalCancellation.get()) {
                throw new java.util.concurrent.CancellationException("local model action cancelled");
            }
            long remaining = deadlineElapsedMs - SystemClock.elapsedRealtime();
            if (remaining <= 0 || remaining > ACTION_TIMEOUT_MS) {
                throw new IllegalStateException("probe deadline expired or invalid");
            }
            checkEntryGates(context);
        }

        private void record(String phase, boolean success, boolean active, Object detail) throws Exception {
            JSONObject event = new JSONObject();
            event.put("schema", "ostadix.aicore-local-probe/v1");
            event.put("request_id", requestId);
            event.put("pid", Process.myPid());
            event.put("uid", Process.myUid());
            event.put("phase", phase);
            event.put("success", success);
            event.put("native_call_active", active);
            event.put("elapsed_realtime_ms", SystemClock.elapsedRealtime());
            event.put("deadline_elapsed_ms", deadlineElapsedMs);
            event.put("inference_dispatched", inferenceDispatched);
            event.put("inference_returned", inferenceReturned);
            if (detail != null) event.put("detail", detail);
            String line = event.toString();
            byte[] bytes = (line + "\n").getBytes(StandardCharsets.UTF_8);
            synchronized (EVENTS_LOCK) {
                // The deployment prepares this app-private directory; registration creates no files.
                if (!directory.isDirectory()) throw new IllegalStateException("probe directory is absent");
                try (FileOutputStream output = new FileOutputStream(new File(directory, "events.jsonl"), true)) {
                    output.write(bytes);
                    output.flush();
                    output.getFD().sync();
                }
            }
            Log.i(TAG, line);
        }

        private void bestEffort(String phase, boolean success, boolean active, Object detail) {
            try { record(phase, success, active, detail); }
            catch (Exception failure) {
                if (evidenceFailure == null) evidenceFailure = failure;
                Log.e(TAG, "Evidence write failed request_id=" + requestId + " phase=" + phase,
                        failure);
            }
        }

        void finish(boolean success, Throwable failure) {
            if (!terminal.compareAndSet(false, true)) return;
            if (failure == null && externalCancellation.get()) {
                failure = new java.util.concurrent.CancellationException("local model action cancelled");
                success = false;
            }
            if (failure == null && evidenceFailure != null) { failure = evidenceFailure; success = false; }
            String detail = failure == null ? "local probe finished" : unwrap(failure).toString();
            bestEffort("complete", success && evidenceFailure == null, false, detail);
            if (reply != null) {
                boolean ok = success && evidenceFailure == null && !externalCancellation.get();
                try {
                    JSONObject result = new JSONObject().put("schema", "ostadix.nano-action-result/v1")
                            .put("ok", ok).put("request_id", requestId)
                            .put("pid", Process.myPid()).put("uid", Process.myUid())
                            .put("inference_dispatched", inferenceDispatched)
                            .put("inference_returned", inferenceReturned)
                            .put("cancel_requested", externalCancellation.get())
                            .put("result", generationResult == null ? JSONObject.NULL : generationResult)
                            .put("error", ok ? JSONObject.NULL : evidenceFailure != null
                                    ? "execution evidence failed: " + evidenceFailure.toString() : detail);
                    Bundle response = new Bundle();
                    response.putString("request_id", requestId);
                    response.putString("result_json", result.toString());
                    reply.send(ok ? 0 : 1, response);
                } catch (Throwable deliveryFailure) {
                    Log.e(TAG, "Result delivery failed; no inference retry request_id=" + requestId,
                            deliveryFailure);
                    rethrowFatal(deliveryFailure);
                }
            }
        }

        private byte[] readManifest() throws Exception {
            try (FileInputStream input = new FileInputStream(new File(directory, "manifest.json"));
                 ByteArrayOutputStream output = new ByteArrayOutputStream()) {
                byte[] buffer = new byte[8192];
                int length;
                while ((length = input.read(buffer)) != -1) {
                    if (output.size() + length > 1024 * 1024) {
                        throw new IllegalArgumentException("manifest exceeds 1MiB");
                    }
                    output.write(buffer, 0, length);
                }
                return output.toByteArray();
            }
        }

        private Object invokeNative(String name, Method method, Object instance, Object... arguments)
                throws Throwable {
            phase = name;
            boolean cleanup = name.equals("session_unload") || name.equals("model_unload")
                    || name.equals("runtime_free");
            if (cleanup) {
                // Cleanup still runs if storage failed after a successful native allocation.
                bestEffort(name + "_enter", true, true, null);
            } else {
                record(name + "_enter", true, true, null);
            }
            try {
                if (!cleanup) checkGates();
                if (name.equals("generation")) inferenceDispatched = true;
                Object result = method.invoke(instance, arguments);
                if (name.equals("generation")) inferenceReturned = true;
                // Preserve returned handle even if recording its return fails, so cleanup owns it.
                bestEffort(name + "_return", true, false, null);
                return result;
            } catch (Throwable failure) {
                Throwable actual = unwrap(failure);
                bestEffort(name + "_failure", false, false, actual.toString());
                throw actual;
            }
        }

        private void verifyEvidence() throws Exception {
            if (evidenceFailure != null) throw new IllegalStateException("evidence recording failed", evidenceFailure);
        }

        private JSONObject saveArtifact(String kind, byte[] identityBytes, JSONObject content)
                throws Exception {
            String identity = sha256(identityBytes);
            String filename = requestId + "." + kind + "." + identity + ".json";
            byte[] bytes = (content.toString() + "\n").getBytes(StandardCharsets.UTF_8);
            if (bytes.length > 6 * 1024 * 1024) {
                throw new IllegalArgumentException("generation artifact exceeds 6MiB");
            }
            try (FileOutputStream output = new FileOutputStream(new File(directory, filename))) {
                output.write(bytes);
                output.flush();
                output.getFD().sync();
            }
            JSONObject evidence = new JSONObject();
            evidence.put("artifact", filename);
            evidence.put("artifact_sha256", sha256(bytes));
            evidence.put("protobuf_sha256", identity);
            evidence.put("protobuf_bytes", identityBytes.length);
            return evidence;
        }

        private void generate(ClassLoader host, Object modelWrapper, long model,
                              Generation parameters) throws Throwable {
            Class<?> sessionClass = Class.forName(
                    "com.google.android.apps.aicore.runtime.wrapper.StatefulSessionWrapper", true, host);
            Class<?> controllerClass = Class.forName(
                    "com.google.android.apps.aicore.runtime.wrapper.StatefulSessionWrapper$Controller", false, host);
            Method create = modelWrapper.getClass().getDeclaredMethod(
                    "nativeCreateSession", long.class, byte[].class);
            Method generate = sessionClass.getDeclaredMethod(
                    "nativeGenerateResponse", long.class, byte[].class, controllerClass);
            Method unload = sessionClass.getDeclaredMethod("nativeUnload", long.class);
            create.setAccessible(true);
            generate.setAccessible(true);
            unload.setAccessible(true);
            Constructor<?> constructor = sessionClass.getDeclaredConstructor(long.class);
            constructor.setAccessible(true);
            // Its stock constructor accepts zero. Create the receiver before acquiring a handle,
            // then pass the owned handle explicitly to every private native method.
            final Object sessionWrapper = constructor.newInstance(0L);
            byte[] runtimeConfig = new Wire().integer(1, 123).integer(2, 1440)
                    .integer(3, 0).integer(4, 1).finish();
            Wire sessionConfigBuilder = new Wire().bytes(100, runtimeConfig)
                    .integer(18, 0).integer(20, 0);
            if (parameters.matformerSignature != null) {
                sessionConfigBuilder.string(9, parameters.matformerSignature);
            }
            byte[] sessionConfig = sessionConfigBuilder.finish();
            byte[] part = new Wire().string(1, parameters.prompt).finish();
            byte[] input = new Wire().bytes(1, part).integer(10, 1).finish();
            byte[] request = new Wire().floating(2, 0.0f).integer(3, parameters.maxOutputTokens)
                    .integer(5, 1).integer(6, 1).bytes(11, input).floating(14, 1.0f).finish();
            JSONObject prepared = new JSONObject();
            prepared.put("schema", "ostadix.nano-generation-request/v1");
            prepared.put("request_id", requestId);
            prepared.put("session_protobuf_base64", Base64.encodeToString(sessionConfig, Base64.NO_WRAP));
            prepared.put("session_protobuf_sha256", sha256(sessionConfig));
            prepared.put("request_protobuf_base64", Base64.encodeToString(request, Base64.NO_WRAP));
            prepared.put("request_protobuf_sha256", sha256(request));
            prepared.put("max_output_tokens", parameters.maxOutputTokens);
            prepared.put("timeout_ms", parameters.timeoutMs);
            prepared.put("max_callbacks", MAX_GENERATION_CALLBACKS);
            prepared.put("action_deadline_elapsed_ms", deadlineElapsedMs);
            prepared.put("budget_scope", "host_configuration_not_measured_model_capacity");
            prepared.put("rng_seed", 123);
            prepared.put("temperature", 0);
            prepared.put("top_k", 1);
            prepared.put("top_p", 1);
            prepared.put("sample_count", 1);
            prepared.put("matformer_signature", parameters.matformerSignature == null
                    ? JSONObject.NULL : parameters.matformerSignature);
            phase = "generation_prepare";
            record("generation_prepared", true, false, saveArtifact("generation-request", request, prepared));
            long session = 0;
            Throwable failure = null;
            String failurePhase = null;
            try {
                session = (Long) invokeNative("session_create", create, modelWrapper, model, sessionConfig);
                if (session == 0) throw new IllegalStateException("nativeCreateSession returned zero");
                verifyEvidence();
                final AtomicInteger callbacks = new AtomicInteger();
                final AtomicBoolean cancel = new AtomicBoolean();
                final long deadline = Math.min(deadlineElapsedMs,
                        SystemClock.elapsedRealtime() + parameters.timeoutMs);
                Object controller = Proxy.newProxyInstance(host, new Class<?>[]{controllerClass},
                        new InvocationHandler() {
                        @Override public Object invoke(Object proxy, Method method, Object[] arguments) {
                            if (method.getDeclaringClass() == Object.class) {
                                if (method.getName().equals("toString")) return "OstadixNanoSyntheticController";
                                if (method.getName().equals("hashCode")) return System.identityHashCode(proxy);
                                if (method.getName().equals("equals")) return proxy == arguments[0];
                            }
                            if (!method.getName().equals("process") || method.getReturnType() != int.class
                                    || method.getParameterCount() != 1
                                    || method.getParameterTypes()[0] != float.class) {
                                throw new UnsupportedOperationException(method.toString());
                            }
                            if (shouldCancelGeneration(externalCancellation.get(), callbacks.incrementAndGet(),
                                    SystemClock.elapsedRealtime(), deadline)) cancel.set(true);
                            return cancel.get() ? 2 : 0;
                        }
                    });
                byte[] response = (byte[]) invokeNative("generation", generate,
                        sessionWrapper, session, request, controller);
                long returnedAt = SystemClock.elapsedRealtime();
                verifyEvidence();
                if (response == null || response.length > 4 * 1024 * 1024) {
                    throw new IllegalStateException("native response is null or exceeds 4MiB");
                }
                JSONObject raw = new JSONObject();
                raw.put("schema", "ostadix.nano-generation-response/v1");
                raw.put("request_id", requestId);
                raw.put("request_protobuf_sha256", sha256(request));
                raw.put("session_protobuf_sha256", sha256(sessionConfig));
                raw.put("response_protobuf_sha256", sha256(response));
                raw.put("response_protobuf_base64", Base64.encodeToString(response, Base64.NO_WRAP));
                raw.put("callback_count", callbacks.get());
                raw.put("max_callbacks", MAX_GENERATION_CALLBACKS);
                raw.put("callback_limit_exceeded", callbacks.get() > MAX_GENERATION_CALLBACKS);
                raw.put("external_cancellation_requested", externalCancellation.get());
                raw.put("cancel_requested", cancel.get());
                raw.put("deadline_elapsed_ms", deadline);
                raw.put("returned_elapsed_ms", returnedAt);
                raw.put("deadline_exceeded", returnedAt >= deadline);
                // Persist actual native bytes before attempting to interpret them.
                JSONObject result = saveArtifact("generation-response", response, raw);
                record("generation_raw_result", true, false, result);
                JSONArray candidates = ResponseDecoder.candidates(response);
                result.put("candidates", candidates);
                result.put("callback_count", callbacks.get());
                result.put("max_output_tokens", parameters.maxOutputTokens);
                result.put("timeout_ms", parameters.timeoutMs);
                result.put("max_callbacks", MAX_GENERATION_CALLBACKS);
                result.put("callback_limit_exceeded", callbacks.get() > MAX_GENERATION_CALLBACKS);
                result.put("cancel_requested", cancel.get());
                boolean deadlineExceeded = returnedAt >= deadline;
                result.put("deadline_exceeded", deadlineExceeded);
                record("generation_result", !cancel.get() && !deadlineExceeded && candidates.length() > 0,
                        false, result);
                if (cancel.get() || deadlineExceeded) {
                    throw new IllegalStateException("generation cancelled or exceeded its callback/time budget; "
                            + "returned data is partial");
                }
                if (candidates.length() == 0) throw new IllegalStateException("native response has no candidates");
                generationResult = result;
            } catch (Throwable error) {
                failure = unwrap(error);
                failurePhase = phase;
                throw failure;
            } finally {
                if (session != 0) {
                    try { invokeNative("session_unload", unload, sessionWrapper, session); }
                    catch (Throwable cleanupError) {
                        Throwable actual = unwrap(cleanupError);
                        if (failure == null) throw actual;
                        if (actual != failure) failure.addSuppressed(actual);
                    }
                }
                if (failurePhase != null) phase = failurePhase;
            }
        }

        void run() {
            ArrayList<ParcelFileDescriptor> openFiles = new ArrayList<ParcelFileDescriptor>();
            long runtime = 0;
            long model = 0;
            Object runtimeWrapper = null;
            Object modelWrapper = null;
            Method nativeFree = null;
            Method nativeUnload = null;
            Throwable failure = null;
            try {
                record("request_enter", true, false, null);
                checkGates();
                byte[] manifestBytes = readManifest();
                JSONObject manifest = new JSONObject(new String(manifestBytes, StandardCharsets.UTF_8));
                if (requestGeneration == null && manifest.has("generation") != allowGeneration) {
                    throw new IllegalArgumentException("generation requires both manifest object and allow_generation broadcast flag");
                }
                Generation generation = requestGeneration != null ? requestGeneration
                        : allowGeneration ? new Generation(manifest.getJSONObject("generation")) : null;
                String modelName = manifest.getString("modelName");
                if (modelName.length() == 0 || modelName.length() > 4096
                        || modelName.startsWith("/") || modelName.contains("..")
                        || modelName.indexOf('\0') >= 0) {
                    throw new IllegalArgumentException("invalid host-owned model location");
                }
                if (manifest.has("files") == manifest.has("baseDirectory")) {
                    throw new IllegalArgumentException("exactly one of files and baseDirectory is required");
                }
                String tokenText = requestGeneration == null && manifest.has("tokenText")
                        ? manifest.getString("tokenText") : null;
                if (tokenText != null && tokenText.getBytes(StandardCharsets.UTF_8).length > 16384) {
                    throw new IllegalArgumentException("tokenText exceeds 16KiB");
                }
                Wire configBuilder = new Wire();
                String fileMapHash = null;
                if (manifest.has("files")) {
                    JSONObject files = manifest.getJSONObject("files");
                    if (files.length() == 0 || files.length() > 256) {
                        throw new IllegalArgumentException("files must contain 1..256 entries");
                    }
                    ArrayList<String> names = new ArrayList<String>();
                    Iterator<String> keys = files.keys();
                    while (keys.hasNext()) names.add(keys.next());
                    Collections.sort(names);
                    ArrayList<File> paths = new ArrayList<File>();
                    Wire mapIdentity = new Wire();
                    // Validate the complete fixed private manifest before opening anything.
                    for (String name : names) {
                        String path = files.getString(name);
                        File file = new File(path);
                        if (name.length() == 0 || name.length() > 4096 || name.startsWith("/")
                                || name.contains("..") || name.indexOf('\0') >= 0
                                || path.length() > 4096 || path.indexOf('\0') >= 0
                                || !file.isAbsolute() || !file.isFile()) {
                            throw new IllegalArgumentException("invalid logical name or absolute regular file in files map");
                        }
                        paths.add(file);
                        mapIdentity.bytes(1, new Wire().string(1, name).string(2, path).finish());
                    }
                    fileMapHash = sha256(mapIdentity.finish());
                    JSONObject opening = new JSONObject();
                    opening.put("file_map_sha256", fileMapHash);
                    opening.put("filename_count", names.size());
                    phase = "descriptors_open";
                    record("descriptors_open_enter", true, false, opening);
                    long totalBytes = 0;
                    for (int index = 0; index < names.size(); index++) {
                        checkGates();
                        ParcelFileDescriptor descriptor = ParcelFileDescriptor.open(
                                paths.get(index), ParcelFileDescriptor.MODE_READ_ONLY);
                        openFiles.add(descriptor);
                        long size = descriptor.getStatSize();
                        if (size < 0 || totalBytes > Long.MAX_VALUE - size) {
                            throw new IllegalArgumentException("invalid descriptor size");
                        }
                        totalBytes += size;
                        configBuilder.bytes(2, new Wire().string(1, names.get(index))
                                .integer(2, descriptor.getFd()).finish());
                    }
                    JSONObject opened = new JSONObject();
                    opened.put("file_map_sha256", fileMapHash);
                    opened.put("filename_count", names.size());
                    opened.put("opened_descriptor_count", openFiles.size());
                    opened.put("total_stat_bytes", totalBytes);
                    record("descriptors_open_return", true, false, opened);
                } else {
                    File base = new File(manifest.getString("baseDirectory"));
                    if (!base.isAbsolute() || !base.isDirectory()) {
                        throw new IllegalArgumentException("absolute base directory required");
                    }
                    configBuilder.string(1, base.getCanonicalPath()).integer(5, 0);
                }
                byte[] config = configBuilder.string(3, modelName).integer(4, 1).finish();
                JSONObject identity = new JSONObject();
                identity.put("manifest_sha256", sha256(manifestBytes));
                identity.put("native_config_sha256", sha256(config));
                identity.put("model_name", modelName);
                identity.put("loader_mode", fileMapHash == null ? "path" : "file_descriptors");
                if (fileMapHash != null) {
                    identity.put("file_map_sha256", fileMapHash);
                    identity.put("opened_descriptor_count", openFiles.size());
                }
                record("validation_complete", true, false, identity);
                ClassLoader loader = context.getClassLoader();
                phase = "library_init";
                record("library_init_enter", true, true, null);
                try {
                    checkGates();
                    Class.forName("cja", true, loader);
                    checkGates();
                    Class.forName("cji", true, loader);
                    checkGates();
                    Class.forName("cje", true, loader);
                    if (generation != null) {
                        checkGates();
                        Class.forName("cjn", true, loader);
                    }
                    bestEffort("library_init_return", true, false, null);
                } catch (Throwable error) {
                    bestEffort("library_init_failure", false, false, unwrap(error).toString());
                    throw error;
                }
                verifyEvidence();
                Class<?> runtimeClass = Class.forName("com.google.android.apps.aicore.runtime.impl.edgetpu.RuntimeEdgetpu", true, loader);
                Class<?> loaderClass = Class.forName("com.google.android.apps.aicore.runtime.wrapper.RuntimeModelLoaderWrapper", true, loader);
                Class<?> modelClass = Class.forName("com.google.android.apps.aicore.runtime.wrapper.LargeLanguageModelWrapper", true, loader);
                Method nativeCreate = runtimeClass.getDeclaredMethod("nativeCreate");
                Method nativeLoad = loaderClass.getDeclaredMethod("nativeLoadModel", long.class, byte[].class);
                nativeFree = loaderClass.getDeclaredMethod("nativeFree", long.class);
                nativeUnload = modelClass.getDeclaredMethod("nativeUnload", long.class);
                Method nativeTokens = modelClass.getDeclaredMethod("nativeGetTokenInfo", long.class, byte[].class);
                nativeCreate.setAccessible(true);
                nativeLoad.setAccessible(true);
                nativeFree.setAccessible(true);
                nativeUnload.setAccessible(true);
                nativeTokens.setAccessible(true);
                Constructor<?> runtimeConstructor = loaderClass.getDeclaredConstructor(Supplier.class, Class.forName("cba", false, loader));
                runtimeConstructor.setAccessible(true);
                runtimeWrapper = runtimeConstructor.newInstance(null, null);
                Constructor<?> modelConstructor = modelClass.getDeclaredConstructor(long.class, Class.forName("bos", false, loader));
                modelConstructor.setAccessible(true);
                checkGates();
                runtime = ((Long) invokeNative("runtime_create", nativeCreate, null)).longValue();
                if (runtime == 0) throw new IllegalStateException("nativeCreate returned zero");
                verifyEvidence();
                checkGates();
                model = ((Long) invokeNative("model_load", nativeLoad, null, runtime, config)).longValue();
                if (model == 0) throw new IllegalStateException("nativeLoadModel returned zero");
                modelWrapper = modelConstructor.newInstance(model, null);
                verifyEvidence();
                record("model_loaded", true, false, "nonzero native model handle");
                if (tokenText != null) {
                    checkGates();
                    byte[] textPart = new Wire().string(1, tokenText).finish();
                    byte[] textInput = new Wire().bytes(1, textPart).integer(10, 1).finish();
                    byte[] request = new Wire().bytes(11, textInput).finish();
                    byte[] result = (byte[]) invokeNative("token_info", nativeTokens, modelWrapper, model, request);
                    verifyEvidence();
                    if (result.length > 4096) throw new IllegalStateException("token-info response exceeds 4096 bytes");
                    JSONObject tokenInfo = new JSONObject();
                    tokenInfo.put("request_sha256", sha256(request));
                    tokenInfo.put("result_sha256", sha256(result));
                    tokenInfo.put("token_count", tokenCount(result));
                    record("token_info_result", true, false, tokenInfo);
                }
                if (generation != null) generate(loader, modelWrapper, model, generation);
            } catch (Throwable error) {
                failure = unwrap(error);
                bestEffort(phase + "_failure", false, false, failure.toString());
            } finally {
                if (model != 0 && nativeUnload != null && modelWrapper != null) {
                    try { invokeNative("model_unload", nativeUnload, modelWrapper, model); }
                    catch (Throwable error) {
                        if (failure == null) failure = unwrap(error);
                    }
                } else if (model != 0) {
                    Throwable error = new IllegalStateException("model handle has no original wrapper for cleanup");
                    bestEffort("model_unload_failure", false, false, error.toString());
                    if (failure == null) failure = error;
                }
                if (runtime != 0 && nativeFree != null && runtimeWrapper != null) {
                    try { invokeNative("runtime_free", nativeFree, runtimeWrapper, runtime); }
                    catch (Throwable error) {
                        if (failure == null) failure = unwrap(error);
                    }
                }
                if (!openFiles.isEmpty()) {
                    int closed = 0;
                    bestEffort("descriptors_close_enter", true, false, openFiles.size());
                    for (ParcelFileDescriptor descriptor : openFiles) {
                        try { descriptor.close(); closed++; }
                        catch (Throwable error) {
                            bestEffort("descriptors_close_failure", false, false, unwrap(error).toString());
                            if (failure == null) failure = unwrap(error);
                        }
                    }
                    bestEffort("descriptors_close_return", closed == openFiles.size(), false, closed);
                }
                finish(failure == null, failure);
            }
            if (failure != null) rethrowFatal(failure);
        }
    }

    static final class Generation {
        final String prompt;
        final int maxOutputTokens;
        final long timeoutMs;
        final String matformerSignature;

        Generation(JSONObject input) throws Exception {
            prompt = input.getString("prompt");
            double requestedMax = input.has("maxOutputTokens") ? input.getDouble("maxOutputTokens") : 32;
            double requestedTimeout = input.has("timeoutMs") ? input.getDouble("timeoutMs") : 10000;
            maxOutputTokens = (int) requestedMax;
            timeoutMs = (long) requestedTimeout;
            matformerSignature = input.has("matformerSignature") ? input.getString("matformerSignature") : null;
            if (prompt.getBytes(StandardCharsets.UTF_8).length > 16384
                    || maxOutputTokens < 1 || maxOutputTokens > MAX_OUTPUT_TOKENS
                    || timeoutMs < 1 || timeoutMs > GENERATION_TIMEOUT_MS
                    || requestedMax != maxOutputTokens || requestedTimeout != timeoutMs
                    || (matformerSignature != null && !matformerSignature.equals("matformer_0")
                        && !matformerSignature.equals("matformer_1"))) {
                throw new IllegalArgumentException("invalid generation bounds or matformerSignature");
            }
            if ((input.has("temperature") && input.getDouble("temperature") != 0)
                    || (input.has("topK") && input.getDouble("topK") != 1)
                    || (input.has("topP") && input.getDouble("topP") != 1)
                    || (input.has("sampleCount") && input.getDouble("sampleCount") != 1)
                    || (input.has("seed") && input.getDouble("seed") != 123)) {
                throw new IllegalArgumentException("generation sampling is fixed: temperature=0, topK=1, topP=1, sampleCount=1, seed=123");
            }
        }
    }

    /** Decodes only the observed ceu/cdw fields; full bytes are preserved first. */
    private static final class ResponseDecoder {
        private final byte[] bytes;
        private final int[] cursor = {0};

        ResponseDecoder(byte[] bytes) { this.bytes = bytes; }

        private byte[] data(long length) {
            if (length < 0 || length > bytes.length - cursor[0]) {
                throw new IllegalArgumentException("truncated response protobuf");
            }
            byte[] result = Arrays.copyOfRange(bytes, cursor[0], cursor[0] + (int) length);
            cursor[0] += (int) length;
            return result;
        }

        private byte[] lengthData() { return data(varint(bytes, cursor)); }

        private void skip(int wire) {
            if (wire == 0) varint(bytes, cursor);
            else if (wire == 1) data(8);
            else if (wire == 2) lengthData();
            else if (wire == 5) data(4);
            else throw new IllegalArgumentException("unsupported response protobuf wire type");
        }

        static JSONArray candidates(byte[] bytes) throws Exception {
            ResponseDecoder outer = new ResponseDecoder(bytes);
            JSONArray result = new JSONArray();
            int remainingText = 6000;
            while (outer.cursor[0] < bytes.length) {
                long tag = varint(bytes, outer.cursor);
                if (tag != 10) { outer.skip((int) tag & 7); continue; }
                if (result.length() >= 16) throw new IllegalArgumentException("more than 16 candidates");
                ResponseDecoder candidate = new ResponseDecoder(outer.lengthData());
                JSONObject item = new JSONObject();
                while (candidate.cursor[0] < candidate.bytes.length) {
                    long field = varint(candidate.bytes, candidate.cursor);
                    if (field == 10) {
                        String text = new String(candidate.lengthData(), StandardCharsets.UTF_8);
                        int count = Math.min(text.length(), remainingText);
                        item.put("text", text.substring(0, count));
                        item.put("text_truncated_in_event", count != text.length());
                        remainingText -= count;
                    } else if (field == 17) {
                        double score = ByteBuffer.wrap(candidate.data(8))
                                .order(ByteOrder.LITTLE_ENDIAN).getDouble();
                        item.put("score", Double.isFinite(score) ? score : JSONObject.NULL);
                    } else if (field == 40) {
                        item.put("finish_reason_enum", varint(candidate.bytes, candidate.cursor));
                    } else candidate.skip((int) field & 7);
                }
                result.put(item);
            }
            return result;
        }
    }

    private static long tokenCount(byte[] response) {
        int[] cursor = {0};
        Long count = null;
        while (cursor[0] < response.length) {
            long tag = varint(response, cursor);
            if (tag == 8) count = varint(response, cursor);
            else {
                int wire = (int) tag & 7;
                if (wire == 0) varint(response, cursor);
                else {
                    long length = wire == 1 ? 8 : wire == 5 ? 4 : wire == 2 ? varint(response, cursor) : -1;
                    if (length < 0 || length > response.length - cursor[0]) throw new IllegalArgumentException("invalid token-info protobuf");
                    cursor[0] += (int) length;
                }
            }
        }
        if (count == null || count < 0 || count > 0xffffffffL) throw new IllegalArgumentException("token-info count missing or out of range");
        return count.longValue();
    }

    private static long varint(byte[] input, int[] cursor) {
        long value = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (cursor[0] >= input.length) throw new IllegalArgumentException("truncated protobuf");
            int next = input[cursor[0]++] & 255;
            value |= (long) (next & 127) << shift;
            if ((next & 128) == 0) return value;
        }
        throw new IllegalArgumentException("oversized protobuf varint");
    }

    private static final class Wire {
        private final ByteArrayOutputStream output = new ByteArrayOutputStream();
        private void varint(long value) {
            while ((value & ~127L) != 0) {
                output.write(((int) value & 127) | 128);
                value >>>= 7;
            }
            output.write((int) value);
        }
        Wire integer(int field, long value) { varint((long) field << 3); varint(value); return this; }
        Wire bytes(int field, byte[] value) {
            varint(((long) field << 3) | 2);
            varint(value.length);
            output.write(value, 0, value.length);
            return this;
        }
        Wire string(int field, String value) { return bytes(field, value.getBytes(StandardCharsets.UTF_8)); }
        Wire floating(int field, float value) {
            varint(((long) field << 3) | 5);
            int bits = Float.floatToIntBits(value);
            for (int index = 0; index < 4; index++) output.write(bits >>> (index * 8));
            return this;
        }
        byte[] finish() { return output.toByteArray(); }
    }
}
