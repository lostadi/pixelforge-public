package org.ostadix.terminal;

import android.app.appfunctions.AppFunctionException;
import android.app.appfunctions.AppFunctionService;
import android.app.appfunctions.ExecuteAppFunctionRequest;
import android.app.appfunctions.ExecuteAppFunctionResponse;
import android.app.appsearch.GenericDocument;
import android.content.pm.SigningInfo;
import android.os.CancellationSignal;
import android.os.OutcomeReceiver;
import android.util.Log;

import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/** Source-first Ostadix tool exposed through Android's system App Functions broker. */
public final class OstadixAppFunctionService extends AppFunctionService {
    public static final String FUNCTION_ID =
            "org.ostadix.terminal.OstadixAppFunctionService#executeO";

    private static final String TAG = "OstadixAppFunction";
    private static final long DEFAULT_TIMEOUT_MS = 120_000L;
    private static final long MAX_TIMEOUT_MS = 900_000L;
    private static final int MAX_SOURCE_BYTES = 64 * 1024;
    private static final int MAX_BINDINGS_BYTES = 1024 * 1024;
    private final AtomicLong nextRequestId = new AtomicLong();
    private final ExecutorService executionExecutor = Executors.newFixedThreadPool(4,
            namedThreadFactory("ostadix-app-function"));
    private final Set<HostMcpClient> clients = java.util.concurrent.ConcurrentHashMap.newKeySet();

    @Override
    public void onExecuteFunction(ExecuteAppFunctionRequest request, String callingPackage,
            SigningInfo callingPackageSigningInfo, CancellationSignal cancellationSignal,
            OutcomeReceiver<ExecuteAppFunctionResponse, AppFunctionException> callback) {
        if (!FUNCTION_ID.equals(request.getFunctionIdentifier())) {
            callback.onError(new AppFunctionException(
                    AppFunctionException.ERROR_FUNCTION_NOT_FOUND,
                    "Unknown Ostadix function: " + request.getFunctionIdentifier()));
            return;
        }

        GenericDocument parameters = request.getParameters();
        String source = parameters.getPropertyString("source");
        String bindingsJson = parameters.getPropertyString("bindingsJson");
        if (bindingsJson == null || bindingsJson.isEmpty()) {
            bindingsJson = "{}";
        }
        long timeoutMs = readOptionalLong(parameters, "timeoutMs", DEFAULT_TIMEOUT_MS);
        String validationError = validateRequest(source, bindingsJson, timeoutMs);
        if (validationError != null) {
            callback.onError(new AppFunctionException(
                    AppFunctionException.ERROR_INVALID_ARGUMENT, validationError));
            return;
        }

        final String acceptedSource = source;
        final String acceptedBindings = bindingsJson;
        final long acceptedTimeoutMs = timeoutMs;
        final String callerId = "android-appfunction/"
                + ((callingPackage == null || callingPackage.isEmpty())
                        ? "platform-agent" : callingPackage);
        final String requestId = "appfn-" + System.currentTimeMillis() + "-"
                + nextRequestId.incrementAndGet();
        final AtomicBoolean responseDone = new AtomicBoolean(false);

        executionExecutor.execute(new Runnable() {
            @Override
            public void run() {
                ExecuteAppFunctionResponse response = null;
                AppFunctionException failure = null;
                HostMcpClient client = new HostMcpClient();
                clients.add(client);
                try {
                    if (cancellationSignal.isCanceled()) {
                        throw new AppFunctionException(AppFunctionException.ERROR_CANCELLED,
                                "Ostadix request was cancelled before dispatch");
                    }
                    Log.i(TAG, "event=request_enter caller=" + callerId
                            + " request=" + requestId + " tool=o_execute");
                    String resultJson = client.execute(OstadixAppFunctionService.this,
                            acceptedSource, acceptedBindings, acceptedTimeoutMs, requestId,
                            cancellationSignal);
                    if (cancellationSignal.isCanceled()) {
                        throw new AppFunctionException(AppFunctionException.ERROR_CANCELLED,
                                "Ostadix request was cancelled");
                    }
                    GenericDocument result = new GenericDocument.Builder<>(
                            "ostadix", requestId, "OstadixMcpResult")
                            .setPropertyString(ExecuteAppFunctionResponse.PROPERTY_RETURN_VALUE, resultJson)
                            .build();
                    response = new ExecuteAppFunctionResponse(result);
                } catch (Throwable error) {
                    if (error instanceof VirtualMachineError) { throw (VirtualMachineError) error; }
                    if (error instanceof ThreadDeath) { throw (ThreadDeath) error; }
                    boolean cancelled = cancellationSignal.isCanceled();
                    failure = new AppFunctionException(cancelled
                            ? AppFunctionException.ERROR_CANCELLED
                            : AppFunctionException.ERROR_APP_UNKNOWN_ERROR,
                            "Ostadix MCP: " + safeMessage(error) + "; no execution retry");
                    Log.i(TAG, "event=" + (cancelled ? "request_cancelled" : "request_failed")
                            + " caller=" + callerId + " request=" + requestId);
                } finally {
                    client.close();
                    clients.remove(client);
                }
                // Callback invocation is outside the execution catch: a callback
                // exception cannot produce a second terminal callback or dispatch.
                if (responseDone.compareAndSet(false, true)) {
                    if (failure != null) { callback.onError(failure); }
                    else {
                        callback.onResult(response);
                        Log.i(TAG, "event=result_returned caller=" + callerId
                                + " request=" + requestId + " tool=o_execute");
                    }
                }
            }
        });
    }

    @Override
    public void onDestroy() {
        for (HostMcpClient client : clients) { client.close(); }
        executionExecutor.shutdownNow();
        super.onDestroy();
    }

    static String validateRequest(String source, String bindingsJson, long timeoutMs) {
        if (source == null || source.trim().isEmpty()) {
            return "source must contain one complete .O program";
        }
        if (source.getBytes(java.nio.charset.StandardCharsets.UTF_8).length > MAX_SOURCE_BYTES) {
            return "source exceeds the 65536-byte host limit";
        }
        if (bindingsJson == null) {
            return "bindingsJson must be a JSON object or omitted";
        }
        if (bindingsJson.getBytes(java.nio.charset.StandardCharsets.UTF_8).length
                > MAX_BINDINGS_BYTES) {
            return "bindingsJson exceeds the 1048576-byte App Function limit";
        }
        if (timeoutMs < 1 || timeoutMs > MAX_TIMEOUT_MS) {
            return "timeoutMs must be between 1 and 900000";
        }
        return null;
    }

    private static long readOptionalLong(GenericDocument parameters, String name,
            long defaultValue) {
        Set<String> names = parameters.getPropertyNames();
        return names.contains(name) ? parameters.getPropertyLong(name) : defaultValue;
    }

    private static ThreadFactory namedThreadFactory(String name) {
        return new ThreadFactory() {
            private final AtomicLong sequence = new AtomicLong();

            @Override
            public Thread newThread(Runnable runnable) {
                Thread thread = new Thread(runnable, name + "-" + sequence.incrementAndGet());
                thread.setDaemon(true);
                return thread;
            }
        };
    }

    private static String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.isEmpty()
                ? error.getClass().getSimpleName() : message;
    }
}
