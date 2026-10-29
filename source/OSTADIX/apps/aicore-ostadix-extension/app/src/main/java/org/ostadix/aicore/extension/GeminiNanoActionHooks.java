package org.ostadix.aicore.extension;

import android.content.Context;
import android.os.CancellationSignal;
import android.provider.Settings;
import android.util.Log;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import io.github.libxposed.api.XposedInterface;

/** Owned local tool action at the ordinary assistant's published-pending-turn callback. */
final class GeminiNanoActionHooks {
    static final String SETTING = "ostadix_gemini_nano_tool_mode";
    private static final AtomicBoolean BUSY = new AtomicBoolean();
    private static final GeminiLocalRoute.Claims LOCAL_INPUTS = new GeminiLocalRoute.Claims();

    static boolean enabled(Context context) {
        String mode = Settings.Global.getString(context.getContentResolver(), SETTING);
        return GeminiLocalRoute.enabledMode(mode) && ExtensionGate.isExplicitlyEnabled(context);
    }

    static boolean selects(String mode, String text) {
        return GeminiLocalRoute.selects(mode, text);
    }

    static XposedInterface.HookHandle installSideStreams(XposedInterface module, Context context)
            throws Throwable {
        ClassLoader loader = context.getClassLoader();
        String[] names = {"audl", "auja", "asgm", "asfw", "aygx", "asff", "ataa", "hdne", "hdkj"};
        Class<?>[] parameters = new Class<?>[names.length];
        for (int i = 0; i < names.length; i++) { parameters[i] = GeminiLocalResponse.type(loader, names[i]); }
        Method method = GeminiLocalResponse.type(loader, "asvw").getDeclaredMethod("c", parameters);
        method.setAccessible(true);
        return module.hook(method).setId("ostadix-gemini/local-turn-side-streams-v1")
                .setExceptionMode(XposedInterface.ExceptionMode.PASSTHROUGH)
                .intercept(chain -> {
                    Prepared prepared = prepare(chain, context, loader, parameters);
                    if (prepared == null) { return chain.proceed(); }
                    LOCAL_INPUTS.put(prepared.input, prepared.conversation);
                    try {
                        Log.i("OstadixGeminiNano", "event=local_side_streams input_sha256="
                                + NanoToolTurn.digest(prepared.text));
                        return chain.proceed(prepared.arguments);
                    } catch (Throwable original) {
                        LOCAL_INPUTS.remove(prepared.input);
                        throw original;
                    }
                });
    }

    /** Any unsupported input or failed preparation retains the original arguments. */
    private static Prepared prepare(XposedInterface.Chain chain, Context context, ClassLoader loader,
            Class<?>[] parameters) throws Throwable {
        try {
            Object input = chain.getArg(1);
            if (input == null) { return null; }
            Object value = GeminiLocalResponse.field(input, "a");
            if (!(value instanceof String) || !GeminiLocalRoute.explicitRequest((String) value)) {
                return null;
            }
            String text = (String) value;
            if (!enabled(context)) { return null; }
            String rejected = GeminiPlainTextInput.rejectionReason(input, parameters[1]);
            if (rejected != null) {
                Log.i("OstadixGeminiNano", "event=explicit_request_delegated reason=" + rejected
                        + " input_sha256=" + NanoToolTurn.digest(text));
                return null;
            }
            Object audio = emptySideStream(loader);
            Object operations = emptySideStream(loader);
            Object streams = Proxy.newProxyInstance(loader,
                    new Class<?>[]{GeminiLocalResponse.type(loader, "asfw")}, (p, m, a) -> {
                if (m.getDeclaringClass() == Object.class) { return objectMethod(p, m, a); }
                if ("l".equals(m.getName())) { return audio; }
                if ("m".equals(m.getName())) { return operations; }
                if ("n".equals(m.getName())) { return null; }
                throw new UnsupportedOperationException(m.toString());
            });
            Object[] arguments = new Object[parameters.length];
            for (int i = 0; i < arguments.length; i++) { arguments[i] = chain.getArg(i); }
            arguments[3] = streams;
            String conversation;
            try {
                Object store = GeminiLocalResponse.field(chain.getThisObject(), "c");
                Object controller = GeminiLocalResponse.field(store, "a");
                Object uuid = GeminiLocalResponse.field(controller, "a");
                if (!(uuid instanceof UUID)) { throw new IllegalStateException("chat identity type changed"); }
                conversation = NanoToolTurn.digest(uuid.toString());
            } catch (ReflectiveOperationException | IllegalStateException failure) {
                // Never use another chat's context if the pinned identity is unavailable.
                conversation = "unavailable-" + UUID.randomUUID();
                Log.w("OstadixGeminiNano", "event=conversation_identity_unavailable", failure);
            }
            return new Prepared(input, text, conversation, arguments);
        } catch (Throwable failure) {
            ExtensionGate.rethrowIfVmFatal(failure);
            Log.w("OstadixGeminiNano", "event=local_preparation_skipped error_class="
                    + failure.getClass().getName());
            return null;
        }
    }

    private static final class Prepared {
        final Object input;
        final String text, conversation;
        final Object[] arguments;
        Prepared(Object input, String text, String conversation, Object[] arguments) {
            this.input = input; this.text = text; this.conversation = conversation;
            this.arguments = arguments;
        }
    }

    private static Object emptySideStream(ClassLoader loader) throws Throwable {
        Class<?> function = GeminiLocalResponse.type(loader, "kotlin.jvm.functions.Function1");
        Method channel = GeminiLocalResponse.type(loader, "hdzk").getDeclaredMethod(
                "f", int.class, int.class, function, int.class);
        Object value = GeminiLocalResponse.call(channel, null, 0, 0, null, 6);
        Method close = GeminiLocalResponse.type(loader, "hdzp").getMethod("e", Throwable.class);
        GeminiLocalResponse.call(close, value, (Object) null);
        return GeminiLocalResponse.type(loader, "hdzx").getDeclaredConstructor(
                GeminiLocalResponse.type(loader, "hdzo"), boolean.class).newInstance(value, false);
    }

    static XposedInterface.HookHandle install(XposedInterface module, Context context) throws Throwable {
        ClassLoader loader = context.getClassLoader();
        // Validate the pinned parser even while the action is disabled.
        GeminiLocalResponse.text(loader, "ostadix-contract-check", "local response contract");
        FlowContract contract = new FlowContract(loader);
        Method method = GeminiLocalResponse.type(loader, "asue").getDeclaredMethod(
                "invoke", Object.class, Object.class);
        method.setAccessible(true);
        return module.hook(method).setId("ostadix-gemini/local-nano-o-action-v1")
                .setExceptionMode(XposedInterface.ExceptionMode.PASSTHROUGH)
                .intercept(new XposedInterface.Hooker() {
                    @Override public Object intercept(XposedInterface.Chain chain) throws Throwable {
                        Object input;
                        try { input = GeminiLocalResponse.field(chain.getThisObject(), "f"); }
                        catch (Throwable failure) {
                            ExtensionGate.rethrowIfVmFatal(failure);
                            Log.w("OstadixGeminiNano", "event=unclaimed_callback_skipped error_class="
                                    + failure.getClass().getName());
                            return chain.proceed();
                        }
                        String conversation = LOCAL_INPUTS.remove(input);
                        if (conversation == null) { return chain.proceed(); }
                        String text = (String) GeminiLocalResponse.field(input, "a");
                        String id = "ostadix-turn-" + UUID.randomUUID();
                        Log.i("OstadixGeminiNano", "event=callback_claimed request_id=" + id
                                + " input_sha256=" + NanoToolTurn.digest(text)
                                + " stock_callback_delegations=0");
                        // Once claimed, any error belongs to this local turn. Never delegate/retry it.
                        return contract.flow(context, id, text, conversation);
                    }
                });
    }

    /** Uses the host's actual Flow, Continuation, Job and dispatcher interfaces. */
    private static final class FlowContract {
        final ClassLoader loader;
        final Class<?> flow, continuation, function;
        final Method getContext, resume, emit, contextGet, jobListen, dispose, fail, checkFailure;
        final Method intercept, release;
        final Object unit, suspended, jobKey, dispatcherKey;

        FlowContract(ClassLoader loader) throws ReflectiveOperationException {
            this.loader = loader;
            flow = type("heab"); continuation = type("hdkj");
            function = type("kotlin.jvm.functions.Function1");
            getContext = continuation.getMethod("getContext");
            resume = continuation.getMethod("resumeWith", Object.class);
            emit = type("heac").getMethod("a", Object.class, continuation);
            contextGet = type("hdkp").getMethod("get", type("hdko"));
            jobListen = type("hdwc").getMethod("l", boolean.class, boolean.class, function);
            dispose = type("hdvj").getMethod(GeminiHostSymbols.current() ? "jj" : "jg");
            fail = type("hdhg").getMethod("b", Throwable.class);
            checkFailure = type("hdhg").getMethod("a", Object.class);
            intercept = type("hdkl").getMethod("g", continuation);
            release = type("hdkl").getMethod("f", continuation);
            unit = GeminiLocalResponse.constant(loader, "hdhs", "a");
            suspended = GeminiLocalResponse.constant(loader, "hdkt", "a");
            jobKey = GeminiLocalResponse.constant(loader, "hdwc", "d");
            dispatcherKey = GeminiLocalResponse.constant(loader, "hdkl", "k");
        }
        Class<?> type(String name) throws ClassNotFoundException { return GeminiLocalResponse.type(loader, name); }
        Object call(Method m, Object owner, Object... args) throws Throwable {
            return GeminiLocalResponse.call(m, owner, args);
        }

        Object flow(Context context, String id, String text, String conversation) {
            AtomicBoolean collected = new AtomicBoolean();
            return Proxy.newProxyInstance(loader, new Class<?>[]{flow}, (proxy, method, args) -> {
                if (method.getDeclaringClass() == Object.class) { return objectMethod(proxy, method, args); }
                if (!(GeminiHostSymbols.current() ? "jz" : "jw").equals(method.getName())) {
                    throw new UnsupportedOperationException(method.toString());
                }
                if (!collected.compareAndSet(false, true)) {
                    throw new IllegalStateException("local assistant flow already collected; no retry");
                }
                return new Collection(context, id, text, conversation, args[0], args[1]).start();
            });
        }

        private final class Collection {
            final Context context;
            final String id, text, conversation;
            final Object collector, completion, coroutineContext, dispatcher;
            final CancellationSignal cancellation = new CancellationSignal();
            final AtomicBoolean finished = new AtomicBoolean();
            final AtomicInteger decision = new AtomicInteger();
            final AtomicReference<Object> terminal = new AtomicReference<>();
            Object registration;

            Collection(Context context, String id, String text, String conversation, Object collector, Object completion)
                    throws Throwable {
                this.context = context; this.id = id; this.text = text; this.conversation = conversation;
                this.collector = collector; this.completion = completion;
                coroutineContext = call(getContext, completion);
                dispatcher = call(contextGet, coroutineContext, dispatcherKey);
                if (dispatcher == null) { throw new IllegalStateException("assistant coroutine dispatcher missing"); }
            }

            Object start() throws Throwable {
                Object job = call(contextGet, coroutineContext, jobKey);
                if (job == null) { throw new IllegalStateException("assistant cancellation job missing; not dispatched"); }
                Object listener = Proxy.newProxyInstance(loader, new Class<?>[]{function}, (p, m, a) -> {
                    if (m.getDeclaringClass() == Object.class) { return objectMethod(p, m, a); }
                    if ("invoke".equals(m.getName()) || "a".equals(m.getName())) {
                        if (a[0] != null) { cancellation.cancel(); }
                        return unit;
                    }
                    throw new UnsupportedOperationException(m.toString());
                });
                // R8 specialized onCancelling=true to its concrete method-reference
                // wrapper. Adapt through the stock completion node and reference.
                java.lang.reflect.Constructor<?> nodeCtor = type("hdwb").getDeclaredConstructor(function);
                nodeCtor.setAccessible(true);
                Object node = nodeCtor.newInstance(listener);
                java.lang.reflect.Constructor<?> referenceCtor = type("hdwf").getDeclaredConstructor(Object.class);
                referenceCtor.setAccessible(true);
                Object reference = referenceCtor.newInstance(node);
                registration = call(jobListen, job, true, true, reference);
                Thread worker = new Thread(() -> work(), "OstadixGeminiToolTurn");
                try { worker.start(); }
                catch (Throwable error) { call(dispose, registration); throw error; }
                if (decision.compareAndSet(0, 1)) { return suspended; }
                Object result = terminal.get();
                call(checkFailure, null, result);
                return unit;
            }

            void work() {
                boolean owned = false;
                try {
                    cancellation.throwIfCanceled();
                    if (!BUSY.compareAndSet(false, true)) {
                        throw new IllegalStateException("local assistant tool busy; not dispatched; no retry");
                    }
                    owned = true;
                    NanoTurnStore.publish(context, id, text, null, null, false);
                    String answer = new NanoToolTurn(context, id, conversation, cancellation).run(text);
                    answer = NanoTurnStore.publish(context, id, text, answer, null, true);
                    cancellation.throwIfCanceled();
                    deliver(GeminiLocalResponse.text(loader, id, answer));
                } catch (Throwable failure) {
                    ExtensionGate.rethrowIfVmFatal(failure);
                    Log.e("OstadixGeminiNano", "event=turn_failed request_id=" + id, failure);
                    try {
                        String savedAnswer = NanoTurnStore.publish(context, id, text, null, failure, true);
                        if (cancellation.isCanceled()) { finish(call(fail, null, failure)); }
                        else { deliver(GeminiLocalResponse.text(loader, id,
                                savedAnswer == null ? "Local Ostadix request failed: " + displayFailure(failure)
                                        + " Execution was not retried." : savedAnswer)); }
                    } catch (Throwable delivery) { failure(delivery); }
                } finally { if (owned) { BUSY.set(false); } }
            }

            void deliver(Object response) throws Throwable {
                dispatch(() -> {
                    try {
                        cancellation.throwIfCanceled();
                        Object next = proxyContinuation(value -> {
                            try { call(checkFailure, null, value); finish(unit); }
                            catch (Throwable failure) { failure(failure); }
                        });
                        Object returned = call(emit, collector, response, next);
                        if (returned != suspended) { finish(unit); }
                        Log.i("OstadixGeminiNano", "event=response_emitted request_id=" + id
                                + " suspended=" + (returned == suspended));
                    } catch (Throwable failure) { failure(failure); }
                });
            }

            void failure(Throwable error) {
                ExtensionGate.rethrowIfVmFatal(error);
                try { finish(call(fail, null, error)); }
                catch (Throwable failure) { Log.e("OstadixGeminiNano", "Continuation completion failed", failure); }
            }

            void finish(Object result) throws Throwable {
                if (!finished.compareAndSet(false, true)) { return; }
                terminal.set(result);
                call(dispose, registration);
                if (decision.getAndSet(2) == 1) {
                    dispatch(() -> {
                        try {
                            call(resume, completion, result);
                            Log.i("OstadixGeminiNano", "event=flow_completion_resumed request_id=" + id);
                        }
                        catch (Throwable error) { Log.e("OstadixGeminiNano", "Assistant completion failed", error); }
                    });
                }
            }

            Object proxyContinuation(java.util.function.Consumer<Object> action) {
                return Proxy.newProxyInstance(loader, new Class<?>[]{continuation}, (p, m, a) -> {
                    if (m.getDeclaringClass() == Object.class) { return objectMethod(p, m, a); }
                    if ("getContext".equals(m.getName())) { return coroutineContext; }
                    if ("resumeWith".equals(m.getName())) { action.accept(a[0]); return null; }
                    throw new UnsupportedOperationException(m.toString());
                });
            }

            void dispatch(Runnable action) throws Throwable {
                AtomicReference<Object> intercepted = new AtomicReference<>();
                Object target = proxyContinuation(value -> {
                    try { call(checkFailure, null, value); action.run(); }
                    catch (Throwable error) { failure(error); }
                    finally {
                        try { call(release, dispatcher, intercepted.get()); }
                        catch (Throwable error) { Log.e("OstadixGeminiNano", "Dispatcher release failed", error); }
                    }
                });
                Object scheduled = call(intercept, dispatcher, target);
                intercepted.set(scheduled);
                call(resume, scheduled, unit);
            }
        }
    }

    static String displayFailure(Throwable failure) {
        String message = failure.getMessage();
        return message == null || message.trim().isEmpty() ? "An internal error occurred." : message;
    }

    private static Object objectMethod(Object proxy, Method method, Object[] args) {
        if ("toString".equals(method.getName())) { return "OstadixLocalAssistantAdapter"; }
        if ("hashCode".equals(method.getName())) { return System.identityHashCode(proxy); }
        if ("equals".equals(method.getName())) { return proxy == args[0]; }
        throw new UnsupportedOperationException(method.toString());
    }
}
