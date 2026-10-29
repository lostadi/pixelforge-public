package org.ostadix.aicore.extension;

import android.app.Application;
import android.content.Context;
import android.util.Log;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

import io.github.libxposed.api.XposedInterface;
import io.github.libxposed.api.XposedModule;

public final class AicoreOstadixModule extends XposedModule {
    private static final String TAG = "OstadixAicoreExperiment";
    private boolean acceptedProcess;
    private boolean initialized;
    private XposedInterface.HookHandle bootstrapHandle;

    @Override
    public void onModuleLoaded(ModuleLoadedParam param) {
        Log.i(TAG, "event=owned_build version=" + OwnedBuildIdentity.VERSION
                + " sources_sha256=" + OwnedBuildIdentity.SOURCES_SHA256
                + " module_apk=" + getModuleApplicationInfo().sourceDir);
        acceptedProcess = !param.isSystemServer()
                && ExtensionGate.matchesEarlyProcess(
                        param.getProcessName(), getFrameworkName(), getApiVersion());
        Log.i(TAG, "event=early_gate process=" + param.getProcessName()
                + " accepted=" + acceptedProcess + " sdk=" + android.os.Build.VERSION.SDK_INT
                + " fingerprint=" + android.os.Build.FINGERPRINT
                + " framework=" + getFrameworkName() + " api=" + getApiVersion());
        if (!acceptedProcess) {
            detach();
        }
    }

    @Override
    public void onPackageReady(PackageReadyParam param) {
        if (!acceptedProcess || !param.isFirstPackage()
                || (!ExtensionGate.ASOSS_PACKAGE.equals(param.getPackageName())
                && !ExtensionGate.ASI_PACKAGE.equals(param.getPackageName())
                && !ExtensionGate.AICORE_PACKAGE.equals(param.getPackageName())
                && !ExtensionGate.GSA_PACKAGE.equals(param.getPackageName()))) {
            return;
        }
        try {
            Method attach = Application.class.getDeclaredMethod("attach", Context.class);
            attach.setAccessible(true);
            bootstrapHandle = hook(attach)
                    .setId("ostadix-aicore/application-attach-bootstrap-v1")
                    .setExceptionMode(XposedInterface.ExceptionMode.PROTECTIVE)
                    .intercept(new XposedInterface.Hooker() {
                        @Override
                        public Object intercept(XposedInterface.Chain chain) throws Throwable {
                            Object attached;
                            try {
                                attached = chain.proceed();
                            } finally {
                                unhookBootstrap();
                            }
                            Object application = chain.getThisObject();
                            if (application instanceof Context) {
                                initialize((Context) application);
                            } else {
                                log(Log.ERROR, TAG,
                                        "Application attach did not expose an AS.OSS context");
                                detach();
                            }
                            return attached;
                        }
                    });
            log(Log.INFO, TAG, "Installed one-shot OSTADIX Application.attach bootstrap");
        } catch (Throwable error) {
            ExtensionGate.rethrowIfVmFatal(error);
            log(Log.ERROR, TAG, "Bootstrap hook installation failed; detaching", error);
            detach();
        }
    }

    private synchronized void unhookBootstrap() {
        XposedInterface.HookHandle handle = bootstrapHandle;
        bootstrapHandle = null;
        if (handle != null) {
            try {
                handle.unhook();
            } catch (Throwable error) {
                ExtensionGate.rethrowIfVmFatal(error);
                log(Log.ERROR, TAG, "Application.attach bootstrap cleanup failed", error);
            }
        }
    }

    private synchronized void initialize(Context context) {
        if (initialized) {
            return;
        }
        initialized = true;
        boolean identityAccepted = ExtensionGate.acceptsInstalledPackages(context);
        boolean activationAccepted = ExtensionGate.isExplicitlyEnabled(context);
        Log.i(TAG, "event=startup_gate identity=" + identityAccepted
                + " activation=" + activationAccepted + " package=" + context.getPackageName());
        if (!identityAccepted || !activationAccepted) {
            log(Log.WARN, TAG, "Startup gate rejected; identity=" + identityAccepted
                    + " activation=" + activationAccepted
                    + " context=" + (context == null ? "none" : context.getPackageName())
                    + "; detaching");
            detach();
            return;
        }
        if (ExtensionGate.ASI_PACKAGE.equals(context.getPackageName())) {
            if (!ExtensionGate.isAsiAutofillExperimentEnabled(context)) {
                log(Log.INFO, TAG, "ASI autofill experiment disabled; original behavior retained");
                detach();
                return;
            }
            initializeAsi(context);
            return;
        }
        if (ExtensionGate.GSA_PACKAGE.equals(context.getPackageName())) {
            initializeGsa(context);
            return;
        }
        if (ExtensionGate.AICORE_PACKAGE.equals(context.getPackageName())) {
            try {
                NanoLocalProbe.register(context);
                log(Log.INFO, TAG, "Installed protected local Nano probe receiver in AICore");
            } catch (Throwable error) {
                ExtensionGate.rethrowIfVmFatal(error);
                log(Log.ERROR, TAG, "Local Nano probe registration failed", error);
            }
            detach();
            return;
        }
        if (!ExtensionGate.isSmartReplyExperimentEnabled(context)) {
            log(Log.INFO, TAG, "Smart Reply experiment disabled; original behavior retained");
            detach();
            return;
        }
        OstadixResultBridge bridge = null;
        List<XposedInterface.HookHandle> installedHooks =
                new ArrayList<XposedInterface.HookHandle>(4);
        try {
            ClassLoader loader = context.getClassLoader();
            Class<?> request = Class.forName(
                    "com.google.android.apps.aicore.aidl.SmartReplyRequest", false, loader);
            Class<?> result = Class.forName(
                    "com.google.android.apps.aicore.aidl.SmartReplyResult", false, loader);
            // JADX displays default-package obfuscated classes under its synthetic
            // `defpackage` source folder. Their installed DEX descriptors are
            // Lflv;, Lflw;, Lfna;, and Lflo;, so runtime lookup uses bare names.
            Class<?> resultCallback = Class.forName("flv", false, loader);
            Class<?> forwarder = Class.forName("flw", false, loader);
            Class<?> callback = Class.forName("fna", false, loader);
            Class<?> cancellation = Class.forName("flo", false, loader);
            Method forward = forwarder.getDeclaredMethod("c", request, resultCallback);
            Method success = callback.getDeclaredMethod("onSmartReplyInferenceSuccess", result);
            Method failure = callback.getDeclaredMethod(
                    "onSmartReplyInferenceFailure", int.class);
            Method cancel = cancellation.getDeclaredMethod("a");
            bridge = new OstadixResultBridge(getModuleApplicationInfo());
            org.ostadix.terminal.OstadixRuntime.Evaluation smoke = bridge.smokeRuntime();
            if (!smoke.ok || smoke.selectedSourceIndex != 1
                    || smoke.selectedScoreMilli != 950) {
                bridge.close();
                throw new IllegalStateException("in-process OSTADIX Smart Reply smoke failed");
            }
            log(Log.INFO, TAG, "event=runtime_smoke source_index="
                    + smoke.selectedSourceIndex + " score_milli=" + smoke.selectedScoreMilli
                    + " elapsed_ms=" + smoke.elapsedMs
                    + " intent_sha256=" + smoke.executionIntentSha256
                    + " result_identity=" + smoke.resultContentIdentity);
            final AicoreHooks hooks = new AicoreHooks(
                    context, bridge, callback);

            installedHooks.add(hook(forward)
                    .setId("ostadix-aicore/flw-smart-reply-forward-v1")
                    .setExceptionMode(XposedInterface.ExceptionMode.PROTECTIVE)
                    .intercept(new XposedInterface.Hooker() {
                        @Override
                        public Object intercept(XposedInterface.Chain chain) throws Throwable {
                            return hooks.interceptForward(chain);
                        }
                    }));
            installedHooks.add(hook(success)
                    .setId("ostadix-aicore/fna-smart-reply-success-v1")
                    .setExceptionMode(XposedInterface.ExceptionMode.PROTECTIVE)
                    .intercept(new XposedInterface.Hooker() {
                        @Override
                        public Object intercept(XposedInterface.Chain chain) throws Throwable {
                            return hooks.interceptSuccess(chain);
                        }
                    }));
            installedHooks.add(hook(failure)
                    .setId("ostadix-aicore/fna-smart-reply-failure-v1")
                    .setExceptionMode(XposedInterface.ExceptionMode.PROTECTIVE)
                    .intercept(new XposedInterface.Hooker() {
                        @Override
                        public Object intercept(XposedInterface.Chain chain) throws Throwable {
                            return hooks.interceptFailure(chain);
                        }
                    }));
            installedHooks.add(hook(cancel)
                    .setId("ostadix-aicore/flo-cancel-v1")
                    .setExceptionMode(XposedInterface.ExceptionMode.PROTECTIVE)
                    .intercept(new XposedInterface.Hooker() {
                        @Override
                        public Object intercept(XposedInterface.Chain chain) throws Throwable {
                            return hooks.interceptCancellation(chain);
                        }
                    }));
            log(Log.INFO, TAG,
                    "Installed four version-pinned active Smart Reply hooks");
            detach();
        } catch (Throwable error) {
            ExtensionGate.rethrowIfVmFatal(error);
            for (int index = installedHooks.size() - 1; index >= 0; index--) {
                try {
                    installedHooks.get(index).unhook();
                } catch (Throwable cleanupError) {
                    ExtensionGate.rethrowIfVmFatal(cleanupError);
                    log(Log.ERROR, TAG, "Partial Smart Reply hook rollback failed", cleanupError);
                }
            }
            if (bridge != null) {
                try {
                    bridge.close();
                } catch (Throwable cleanupError) {
                    ExtensionGate.rethrowIfVmFatal(cleanupError);
                    log(Log.ERROR, TAG, "OSTADIX runtime cleanup failed", cleanupError);
                }
            }
            log(Log.ERROR, TAG, "Hook installation failed; original AS.OSS path retained", error);
            detach();
        }
    }

    private void initializeAsi(Context context) {
        OstadixResultBridge bridge = null;
        XposedInterface.HookHandle installedHook = null;
        try {
            ClassLoader loader = context.getClassLoader();
            Class<?> connector = Class.forName("jht", false, loader);
            Class<?> request = Class.forName("ksf", false, loader);
            Class<?> requestData = Class.forName("isj", false, loader);
            Class<?> wrapper = Class.forName("ffg", false, loader);
            Method buildResponse = connector.getDeclaredMethod("d", request, requestData,
                    wrapper, wrapper, List.class);
            bridge = new OstadixResultBridge(getModuleApplicationInfo());
            org.ostadix.terminal.OstadixRuntime.Evaluation smoke = bridge.smokeRuntime();
            if (!smoke.ok || smoke.selectedSourceIndex != 1
                    || smoke.selectedScoreMilli != 950) {
                bridge.close();
                throw new IllegalStateException("in-process OSTADIX ASI smoke failed");
            }
            final AsiHooks hooks = new AsiHooks(context, bridge);
            installedHook = hook(buildResponse)
                    .setId("ostadix-asi/jht-fill-response-v1")
                    .setExceptionMode(XposedInterface.ExceptionMode.PASSTHROUGH)
                    .intercept(new XposedInterface.Hooker() {
                        @Override
                        public Object intercept(XposedInterface.Chain chain) throws Throwable {
                            return hooks.interceptFillResponse(chain);
                        }
                    });
            log(Log.INFO, TAG, "event=asi_runtime_smoke source_index="
                    + smoke.selectedSourceIndex + " score_milli=" + smoke.selectedScoreMilli
                    + " elapsed_ms=" + smoke.elapsedMs
                    + " intent_sha256=" + smoke.executionIntentSha256
                    + " result_identity=" + smoke.resultContentIdentity);
            log(Log.INFO, TAG, "Installed version-pinned ASI FillResponse hook");
            detach();
        } catch (Throwable error) {
            ExtensionGate.rethrowIfVmFatal(error);
            if (installedHook != null) {
                try {
                    installedHook.unhook();
                } catch (Throwable cleanupError) {
                    ExtensionGate.rethrowIfVmFatal(cleanupError);
                }
            }
            if (bridge != null) {
                bridge.close();
            }
            log(Log.ERROR, TAG, "ASI hook installation failed; original path retained", error);
            detach();
        }
    }

    private void initializeGsa(Context context) {
        XposedInterface.HookHandle installedHook = null;
        XposedInterface.HookHandle nanoAction = null;
        XposedInterface.HookHandle nanoSideStreams = null;
        List<XposedInterface.HookHandle> observations = null;
        try {
            GeminiHostSymbols.configure(context.getPackageManager().getPackageInfo(
                    ExtensionGate.GSA_PACKAGE, 0).getLongVersionCode());
            ClassLoader loader = context.getClassLoader();
            Class<?> inventory = Class.forName(
                    "com.google.android.appfunctions.schema.agent.internal."
                            + "SchemaFunctionInventory_Impl",
                    false, loader);
            Method getSchemaFunctionMap = inventory.getDeclaredMethod("getSchemaFunctionMap");
            getSchemaFunctionMap.setAccessible(true);
            final GeminiAppFunctionSchemaHooks hooks =
                    new GeminiAppFunctionSchemaHooks(loader);
            installedHook = hook(getSchemaFunctionMap)
                    .setId("ostadix-gemini/schema-function-inventory-v1")
                    .setExceptionMode(XposedInterface.ExceptionMode.PROTECTIVE)
                    .intercept(new XposedInterface.Hooker() {
                        @Override
                        public Object intercept(XposedInterface.Chain chain) throws Throwable {
                            return hooks.interceptInventory(chain);
                        }
                    });
            log(Log.INFO, TAG,
                    "Installed version-pinned Gemini AppFunction schema inventory hook");
            observations = GeminiRouteObservation.install(this, context);
            NanoActionClient.registerDiagnostic(context);
            nanoAction = GeminiNanoActionHooks.install(this, context);
            nanoSideStreams = GeminiNanoActionHooks.installSideStreams(this, context);
            Log.i(TAG, "event=nano_assistant_action_installed");
            Log.i(TAG, "event=gsa_hooks_installed observer_hooks=" + observations.size());
            detach();
        } catch (Throwable error) {
            ExtensionGate.rethrowIfVmFatal(error);
            Log.e(TAG, "event=gsa_hooks_failed", error);
            if (nanoAction != null) { nanoAction.unhook(); }
            if (nanoSideStreams != null) { nanoSideStreams.unhook(); }
            if (observations != null) {
                for (XposedInterface.HookHandle observation : observations) {
                    observation.unhook();
                }
            }
            if (installedHook != null) {
                try {
                    installedHook.unhook();
                } catch (Throwable cleanupError) {
                    ExtensionGate.rethrowIfVmFatal(cleanupError);
                }
            }
            log(Log.ERROR, TAG,
                    "Gemini AppFunction schema hook installation failed; original catalog retained",
                    error);
            detach();
        }
    }
}
