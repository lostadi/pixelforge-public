# OSTADIX AICore Smart Reply experiment

Normal Gemini requests retain Google’s original behavior. Start a plain typed
request with `Use Ostadix` to select the local Nano → Ostadix route. For the
persistent result viewer and repeatable deployment, see
[Using the assistant](ASSISTANT-USE.md). Smart Reply candidate selection and ASI
autofill experiments are disabled by default; they separately require the global
settings `ostadix_asoss_smart_reply_experiment` and
`ostadix_asi_autofill_experiment`, respectively, to equal `enabled-v1`, in addition
to the existing activation and compatibility gates.

This is an explicitly activated libxposed API 102 experiment for the
exact installed AS.OSS/AICore versions documented in
`audits/pixel-ai-20260915/OSTADIX-AICORE-INTEGRATION.md`.

It is not a supported Google plug-in. Its static scope contains the four exact
hosts used by the experiment: `com.google.android.as.oss` for the AICore result
callback and `com.google.android.as` for the conventional ASI Smart Reply
response path, plus the `:search` process of
`com.google.android.googlequicksearchbox` for Gemini's AppFunctions schema
inventory, and `com.google.android.aicore` for an explicitly requested local
factory-model loading diagnostic. All paths are guarded by pinned firmware, package, APK, signer,
framework API, activation, and thermal checks.

The ASI path hooks the installed `jht.d` response seam after candidate providers
finish and before ASI creates Android `Dataset` and `FillResponse` objects. It
passes only candidate order, text presence, and an already accepted safety bit
to OSTADIX. Generated text stays in the original ASI candidate object. A valid
selection is returned as a singleton list to ASI's original response builder;
an error calls the original method with the original list.
At runtime it rejects any firmware, version, base-APK hash, or signer mismatch
and requires both the exact `ostadix_aicore_extension_token` global setting and
the corresponding candidate-experiment opt-in before installing hooks.
Both settings are rechecked at request entry and completion, so removing either
stops result transformation even before the host process is restarted.

Reverse engineering of the live ASI flags selects its enabled Smart Reply route
(GenAiInferenceService transaction 6), rather than the disabled open-prompt LLM
route (transaction 5). When enabled on the pinned build, four protected hooks
preserve that active path:

```text
flw.c(SmartReplyRequest, flv)          capture original Binder caller UID
fna.onSmartReplyInferenceSuccess       run OSTADIX and replace SmartReplyResult
fna.onSmartReplyInferenceFailure       record failure and release correlation
flo.a()                                propagate cancellation to both runtimes
```

The installed host descriptors are the default-package names `Lflv;`, `Lflw;`,
`Lfna;`, and `Lflo;`; JADX's `defpackage` directory is only a source-display
convention. Because libxposed reports package readiness before Android attaches
the package `Application`, a one-shot `Application.attach` bootstrap obtains
the real AS.OSS context and class loader, then unhooks itself. Hook installation
keeps every returned handle so a later registration failure can unhook the
earlier handles and close the runtime.

The replacement preserves the inference trace and narrows the reply list to the
typed index chosen by OSTADIX. Its dedicated Smart Reply JNI contract gives O
only `score_milli`, `has_text`, and the exact `SafetyClassificationResult` for
each source index; value 0 is eligible and every other value is treated as
unsafe. Any
gate, thermal check, JNI, timeout, reflection, or OSTADIX failure calls the
original AS.OSS completion unchanged. Generated text, model sessions, FDs, and
native buffers never enter OSTADIX; only bounded scalar reply metadata does.
The JNI library and O/Bash launchers run directly from immutable module package
storage; module initialization creates no file in AS.OSS private storage.
An explicitly enabled candidate experiment runs one fixed scalar Smart Reply
smoke before hook registration. Hooks are installed only when that in-process
OSTADIX execution selects source 1 at score 950. With its experiment opt-in
absent, ASI or AS.OSS detaches before loading the native runtime or running the smoke.

The Gemini hook adds the `ostadix/executeO/1` schema to the installed Google
AppFunctions agent inventory. The stock lister drops a function whose schema
is absent from that compiled inventory, even when Android has indexed and
enabled it. The hook returns a copied immutable map with one source-first
Ostadix descriptor. Android brokers the AppFunction request to the standalone
`org.ostadix.terminal` AppFunctionService, which delegates over pinned local HTTPS
to the configured MCP host. Ostadix code executes under that host's Termux UID
and SELinux domain. The explicit local Nano route uses the same host directly.

## Local Nano loading diagnostic

Version 3 also registers a diagnostic receiver in the **real AICore main
process**, with its original class loader, UID and SELinux context. It reuses
the installed native libraries and recovered factory model; this does not
recompile Google's native implementation or retrain weights. The previously
tested external loader was denied by the vendor TPU service's caller checks.

The dedicated activation setting is `ostadix_nano_local_probe_token`, with
value `local-factory-234-10745-v1`. An explicit
`org.ostadix.aicore.NANO_LOCAL_PROBE` broadcast requires the sender's `DUMP`
permission. Input is the host-staged file
`files/ostadix-nano-probe/manifest.json` in AICore's private directory. Loading
runs once on a worker thread, with no automatic retry; root's separate watchdog
bounds the native call. The optional tokenizer check and model/runtime cleanup
are recorded in `events.jsonl` with request ID, process ID and UID.

This diagnostic does not register a public Nano feature or attach ordinary
assistant requests. Loading, generation and the assistant/tool connection must
each be demonstrated separately. The retained reconstruction observations are
in `audits/gemini-nano-rebuild-20260917/`.

Build after the Android terminal artifacts exist:

```bash
./apps/aicore-ostadix-extension/build.sh
```

The build does not install the APK or alter Vector configuration. The local ASI
provider 12 path is proven on-device through the caller's framework callback;
see `audits/pixel-ai-20260915/OSTADIX-ASI-ACCEPTED-REQUEST-20260917.txt`.
The separate AICore provider 25 path remains unproven because AICore has no
server-authorized model feature on this device.

`verify-source-policy.sh` also makes the build fail if any shared AICore or
AS.OSS version/APK/signer value diverges from the live compatibility policy;
the build additionally pins ASI's version, APK, signer, and runtime seam.
All seven packaged native files are separately hash-pinned and their manifest
is embedded as `META-INF/ostadix/native-sha256.txt`.
