# Local factory reader

Owned Java/Python code reconstructs the file-loading contract observed in the
installed AICore application on this device. It calls the existing local
`vendor.google.plat_security.ITrustyDecrypt/default` service. Google's native
implementation and model weights are reused, not rebuilt.

## Implementation

- `FactoryReader.java` accepts a basename under `/data/vendor/intelligence`,
  uses read-only input and shared-memory output descriptors, validates the
  decoded length footer, and writes one new private file.
- `materialize.py` follows the recovered manifest. Large files stay linked to
  the read-only factory mount. Smaller files use the reader. Every available
  result must match the manifest's byte size and SHA-1. Missing files are
  recorded; calls are not automatically retried.
- `run_native_probe.py` runs the separately built JNI loader with a process
  deadline, memory and thermal guards, and retained stdout/stderr. Read its
  JSON `exit_code` and probe events to determine success; the runner can finish
  successfully after recording a failed native probe.

Build with `bash tools/nano-factory-reader/build.sh`. The reader needs the
normal Android ART environment; `android_environment()` recovers only named
Android runtime variables from zygote. It does not export unrelated process
environment values. The native probe's staged DEX/JAR must be read-only.

These diagnostics currently run from a rooted host. A Unix app UID inherited
through `su` retains the KernelSU SELinux context; this does not test ordinary
Android application permissions.

## Live observations

See `audits/gemini-nano-rebuild-20260917/`: local decoding returned status 0;
191 available payloads match the recovered manifest. The remaining manifest
entry, `dvfs_manager_params.binarypb`, is absent. The native runtime contains a
fallback for its absence. Raw model files remain outside this repository.

The first Java/native model-load attempt reached the original loader and failed
obtaining an EdgeTPU service descriptor, then released the runtime. It did not
run inference. The earlier writable-DEX ART abort occurred before Java started
and was corrected by staging the probe JAR read-only.

## Unverified

Model loading, inference, public Nano release identity, ordinary app execution,
and the ordinary assistant-to-Nano-to-Ostadix chain have not been established by
these file-recovery results. The existing public Prompt API still reported
unavailable features in its separately retained tests.
