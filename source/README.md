# Integration source

The voice interface is exported as reviewable application source in
[ostadix-voice](ostadix-voice/README.md). It includes the PixelForge conversation
adapter, validated host-tool workflow, Android bridge source, cancellation, and
main-generation timeout changes. The repository's `SOURCE_MANIFEST.json` records
exported source hashes; it does not contain model weights or deployment state.

| Directory | Purpose |
| --- | --- |
| `ostadix-voice/` | Python 3.11+ terminal/browser interface, model clients and offline tests |
| `ostadix-voice/android_voice/` | Android microphone/recognition/TTS bridge source |
| `ostadix-voice/runtime/npu/` | LiteRT native-host source, headers and deterministic protocol test |
| `device/` | Earlier device recovery integration source; installation is optional |
| `OSTADIX/` | Captured OSTADIX Android/AppFunction integration source |

Install the source `c` shortcut without unpacking the older private payload:

```sh
python3 source/ostadix-voice/install_shortcut.py --shell both
```

That command installs only a launcher and optional shell aliases. It preserves
existing user commands/configuration and does not start a model, audio service,
root watchdog, or download. Engine setup remains explicit: see the
[voice setup guide](ostadix-voice/README.md) and [PixelForge chat guide](../chat/README.md).
Runtime profiles, model banks, compiler/runtime binaries, credentials, signing
keys, conversation stores, user history, and local verification receipts are not
part of the source export. Diagnostic scripts require explicit `--live` opt-in
where applicable and are not run by the offline test commands.

The Python interface's base offline suite uses only the standard library. Android
builds additionally need local SDK/JDK tools; native protocol tests need a C
compiler. Complete OSTADIX Android builds need the upstream compiler checkout,
SDK, libxposed API AAR and applicable runtime artifacts. Running a Tensor NPU model
requires compatible device hardware/vendor services and locally compiled banks;
source availability alone does not establish hardware execution or performance.

The OSTADIX subtree retains its original license and notices. The included
LiteRT C API headers retain their Apache-2.0 notices and license. See the repository
license and third-party notices for the complete distribution terms.
