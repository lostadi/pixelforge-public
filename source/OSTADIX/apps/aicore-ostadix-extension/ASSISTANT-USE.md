# Using local Nano and Ostadix on the configured Pixel

## Daily use

1. Invoke Gemini with the usual assistant gesture or power button.
2. Use Gemini normally for ordinary requests. To select local Nano and Ostadix,
   start that request with **Use Ostadix**, for example:
   `Use Ostadix: compute 17 + 25 and return the result.` The enabled `local-o-v1`
   route requires this prefix on each local request, including follow-ups.
3. Open **Ostadix Results** from the app drawer, or tap its result notification.
   The saved output appears first. **All results** opens earlier requests.
4. **Copy answer** copies the displayed answer. **Show program and execution
   details** reveals the generated source, actual tool result and evidence.

Ordinary requests continue through Google's normal Gemini flow. Google's existing
tool inventory remains available, with Ostadix's AppFunction added to it. Local
selection does not apply to an entire chat or persist to an unprefixed follow-up.
The older `local-all-v1` catch-all setting is no longer supported.

The viewer is local history. It does not populate Google's cloud conversation
history. Closing Gemini does not delete saved records. A cancelled request can
save a stopped/error record; cancellation does not guarantee rollback of effects.
An interrupted process may leave its last recorded stage without a final result.
Records persist across app restarts/updates until app data is cleared or the app
is uninstalled. They are excluded from Android backup.

The displayed **Ostadix output** comes from the runtime. **Nano's explanation**
is generated separately. A program can execute successfully and still calculate
the wrong thing: the September 18 test caught Nano duplicating `17 + 25` and
adding the copies. Its actual output, 84, remains in history; the host did not
replace it with an expected answer. Generation guidance was subsequently changed.
Another generated document was bare Python, which O returned as text. The adapter
now rejects missing executable block delimiters before execution and can request
its one correction. Model explanations are explicitly labeled unverified.

The adapter gives Nano the complete current request and bounded prior turns from
the same chat. Oversized required input is rejected before generation, rather
than silently truncated. State needed inputs explicitly when starting a new chat. Voice, attachments, Gemini Live, arbitrary external services,
and unrestricted task correctness have not been established by the text tests.

## Rebuild and redeploy this configuration

This is a repeatable procedure for the **already configured, pinned Pixel**,
not an installer for an arbitrary phone. It requires the existing Vector module
scope and activation, recovered Nano factory assets, original compatible Google
packages, private host configuration, native artifacts, and signing keystore.
Google's model and AICore binaries are reused; this does not rebuild them.

Use the canonical checkout, not the older `Ostadix-lang` directory:

```bash
cd /data/data/com.termux/files/home/OSTADIX
bash apps/aicore-ostadix-extension/build.sh
bash apps/aicore-ostadix-extension/test-result-output.sh
python -m unittest discover -s tests -p test_appfunction_host.py -v
```

The build verifies Android SDK/tool availability, compile-only libxposed API
102's hash, all seven native files, Google package policy and APK signing. Its
native closure comes from the existing Android terminal build. Follow that
app's build instructions if those artifacts are absent; changed hashes require
compatibility investigation, not blindly updating the pins.

From the authorized root terminal, after requests have finished:

```bash
python scripts/ostadix_assistant_results.py install
python scripts/ostadix_assistant_service.py install
python scripts/ostadix_assistant_service.py start
python scripts/ostadix_assistant_service.py check
python scripts/ostadix_assistant_results.py open
```

The result installer requires recorded Nano cleanup and a terminal assistant
turn before replacing the APK. It grants result notifications, verifies the
installed APK hash, restarts the identified AICore main, GSA `:search`, ASI and
AS.OSS processes so earlier hooks cannot survive the upgrade, and recovers
earlier private turn records without executing them. Recovery can
be run again with `python scripts/ostadix_assistant_results.py recover`.

The service installer preserves existing credentials. The root-owned boot
entry waits for boot completion and unlocked Termux storage, then launches the
MCP host as the configuration owner's Termux UID. Its supervisor restarts a
crashed host, but never resubmits requests. Repeating installation with identical
scripts and repeating start are supported. Upgrading deployed host scripts while
the old supervisor is running is refused. Physical reboot recovery remains a
separate check; invoking the boot entry on an unlocked phone has been tested.

`start` enables only the explicit `local-o-v1` route and replaces any older
catch-all setting, including when the host is already running. APK installation
alone retains the existing setting; run `start` as shown above to migrate it.

`check` executes a new harmless `1 + 1` program over pinned local HTTPS and prints
the actual typed result and execution evidence. It does not test Nano generation.

## Diagnosis and disabling the route

```bash
python scripts/ostadix_assistant_service.py status
python scripts/check_assistant_permissions.py --probe-transport
python scripts/ostadix_assistant_service.py check
python scripts/ostadix_assistant_service.py disable
```

`disable` removes local selection for future assistant requests and leaves the
MCP service and saved results available. `start` enables explicit **Use Ostadix**
selection again; ordinary Gemini requests continue normally. It
does not revoke an already executing request. The existing explicit activation,
version/hash/signature, thermal and cancellation gates still apply.

Service metadata and logs are in `~/.config/ostadix/gemini/`; configuration there
contains credentials and must not be shared. Boot files are
`/data/adb/service.d/40-ostadix-assistant.sh` and
`/data/local/ostadix-assistant/`. History is private app data; other apps do not
receive a read API. Its writer verifies the pinned GSA identity. Notifications
respect Android notification/channel settings.

## Execution behavior and limits

Nano returns a complete source document. The host uses primary MCP `o_execute`
for parse-only validation, then executes once. A confirmed parse or source-contract
rejection can request one corrected source from Nano and validate it again.
Source checks include executable blocks, required initial bindings, and the
syntax and result capture of a sole static Python block. Execution
errors, transport errors and timeouts do not cause automatic execution retries.
A normal turn therefore has two MCP calls (check, execute); a corrected turn
has three (check, check, execute). These are not extra execution attempts.

The host owns deadlines, credentials and placement. Current generation is
bounded to 2048 tokens, parse checking to 15 seconds and execution to 120 seconds;
this assistant adapter is not proof that every Ostadix workload fits those limits.
History records are limited to 16 MiB and a save failure is reported in the
answer when delivery is possible. Notification previews are abbreviated; the
viewer keeps the full saved answer. Model-generated code runs with the existing
host permissions. Parse validation checks syntax, not intent or safety.

The dated 2026-09-18 Nano audit is part of the original private device work;
it is not included in this source export. This procedure describes that
integration and is not fresh inference evidence for the public release.

## Package updates and permission diagnosis

The repaired integration accepts the exact reviewed Google app 17.56.15 and
17.58.16 APKs, and the exact reviewed AICore RC11 and RC13 APKs. Package identity
and signer checks remain enforced. Each Google version selects its own checked
class and coroutine method names. A different update requires a compatibility
review; granting additional Android permissions cannot repair changed methods.

`check_assistant_permissions.py --probe-transport` checks every permission
required by this integration, reports all declared permission states, validates
Google/AICore and installed integration identities, checks private credential
ownership and agreement, checks service/boot ownership, and proves unauthenticated
transport calls are rejected. It prints no tokens and dispatches no program.
Optional Google app permissions are reported separately and are not required
for typed local Ostadix requests. The route check requires `local-o-v1` and flags
the older catch-all configuration as unsafe; a disabled route is reported as
not ready for explicit local requests.

Error cards show the underlying failure briefly. **Show program and execution
details** retains the complete program, MCP result, and traceback. Diagnostic
Python files use private unique names so earlier root-owned files cannot block
error reporting. Disconnecting a request sends a single cancellation and waits
for settlement; it never repeats the execution.
