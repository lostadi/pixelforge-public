# Android voice bridge source

Package `org.ostadix.voicebridge` provides a foreground microphone, on-device
recognition and offline TTS bridge for the Termux interface. Opening the interface
in text mode or importing the bridge does not start capture. A live authenticated
session owns the microphone; socket loss, expiry or the notification Stop action
releases it. Android displays its microphone indicator while capture is active.

## Build locally

Supply a JDK, `d8`, `aapt2`, `apksigner`, and an Android SDK platform jar. Set
`ANDROID_JAR` directly or configure `ANDROID_HOME`/`ANDROID_SDK_ROOT`. Tools are
resolved from the Termux prefix or `PATH`; no SDK is downloaded:

```sh
python3 android_voice/build.py
```

The APK and classes are written to the ignored `build/` directory. A local
`local-signing.p12` is generated for development updates and is never distributed.
The development key's password is not a production signing-security mechanism.
Use the Android package installer for the resulting APK. `--install` is an
explicit rooted-Android option that installs this package and grants its
microphone/notification permissions; building alone does not do that or record.

## Transport and limitations

The Python interface and helper authenticate with a fresh per-session token.
PCM16 audio stays in the helper and feeds the installed on-device recognizer.
There is no automatic online recognition fallback. The speech engine, language
availability, voice selection, acoustic echo cancellation and overlapping-speech
accuracy depend on the device. Reported capability or an NPU backend flag is not
proof of speech quality or that ASR executes on an NPU.

Run the fake-peer transport tests without audio or an Android device:

```sh
python3 -m unittest android_voice.test_bridge -q
```

`verify_stop.py` is a separate hardware probe; read it before running it on a phone.
No probe output, signing material, APK, captured audio or device report is included.
