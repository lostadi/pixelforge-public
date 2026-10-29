package org.ostadix.voicebridge;

import android.app.*;
import android.content.*;
import android.content.pm.ServiceInfo;
import android.media.*;
import android.media.audiofx.*;
import android.os.*;
import android.speech.*;
import android.speech.tts.*;
import org.json.*;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;

/** Owns the microphone for one authenticated terminal session, including TTS playback. */
public final class VoiceService extends Service {
    static final String CHANNEL = "ostadix-live-voice";
    static final int NOTIFICATION = 401;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final ArrayBlockingQueue<String> outbound = new ArrayBlockingQueue<>(256);
    private final ArrayBlockingQueue<byte[]> pcm = new ArrayBlockingQueue<>(100);
    private volatile boolean running, capturing, closed, microphoneMuted;
    private volatile int captureGeneration;
    private boolean conversationStarted;
    private Socket socket;
    private BufferedWriter socketWriter;
    private final Object writeLock=new Object();
    private AudioRecord record;
    private AcousticEchoCanceler aec;
    private NoiseSuppressor ns;
    private SpeechRecognizer recognizer;
    private TextToSpeech tts;
    private boolean ttsReady, speaking, onDevice, segmentedSeen, readySeen;
    private volatile long frames, dropped;
    private long sequence;
    private int generation, failures;
    private PipeSink sink;
    private String language = "en-US";
    private String pendingSpeech, pendingSpeechId;
    private String previousPartial = "";
    private boolean speechStarted;
    private long lastReady, recognitionStarted;
    private boolean modeOwned;

    static JSONObject json(Object... values) {
        JSONObject data = new JSONObject();
        try { for (int i=0; i<values.length; i+=2) data.put((String)values[i], values[i+1]); }
        catch (JSONException ignored) {}
        return data;
    }
    static JSONObject capabilities(Context context) {
        return json("type", "capabilities", "protocol", 1, "sdk", Build.VERSION.SDK_INT,
            "on_device_available", SpeechRecognizer.isOnDeviceRecognitionAvailable(context),
            "recognition_available", SpeechRecognizer.isRecognitionAvailable(context),
            "aec_available", AcousticEchoCanceler.isAvailable(), "mic_active", false,
            "capture_continuous", false, "microphone_muted", false, "microphone_mute_supported", true, "source_acceptance", "not_tested",
            "segmented_confirmed", false, "package", context.getPackageName());
    }
    static void probe(Context context, Intent intent) {
        try (Socket connection = new Socket()) {
            int port = intent.getIntExtra("port", 0);
            String token = intent.getStringExtra("token");
            if (port < 1024 || port > 65535 || token == null || token.length() < 32) return;
            connection.connect(new InetSocketAddress("127.0.0.1", port), 5000);
            Writer writer = new OutputStreamWriter(connection.getOutputStream(), StandardCharsets.UTF_8);
            writer.write(json("type", "hello", "protocol", 1, "token", token).toString()+"\n");
            writer.write(capabilities(context).toString()+"\n");
            writer.flush();
        } catch (Exception ignored) {}
    }
    @Override public void onCreate() {
        super.onCreate();
        NotificationManager nm = getSystemService(NotificationManager.class);
        nm.createNotificationChannel(new NotificationChannel(CHANNEL, "Live voice conversation", NotificationManager.IMPORTANCE_LOW));
    }
    @Override public IBinder onBind(Intent intent) { return null; }
    @Override public int onStartCommand(Intent intent, int flags, int id) {
        if (intent == null || "stop".equals(intent.getAction())) { stopSelf(); return START_NOT_STICKY; }
        if (running) { stopSelf(); return START_NOT_STICKY; }
        int port = intent.getIntExtra("port", 0);
        String token = intent.getStringExtra("token");
        if (port < 1024 || port > 65535 || token == null || token.length() < 32) { stopSelf(); return START_NOT_STICKY; }
        language = intent.getStringExtra("language");
        if (language == null || language.length()>30) language = "en-US";
        Intent stop = new Intent(this, VoiceService.class).setAction("stop");
        PendingIntent action = PendingIntent.getService(this, 0, stop, PendingIntent.FLAG_IMMUTABLE);
        Notification notification = new Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now).setContentTitle("Ostadix voice conversation")
            .setContentText("Microphone stays on during replies. Tap Stop to end.").setOngoing(true)
            .addAction(new Notification.Action.Builder(null, "Stop", action).build()).build();
        startForeground(NOTIFICATION, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE | ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK);
        running = true;
        new Thread(() -> connect(port, token), "ostadix-voice-link").start();
        return START_NOT_STICKY;
    }
    private void connect(int port, String token) {
        try {
            socket = new Socket();
            socket.connect(new InetSocketAddress("127.0.0.1", port), 5000);
            socket.setSoTimeout(20000);
            socket.setTcpNoDelay(true);
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8));
            writer.write(json("type", "hello", "protocol", 1, "token", token).toString()+"\n");
            writer.flush();
            socketWriter=writer;
            new Thread(() -> {
                try {
                    while (running) {
                        String line = outbound.poll(2, TimeUnit.SECONDS);
                        if (line != null) { synchronized(writeLock) { writer.write(line); writer.write('\n'); writer.flush(); } }
                    }
                } catch (Exception error) { main.post(this::stopSelf); }
            }, "ostadix-voice-events").start();
            emit(capabilities(this));
            BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
            String line;
            while (running && (line = reader.readLine()) != null) {
                if (line.length()>32768) break;
                JSONObject command = new JSONObject(line);
                main.post(() -> command(command));
            }
        } catch (Exception error) { /* socket closes when terminal disappears */ }
        finally { main.post(this::stopSelf); }
    }
    private synchronized void emit(JSONObject data) {
        try { data.put("seq", ++sequence); data.put("time_ms", SystemClock.elapsedRealtime()); }
        catch (JSONException ignored) {}
        if (!outbound.offer(data.toString())) { outbound.poll(); outbound.offer(data.toString()); }
    }
    private void command(JSONObject command) {
        if (!running) return;
        switch (command.optString("cmd")) {
            case "ping": emit(json("type", "pong", "mic_active", capturing, "microphone_muted", microphoneMuted, "frames", frames, "dropped_frames", dropped)); break;
            case "start":
                conversationStarted=true;
                if(command.optBoolean("microphone_muted", false))microphoneMuted=true;
                initializeSpeech();
                if(microphoneMuted)emit(json("type", "ready", "mic_active", false, "microphone_muted", true));
                else if(!capturing)startCapture();
                emitStatus();
                break;
            case "microphone_muted": setMicrophoneMuted(command.optBoolean("muted", true)); break;
            case "speak": speak(command.optString("text"), command.optString("id", "answer")); break;
            case "stop_speaking": pendingSpeech=null; pendingSpeechId=null; if (tts != null) tts.stop(); speaking = false; emit(json("type", "tts_done", "interrupted", true)); break;
            case "status": emitStatus(); break;
            case "stop": stopSelf(); break;
        }
    }
    private void emitStatus() {
        AudioRecordingConfiguration config = record==null ? null : record.getActiveRecordingConfiguration();
        emit(json("type", "capabilities", "protocol", 1, "on_device_available", SpeechRecognizer.isOnDeviceRecognitionAvailable(this),
            "recognizer", onDevice ? "android_on_device" : "android_default", "mic_active", capturing,
            "capture_continuous", capturing, "microphone_muted", microphoneMuted, "microphone_mute_supported", true, "audio_mode_owned", modeOwned, "mic_silenced", config==null ? JSONObject.NULL : config.isClientSilenced(), "sample_rate", 16000, "audio_source", "VOICE_COMMUNICATION",
            "aec_available", AcousticEchoCanceler.isAvailable(), "aec_enabled", aec != null && aec.getEnabled(),
            "noise_suppression_enabled", ns != null && ns.getEnabled(), "source_acceptance", readySeen ? "recognizer_ready" : "not_confirmed",
            "segmented_confirmed", segmentedSeen, "tts_engine", "com.google.android.tts", "frames", frames, "dropped_frames", dropped, "tts_ready", ttsReady, "speaking", speaking, "tts_voice", ttsReady && tts.getVoice()!=null ? tts.getVoice().getName() : JSONObject.NULL, "tts_pitch", 1.08, "tts_rate", 0.95));
    }
    private void startCapture() {
        if(!running || capturing || microphoneMuted)return;
        try {
            AudioManager manager = getSystemService(AudioManager.class);
            // Acquire a communication-mode request only for active capture.
            // An existing mode selected by another app is never adopted.
            if (!modeOwned && manager.getMode() == AudioManager.MODE_NORMAL) {
                manager.setMode(AudioManager.MODE_IN_COMMUNICATION); modeOwned = true;
            }
            int min = AudioRecord.getMinBufferSize(16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
            if (min < 0) throw new IOException("16 kHz microphone format is unavailable");
            record = new AudioRecord.Builder().setAudioSource(MediaRecorder.AudioSource.VOICE_COMMUNICATION)
                .setAudioFormat(new AudioFormat.Builder().setSampleRate(16000).setChannelMask(AudioFormat.CHANNEL_IN_MONO).setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
                .setBufferSizeInBytes(Math.max(min*2, 6400)).setPrivacySensitive(true).build();
            if (record.getState()!=AudioRecord.STATE_INITIALIZED) throw new IOException("Android could not initialize the microphone");
            if (AcousticEchoCanceler.isAvailable()) { aec = AcousticEchoCanceler.create(record.getAudioSessionId()); if (aec != null) aec.setEnabled(true); }
            if (NoiseSuppressor.isAvailable()) { ns = NoiseSuppressor.create(record.getAudioSessionId()); if (ns != null) ns.setEnabled(true); }
            record.startRecording();
            capturing = true;
            final AudioRecord activeRecord=record;
            final int currentCapture=++captureGeneration;
            new Thread(() -> capture(activeRecord,currentCapture), "ostadix-continuous-mic").start();
            onDevice = SpeechRecognizer.isOnDeviceRecognitionAvailable(this);
            if (!onDevice) throw new IOException("Android on-device speech recognition is unavailable; no online fallback was started");
            beginRecognition();
            emitStatus();
        } catch (Exception error) {
            emit(json("type", "error", "scope", "capture", "message", error.toString(), "fatal", true));
            main.postDelayed(this::stopSelf, 250);
        }
    }
    private void initializeSpeech() {
        if(tts!=null)return;
        try {
            tts = new TextToSpeech(this, status -> main.post(() -> {
                if(!running || closed)return;
                ttsReady = status==TextToSpeech.SUCCESS;
                if (ttsReady) {
                    tts.setAudioAttributes(new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build());
                    tts.setLanguage(Locale.forLanguageTag(language));
                    Voice selected = null;
                    Voice preferred = null;
                    if (tts.getVoices()!=null) for (Voice voice : tts.getVoices()) {
                        if (!voice.isNetworkConnectionRequired() && voice.getLocale().toLanguageTag().equalsIgnoreCase(language)) {
                            if (selected==null || voice.getQuality()>selected.getQuality()) selected=voice;
                            if ("en-us-x-iom-local".equals(voice.getName())) preferred=voice;
                        }
                    }
                    if (preferred!=null) selected=preferred;
                    if (selected!=null) {
                        ttsReady=tts.setVoice(selected)==TextToSpeech.SUCCESS;
                        if (ttsReady) {
                            // User-selected softer delivery, confined to this app's TTS instance.
                            tts.setPitch(1.08f);
                            tts.setSpeechRate(0.95f);
                        }
                    }
                    else {
                        ttsReady=false;
                        emit(json("type", "error", "scope", "tts", "message", "No installed offline speech voice matches "+language));
                    }
                    tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                        @Override public void onStart(String id) { main.post(() -> { speaking=true; emit(json("type", "tts_start", "id", id)); }); }
                        @Override public void onDone(String id) { main.post(() -> { speaking=false; emit(json("type", "tts_done", "id", id)); }); }
                        @Override public void onError(String id) { main.post(() -> { speaking=false; emit(json("type", "error", "scope", "tts", "message", "Android speech playback failed", "id", id)); }); }
                        @Override public void onStop(String id, boolean interrupted) { main.post(() -> { speaking=false; emit(json("type", "tts_done", "id", id, "interrupted", true)); }); }
                    });
                }
                emitStatus();
                if (ttsReady && pendingSpeech!=null) {
                    String text=pendingSpeech, id=pendingSpeechId;
                    pendingSpeech=null; pendingSpeechId=null;
                    speak(text,id);
                } else if (!ttsReady) emit(json("type", "error", "scope", "tts", "message", "The offline Android speech engine is unavailable"));
            }), "com.google.android.tts");
        } catch (Exception error) {
            emit(json("type", "error", "scope", "tts", "message", error.toString()));
        }
    }
    private void setMicrophoneMuted(boolean muted) {
        microphoneMuted=muted;
        if(muted)releaseCapture();
        else if(conversationStarted && !capturing)startCapture();
        // AudioRecord and the recognizer have been released before confirming
        // mute. Playback belongs to the still-connected service and is untouched.
        emit(json("type", "microphone_muted", "muted", microphoneMuted, "mic_active", capturing, "audio_mode_owned", modeOwned));
        emitStatus();
    }
    private void releaseAudioMode() {
        if(!modeOwned)return;
        try {
            // MODE_NORMAL removes this caller's AudioService mode request; it
            // does not restore a cached global mode or replace another owner's
            // request. Release before acknowledging mute, while TTS stays alive.
            getSystemService(AudioManager.class).setMode(AudioManager.MODE_NORMAL);
            modeOwned=false;
        } catch(Exception error) {
            emit(json("type", "error", "scope", "audio_mode", "message", "Microphone capture stopped, but Android did not release its communication mode: "+error));
        }
    }
    private void releaseCapture() {
        capturing=false; captureGeneration++; generation++;
        if(recognizer!=null) { try { recognizer.cancel(); recognizer.destroy(); } catch(Exception ignored) {} recognizer=null; }
        if(sink!=null) { sink.close(); sink=null; }
        AudioRecord previous=record; record=null;
        if(previous!=null) { try { previous.stop(); } catch(Exception ignored) {} previous.release(); }
        if(aec!=null) { aec.release(); aec=null; }
        if(ns!=null) { ns.release(); ns=null; }
        pcm.clear(); previousPartial=""; speechStarted=false; readySeen=false;
        releaseAudioMode();
    }
    private void capture(AudioRecord activeRecord, int currentCapture) {
        try {
            while (capturing && running && !microphoneMuted && currentCapture==captureGeneration) {
                byte[] buffer = new byte[640]; // 20 ms PCM16 mono, never written to disk.
                int count = activeRecord.read(buffer, 0, buffer.length, AudioRecord.READ_BLOCKING);
                if(!capturing || !running || microphoneMuted || currentCapture!=captureGeneration)break;
                if (count < 0) throw new IOException("Microphone read failed: "+count);
                if (count == 0) continue;
                if (count != buffer.length) buffer = Arrays.copyOf(buffer, count);
                frames += count/2;
                if (!pcm.offer(buffer)) { pcm.poll(); pcm.offer(buffer); dropped += count/2; }
            }
        } catch (Exception error) {
            if (running && capturing && !microphoneMuted && currentCapture==captureGeneration) main.post(() -> {
                if(!running || !capturing || microphoneMuted || currentCapture!=captureGeneration)return;
                emit(json("type", "error", "scope", "capture", "message", error.toString(), "fatal", true)); main.postDelayed(this::stopSelf,250);
            });
        }
    }
    private final class PipeSink {
        final ParcelFileDescriptor read, write;
        final OutputStream output;
        volatile boolean active = true;
        Thread worker;
        PipeSink() throws IOException {
            ParcelFileDescriptor[] pipe = ParcelFileDescriptor.createPipe();
            read=pipe[0]; write=pipe[1]; output=new ParcelFileDescriptor.AutoCloseOutputStream(write);
        }
        void start() {
            worker = new Thread(() -> {
                try { while (active && running) { byte[] frame=pcm.poll(1,TimeUnit.SECONDS); if (frame!=null && active) output.write(frame); } }
                catch (Exception ignored) {}
            }, "ostadix-asr-pcm");
            worker.start();
        }
        void close() {
            active=false;
            try { output.close(); } catch (IOException ignored) {}
            try { read.close(); } catch (IOException ignored) {}
            if(worker!=null)worker.interrupt();
        }
    }
    private void beginRecognition() {
        if (!running || !capturing || microphoneMuted) return;
        int current = ++generation;
        try {
            if (sink!=null) { sink.close(); sink=null; }
            if (recognizer!=null) { recognizer.destroy(); recognizer=null; }
            previousPartial=""; speechStarted=false;
            recognizer=onDevice ? SpeechRecognizer.createOnDeviceSpeechRecognizer(this) : SpeechRecognizer.createSpeechRecognizer(this);
            sink=new PipeSink();
            Intent request=new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
                .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                .putExtra(RecognizerIntent.EXTRA_LANGUAGE, language)
                .putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                .putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
                .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE, sink.read)
                .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_CHANNEL_COUNT, 1)
                .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_ENCODING, AudioFormat.ENCODING_PCM_16BIT)
                .putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_SAMPLING_RATE, 16000)
                .putExtra(RecognizerIntent.EXTRA_SEGMENTED_SESSION, RecognizerIntent.EXTRA_AUDIO_SOURCE)
                .putExtra(RecognizerIntent.EXTRA_MASK_OFFENSIVE_WORDS, false);
            recognizer.setRecognitionListener(new RecognitionListener() {
                private boolean valid() { return running && capturing && !microphoneMuted && current==generation; }
                @Override public void onReadyForSpeech(Bundle params) {
                    if (!valid())return;
                    readySeen=true; lastReady=SystemClock.elapsedRealtime(); failures=0;
                    emit(json("type", "ready", "mic_active", true, "microphone_muted", false, "recognizer", onDevice ? "android_on_device" : "android_default")); emitStatus();
                }
                @Override public void onBeginningOfSpeech() { if(valid()) speechStart(); }
                @Override public void onEndOfSpeech() { if(valid()) emit(json("type", "speech_end")); }
                @Override public void onRmsChanged(float rms) {}
                @Override public void onBufferReceived(byte[] data) {}
                @Override public void onEvent(int event, Bundle params) {}
                @Override public void onPartialResults(Bundle results) { if(valid()) transcript(results,false); }
                @Override public void onResults(Bundle results) {
                    if(!valid())return;
                    transcript(results,true);
                    scheduleRecognition(120, current);
                }
                @Override public void onSegmentResults(Bundle results) {
                    if(!valid())return;
                    if(!segmentedSeen) { segmentedSeen=true; emitStatus(); }
                    transcript(results,true);
                }
                @Override public void onEndOfSegmentedSession() { if(valid()) scheduleRecognition(120,current); }
                @Override public void onError(int error) {
                    if(!valid())return;
                    failures++;
                    boolean transientError=error==SpeechRecognizer.ERROR_NO_MATCH || error==SpeechRecognizer.ERROR_SPEECH_TIMEOUT;
                    emit(json("type", "error", "scope", "recognition", "code", error, "recovering", true,
                        "message", "Android recognition error "+error, "silence", transientError));
                    scheduleRecognition(transientError ? 200 : Math.min(10000,500*failures),current);
                }
            });
            recognitionStarted=SystemClock.elapsedRealtime();
            recognizer.startListening(request);
            sink.start();
        } catch (Exception error) {
            emit(json("type", "error", "scope", "recognition", "message", error.toString(), "recovering", true));
            scheduleRecognition(2000,current);
        }
    }
    private void scheduleRecognition(long delay, int current) {
        if(current!=generation)return;
        generation++;
        if(sink!=null) { sink.close(); sink=null; }
        final int scheduled=generation;
        main.postDelayed(() -> { if(running && capturing && !microphoneMuted && scheduled==generation)beginRecognition(); },delay);
    }
    private void speechStart() {
        if(!capturing || microphoneMuted || speechStarted)return;
        speechStarted=true;
        emit(json("type", "speech_start", "during_playback", speaking));
    }
    private void transcript(Bundle results, boolean finalResult) {
        if(!capturing || microphoneMuted)return;
        ArrayList<String> texts=results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
        if(texts==null || texts.isEmpty())return;
        String text=texts.get(0).trim();
        if(text.isEmpty())return;
        if(!finalResult && text.equals(previousPartial))return;
        speechStart();
        previousPartial=finalResult ? "" : text;
        emit(json("type", finalResult ? "final" : "partial", "text", text, "during_playback", speaking));
        if(finalResult)speechStarted=false;
    }
    private void speak(String text, String id) {
        if(text==null || text.trim().isEmpty())return;
        if(!ttsReady) { pendingSpeech=text; pendingSpeechId=id; emit(json("type", "tts_waiting", "id", id)); return; }
        int rc=tts.speak(text.substring(0,Math.min(text.length(),3900)),TextToSpeech.QUEUE_FLUSH,new Bundle(),id);
        if(rc!=TextToSpeech.SUCCESS)emit(json("type", "error", "scope", "tts", "message", "Android refused speech playback", "id", id));
    }
    @Override public void onDestroy() {
        if(closed) { super.onDestroy(); return; }
        closed=true; running=false;
        main.removeCallbacksAndMessages(null);
        releaseCapture();
        if(tts!=null) { tts.stop(); tts.shutdown(); }
        // Android forbids socket writes on the main lifecycle thread. Notify
        // after all audio handles are released, before the service loses its
        // foreground lifetime, and keep this join bounded.
        Thread notifier = new Thread(() -> {
            try {
                synchronized(writeLock) {
                    if(socketWriter!=null) {
                        socketWriter.write(json("type","stopped","mic_active",false,"speaking",false).toString());
                        socketWriter.write('\n'); socketWriter.flush();
                    }
                }
            } catch(IOException ignored) {}
            finally { try { if(socket!=null)socket.close(); } catch(IOException ignored) {} }
        }, "ostadix-audio-released");
        notifier.start();
        try { notifier.join(1500); } catch(InterruptedException ignored) { Thread.currentThread().interrupt(); }
        stopForeground(STOP_FOREGROUND_REMOVE);
        super.onDestroy();
    }
}
