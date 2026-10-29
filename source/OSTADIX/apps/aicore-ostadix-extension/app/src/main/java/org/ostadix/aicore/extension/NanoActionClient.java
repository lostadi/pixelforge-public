package org.ostadix.aicore.extension;

import android.app.BroadcastOptions;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.CancellationSignal;
import android.os.ResultReceiver;
import android.os.SystemClock;

import org.json.JSONObject;

import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/** One local-model request; callers run this on an owned worker, never the UI thread. */
final class NanoActionClient {
    private static final AtomicBoolean DIAGNOSTIC_BUSY = new AtomicBoolean();
    static void registerDiagnostic(Context context) {
        context.registerReceiver(new android.content.BroadcastReceiver() {
            @Override public void onReceive(Context receiverContext, Intent intent) {
                String prompt = intent == null ? null : intent.getStringExtra("prompt");
                if (prompt == null || !ExtensionGate.isExplicitlyEnabled(context)) { return; }
                try {
                    StringBuilder digest = new StringBuilder();
                    for (byte value : java.security.MessageDigest.getInstance("SHA-256").digest(
                            prompt.getBytes(java.nio.charset.StandardCharsets.UTF_8))) {
                        digest.append(String.format(java.util.Locale.ROOT, "%02x", value & 255));
                    }
                    if (!digest.toString().equals(android.provider.Settings.Global.getString(
                            context.getContentResolver(), "ostadix_nano_action_probe_sha256"))) { return; }
                } catch (Exception failure) { return; }
                if (!DIAGNOSTIC_BUSY.compareAndSet(false, true)) { return; }
                Thread worker = new Thread(new Runnable() {
                    @Override public void run() {
                        JSONObject record = new JSONObject();
                        try {
                            record.put("caller_pid", android.os.Process.myPid());
                            record.put("caller_uid", android.os.Process.myUid());
                            record.put("ordinary_assistant_request", false);
                            record.put("result", generate(context, prompt, new CancellationSignal()));
                        } catch (Exception failure) {
                            try { record.put("error", failure.toString()); }
                            catch (org.json.JSONException ignored) { }
                        }
                        try (java.io.FileOutputStream output = context.openFileOutput(
                                "ostadix-nano-action-probe.json", Context.MODE_PRIVATE)) {
                            output.write(record.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
                            output.getFD().sync();
                            android.util.Log.i("OstadixNanoAction", "event=diagnostic_finished");
                        } catch (Exception failure) {
                            android.util.Log.e("OstadixNanoAction", "Diagnostic evidence write failed", failure);
                        } finally { DIAGNOSTIC_BUSY.set(false); }
                    }
                }, "OstadixNanoActionDiagnostic");
                try { worker.start(); }
                catch (RuntimeException failure) { DIAGNOSTIC_BUSY.set(false); throw failure; }
            }
        }, new android.content.IntentFilter("org.ostadix.gemini.NANO_ACTION_PROBE"),
                "android.permission.DUMP", null, Context.RECEIVER_EXPORTED);
    }

    static JSONObject generate(Context context, String prompt, CancellationSignal cancellation)
            throws Exception {
        if (!ExtensionGate.GSA_PACKAGE.equals(context.getPackageName())) {
            throw new IllegalArgumentException("original Google application context required");
        }
        final String id = "nano-action-" + UUID.randomUUID();
        final CountDownLatch ready = new CountDownLatch(1);
        final AtomicReference<Bundle> answer = new AtomicReference<>();
        final AtomicBoolean done = new AtomicBoolean();
        ResultReceiver reply = new ResultReceiver(null) {
            @Override protected void onReceiveResult(int code, Bundle data) {
                if (data != null && id.equals(data.getString("request_id"))
                        && done.compareAndSet(false, true)) {
                    answer.set(data);
                    ready.countDown();
                }
            }
        };
        Intent request = new Intent(NanoActionReceiver.GENERATE)
                .setPackage(ExtensionGate.AICORE_PACKAGE)
                .putExtra("request_id", id).putExtra("prompt", prompt).putExtra("reply", reply);
        cancellation.throwIfCanceled();
        AtomicBoolean sent = new AtomicBoolean();
        cancellation.setOnCancelListener(new CancellationSignal.OnCancelListener() {
            @Override public void onCancel() {
                if (sent.get()) { cancel(context, id); }
                ready.countDown();
            }
        });
        boolean received = false;
        try {
            cancellation.throwIfCanceled();
            // Identity sharing is required for the AICore receiver's UID/package gate.
            sent.set(true);
            context.sendBroadcast(request, null, BroadcastOptions.makeBasic()
                    .setShareIdentityEnabled(true).toBundle());
            long deadline = SystemClock.elapsedRealtime() + NanoLocalProbe.CLIENT_REPLY_TIMEOUT_MS;
            while (!ready.await(Math.min(1000L, Math.max(1L,
                    deadline - SystemClock.elapsedRealtime())), TimeUnit.MILLISECONDS)) {
                if (SystemClock.elapsedRealtime() >= deadline) {
                    throw new java.io.IOException("local Nano result timeout; completion unknown; no retry");
                }
                cancellation.throwIfCanceled();
            }
            cancellation.throwIfCanceled();
            // The process may have been suspended while the Binder reply or
            // latch was pending. Waking with a value does not extend the budget.
            if (SystemClock.elapsedRealtime() >= deadline) {
                throw new java.io.IOException("local Nano result arrived after deadline; no retry");
            }
            Bundle response = answer.get();
            if (response == null) {
                throw new java.io.IOException("local Nano cancelled before result; no retry");
            }
            received = true;
            String json = response.getString("result_json");
            if (json == null) {
                throw new java.io.IOException(response.getString("error", "local Nano omitted result; no retry"));
            }
            JSONObject result = new JSONObject(json);
            if (!id.equals(result.getString("request_id"))) {
                throw new java.io.IOException("local Nano response correlation mismatch; no retry");
            }
            return result;
        } finally {
            cancellation.setOnCancelListener(null);
            done.set(true);
            if (sent.get() && !received) { cancel(context, id); }
        }
    }

    private static void cancel(Context context, String id) {
        context.sendBroadcast(new Intent(NanoActionReceiver.CANCEL)
                .setPackage(ExtensionGate.AICORE_PACKAGE).putExtra("request_id", id), null,
                BroadcastOptions.makeBasic().setShareIdentityEnabled(true).toBundle());
    }
}
