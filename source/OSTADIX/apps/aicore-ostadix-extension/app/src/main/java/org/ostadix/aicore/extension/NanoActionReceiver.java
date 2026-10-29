package org.ostadix.aicore.extension;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.os.ResultReceiver;
import android.util.Log;

/** GSA-to-AICore transport. Android supplies the sender identity; extras do not. */
final class NanoActionReceiver {
    static final String GENERATE = "org.ostadix.aicore.NANO_ACTION_GENERATE";
    static final String CANCEL = "org.ostadix.aicore.NANO_ACTION_CANCEL";
    private static final String TAG = "OstadixNanoAction";

    static void register(Context context) {
        IntentFilter filter = new IntentFilter(GENERATE);
        filter.addAction(CANCEL);
        context.registerReceiver(new BroadcastReceiver() {
            @Override public void onReceive(Context receiverContext, Intent intent) {
                if (intent == null) { return; }
                // The sender must opt into identity sharing with BroadcastOptions.
                // No fallback to an intent extra, caller label or guessed UID.
                int uid = getSentFromUid();
                String caller = getSentFromPackage();
                if (!ExtensionGate.acceptsGsaCaller(context, uid, caller)) {
                    Log.w(TAG, "event=caller_rejected uid=" + uid);
                    return;
                }
                String id = intent.getStringExtra("request_id");
                if (id == null || !id.matches("[A-Za-z0-9._:-]{1,80}")) { return; }
                if (CANCEL.equals(intent.getAction())) {
                    NanoLocalProbe.cancelAction(id);
                    return;
                }
                if (!GENERATE.equals(intent.getAction())) { return; }
                ResultReceiver reply = intent.getParcelableExtra("reply", ResultReceiver.class);
                if (reply == null) { return; }
                String prompt = intent.getStringExtra("prompt");
                if (prompt == null || prompt.isEmpty()
                        || prompt.getBytes(java.nio.charset.StandardCharsets.UTF_8).length > 16384) {
                    Bundle error = new Bundle();
                    error.putString("request_id", id);
                    error.putString("error", "prompt must contain 1..16384 UTF-8 bytes; not dispatched");
                    reply.send(1, error);
                    return;
                }
                Log.i(TAG, "event=caller_accepted request_id=" + id + " uid=" + uid
                        + " caller_package=" + caller + " identity_source=android_broadcast");
                NanoLocalProbe.submitAction(context, id, prompt, reply);
            }
        }, filter, null, null, Context.RECEIVER_EXPORTED);
        Log.i(TAG, "event=receiver_registered");
    }
}
