package org.ostadix.voicebridge;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.widget.TextView;

/** Visible entry point. Only an authenticated live terminal can request capture. */
public final class BridgeActivity extends Activity {
    private boolean dispatched;
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        TextView text = new TextView(this);
        text.setTextSize(22);
        text.setPadding(32, 56, 32, 32);
        text.setText("Ostadix voice\n\nOpen c in Termux to talk. Microphone capture is active only during an attached voice conversation.");
        setContentView(text);
        if (getIntent().getIntExtra("port", 0) == 0) return;
        if (getIntent().getBooleanExtra("probe", false)) {
            new Thread(() -> {
                VoiceService.probe(this, getIntent());
                runOnUiThread(this::finish);
            }, "ostadix-probe").start();
            return;
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO, Manifest.permission.POST_NOTIFICATIONS}, 1);
        } else startBridge();
    }
    private void startBridge() {
        if (dispatched) return;
        dispatched = true;
        Intent service = new Intent(this, VoiceService.class);
        service.putExtras(getIntent());
        stopService(new Intent(this, VoiceService.class));
        startForegroundService(service);
        finish();
    }
    @Override public void onRequestPermissionsResult(int request, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(request, permissions, results);
        if (request == 1 && checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) startBridge();
        else finish();
    }
}
