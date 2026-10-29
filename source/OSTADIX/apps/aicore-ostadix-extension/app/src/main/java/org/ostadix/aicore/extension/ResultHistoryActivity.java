package org.ostadix.aicore.extension;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.database.ContentObserver;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.provider.Settings;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import java.io.File;
import java.text.DateFormat;
import java.util.Arrays;
import java.util.Date;
import org.json.JSONObject;

/** Persistent local answer history, independent of Gemini's server conversation history. */
public final class ResultHistoryActivity extends Activity {
    private LinearLayout page;
    private String selected;
    private int shown = 30;
    private final ContentObserver observer = new ContentObserver(new Handler()) {
        @Override public void onChange(boolean selfChange) { render(); }
    };

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        selected = saved == null ? getIntent().getStringExtra("request_id") : saved.getString("request_id");
        getContentResolver().registerContentObserver(Uri.parse("content://" + ResultHistory.AUTHORITY), true, observer);
        render();
    }
    @Override protected void onResume() { super.onResume(); render(); }
    @Override protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent); setIntent(intent); selected = intent.getStringExtra("request_id"); render();
    }
    @Override protected void onSaveInstanceState(Bundle state) { super.onSaveInstanceState(state); state.putString("request_id", selected); }
    @Override protected void onDestroy() { getContentResolver().unregisterContentObserver(observer); super.onDestroy(); }
    @Override public void onBackPressed() { if (selected != null) { selected = null; render(); } else { super.onBackPressed(); } }

    private int dp(int n) { return Math.round(n * getResources().getDisplayMetrics().density); }
    private void label(String value, int size) {
        TextView view = new TextView(this); view.setText(value); view.setTextSize(size);
        view.setTextColor(Color.rgb(24, 35, 47)); view.setTextIsSelectable(true); view.setPadding(0, dp(8), 0, dp(12));
        page.addView(view, new LinearLayout.LayoutParams(-1, -2));
    }
    private void button(String text, View.OnClickListener action) {
        Button button = new Button(this); button.setText(text); button.setAllCaps(false); button.setOnClickListener(action);
        page.addView(button, new LinearLayout.LayoutParams(-1, -2));
    }
    private void copy(String text) {
        try {
            ((ClipboardManager) getSystemService(CLIPBOARD_SERVICE)).setPrimaryClip(ClipData.newPlainText("Ostadix result", text));
            Toast.makeText(this, "Copied", Toast.LENGTH_SHORT).show();
        } catch (RuntimeException failure) {
            Toast.makeText(this, "Too large for the Android clipboard. You can still view the saved record here.", Toast.LENGTH_LONG).show();
        }
    }

    private void render() {
        ScrollView scroll = new ScrollView(this); scroll.setFillViewport(true);
        page = new LinearLayout(this); page.setOrientation(LinearLayout.VERTICAL);
        page.setPadding(dp(22), dp(28), dp(22), dp(36)); page.setBackgroundColor(Color.rgb(247, 249, 252));
        scroll.addView(page); setContentView(scroll);
        label("Ostadix Results", 28);
        try {
            if (selected != null) { showRecord(); return; }
            label("Answers saved on this phone. Open any request to see its output and execution details. These records are separate from Gemini chat history.", 16);
            button("Refresh", v -> render());
            if (!getSystemService(android.app.NotificationManager.class).areNotificationsEnabled()) {
                label("Notifications are off. Your results are still saved here.", 16);
                button("Enable result notifications", v -> startActivity(new Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS)
                        .putExtra(Settings.EXTRA_APP_PACKAGE, getPackageName())));
            }
            File[] records = ResultHistory.directory(this).listFiles((dir, name) -> name.matches("ostadix-turn-[0-9a-f-]+\\.json"));
            if (records == null || records.length == 0) {
                label("No saved requests yet. Ask Gemini a question while the local Ostadix route is enabled, then return here.", 18); return;
            }
            Arrays.sort(records, (a, b) -> Long.compare(b.lastModified(), a.lastModified()));
            for (int i = 0; i < Math.min(shown, records.length); i++) {
                File file = records[i];
                try {
                    JSONObject record = ResultHistory.read(file);
                    String id = ResultHistory.checkedId(record.getString("request_id"));
                    String prompt = record.optString("user_text", "Ostadix request");
                    String status = record.optString("phase", "saved").replace('_', ' ');
                    String preview = ResultHistory.display(record);
                    label(DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(new Date(file.lastModified()))
                            + " · " + status, 13);
                    button(prompt.substring(0, Math.min(180, prompt.length())), v -> { selected = id; render(); });
                    label(preview.substring(0, Math.min(220, preview.length())), 16);
                } catch (Exception error) { label("A saved record could not be read: " + file.getName(), 14); }
            }
            if (records.length > shown) { button("Show older requests", v -> { shown += 30; render(); }); }
        } catch (Exception error) { label("Could not open history: " + error.getMessage(), 18); button("All results", v -> { selected = null; render(); }); }
    }

    private void showRecord() throws Exception {
        String id = ResultHistory.checkedId(selected);
        JSONObject record = ResultHistory.read(new File(ResultHistory.directory(this), id + ".json"));
        button("All results", v -> { selected = null; render(); });
        label("Saved output", 14);
        String output = ResultHistory.display(record);
        label(output, 20);
        button("Copy answer", v -> copy(output));
        label("Your request", 14); label(record.optString("user_text", "Ostadix request"), 18);
        button("Copy execution record", v -> copy(record.toString()));
        button("Show program and execution details", v -> {
            v.setEnabled(false);
            try { label(record.toString(2), 13); }
            catch (Exception failure) { label("Could not format execution details", 16); }
        });
        label("Request: " + id, 12);
    }
}
