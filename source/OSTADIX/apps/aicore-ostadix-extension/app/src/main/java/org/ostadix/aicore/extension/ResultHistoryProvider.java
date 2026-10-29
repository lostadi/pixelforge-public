package org.ostadix.aicore.extension;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.ContentProvider;
import android.content.ContentValues;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Binder;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.os.Process;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.RandomAccessFile;
import org.json.JSONObject;

/** Write-only handoff from the pinned Google app to the owned result viewer. */
public final class ResultHistoryProvider extends ContentProvider {
    @Override public boolean onCreate() { return true; }

    private synchronized void enforceWriter() {
        int uid = Binder.getCallingUid();
        if (uid == Process.myUid()) { return; }
        String name = getCallingPackage();
        if (!ExtensionGate.acceptsGsaCaller(getContext(), uid, name)) {
            throw new SecurityException("Only the pinned Gemini host may write Ostadix results");
        }
    }

    @Override public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        enforceWriter();
        if (!"w".equals(mode) || !ResultHistory.AUTHORITY.equals(uri.getAuthority())
                || uri.getPathSegments().size() != 1) { throw new SecurityException("Result handoff is write-only"); }
        try {
            File file = new File(ResultHistory.directory(getContext()), ResultHistory.checkedId(uri.getLastPathSegment()) + ".pending");
            return ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_WRITE_ONLY
                    | ParcelFileDescriptor.MODE_CREATE | ParcelFileDescriptor.MODE_TRUNCATE);
        } catch (java.io.IOException failure) { throw new FileNotFoundException(failure.toString()); }
    }

    @Override public synchronized Bundle call(String method, String arg, Bundle extras) {
        enforceWriter();
        if (!"commit".equals(method)) { throw new IllegalArgumentException("Unsupported history operation"); }
        String id = ResultHistory.checkedId(arg);
        try {
            File directory = ResultHistory.directory(getContext());
            File pending = new File(directory, id + ".pending");
            JSONObject record = ResultHistory.read(pending);
            if (!id.equals(record.getString("request_id"))) { throw new IllegalArgumentException("Result identity mismatch"); }
            try (RandomAccessFile file = new RandomAccessFile(pending, "rw")) { file.getFD().sync(); }
            File target = new File(directory, id + ".json");
            if (!pending.renameTo(target)) { throw new java.io.IOException("Could not commit result history"); }
            getContext().getContentResolver().notifyChange(Uri.parse("content://" + ResultHistory.AUTHORITY), null);
            Bundle response = new Bundle(); response.putBoolean("saved", true);
            try { response.putBoolean("notified", notifyResult(id, record)); }
            catch (RuntimeException notificationError) { response.putBoolean("notified", false); }
            return response;
        } catch (Exception failure) { throw new IllegalStateException("Could not save Ostadix result", failure); }
    }

    private boolean notifyResult(String id, JSONObject record) throws Exception {
        NotificationManager manager = getContext().getSystemService(NotificationManager.class);
        manager.createNotificationChannel(new NotificationChannel("ostadix_results", "Ostadix results",
                NotificationManager.IMPORTANCE_DEFAULT));
        if (!manager.areNotificationsEnabled()) { return false; }
        boolean done = record.optBoolean("history_terminal");
        String text = ResultHistory.display(record);
        Intent intent = new Intent(getContext(), ResultHistoryActivity.class)
                .setData(Uri.parse("ostadix-result:" + id)).putExtra("request_id", id)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        PendingIntent open = PendingIntent.getActivity(getContext(), 0, intent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        manager.notify(id, 1, new Notification.Builder(getContext(), "ostadix_results")
                .setSmallIcon(android.R.drawable.stat_notify_more)
                .setContentTitle(done ? "Ostadix result saved" : "Ostadix is working")
                .setContentText(text.substring(0, Math.min(text.length(), 160)))
                .setStyle(new Notification.BigTextStyle().bigText(text.substring(0, Math.min(text.length(), 4000))))
                .setContentIntent(open).setAutoCancel(true).setOnlyAlertOnce(true)
                .setVisibility(Notification.VISIBILITY_PRIVATE).build());
        return true;
    }

    private SecurityException privateHistory() { return new SecurityException("Open Ostadix Results to view private history"); }
    @Override public Cursor query(Uri u, String[] p, String s, String[] a, String o) { throw privateHistory(); }
    @Override public String getType(Uri u) { return "application/json"; }
    @Override public Uri insert(Uri u, ContentValues v) { throw privateHistory(); }
    @Override public int update(Uri u, ContentValues v, String s, String[] a) { throw privateHistory(); }
    @Override public int delete(Uri u, String s, String[] a) { throw privateHistory(); }
}
