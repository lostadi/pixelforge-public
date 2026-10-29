package org.ostadix.terminal;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;

/**
 * Foreground lifetime anchor for a terminal session explicitly started by the
 * user. The service never starts at boot and is deliberately not sticky.
 */
public final class TerminalSessionService extends Service {
    private static final String ACTION_START =
            "org.ostadix.terminal.action.START_SESSION_PROTECTION";
    private static final String EXTRA_SESSION_NAME = "session_name";
    private static final String CHANNEL_ID = "active_terminal_sessions";
    private static final int NOTIFICATION_ID = 0x0F57AD1;

    public static void start(Context context, CharSequence sessionName) {
        Intent intent = new Intent(context, TerminalSessionService.class)
                .setAction(ACTION_START)
                .putExtra(EXTRA_SESSION_NAME, sessionName.toString());
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            context.startForegroundService(intent);
        } else {
            context.startService(intent);
        }
    }

    public static void stop(Context context) {
        context.stopService(new Intent(context, TerminalSessionService.class));
    }

    @Override
    public void onCreate() {
        super.onCreate();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID,
                    getString(R.string.terminal_session_channel),
                    NotificationManager.IMPORTANCE_LOW);
            channel.setDescription(getString(R.string.terminal_session_channel_description));
            channel.setShowBadge(false);
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) {
                manager.createNotificationChannel(channel);
            }
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null || !ACTION_START.equals(intent.getAction())) {
            stopSelf(startId);
            return START_NOT_STICKY;
        }
        String sessionName = intent.getStringExtra(EXTRA_SESSION_NAME);
        if (sessionName == null || sessionName.trim().isEmpty()) {
            sessionName = getString(R.string.app_name);
        }
        Notification notification = buildNotification(sessionName);
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(
                    NOTIFICATION_ID,
                    notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        } else {
            startForeground(NOTIFICATION_ID, notification);
        }
        return START_NOT_STICKY;
    }

    @Override
    public void onDestroy() {
        stopForeground(STOP_FOREGROUND_REMOVE);
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    private Notification buildNotification(String sessionName) {
        Intent openTerminal = new Intent(this, MainActivity.class)
                .setAction(Intent.ACTION_MAIN)
                .addCategory(Intent.CATEGORY_LAUNCHER)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        int pendingFlags = PendingIntent.FLAG_UPDATE_CURRENT;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            pendingFlags |= PendingIntent.FLAG_IMMUTABLE;
        }
        PendingIntent contentIntent = PendingIntent.getActivity(
                this, 0, openTerminal, pendingFlags);
        Notification.Builder builder = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
                ? new Notification.Builder(this, CHANNEL_ID)
                : new Notification.Builder(this);
        return builder
                .setSmallIcon(R.drawable.ic_terminal_notification)
                .setContentTitle(getString(R.string.app_name))
                .setContentText(getString(R.string.terminal_session_notification, sessionName))
                .setContentIntent(contentIntent)
                .setCategory(Notification.CATEGORY_SERVICE)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .setShowWhen(false)
                .build();
    }
}
