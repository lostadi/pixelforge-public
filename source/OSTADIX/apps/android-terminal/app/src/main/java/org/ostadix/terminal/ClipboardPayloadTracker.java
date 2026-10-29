package org.ostadix.terminal;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.net.Uri;

/** Tracks the one URI-backed payload that is valid while it remains the primary clipboard item. */
final class ClipboardPayloadTracker {
    private static final Object LOCK = new Object();

    private static ClipboardManager activeManager;
    private static ClipboardManager.OnPrimaryClipChangedListener activeListener;
    private static ClipboardFileStore.StagedPayload activePayload;

    private ClipboardPayloadTracker() {
    }

    static void onLargeClipPublished(
            final ClipboardManager manager,
            final ClipboardFileStore.StagedPayload payload,
            final Uri expectedUri) {
        ClipboardFileStore.StagedPayload previous;
        synchronized (LOCK) {
            detachListenerLocked();
            previous = activePayload;
            activePayload = payload;
            activeManager = manager;
            activeListener = new ClipboardManager.OnPrimaryClipChangedListener() {
                @Override
                public void onPrimaryClipChanged() {
                    if (isStillPrimary(manager, expectedUri)) {
                        return;
                    }
                    releaseIfActive(payload);
                }
            };
            try {
                activeManager.addPrimaryClipChangedListener(activeListener);
            } catch (RuntimeException unavailable) {
                activeManager = null;
                activeListener = null;
            }
        }
        if (previous != null && previous != payload
                && !previous.file.equals(payload.file)) {
            previous.delete();
        }
    }

    static void clearActivePayload() {
        ClipboardFileStore.StagedPayload previous;
        synchronized (LOCK) {
            detachListenerLocked();
            previous = activePayload;
            activePayload = null;
        }
        if (previous != null) {
            previous.delete();
        }
    }

    /**
     * Clipboard callbacks are queued. A callback triggered by setPrimaryClip can therefore run
     * after this listener is installed; checking the current URI keeps it from deleting its own
     * newly published payload.
     */
    private static boolean isStillPrimary(ClipboardManager manager, Uri expectedUri) {
        try {
            ClipData current = manager.getPrimaryClip();
            if (current == null) {
                // A background app can receive null even while its URI is still the primary clip.
                return true;
            }
            return current.getItemCount() > 0
                    && expectedUri.equals(current.getItemAt(0).getUri());
        } catch (RuntimeException unavailable) {
            // Keep the payload usable if Android temporarily denies clipboard reads. Startup
            // expiry will remove it later if no subsequent callback can verify replacement.
            return true;
        }
    }

    private static void releaseIfActive(ClipboardFileStore.StagedPayload expected) {
        ClipboardFileStore.StagedPayload stale;
        synchronized (LOCK) {
            if (activePayload != expected) {
                return;
            }
            detachListenerLocked();
            stale = activePayload;
            activePayload = null;
        }
        if (stale != null) {
            stale.delete();
        }
    }

    private static void detachListenerLocked() {
        if (activeManager != null && activeListener != null) {
            try {
                activeManager.removePrimaryClipChangedListener(activeListener);
            } catch (RuntimeException unavailable) {
                // The backing file still expires on a later app startup.
            }
        }
        activeManager = null;
        activeListener = null;
    }
}
