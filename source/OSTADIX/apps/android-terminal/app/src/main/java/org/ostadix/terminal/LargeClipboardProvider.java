package org.ostadix.terminal;

import android.content.ContentProvider;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.database.Cursor;
import android.database.MatrixCursor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

/** Read-only provider for large clipboard text stored in the app's private files directory. */
public final class LargeClipboardProvider extends ContentProvider {
    static final String AUTHORITY = "org.ostadix.terminal.largeclipboard";
    private static final String PAYLOAD_PATH = "payload";
    private static final String MIME_TYPE = "text/plain";

    static Uri uriFor(File payload) {
        if (payload == null || !ClipboardFileStore.isFinishedPayloadName(payload.getName())) {
            throw new IllegalArgumentException("payload is not a staged clipboard file");
        }
        return new Uri.Builder()
                .scheme(ContentResolver.SCHEME_CONTENT)
                .authority(AUTHORITY)
                .appendPath(PAYLOAD_PATH)
                .appendPath(payload.getName())
                .build();
    }

    static String payloadFileName(Uri uri) {
        if (uri == null
                || !ContentResolver.SCHEME_CONTENT.equals(uri.getScheme())
                || !AUTHORITY.equals(uri.getAuthority())) {
            return null;
        }
        List<String> segments = uri.getPathSegments();
        if (segments.size() != 2 || !PAYLOAD_PATH.equals(segments.get(0))) {
            return null;
        }
        String fileName = segments.get(1);
        return ClipboardFileStore.isFinishedPayloadName(fileName) ? fileName : null;
    }

    @Override
    public boolean onCreate() {
        return true;
    }

    @Override
    public String getType(Uri uri) {
        requirePayload(uri);
        return MIME_TYPE;
    }

    @Override
    public Cursor query(
            Uri uri,
            String[] projection,
            String selection,
            String[] selectionArguments,
            String sortOrder) {
        File payload = requirePayload(uri);
        String[] columns = projection == null
                ? new String[] {OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE}
                : projection;
        MatrixCursor cursor = new MatrixCursor(columns, 1);
        MatrixCursor.RowBuilder row = cursor.newRow();
        for (String column : columns) {
            if (OpenableColumns.DISPLAY_NAME.equals(column)) {
                row.add(column, "Ostadix terminal selection.txt");
            } else if (OpenableColumns.SIZE.equals(column)) {
                row.add(column, payload.length());
            } else {
                row.add(column, null);
            }
        }
        return cursor;
    }

    @Override
    public ParcelFileDescriptor openFile(Uri uri, String mode) throws FileNotFoundException {
        if (!"r".equals(mode)) {
            throw new FileNotFoundException("clipboard payloads are read-only");
        }
        File payload = requirePayload(uri);
        if (!payload.isFile() || !payload.canRead()) {
            throw new FileNotFoundException("clipboard payload is no longer available");
        }
        if (payload.length() > ClipboardFileStore.MAX_PAYLOAD_BYTES) {
            throw new FileNotFoundException("clipboard payload exceeds the 600 MiB limit");
        }
        return ParcelFileDescriptor.open(payload, ParcelFileDescriptor.MODE_READ_ONLY);
    }

    @Override
    public Uri insert(Uri uri, ContentValues values) {
        throw new UnsupportedOperationException("clipboard payloads are read-only");
    }

    @Override
    public int delete(Uri uri, String selection, String[] selectionArguments) {
        throw new UnsupportedOperationException("clipboard payloads are read-only");
    }

    @Override
    public int update(
            Uri uri,
            ContentValues values,
            String selection,
            String[] selectionArguments) {
        throw new UnsupportedOperationException("clipboard payloads are read-only");
    }

    private File requirePayload(Uri uri) {
        String fileName = payloadFileName(uri);
        if (fileName == null) {
            throw new IllegalArgumentException("unknown clipboard URI");
        }
        if (getContext() == null) {
            throw new IllegalStateException("clipboard provider is not attached");
        }
        try {
            return ClipboardFileStore.resolvePayload(
                    getContext().getFilesDir(), fileName);
        } catch (IOException invalid) {
            throw new IllegalArgumentException("invalid clipboard payload", invalid);
        }
    }
}
