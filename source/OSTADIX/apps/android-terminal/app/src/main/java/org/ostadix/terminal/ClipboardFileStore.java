package org.ostadix.terminal;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

/** Disk-backed clipboard payloads that never put large text in a Binder transaction. */
final class ClipboardFileStore {
    static final long MAX_PAYLOAD_BYTES = 600L * 1024L * 1024L;
    static final int MAX_INLINE_BYTES = 256 * 1024;

    private static final long PARTIAL_MAX_AGE_MILLIS = 60L * 60L * 1000L;
    private static final long FINISHED_MAX_AGE_MILLIS = 24L * 60L * 60L * 1000L;
    private static final String DIRECTORY_NAME = "clipboard_exports";
    private static final int COPY_BUFFER_BYTES = 256 * 1024;

    interface PayloadWriter {
        void writeTo(Writer destination) throws IOException;
    }

    static final class StagedPayload {
        final File file;
        final long byteCount;

        private StagedPayload(File file, long byteCount) {
            this.file = file;
            this.byteCount = byteCount;
        }

        String readInlineText() throws IOException {
            if (byteCount > MAX_INLINE_BYTES) {
                throw new IOException("payload is too large to materialize as inline text");
            }
            byte[] contents = new byte[(int) byteCount];
            int offset = 0;
            try (InputStream input = new BufferedInputStream(
                    new FileInputStream(file), COPY_BUFFER_BYTES)) {
                while (offset < contents.length) {
                    int read = input.read(contents, offset, contents.length - offset);
                    if (read < 0) {
                        throw new IOException("clipboard payload ended before its declared size");
                    }
                    offset += read;
                }
                if (input.read() != -1) {
                    throw new IOException("clipboard payload grew while it was being read");
                }
            }
            return new String(contents, StandardCharsets.UTF_8);
        }

        void delete() {
            file.delete();
        }
    }

    static final class PayloadTooLargeException extends IOException {
        PayloadTooLargeException(long maximumBytes) {
            super("clipboard payload exceeds " + maximumBytes + " bytes");
        }
    }

    private ClipboardFileStore() {
    }

    static StagedPayload stage(File appFilesDirectory, PayloadWriter source) throws IOException {
        return stage(appFilesDirectory, MAX_PAYLOAD_BYTES, source);
    }

    static StagedPayload stage(
            File appFilesDirectory,
            long maximumBytes,
            PayloadWriter source) throws IOException {
        if (appFilesDirectory == null) {
            throw new IllegalArgumentException("appFilesDirectory must not be null");
        }
        if (source == null) {
            throw new IllegalArgumentException("source must not be null");
        }
        if (maximumBytes < 0) {
            throw new IllegalArgumentException("maximumBytes must not be negative");
        }

        File directory = exportsDirectory(appFilesDirectory);
        if ((!directory.exists() && !directory.mkdirs()) || !directory.isDirectory()) {
            throw new IOException("unable to create clipboard export directory");
        }

        String identifier = UUID.randomUUID().toString();
        File partial = new File(directory, identifier + ".part");
        File finished = new File(directory, identifier + ".txt");
        boolean complete = false;
        try {
            CountingBoundedOutputStream bounded;
            try (OutputStream fileOutput = new FileOutputStream(partial);
                    OutputStream buffered = new BufferedOutputStream(
                            fileOutput, COPY_BUFFER_BYTES)) {
                bounded = new CountingBoundedOutputStream(buffered, maximumBytes);
                try (Writer destination = new OutputStreamWriter(
                        bounded, StandardCharsets.UTF_8)) {
                    source.writeTo(destination);
                }
            }
            long byteCount = bounded.getByteCount();
            if (partial.length() != byteCount) {
                throw new IOException("clipboard payload size changed while staging");
            }
            if (!partial.renameTo(finished)) {
                throw new IOException("unable to publish staged clipboard payload");
            }
            complete = true;
            return new StagedPayload(finished, byteCount);
        } finally {
            if (!complete) {
                partial.delete();
                finished.delete();
            }
        }
    }

    static StagedPayload reopen(File appFilesDirectory, String fileName) throws IOException {
        File payload = resolvePayload(appFilesDirectory, fileName);
        if (!payload.isFile() || !payload.canRead()) {
            throw new IOException("clipboard payload is no longer available");
        }
        long byteCount = payload.length();
        if (byteCount > MAX_PAYLOAD_BYTES) {
            throw new PayloadTooLargeException(MAX_PAYLOAD_BYTES);
        }
        return new StagedPayload(payload, byteCount);
    }

    static File resolvePayload(File appFilesDirectory, String fileName) throws IOException {
        if (!isFinishedPayloadName(fileName)) {
            throw new IOException("invalid clipboard payload name");
        }
        File directory = exportsDirectory(appFilesDirectory).getCanonicalFile();
        File candidate = new File(directory, fileName).getCanonicalFile();
        if (!directory.equals(candidate.getParentFile())) {
            throw new IOException("clipboard payload escaped its private directory");
        }
        return candidate;
    }

    /** Deletes process-death residue. Call this on the serialized clipboard worker. */
    static void deleteExpiredPayloads(File appFilesDirectory, long nowMillis) {
        File directory = exportsDirectory(appFilesDirectory);
        File[] files = directory.listFiles();
        if (files == null) {
            return;
        }
        for (File candidate : files) {
            if (!candidate.isFile()) {
                continue;
            }
            String name = candidate.getName();
            long modified = candidate.lastModified();
            long age = modified <= 0 || modified > nowMillis ? 0 : nowMillis - modified;
            if ((isPartialPayloadName(name) && age >= PARTIAL_MAX_AGE_MILLIS)
                    || (isFinishedPayloadName(name) && age >= FINISHED_MAX_AGE_MILLIS)) {
                candidate.delete();
            }
        }
    }

    /**
     * Removes files that cannot be referenced by the one primary clipboard item. Run this only
     * after inspecting the clipboard and on the same serialized worker used for staging.
     */
    static void deleteOrphanedPayloads(
            File appFilesDirectory,
            String protectedFileName) {
        File directory = exportsDirectory(appFilesDirectory);
        File[] files = directory.listFiles();
        if (files == null) {
            return;
        }
        for (File candidate : files) {
            String name = candidate.getName();
            if (candidate.isFile()
                    && (isPartialPayloadName(name)
                            || (isFinishedPayloadName(name)
                                    && !name.equals(protectedFileName)))) {
                candidate.delete();
            }
        }
    }

    static boolean isFinishedPayloadName(String fileName) {
        return hasUuidSuffix(fileName, ".txt");
    }

    private static boolean isPartialPayloadName(String fileName) {
        return hasUuidSuffix(fileName, ".part");
    }

    private static boolean hasUuidSuffix(String fileName, String suffix) {
        if (fileName == null || !fileName.endsWith(suffix)) {
            return false;
        }
        String identifier = fileName.substring(0, fileName.length() - suffix.length());
        try {
            return UUID.fromString(identifier).toString().equals(identifier);
        } catch (IllegalArgumentException invalid) {
            return false;
        }
    }

    private static File exportsDirectory(File appFilesDirectory) {
        return new File(appFilesDirectory, DIRECTORY_NAME);
    }

    private static final class CountingBoundedOutputStream extends FilterOutputStream {
        private final long maximumBytes;
        private long byteCount;

        CountingBoundedOutputStream(OutputStream destination, long maximumBytes) {
            super(destination);
            this.maximumBytes = maximumBytes;
        }

        @Override
        public void write(int value) throws IOException {
            requireCapacity(1);
            out.write(value);
            byteCount++;
        }

        @Override
        public void write(byte[] bytes, int offset, int length) throws IOException {
            if (bytes == null) {
                throw new NullPointerException("bytes");
            }
            if (offset < 0 || length < 0 || length > bytes.length - offset) {
                throw new IndexOutOfBoundsException();
            }
            requireCapacity(length);
            out.write(bytes, offset, length);
            byteCount += length;
        }

        long getByteCount() {
            return byteCount;
        }

        private void requireCapacity(int requestedBytes) throws PayloadTooLargeException {
            if (requestedBytes > maximumBytes - byteCount) {
                throw new PayloadTooLargeException(maximumBytes);
            }
        }
    }
}
