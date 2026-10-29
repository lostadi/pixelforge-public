package org.ostadix.terminal;

import java.io.File;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.StringWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;

/** Dependency-free host smoke tests for the terminal parser and screen model. */
public final class TerminalCoreSelfTest {
    private TerminalCoreSelfTest() {
    }

    public static void main(String[] arguments) {
        testCpuPolicyDefaults();
        testTextAndCursor();
        testAnsiColors();
        testSplitUtf8();
        testRejectNonScalarCodePoints();
        testOscTitle();
        testBoundedScrollback();
        testAlternateScreenRestore();
        testTerminalQueryReplies();
        testWordSelectionAndReverseExtraction();
        testSoftWrapAndWideSelection();
        testSelectionAfterHistoryEviction();
        testStreamingSelection();
        testVoiceInputSemantics();
        testClipboardFileStore();
        System.out.println("Terminal core self-tests passed");
    }

    private static void testCpuPolicyDefaults() {
        AppPreferences.Snapshot defaults = AppPreferences.defaults();
        equal(AppPreferences.CPU_MODE_BALANCED, defaults.cpuMode,
                "graph-safe default CPU policy");
        if (defaults.isPrimeCpu7Enabled()) {
            throw new AssertionError("default CPU policy must not pin graph workers to CPU7");
        }
        equal(
                AppPreferences.CPU_MODE_BALANCED,
                AppPreferences.migrateLegacyCpuMode(1, AppPreferences.CPU_MODE_PRIME_CPU7),
                "legacy CPU7 default safety migration");
        equal(
                AppPreferences.CPU_MODE_PRIME_CPU7,
                AppPreferences.migrateLegacyCpuMode(2, AppPreferences.CPU_MODE_PRIME_CPU7),
                "current explicit Prime policy preservation");

        AppPreferences.Snapshot explicitPrime = new AppPreferences.Snapshot(
                defaults.theme,
                defaults.fontSizeSp,
                defaults.scrollbackLines,
                defaults.cursorStyle,
                AppPreferences.CPU_MODE_PRIME_CPU7,
                defaults.keepScreenAwake,
                defaults.hapticsEnabled,
                defaults.startupMode
        );
        if (!explicitPrime.isPrimeCpu7Enabled()) {
            throw new AssertionError("explicit Prime CPU7 policy was not preserved");
        }
    }

    private static void testTextAndCursor() {
        TerminalBuffer buffer = new TerminalBuffer(12, 4, 20);
        AnsiParser parser = new AnsiParser(buffer);
        parser.feed("abc\r\nZ".getBytes(StandardCharsets.UTF_8));
        TerminalBuffer.Snapshot screen = buffer.snapshot(0);
        equal((int) 'a', screen.codePointAt(0, 0), "first character");
        equal((int) 'c', screen.codePointAt(0, 2), "third character");
        equal((int) 'Z', screen.codePointAt(1, 0), "next line");
        equal(1, screen.cursorColumn, "cursor column");
        equal(1, screen.cursorRow, "cursor row");
    }

    private static void testAnsiColors() {
        TerminalBuffer buffer = new TerminalBuffer(12, 4, 20);
        AnsiParser parser = new AnsiParser(buffer);
        parser.feed("\u001b[31mR\u001b[38;5;196mX".getBytes(StandardCharsets.UTF_8));
        TerminalBuffer.Snapshot screen = buffer.snapshot(0);
        equal(TerminalBuffer.indexedColor(1),
                screen.foregroundSpecificationAt(0, 0), "ANSI red");
        equal(TerminalBuffer.indexedColor(196),
                screen.foregroundSpecificationAt(0, 1), "ANSI 256 color");
    }

    private static void testSplitUtf8() {
        TerminalBuffer buffer = new TerminalBuffer(12, 4, 20);
        AnsiParser parser = new AnsiParser(buffer);
        byte[] encoded = "λ".getBytes(StandardCharsets.UTF_8);
        parser.feed(encoded, 0, 1);
        parser.feed(encoded, 1, encoded.length - 1);
        equal(0x03bb, buffer.snapshot(0).codePointAt(0, 0), "split UTF-8 scalar");
    }

    private static void testRejectNonScalarCodePoints() {
        TerminalBuffer buffer = new TerminalBuffer(4, 2, 4);
        buffer.putCodePoint(Character.MIN_SURROGATE);
        buffer.putCodePoint(Character.MAX_SURROGATE);
        TerminalBuffer.Snapshot screen = buffer.snapshot(0);
        equal(0, screen.codePointAt(0, 0), "surrogate code point rejection");
        equal(0, screen.cursorColumn, "surrogate does not advance cursor");
    }

    private static void testOscTitle() {
        TerminalBuffer buffer = new TerminalBuffer(12, 4, 20);
        AnsiParser parser = new AnsiParser(buffer);
        final String[] title = {null};
        parser.setTitleListener(new AnsiParser.TitleListener() {
            @Override
            public void onTitleChanged(String value) {
                title[0] = value;
            }
        });
        parser.feed("\u001b]0;Ostadix\u0007".getBytes(StandardCharsets.UTF_8));
        equal("Ostadix", title[0], "OSC title");
    }

    private static void testBoundedScrollback() {
        TerminalBuffer buffer = new TerminalBuffer(8, 2, 3);
        AnsiParser parser = new AnsiParser(buffer);
        parser.feed("1\r\n2\r\n3\r\n4\r\n5".getBytes(StandardCharsets.UTF_8));
        equal(3, buffer.getScrollbackSize(), "scrollback bound");
    }

    private static void testAlternateScreenRestore() {
        TerminalBuffer buffer = new TerminalBuffer(12, 4, 20);
        AnsiParser parser = new AnsiParser(buffer);
        parser.feed(("primary\u001b(0\u001b[?1049h\u001b(B\u001b[31m\u001b[?7l"
                + "\u001b[?25lfullscreen").getBytes(StandardCharsets.UTF_8));
        if (!buffer.isAlternateScreen()) {
            throw new AssertionError("alternate screen was not entered");
        }
        equal((int) 'f', buffer.snapshot(0).codePointAt(0, 0), "alternate contents");
        parser.feed("\u001b[?1049lq".getBytes(StandardCharsets.UTF_8));
        if (buffer.isAlternateScreen()) {
            throw new AssertionError("alternate screen was not exited");
        }
        equal((int) 'p', buffer.snapshot(0).codePointAt(0, 0), "restored primary contents");
        equal(0x2500, buffer.snapshot(0).codePointAt(0, 7), "restored DEC charset");
        equal(TerminalBuffer.COLOR_DEFAULT,
                buffer.snapshot(0).foregroundSpecificationAt(0, 7), "restored rendition");
        if (!buffer.snapshot(0).cursorVisible) {
            throw new AssertionError("primary cursor visibility was not restored");
        }
        equal(8, buffer.getCursorColumn(), "restored primary cursor");
    }

    private static void testTerminalQueryReplies() {
        TerminalBuffer buffer = new TerminalBuffer(12, 4, 20);
        AnsiParser parser = new AnsiParser(buffer);
        final StringBuilder replies = new StringBuilder();
        parser.setResponseListener(new AnsiParser.ResponseListener() {
            @Override
            public void onResponse(byte[] response) {
                replies.append(new String(response, StandardCharsets.UTF_8));
            }
        });
        parser.feed("ab\u001b[6n\u001b[c".getBytes(StandardCharsets.UTF_8));
        equal("\u001b[1;3R\u001b[?1;2c", replies.toString(), "terminal query replies");
    }

    private static void testWordSelectionAndReverseExtraction() {
        TerminalBuffer buffer = new TerminalBuffer(20, 3, 20);
        AnsiParser parser = new AnsiParser(buffer);
        parser.feed("run /data/app/demo\r\nnext".getBytes(StandardCharsets.UTF_8));
        TerminalBuffer.WordRange word = buffer.wordRangeAt(0, 9);
        equal(4, word.startColumn, "path word start");
        equal(17, word.endColumn, "path word end");
        equal("/data/app/demo\nnext",
                buffer.extractText(1, 3, 0, 4), "reverse multi-row selection");
    }

    private static void testSoftWrapAndWideSelection() {
        TerminalBuffer wrapped = new TerminalBuffer(4, 3, 20);
        new AnsiParser(wrapped).feed("abcdef".getBytes(StandardCharsets.UTF_8));
        equal("abcdef", wrapped.extractText(0, 0, 1, 3), "soft wrap joins rows");

        TerminalBuffer hardBreak = new TerminalBuffer(4, 3, 20);
        new AnsiParser(hardBreak).feed("ab\r\ncd".getBytes(StandardCharsets.UTF_8));
        equal("ab\ncd", hardBreak.extractText(0, 0, 1, 3), "hard break stays newline");

        TerminalBuffer wide = new TerminalBuffer(8, 2, 20);
        new AnsiParser(wide).feed("A界B".getBytes(StandardCharsets.UTF_8));
        equal("界", wide.extractText(0, 2, 0, 2), "wide continuation selects glyph");
    }

    private static void testSelectionAfterHistoryEviction() {
        TerminalBuffer buffer = new TerminalBuffer(6, 2, 3);
        new AnsiParser(buffer).feed("1\r\n2\r\n3\r\n4\r\n5".getBytes(StandardCharsets.UTF_8));
        equal(5, buffer.getDocumentLineCount(), "bounded selection document size");
        equal(3, buffer.snapshot(0).firstDocumentLine, "live viewport document origin");
        equal(1, buffer.snapshot(2).firstDocumentLine, "scrolled viewport document origin");
        equal("2\n3\n4\n5",
                buffer.extractText(4, 0, 1, 0), "selection after history eviction");
    }

    private static void testStreamingSelection() {
        TerminalBuffer buffer = new TerminalBuffer(4, 4, 20);
        AnsiParser parser = new AnsiParser(buffer);
        parser.feed("ab\r\n\r\ncd".getBytes(StandardCharsets.UTF_8));
        StringWriter streamed = new StringWriter();
        try {
            buffer.writeText(2, 3, 0, 0, streamed);
        } catch (IOException error) {
            throw new AssertionError("streamed selection failed", error);
        }
        equal("ab\n\ncd", streamed.toString(), "streamed selection formatting");
        equal(
                buffer.extractText(2, 3, 0, 0),
                streamed.toString(),
                "materialized and streamed selections agree");
        equal(
                (long) streamed.toString().getBytes(StandardCharsets.UTF_8).length,
                buffer.captureText(2, 3, 0, 0).utf8ByteCount(),
                "selection snapshot UTF-8 byte count");

        TerminalBuffer.TextSelection snapshot = buffer.captureText(2, 3, 0, 0);
        if (snapshot.maximumUtf8ByteCount() < snapshot.utf8ByteCount()) {
            throw new AssertionError("selection UTF-8 upper bound is too small");
        }
        buffer.reset();
        equal("ab\n\ncd", snapshot.asString(), "selection snapshot survives screen mutation");

        boolean interrupted = false;
        Thread.currentThread().interrupt();
        try {
            snapshot.writeTo(new StringWriter());
        } catch (InterruptedIOException expected) {
            interrupted = true;
        } catch (IOException error) {
            throw new AssertionError("unexpected cancellation error", error);
        } finally {
            Thread.interrupted();
        }
        if (!interrupted) {
            throw new AssertionError("selection streaming ignored cancellation");
        }
    }

    private static void testClipboardFileStore() {
        equal(629_145_600L, ClipboardFileStore.MAX_PAYLOAD_BYTES,
                "600 MiB clipboard payload limit");
        File testRoot = new File(
                System.getProperty("java.io.tmpdir"),
                "ostadix-clipboard-test-" + System.nanoTime());
        if (!testRoot.mkdirs()) {
            throw new AssertionError("unable to create clipboard test directory");
        }
        ClipboardFileStore.StagedPayload payload = null;
        try {
            final String expected = "clipboard 界";
            payload = ClipboardFileStore.stage(
                    testRoot,
                    new ClipboardFileStore.PayloadWriter() {
                        @Override
                        public void writeTo(Writer destination) throws IOException {
                            destination.write(expected);
                        }
                    });
            equal(
                    (long) expected.getBytes(StandardCharsets.UTF_8).length,
                    payload.byteCount,
                    "UTF-8 clipboard payload byte count");
            equal(expected, payload.readInlineText(), "inline clipboard payload");
            if (!ClipboardFileStore.isFinishedPayloadName(payload.file.getName())) {
                throw new AssertionError("staged clipboard payload name is not provider-safe");
            }
            equal(
                    payload.file.getCanonicalFile(),
                    ClipboardFileStore.resolvePayload(testRoot, payload.file.getName()),
                    "clipboard payload resolution");
            equal(
                    payload.byteCount,
                    ClipboardFileStore.reopen(
                            testRoot, payload.file.getName()).byteCount,
                    "clipboard payload restart recovery");

            ClipboardFileStore.StagedPayload orphan = ClipboardFileStore.stage(
                    testRoot,
                    new ClipboardFileStore.PayloadWriter() {
                        @Override
                        public void writeTo(Writer destination) throws IOException {
                            destination.write("orphan");
                        }
                    });
            ClipboardFileStore.deleteOrphanedPayloads(
                    testRoot, payload.file.getName());
            if (!payload.file.exists()) {
                throw new AssertionError("current clipboard payload was not preserved");
            }
            if (orphan.file.exists()) {
                throw new AssertionError("orphaned clipboard payload was not deleted");
            }

            boolean rejected = false;
            try {
                ClipboardFileStore.stage(
                        testRoot,
                        4,
                        new ClipboardFileStore.PayloadWriter() {
                            @Override
                            public void writeTo(Writer destination) throws IOException {
                                destination.write("12345");
                            }
                        });
            } catch (ClipboardFileStore.PayloadTooLargeException expectedFailure) {
                rejected = true;
            }
            if (!rejected) {
                throw new AssertionError("clipboard payload limit was not enforced");
            }

            if (!payload.file.setLastModified(1)) {
                throw new AssertionError("unable to age clipboard test payload");
            }
            ClipboardFileStore.deleteExpiredPayloads(
                    testRoot, 8L * 24L * 60L * 60L * 1000L);
            if (payload.file.exists()) {
                throw new AssertionError("expired clipboard payload was not deleted");
            }
        } catch (IOException error) {
            throw new AssertionError("clipboard file-store test failed", error);
        } finally {
            deleteRecursively(testRoot);
        }
    }

    private static void testVoiceInputSemantics() {
        equal("next command",
                TerminalView.trailingInputLine("first command\r\nnext command"),
                "voice input CRLF normalization state");
        equal("λ world",
                TerminalView.trailingInputLine("ignored\nλ world"),
                "voice input Unicode trailing line");
        if (!TerminalView.containsLineBreak("one\ntwo")) {
            throw new AssertionError("voice input line break was not detected");
        }
        equal(
                "hello ".length(),
                TerminalView.commonPrefixAtCodePointBoundary("hello world", "hello there"),
                "voice replacement common prefix");
        equal(
                "A😀".length(),
                TerminalView.commonPrefixAtCodePointBoundary("A😀x", "A😀y"),
                "voice replacement Unicode boundary");
        TerminalView.InputReplacement append = TerminalView.planInputReplacement(
                "echo", "echo hello world");
        equal(0, append.deleteCodePoints, "voice append deletion count");
        equal(" hello world", append.suffix, "voice append suffix");
        TerminalView.InputReplacement correction = TerminalView.planInputReplacement(
                "hello world", "hello there");
        equal(5, correction.deleteCodePoints, "voice correction deletion count");
        equal("there", correction.suffix, "voice correction suffix");
        TerminalView.InputReplacement unicode = TerminalView.planInputReplacement(
                "A😀x", "A😀y");
        equal(1, unicode.deleteCodePoints, "voice Unicode correction deletion count");
        equal("y", unicode.suffix, "voice Unicode correction suffix");
    }

    private static void deleteRecursively(File target) {
        File[] children = target.listFiles();
        if (children != null) {
            for (File child : children) {
                deleteRecursively(child);
            }
        }
        if (target.exists() && !target.delete()) {
            throw new AssertionError("unable to remove test path: " + target);
        }
    }

    private static void equal(Object expected, Object actual, String label) {
        if (!expected.equals(actual)) {
            throw new AssertionError(label + ": expected " + expected + ", got " + actual);
        }
    }
}
