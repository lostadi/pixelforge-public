package org.ostadix.aicore.extension;

import java.io.ByteArrayOutputStream;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.util.List;

/** Minimal plain-text wire adapter for the exact Google APK pinned by ExtensionGate. */
final class GeminiLocalResponse {
    static Class<?> type(ClassLoader loader, String name) throws ClassNotFoundException {
        return Class.forName(GeminiHostSymbols.name(name), false, loader);
    }

    static Object field(Object owner, String name) throws ReflectiveOperationException {
        Field f = owner.getClass().getDeclaredField(name);
        f.setAccessible(true);
        return f.get(owner);
    }

    static Object constant(ClassLoader loader, String owner, String name)
            throws ReflectiveOperationException {
        Field f = type(loader, owner).getDeclaredField(name);
        f.setAccessible(true);
        return f.get(null);
    }

    static Object call(Method m, Object receiver, Object... args) throws Throwable {
        try { return m.invoke(receiver, args); }
        catch (InvocationTargetException e) { throw e.getTargetException(); }
    }

    static Object text(ClassLoader loader, String id, String text) throws Throwable {
        // gaik.candidates(1) -> gbdu.content(5) -> gbdj.parts(1)
        // -> gaet.text(1) -> gbeu.richText(1) -> gadw.text(1).
        byte[] rich = new Wire().string(1, text).bytes();
        byte[] part = new Wire().message(1, new Wire().message(1, rich).bytes()).bytes();
        byte[] content = new Wire().message(1, part).bytes();
        byte[] candidate = new Wire().string(1, id).message(5, content).bytes();
        // No server conversation ID, fabricated action, or Google receipt.
        byte[] response = new Wire().message(1, candidate).integer(9, 1).bytes();
        Class<?> proto = type(loader, "goqd");
        Method parse = proto.getDeclaredMethod("parseFrom", proto, byte[].class);
        parse.setAccessible(true);
        Object result = call(parse, null, constant(loader, "gaik", "a"), response);
        // Round trip through the stock parser before entering its response stream.
        Object parsedCandidate = ((List<?>) field(result, "d")).get(0);
        Object parsedPart = ((List<?>) field(field(parsedCandidate, "f"), "b")).get(0);
        if (!id.equals(field(parsedCandidate, "c"))
                || !Integer.valueOf(1).equals(field(parsedPart, "b"))
                || !text.equals(field(field(field(parsedPart, "c"), "c"), "c"))) {
            throw new IllegalStateException("pinned Gemini text response contract mismatch");
        }
        return result;
    }

    private static final class Wire {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();
        private void varint(long n) {
            while ((n & ~127L) != 0) { out.write(((int) n & 127) | 128); n >>>= 7; }
            out.write((int) n);
        }
        Wire integer(int field, long n) { varint((long) field << 3); varint(n); return this; }
        Wire message(int field, byte[] b) {
            varint(((long) field << 3) | 2); varint(b.length); out.write(b, 0, b.length); return this;
        }
        Wire string(int field, String s) { return message(field, s.getBytes(StandardCharsets.UTF_8)); }
        byte[] bytes() { return out.toByteArray(); }
    }
}
