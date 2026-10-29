package org.ostadix.aicore.extension;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/** Checks that local routing cannot discard attachments, voice or tool/context data. */
public final class GeminiPlainTextInputSelfTest {
    private enum Source { TEXT, VOICE, CONVERSATION_MODE, OTHER, POST_CONVERSATION_MODE_SUMMARY, MEMORY_SUMMARY_REQUEST }
    private static final String LISTS = "c d e W x z A L N ae af ag";
    private static final String OBJECTS = "g m X Z t aa y B D E ah K ai O am an ao ap Q U";
    private static final String STRINGS = "h Y F ac ad H I J aq";
    private static final String FLAGS = "f n s G M aj ak al S";
    private static final Object DEFAULT_AUDIO = new Object();

    public static void main(String[] args) throws Exception {
        require(GeminiPlainTextInput.reviewedSuggestionReference(null, Object.class)
                && !GeminiPlainTextInput.reviewedSuggestionReference(new Object(), Object.class)
                && !GeminiPlainTextInput.reviewedSuggestionReference("reference", String.class),
                "unreviewed suggestion reference type admitted");
        Map<String, Object> plain = plain();
        require(eligible(plain), "reviewed plain typed input rejected");
        require(GeminiPlainTextInput.rejectionFields(plain::get, DEFAULT_AUDIO) == null,
                "eligible input has a rejection diagnostic");
        Map<String, Object> several = plain();
        several.put("c", Collections.singletonList("PRIVATE_ATTACHMENT_VALUE"));
        several.put("aq", "PRIVATE_CONFIRMATION_VALUE");
        several.put("p", 314159);
        String reason = GeminiPlainTextInput.rejectionFields(several::get, DEFAULT_AUDIO);
        require("fields:c,aq,p".equals(reason) && !reason.contains("PRIVATE") && !reason.contains("314159")
                && !reason.contains((String) several.get("a")) && !eligible(several),
                "rejection diagnostic failed to combine field names without exposing values");
        require("input_type".equals(GeminiPlainTextInput.rejectionReason(null, Object.class))
                && !GeminiPlainTextInput.eligible(null, Object.class)
                && "input_layout".equals(GeminiPlainTextInput.rejectionReason(new Object(), Object.class))
                && !GeminiPlainTextInput.eligible(new Object(), Object.class),
                "public eligibility and rejection reasons disagree");
        for (Source source : Source.values()) {
            Map<String, Object> candidate = plain(); candidate.put("b", source);
            require(eligible(candidate) == (source == Source.TEXT), "non-text source admitted: " + source);
        }
        reject("b", "TEXT"); reject("b", null); reject("a", " "); reject("a", null);
        for (String name : LISTS.split(" ")) {
            reject(name, Arrays.asList("attached/context/tool value"));
            reject(name, null); reject(name, "[]");
        }
        for (String name : OBJECTS.split(" ")) {
            reject(name, new Object()); reject(name, Collections.emptyList());
        }
        for (String name : STRINGS.split(" ")) {
            reject(name, "context value"); reject(name, 0);
            Map<String, Object> empty = plain(); empty.put(name, "");
            require(eligible(empty), "empty optional text metadata treated as a supplied context");
        }
        for (String name : FLAGS.split(" ")) { reject(name, true); reject(name, null); reject(name, 0); }
        reject("o", null); reject("o", new Object());
        reject("p", 0); reject("p", null); reject("p", "-1");
        reject("ar", 1); reject("ar", null); reject("ar", "0");
        Map<String, Object> missing = plain(); missing.remove("x");
        try { eligible(missing); throw new AssertionError("missing field treated as empty"); }
        catch (NoSuchFieldException expected) { }
        require(!GeminiPlainTextInput.eligible(new Object(), Object.class)
                && !GeminiPlainTextInput.eligible(null, Object.class)
                && !GeminiPlainTextInput.eligible(new Object(), String.class), "unreviewed input class/layout admitted");
        Set<String> names = new HashSet<>();
        for (char c = 'a'; c <= 'z'; c++) { names.add("" + c); names.add("" + Character.toUpperCase(c)); }
        for (char c = 'a'; c <= 'r'; c++) { names.add("a" + c); }
        require(GeminiPlainTextInput.reviewedFieldNames(names), "reviewed MessageInput field set rejected");
        names.add("newAttachment");
        require(!GeminiPlainTextInput.reviewedFieldNames(names), "unknown field layout admitted");
        names.remove("newAttachment"); names.remove("c");
        require(!GeminiPlainTextInput.reviewedFieldNames(names), "missing attachment field admitted");
        System.out.println("Plain-text eligibility: typed source, every attachment/tool/context guard, "
                + "nullable metadata, default audio and unknown-layout stock fallback passed");
    }

    private static Map<String, Object> plain() {
        Map<String, Object> values = new HashMap<>();
        values.put("a", "Use Ostadix for this request"); values.put("b", Source.TEXT); values.put("o", DEFAULT_AUDIO);
        values.put("p", -1); values.put("ar", 0);
        for (String name : LISTS.split(" ")) { values.put(name, Collections.emptyList()); }
        for (String name : OBJECTS.split(" ")) { values.put(name, null); }
        for (String name : STRINGS.split(" ")) { values.put(name, null); }
        for (String name : FLAGS.split(" ")) { values.put(name, false); }
        return values;
    }

    private static boolean eligible(Map<String, Object> values) throws ReflectiveOperationException {
        return GeminiPlainTextInput.eligibleFields(name -> {
            if (!values.containsKey(name)) { throw new NoSuchFieldException(name); }
            return values.get(name);
        }, DEFAULT_AUDIO);
    }

    private static void reject(String name, Object value) throws Exception {
        Map<String, Object> values = plain(); values.put(name, value);
        require(!eligible(values), "unsupported input admitted at " + name);
    }

    private static void require(boolean condition, String message) {
        if (!condition) { throw new AssertionError(message); }
    }
}
