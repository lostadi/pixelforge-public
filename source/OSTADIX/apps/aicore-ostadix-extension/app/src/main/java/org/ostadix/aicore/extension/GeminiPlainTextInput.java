package org.ostadix.aicore.extension;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.HashSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/** Eligibility only: unsupported inputs retain Google's original request and streams. */
final class GeminiPlainTextInput {
    // Reviewed from both pinned MessageInput implementations (auja/aumw).
    private static final String[] EMPTY_LISTS = {
        "c", "d", "e", "W", "x", "z", "A", "L", "N", "ae", "af", "ag"
    };
    private static final String[] ABSENT_OBJECTS = {
        "g", "m", "X", "Z", "t", "aa", "y", "B", "D", "E", "ah", "K", "ai", "O",
        "am", "an", "ao", "ap", "Q", "U"
    };
    private static final String[] EMPTY_STRINGS = {"h", "Y", "F", "ac", "ad", "H", "I", "J", "aq"};
    private static final String[] FALSE_FLAGS = {"f", "n", "s", "G", "M", "aj", "ak", "al", "S"};

    /** pinnedInputType must come from the already version/hash-checked Google class loader. */
    static boolean eligible(Object input, Class<?> pinnedInputType) {
        return rejectionReason(input, pinnedInputType) == null;
    }

    /** Only fixed field names are returned; never request values or exception messages. */
    static String rejectionReason(Object input, Class<?> pinnedInputType) {
        if (input == null || pinnedInputType == null || input.getClass() != pinnedInputType) { return "input_type"; }
        try {
            if (!reviewedLayout(pinnedInputType)) { return "input_layout"; }
            Field audio = pinnedInputType.getDeclaredField("o");
            if (!audio.getType().getName().equals(
                    "com.google.android.apps.search.assistant.surfaces.voice.robin.speech.AudioSessionConfig")) {
                return "audio_type";
            }
            Field defaultAudio = audio.getType().getDeclaredField("a");
            if (!Modifier.isStatic(defaultAudio.getModifiers()) || defaultAudio.getType() != audio.getType()) {
                return "audio_default";
            }
            defaultAudio.setAccessible(true);
            return rejectionFields(name -> {
                Field field = pinnedInputType.getDeclaredField(name);
                field.setAccessible(true);
                Object value = field.get(input);
                // The typed overlay supplies a numeric suggestion-tree tracking reference.
                // Keep the original input object intact; X/t below reject selected
                // suggestions and screen context before local execution can start.
                return "m".equals(name) && reviewedSuggestionReference(value, field.getType()) ? null : value;
            }, defaultAudio.get(null));
        } catch (ReflectiveOperationException | RuntimeException | LinkageError error) {
            return "unavailable_fields";
        }
    }

    static boolean reviewedSuggestionReference(Object value, Class<?> pinnedType) {
        if (value == null) { return true; }
        return pinnedType != null && value.getClass() == pinnedType
                && ("grbq".equals(pinnedType.getName()) || "gpxw".equals(pinnedType.getName()));
    }

    static boolean reviewedLayout(Class<?> type) {
        Set<String> names = new HashSet<>();
        for (Field field : type.getDeclaredFields()) {
            if (Modifier.isStatic(field.getModifiers())) { return false; }
            names.add(field.getName());
        }
        return reviewedFieldNames(names);
    }

    static boolean reviewedFieldNames(Set<String> names) {
        if (names.size() != 70) { return false; }
        for (char name = 'a'; name <= 'z'; name++) {
            if (!names.contains(String.valueOf(name)) || !names.contains(String.valueOf(Character.toUpperCase(name)))) {
                return false;
            }
        }
        for (char name = 'a'; name <= 'r'; name++) {
            if (!names.contains("a" + name)) { return false; }
        }
        return true;
    }

    interface Fields { Object get(String name) throws ReflectiveOperationException; }

    /** Pure value checks shared with the JVM regression test; no Google method is invoked except value equality. */
    static boolean eligibleFields(Fields fields, Object defaultAudio) throws ReflectiveOperationException {
        return rejectionFields(fields, defaultAudio) == null;
    }

    static String rejectionFields(Fields fields, Object defaultAudio) throws ReflectiveOperationException {
        List<String> rejected = new ArrayList<>();
        Object text = fields.get("a");
        Object source = fields.get("b");
        if (!(text instanceof String) || ((String) text).trim().isEmpty()) { rejected.add("a"); }
        if (!(source instanceof Enum) || !"TEXT".equals(((Enum<?>) source).name())) { rejected.add("b"); }
        for (String name : EMPTY_LISTS) {
            Object value = fields.get(name);
            if (!(value instanceof List) || !((List<?>) value).isEmpty()) { rejected.add(name); }
        }
        for (String name : ABSENT_OBJECTS) { if (fields.get(name) != null) { rejected.add(name); } }
        for (String name : EMPTY_STRINGS) {
            Object value = fields.get(name);
            if (value != null && (!(value instanceof String) || !((String) value).isEmpty())) { rejected.add(name); }
        }
        for (String name : FALSE_FLAGS) { if (!Boolean.FALSE.equals(fields.get(name))) { rejected.add(name); } }
        if (!Integer.valueOf(-1).equals(fields.get("p"))) { rejected.add("p"); }
        if (!Integer.valueOf(0).equals(fields.get("ar"))) { rejected.add("ar"); }
        Object audio = fields.get("o");
        if (defaultAudio == null || audio == null || audio.getClass() != defaultAudio.getClass()
                || !defaultAudio.equals(audio)) { rejected.add("o"); }
        return rejected.isEmpty() ? null : "fields:" + String.join(",", rejected);
    }

    private GeminiPlainTextInput() {}
}
