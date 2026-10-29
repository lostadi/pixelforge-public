package org.ostadix.aicore.extension;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;

/** Explicit local requests only; ordinary Gemini turns never acquire a claim. */
final class GeminiLocalRoute {
    private static final Pattern PREFIX = Pattern.compile(
            "^\\s*use\\s+ostadix(?=$|[\\s:,.!?;])", Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE);

    static boolean enabledMode(String mode) { return "local-o-v1".equals(mode); }

    static boolean explicitRequest(String text) {
        return text != null && PREFIX.matcher(text).find();
    }

    static boolean selects(String mode, String text) {
        return enabledMode(mode) && explicitRequest(text);
    }

    /** Host input objects are Kotlin values: equality must not join separate sends. */
    static final class Claims {
        private final List<Entry> entries = new ArrayList<>();

        synchronized void put(Object input, String conversation) {
            remove(input);
            entries.add(new Entry(input, conversation));
        }

        synchronized String remove(Object input) {
            String found = null;
            Iterator<Entry> iterator = entries.iterator();
            while (iterator.hasNext()) {
                Entry entry = iterator.next();
                Object key = entry.input.get();
                if (key == null || key == input) {
                    iterator.remove();
                    if (key != null) { found = entry.conversation; }
                }
            }
            return found;
        }

        private static final class Entry {
            final WeakReference<Object> input;
            final String conversation;
            Entry(Object input, String conversation) {
                this.input = new WeakReference<>(input);
                this.conversation = conversation;
            }
        }
    }

    private GeminiLocalRoute() {}
}
