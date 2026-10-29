package org.ostadix.aicore.extension;

public final class GeminiLocalRouteSelfTest {
    public static void main(String[] args) {
        String[] ordinary = {null, "", " ", "What is two plus two?", "Set a timer for ten minutes",
                "Write a poem", "What is Ostadix?", "Explain the last error", "use ostadixian tools",
                "use ostadix_like syntax", "use ostadix-like colors", "Say: Use Ostadix to calculate 2+2",
                "\"Use Ostadix\" is the title", "Do not use Ostadix for this"};
        for (String text : ordinary) {
            require(!GeminiLocalRoute.selects("local-o-v1", text), "ordinary request intercepted: " + text);
        }
        for (String text : new String[]{"Use Ostadix to calculate 2+2", " USE OSTADIX: run this",
                "use\tOstadix, please calculate 2+2", "Use Ostadix"}) {
            require(GeminiLocalRoute.selects("local-o-v1", text), "explicit request rejected");
            for (String mode : new String[]{null, "", "off", "local-all-v1", "unknown"}) {
                require(!GeminiLocalRoute.selects(mode, text), "unsupported/catch-all mode accepted");
            }
        }
        Object first = new String("equal-input");
        Object equalButDifferent = new String("equal-input");
        GeminiLocalRoute.Claims claims = new GeminiLocalRoute.Claims();
        claims.put(first, "chat-one");
        require(claims.remove(equalButDifferent) == null, "equal unclaimed input stole another turn");
        claims.put(equalButDifferent, "chat-two");
        require("chat-one".equals(claims.remove(first)), "identity claim was overwritten");
        require(claims.remove(first) == null, "request claimed twice");
        require("chat-two".equals(claims.remove(equalButDifferent)), "second claim lost");
        claims.put(first, "old"); claims.put(first, "new");
        require("new".equals(claims.remove(first)) && claims.remove(first) == null,
                "same input had duplicate stale claims");
        System.out.println("Gemini routing: ordinary requests and legacy catch-all pass through; explicit prefix and single identity claims verified");
    }

    private static void require(boolean value, String message) {
        if (!value) { throw new AssertionError(message); }
    }
}
