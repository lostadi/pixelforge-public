package org.ostadix.aicore.extension;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/** Reviewed names for the two APKs accepted by ExtensionGate, never a best-effort fallback. */
final class GeminiHostSymbols {
    private static volatile long version;
    private static final Map<String, String> CURRENT;
    static {
        Map<String, String> names = new HashMap<>();
        String[][] pairs = {
            {"asew", "ashd"}, {"asff", "ashm"}, {"asfw", "asid"}, {"asgm", "asit"},
            {"asue", "asxd"}, {"asuo", "asxn"}, {"asvw", "asyv"}, {"asws", "aszs"},
            {"ataa", "atdf"}, {"atma", "atpw"}, {"audl", "auhh"}, {"auja", "aumw"},
            {"aygj", "aymv"}, {"aygx", "aynj"}, {"ayij", "ayov"}, {"ayiz", "aypl"},
            {"aykm", "ayqy"}, {"ayko", "ayra"}, {"aylb", "ayrn"}, {"ayld", "ayrp"},
            {"gaik", "gbjv"}, {"goqd", "gptx"}, {"vko", "vli"},
            {"hdhg", "henb"}, {"hdhs", "henn"}, {"hdkj", "heqe"}, {"hdkl", "heqg"},
            {"hdko", "heqj"}, {"hdkp", "heqk"}, {"hdkt", "heqo"}, {"hdne", "hesz"},
            {"hdvj", "hfbb"}, {"hdwb", "hfbt"}, {"hdwc", "hfbu"}, {"hdwf", "hfbx"},
            {"hdzk", "hffc"}, {"hdzo", "hffg"}, {"hdzp", "hffh"}, {"hdzx", "hffp"},
            {"heab", "hfft"}, {"heac", "hffu"}
        };
        for (String[] pair : pairs) { names.put(pair[0], pair[1]); }
        CURRENT = Collections.unmodifiableMap(names);
    }

    static void configure(long code) {
        if (code != 301803623L && code != 301806951L) {
            throw new IllegalStateException("Unsupported Google app version: " + code);
        }
        version = code;
    }

    static boolean current() {
        if (version == 0) { throw new IllegalStateException("Google app symbols not configured"); }
        return version == 301806951L;
    }

    static String name(String original) {
        if (!current() || original.indexOf('.') >= 0) { return original; }
        String mapped = CURRENT.get(original);
        if (mapped == null) { throw new IllegalArgumentException("Unreviewed Google app symbol: " + original); }
        return mapped;
    }

    private GeminiHostSymbols() {}
}
