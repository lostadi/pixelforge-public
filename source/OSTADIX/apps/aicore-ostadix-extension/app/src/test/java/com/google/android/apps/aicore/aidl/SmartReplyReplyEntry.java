package com.google.android.apps.aicore.aidl;

public final class SmartReplyReplyEntry {
    private final String text;
    private final int safety;
    private final float score;

    public SmartReplyReplyEntry(String text, int safety, float score) {
        this.text = text;
        this.safety = safety;
        this.score = score;
    }

    public String getText() { return text; }
    public int getSafetyClassificationResult() { return safety; }
    public float getScore() { return score; }
}
