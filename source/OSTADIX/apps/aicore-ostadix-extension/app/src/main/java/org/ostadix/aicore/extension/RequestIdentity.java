package org.ostadix.aicore.extension;

final class RequestIdentity {
    final int uid;
    final long sequence;
    final String requestId;
    private final String callerPrefix;

    RequestIdentity(int uid, long sequence) {
        this("asoss-smart-reply", "asoss-caller", uid, sequence);
    }

    RequestIdentity(String requestPrefix, String callerPrefix, int uid, long sequence) {
        this.uid = uid;
        this.sequence = sequence;
        this.requestId = requestPrefix + "-" + uid + "-" + sequence;
        this.callerPrefix = callerPrefix;
    }

    String callerId() {
        return callerPrefix + "/uid-" + uid;
    }
}
