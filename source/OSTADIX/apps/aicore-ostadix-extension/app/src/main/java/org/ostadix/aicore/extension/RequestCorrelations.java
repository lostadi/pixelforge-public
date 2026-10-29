package org.ostadix.aicore.extension;

import java.util.Iterator;
import java.util.Map;
import java.util.WeakHashMap;

/** Correlates one Smart Reply callback and cancellation handle with its terminal state. */
final class RequestCorrelations {
    static final class State {
        final RequestIdentity identity;
        private boolean cancelled;
        private boolean completed;

        State(RequestIdentity identity) {
            this.identity = identity;
        }
    }

    private final Map<Object, State> callbacks = new WeakHashMap<Object, State>();
    private final Map<Object, State> cancellations = new WeakHashMap<Object, State>();

    synchronized State begin(Object callback, RequestIdentity identity) {
        State state = new State(identity);
        if (callback != null) {
            callbacks.put(callback, state);
        }
        return state;
    }

    synchronized void registerCancellation(Object cancellation, State state) {
        if (cancellation != null && !state.cancelled && !state.completed) {
            cancellations.put(cancellation, state);
        }
    }

    synchronized State claimCallback(Object callback) {
        if (callback == null) {
            return null;
        }
        State state = callbacks.remove(callback);
        if (state == null || state.completed) {
            return null;
        }
        return state;
    }

    synchronized State cancel(Object cancellation) {
        State state = cancellations.get(cancellation);
        if (state == null || state.completed) {
            return null;
        }
        state.cancelled = true;
        removeCorrelations(state);
        return state;
    }

    synchronized boolean isCancelled(State state) {
        return state.cancelled;
    }

    /** Atomically decides whether replacement delivery may begin and closes correlation. */
    synchronized boolean commitReplacement(State state) {
        if (state.cancelled || state.completed) {
            state.completed = true;
            removeCorrelations(state);
            return false;
        }
        state.completed = true;
        removeCorrelations(state);
        return true;
    }

    synchronized void complete(State state) {
        state.completed = true;
        removeCorrelations(state);
    }

    synchronized int callbackCountForTest() {
        return callbacks.size();
    }

    synchronized int cancellationCountForTest() {
        return cancellations.size();
    }

    private void removeCorrelations(State state) {
        removeState(callbacks, state);
        removeState(cancellations, state);
    }

    private static void removeState(Map<Object, State> correlations, State state) {
        Iterator<State> values = correlations.values().iterator();
        while (values.hasNext()) {
            if (values.next() == state) {
                values.remove();
            }
        }
    }
}
