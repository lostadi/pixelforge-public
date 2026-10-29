package org.ostadix.terminal;

import android.content.Context;
import android.os.CancellationSignal;
import org.json.JSONObject;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.security.MessageDigest;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLContext;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;

/** Translates the AppFunction payload to the primary MCP tool through a pinned local host. */
public final class HostMcpClient implements AutoCloseable {
    private volatile HttpsURLConnection connection;
    private volatile boolean closed;

    public String execute(Context context, String source, String bindingsJson, long timeoutMs,
            String requestId, CancellationSignal cancellation) throws Exception {
        return request(context, source, bindingsJson, timeoutMs, requestId, cancellation, false);
    }

    /** Native parse-only validation: no executable language block is evaluated. */
    public String check(Context context, String source, long timeoutMs,
            String requestId, CancellationSignal cancellation) throws Exception {
        return request(context, source, "{}", timeoutMs, requestId, cancellation, true);
    }

    private String request(Context context, String source, String bindingsJson, long timeoutMs,
            String requestId, CancellationSignal cancellation, boolean checkOnly) throws Exception {
        File configuration = new File(context.getFilesDir(), "ostadix-mcp-host.json");
        if (!configuration.isFile()) {
            throw new IOException("Ostadix MCP host is not configured; nothing was dispatched");
        }
        JSONObject config;
        try (InputStream input = new FileInputStream(configuration)) {
            config = new JSONObject(read(input, 8192));
        }
        final String fingerprint = config.getString("certificate_sha256");
        String token = config.getString("token");
        int port = config.getInt("port");
        if (!fingerprint.matches("[0-9a-f]{64}") || token.length() < 64
                || port < 1 || port > 65535) {
            throw new IOException("Invalid host configuration; nothing was dispatched");
        }
        SSLContext tls = SSLContext.getInstance("TLS");
        tls.init(null, new TrustManager[] { new X509TrustManager() {
            public X509Certificate[] getAcceptedIssuers() { return new X509Certificate[0]; }
            public void checkClientTrusted(X509Certificate[] chain, String authType)
                    throws CertificateException { throw new CertificateException("Client trust unused"); }
            public void checkServerTrusted(X509Certificate[] chain, String authType)
                    throws CertificateException {
                try {
                    if (chain.length != 1) { throw new CertificateException("Unexpected host chain"); }
                    byte[] digest = MessageDigest.getInstance("SHA-256").digest(chain[0].getEncoded());
                    StringBuilder actual = new StringBuilder();
                    for (byte value : digest) { actual.append(String.format(java.util.Locale.ROOT, "%02x", value & 255)); }
                    if (!fingerprint.equals(actual.toString())) { throw new CertificateException("Host pin mismatch"); }
                } catch (CertificateException error) { throw error; }
                catch (Exception error) { throw new CertificateException(error); }
            }
        } }, null);
        HttpsURLConnection active = (HttpsURLConnection) new URL(
                "https://127.0.0.1:" + port + "/o_execute").openConnection();
        active.setSSLSocketFactory(tls.getSocketFactory());
        active.setHostnameVerifier(new javax.net.ssl.HostnameVerifier() {
            public boolean verify(String hostname, javax.net.ssl.SSLSession session) {
                return "127.0.0.1".equals(hostname);
            }
        });
        active.setInstanceFollowRedirects(false);
        active.setRequestMethod("POST");
        active.setConnectTimeout(3000);
        active.setReadTimeout((int) Math.min(timeoutMs + 10000L, 910000L));
        active.setRequestProperty("Authorization", "Bearer " + token);
        active.setRequestProperty("Content-Type", "application/json");
        active.setRequestProperty("X-Ostadix-Request", requestId);
        active.setDoOutput(true);
        JSONObject body = new JSONObject().put("source", source)
                .put("bindings", new JSONObject(bindingsJson))
                .put("constraints", new JSONObject().put("timeout_ms", timeoutMs));
        if (checkOnly) { body.put("action", "check"); }
        byte[] bytes = body.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8);
        active.setFixedLengthStreamingMode(bytes.length);
        connection = active;
        cancellation.setOnCancelListener(new CancellationSignal.OnCancelListener() {
            public void onCancel() { close(); }
        });
        if (closed || cancellation.isCanceled()) { close(); throw new IOException("Cancelled before dispatch"); }
        try {
            try (java.io.OutputStream output = active.getOutputStream()) { output.write(bytes); }
            int status = active.getResponseCode();
            InputStream input = status == 200 ? active.getInputStream() : active.getErrorStream();
            String response;
            try (InputStream result = input) { response = read(result, 8 * 1024 * 1024); }
            if (status != 200) { throw new IOException("MCP host returned " + status + ": " + response); }
            JSONObject toolResult = new JSONObject(response);
            if (!toolResult.has("structuredContent")) { throw new IOException("MCP omitted its structured result; no retry"); }
            return toolResult.getJSONObject("structuredContent").toString();
        } finally { close(); }
    }

    @Override public void close() {
        closed = true;
        HttpsURLConnection active = connection;
        if (active != null) { active.disconnect(); }
    }

    private static String read(InputStream input, int maximum) throws IOException {
        if (input == null) { throw new IOException("Missing response body"); }
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int count;
        while ((count = input.read(buffer)) != -1) {
            if (output.size() + count > maximum) { throw new IOException("Response exceeds host boundary"); }
            output.write(buffer, 0, count);
        }
        return new String(output.toByteArray(), java.nio.charset.StandardCharsets.UTF_8);
    }
}
