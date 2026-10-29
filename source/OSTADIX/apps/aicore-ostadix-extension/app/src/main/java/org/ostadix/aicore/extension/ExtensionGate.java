package org.ostadix.aicore.extension;

import android.content.Context;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.Signature;
import android.os.Build;
import android.os.PowerManager;
import android.provider.Settings;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.security.MessageDigest;
import java.util.Locale;

final class ExtensionGate {
    static final String ASOSS_PACKAGE = "com.google.android.as.oss";
    static final String ASOSS_PROCESS = "com.google.android.as.oss";
    static final String ASI_PACKAGE = "com.google.android.as";
    static final String ASI_PROCESS = "com.google.android.as";
    static final String GSA_PACKAGE = "com.google.android.googlequicksearchbox";
    static final String GSA_PROCESS = "com.google.android.googlequicksearchbox:search";
    static final String MODULE_PACKAGE = "org.ostadix.aicore.extension";

    static final String AICORE_PACKAGE = "com.google.android.aicore";
    private static final String EXPECTED_FRAMEWORK = "Vector";
    private static final String EXPECTED_FINGERPRINT =
            "google/blazer/blazer:17/CP2A.260805.005/15828068:user/release-keys";
    private static final long EXPECTED_AICORE_CODE = 494417L;
    private static final String EXPECTED_AICORE_NAME =
            "0.release.prod_aicore_20260723.00_RC11.964081323";
    private static final String EXPECTED_AICORE_APK =
            "67aa6c6cc457163d18b8cff35706eeffd60ac234fa10bf4f3b4b7bd8e1d45f57";
    private static final String EXPECTED_AICORE_SIGNER =
            "b7971ccc10a03932e14a3557a1b4c2a84be0ecb506777f0c72dd46cf5d7093c6";
    // RC13 retains the reviewed Java/native seams and byte-identical Nano
    // runtime libraries. Keep the previous reviewed release in the allowlist.
    private static final long REVIEWED_AICORE_RC13_CODE = 494585L;
    private static final String REVIEWED_AICORE_RC13_NAME =
            "0.release.prod_aicore_20260723.00_RC13.981368508";
    private static final String REVIEWED_AICORE_RC13_APK =
            "d3f749159f6d4b691093d6c8118b64fe3e7892b77180e8d473577e79114c864e";
    private static final long EXPECTED_ASOSS_CODE = 143685L;
    private static final String EXPECTED_ASOSS_NAME = "1.0.release.962568596";
    private static final String EXPECTED_ASOSS_APK =
            "c08952bafbd19a4bcb4399aaebc17ae7b1685c8b20cad033a41b435fbfe3620c";
    private static final String EXPECTED_ASOSS_SIGNER =
            "071f09456bf1a8e8ad2e808ffe6a0ebc13582a7e6f9aba13e47280ad9a85d833";
    private static final long EXPECTED_ASI_CODE = 16934935L;
    private static final String EXPECTED_ASI_NAME = "C.6.playstore.pixel11.961955194";
    private static final String EXPECTED_ASI_APK =
            "16d2b265fbea8c8abc46b537b6191628f7a855d7e3fd760b4a60bb6ea9c93e68";
    private static final String EXPECTED_ASI_SIGNER =
            "3af39ab967aaa5d279e49b5f769cb66e40799838bc8799343ee57ae435d2455b";
    private static final long EXPECTED_GSA_CODE = 301803623L;
    private static final String EXPECTED_GSA_NAME = "17.56.15.sa.arm64";
    private static final String EXPECTED_GSA_APK =
            "c227beb9468f1740c395e457d5f06fb780f288a89156d8487ef069953c1c359a";
    private static final long REVIEWED_GSA_CODE = 301806951L;
    private static final String REVIEWED_GSA_NAME = "17.58.16.sa.arm64";
    private static final String REVIEWED_GSA_APK =
            "711714ac6df264bf5d319358f9dc8bae867f8a89ee13e7fc5d0e4d3df2611ff9";
    private static final String EXPECTED_GSA_SIGNER =
            "7ce83c1b71f3d572fed04c8d40c5cb10ff75e6d87d9df6fbd53f0468c2905053";
    private static final String ACTIVATION_SETTING = "ostadix_aicore_extension_token";
    private static final String ACTIVATION_TOKEN = EXPECTED_FINGERPRINT
            + ":asoss-smart-reply-result-v1:" + EXPECTED_ASOSS_APK;
    static final String SMART_REPLY_EXPERIMENT_SETTING = "ostadix_asoss_smart_reply_experiment";
    static final String ASI_AUTOFILL_EXPERIMENT_SETTING = "ostadix_asi_autofill_experiment";

    private ExtensionGate() {}

    static boolean matchesEarlyProcess(String processName, String framework, int apiVersion) {
        return Build.VERSION.SDK_INT == 37
                && EXPECTED_FINGERPRINT.equals(Build.FINGERPRINT)
                && (ASOSS_PROCESS.equals(processName) || ASI_PROCESS.equals(processName)
                        || GSA_PROCESS.equals(processName) || AICORE_PACKAGE.equals(processName))
                && EXPECTED_FRAMEWORK.equals(framework)
                && apiVersion == 102;
    }

    static boolean acceptsInstalledPackages(Context context) {
        if (context == null) {
            return false;
        }
        boolean hostMatches;
        if (AICORE_PACKAGE.equals(context.getPackageName())) {
            hostMatches = aicorePackageMatches(context);
        } else if (ASOSS_PACKAGE.equals(context.getPackageName())) {
            hostMatches = packageMatches(context, ASOSS_PACKAGE, EXPECTED_ASOSS_CODE,
                        EXPECTED_ASOSS_NAME, EXPECTED_ASOSS_APK, EXPECTED_ASOSS_SIGNER)
                    && aicorePackageMatches(context);
        } else if (ASI_PACKAGE.equals(context.getPackageName())) {
            hostMatches = packageMatches(context, ASI_PACKAGE, EXPECTED_ASI_CODE,
                    EXPECTED_ASI_NAME, EXPECTED_ASI_APK, EXPECTED_ASI_SIGNER);
        } else if (GSA_PACKAGE.equals(context.getPackageName())) {
            hostMatches = gsaMatches(context);
        } else {
            hostMatches = false;
        }
        return hostMatches;
    }

    private static boolean aicorePackageMatches(Context context) {
        return packageMatches(context, AICORE_PACKAGE, EXPECTED_AICORE_CODE,
                EXPECTED_AICORE_NAME, EXPECTED_AICORE_APK, EXPECTED_AICORE_SIGNER)
                || packageMatches(context, AICORE_PACKAGE, REVIEWED_AICORE_RC13_CODE,
                        REVIEWED_AICORE_RC13_NAME, REVIEWED_AICORE_RC13_APK,
                        EXPECTED_AICORE_SIGNER);
    }

    private static boolean gsaMatches(Context context) {
        return packageMatches(context, GSA_PACKAGE, EXPECTED_GSA_CODE,
                EXPECTED_GSA_NAME, EXPECTED_GSA_APK, EXPECTED_GSA_SIGNER)
                || packageMatches(context, GSA_PACKAGE, REVIEWED_GSA_CODE,
                REVIEWED_GSA_NAME, REVIEWED_GSA_APK, EXPECTED_GSA_SIGNER);
    }

    static boolean acceptsGsaCaller(Context context, int uid, String packageName) {
        if (!GSA_PACKAGE.equals(packageName) || uid < 0) { return false; }
        try {
            return context.getPackageManager().getApplicationInfo(GSA_PACKAGE, 0).uid == uid
                    && gsaMatches(context);
        } catch (PackageManager.NameNotFoundException missing) { return false; }
    }

    static boolean isExplicitlyEnabled(Context context) {
        try {
            if (AICORE_PACKAGE.equals(context.getPackageName())) {
                return "local-factory-234-10745-v1".equals(Settings.Global.getString(
                        context.getContentResolver(), "ostadix_nano_local_probe_token"));
            }
            return ACTIVATION_TOKEN.equals(Settings.Global.getString(
                    context.getContentResolver(), ACTIVATION_SETTING));
        } catch (Throwable error) {
            rethrowIfVmFatal(error);
            return false;
        }
    }

    static boolean isSmartReplyExperimentEnabled(Context context) {
        return candidateExperimentEnabled(context, SMART_REPLY_EXPERIMENT_SETTING);
    }

    static boolean isAsiAutofillExperimentEnabled(Context context) {
        return candidateExperimentEnabled(context, ASI_AUTOFILL_EXPERIMENT_SETTING);
    }

    private static boolean candidateExperimentEnabled(Context context, String setting) {
        try {
            if (context == null || !isExplicitlyEnabled(context)) {
                return false;
            }
            return candidateExperimentAllows(context.getPackageName(), true, setting,
                    Settings.Global.getString(context.getContentResolver(), setting));
        } catch (Throwable error) {
            rethrowIfVmFatal(error);
            return false;
        }
    }

    // Candidate selection is a separate experiment. Enabling the Assistant
    // integration or local Nano receiver must never opt into replacing replies.
    static boolean candidateExperimentAllows(String hostPackage, boolean activated,
            String setting, String value) {
        return activated && "enabled-v1".equals(value)
                && ((ASOSS_PACKAGE.equals(hostPackage)
                        && SMART_REPLY_EXPERIMENT_SETTING.equals(setting))
                    || (ASI_PACKAGE.equals(hostPackage)
                        && ASI_AUTOFILL_EXPERIMENT_SETTING.equals(setting)));
    }

    static boolean thermalPolicyAllows(Context context) {
        try {
            PowerManager power = context.getSystemService(PowerManager.class);
            return power != null
                    && power.getCurrentThermalStatus() < PowerManager.THERMAL_STATUS_SEVERE;
        } catch (Throwable error) {
            rethrowIfVmFatal(error);
            return false;
        }
    }

    static Context currentApplication() {
        try {
            Class<?> activityThread = Class.forName("android.app.ActivityThread");
            return (Context) activityThread.getDeclaredMethod("currentApplication").invoke(null);
        } catch (Throwable error) {
            rethrowIfVmFatal(error);
            return null;
        }
    }

    static void rethrowIfVmFatal(Throwable error) {
        if (error instanceof VirtualMachineError) {
            throw (VirtualMachineError) error;
        }
    }

    private static boolean packageMatches(Context context, String packageName, long code,
            String name, String apkSha256, String signerSha256) {
        try {
            PackageInfo info = context.getPackageManager().getPackageInfo(
                    packageName, PackageManager.GET_SIGNING_CERTIFICATES);
            if (info.getLongVersionCode() != code || !name.equals(info.versionName)
                    || info.signingInfo == null) {
                return false;
            }
            Signature[] signers = info.signingInfo.getApkContentsSigners();
            if (signers == null || signers.length != 1
                    || !signerSha256.equals(sha256(signers[0].toByteArray()))) {
                return false;
            }
            ApplicationInfo application = info.applicationInfo;
            return application != null && application.sourceDir != null
                    && apkSha256.equals(sha256(new File(application.sourceDir)));
        } catch (Throwable error) {
            rethrowIfVmFatal(error);
            return false;
        }
    }

    private static String sha256(File file) {
        try (InputStream input = new FileInputStream(file)) {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] buffer = new byte[64 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) {
                digest.update(buffer, 0, count);
            }
            return hex(digest.digest());
        } catch (Throwable error) {
            rethrowIfVmFatal(error);
            return "";
        }
    }

    private static String sha256(byte[] value) {
        try {
            return hex(MessageDigest.getInstance("SHA-256").digest(value));
        } catch (Throwable error) {
            rethrowIfVmFatal(error);
            return "";
        }
    }

    private static String hex(byte[] value) {
        StringBuilder result = new StringBuilder(value.length * 2);
        for (byte item : value) {
            result.append(String.format(Locale.ROOT, "%02x", item & 0xff));
        }
        return result.toString();
    }
}
