import importlib.util
from pathlib import Path
import tempfile
import unittest
import os

spec = importlib.util.spec_from_file_location('assistant_permissions',
    Path(__file__).resolve().parents[1] / 'scripts/check_assistant_permissions.py')
doctor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doctor)


class AssistantPermissionTests(unittest.TestCase):
    def test_route_requires_explicit_selection_and_identifies_legacy_catch_all(self):
        self.assertTrue(doctor.assistant_route_status('local-o-v1')['passed'])
        for mode in ('local-all-v1', 'null', '', 'off', 'unknown', 'local-o-v2'):
            with self.subTest(mode=mode):
                report = doctor.assistant_route_status(mode)
                self.assertFalse(report['passed'])
                self.assertEqual(report['state'], mode)
                self.assertEqual(report['expected'], 'local-o-v1')
                self.assertEqual(report['unsafe_legacy_catch_all'], mode == 'local-all-v1')

    def test_android_signer_selection_honors_sdk_rotation_and_ignores_source_stamp(self):
        output = ('V3.1 Signer: (minSdkVersion=33, maxSdkVersion=2147483647) '
                  'certificate SHA-256 digest: ' + 'a' * 64 + '\n'
                  'V3.0 Signer: (minSdkVersion=24, maxSdkVersion=32) '
                  'certificate SHA-256 digest: ' + 'b' * 64 + '\n'
                  'Source Stamp Signer: certificate SHA-256 digest: ' + 'c' * 64 + '\n')
        self.assertEqual(doctor.signers_for_sdk(output, 37), ['a' * 64])
        self.assertEqual(doctor.signers_for_sdk(output, 30), ['b' * 64])
        self.assertEqual(doctor.signers_for_sdk(
            'V3.0 Signer: certificate SHA-256 digest: ' + 'd' * 64, 37), ['d' * 64])

    def test_gate_requires_complete_reviewed_tuple_not_unused_constants(self):
        source = '''static final String GSA_PACKAGE = "gsa";
        private static final long OLD_CODE = 1L;
        private static final String OLD_NAME = "old";
        private static final String OLD_APK = "old-hash";
        private static final long NEW_CODE = 2L;
        private static final String NEW_NAME = "new";
        private static final String NEW_APK = "new-hash";
        private static final String SIGNER = "trusted";
        return packageMatches(context, GSA_PACKAGE, OLD_CODE, OLD_NAME, OLD_APK, SIGNER)
            || packageMatches(context, GSA_PACKAGE, NEW_CODE, NEW_NAME, NEW_APK, SIGNER);'''
        identities = doctor.allowed_package_identities(source)['gsa']
        self.assertEqual(len(identities), 2)
        old = dict(version_code=1, version_name='old')
        new = dict(version_code=2, version_name='new')
        self.assertTrue(doctor.identity_matches(old, 'old-hash', ['trusted'], identities))
        self.assertTrue(doctor.identity_matches(new, 'new-hash', ['trusted'], identities))
        self.assertFalse(doctor.identity_matches(new, 'old-hash', ['trusted'], identities))
        self.assertFalse(doctor.identity_matches(new, 'new-hash', ['untrusted'], identities))
        self.assertFalse(doctor.identity_matches(new, 'new-hash', ['trusted', 'extra'], identities))
        without_call = source.split('||')[0]
        self.assertEqual(len(doctor.allowed_package_identities(without_call)['gsa']), 1)

    def test_active_package_permissions_exclude_hidden_backup(self):
        dump = '''Packages:
  Package [example]:
    appId=10191
    versionCode=123 minSdk=32
    versionName=current
    requested permissions:
      android.permission.INTERNET
      android.permission.POST_NOTIFICATIONS
      android.permission.OPTIONAL
    install permissions:
      android.permission.INTERNET: granted=true
    User 0:
      runtime permissions:
        android.permission.POST_NOTIFICATIONS: granted=false, flags=[]
Hidden system packages:
  Package [example]:
    appId=10191
    versionName=old
    runtime permissions:
      android.permission.POST_NOTIFICATIONS: granted=true
'''
        result = doctor.package_snapshot(dump)
        self.assertEqual(result['uid'], 10191)
        self.assertEqual(result['version_name'], 'current')
        self.assertEqual(result['declared_permissions'], {
            'android.permission.INTERNET': True,
            'android.permission.POST_NOTIFICATIONS': False,
            'android.permission.OPTIONAL': None})

    def test_credentials_require_all_three_fields_without_serializing_secrets(self):
        token = 'private-secret-' * 8
        host = dict(token=token, port=12345)
        client = dict(token=token, port=12345, certificate_sha256='f' * 64)
        self.assertTrue(doctor.client_matches(client, host, 'f' * 64))
        for field, replacement in [('token', 'different'), ('port', 12346),
                                   ('certificate_sha256', 'a' * 64)]:
            changed = dict(client, **{field: replacement})
            self.assertFalse(doctor.client_matches(changed, host, 'f' * 64))
        self.assertFalse(doctor.client_matches(dict(client, token=123), host, 'f' * 64))

    def test_private_permissions_reject_world_readability_and_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'configuration'
            path.write_text('not a real credential')
            path.chmod(0o600)
            self.assertTrue(doctor.file_status(path, os.getuid(), private=True)['passed'])
            self.assertFalse(doctor.file_status(path, os.getuid() + 1, private=True)['passed'])
            path.chmod(0o644)
            self.assertFalse(doctor.file_status(path, os.getuid(), private=True)['passed'])
            path.chmod(0o600)
            link = Path(temporary) / 'link'
            link.symlink_to(path)
            self.assertFalse(doctor.file_status(link, os.getuid(), private=True)['passed'])

    def test_boot_entry_requires_executable_and_protected_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'boot.sh'
            path.write_text('#!/system/bin/sh\n')
            path.chmod(0o644)
            self.assertFalse(doctor.file_status(path, os.getuid(), executable=True)['passed'])
            path.chmod(0o755)
            self.assertTrue(doctor.file_status(path, os.getuid(), executable=True)['passed'])
            path.chmod(0o775)
            self.assertFalse(doctor.file_status(path, os.getuid(), executable=True)['passed'])


if __name__ == '__main__':
    unittest.main()
