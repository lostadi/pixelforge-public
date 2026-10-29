import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import agent
import ostadix_pair as pair


class PairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.home = Path(self.temporary.name)
        self.home_patch = patch.object(pair, 'PAIR_HOME', self.home)
        self.home_patch.start()
        self.addCleanup(self.home_patch.stop)
        self.network_patch = patch.object(pair, '_network_environment', return_value={})
        self.network_patch.start()
        self.addCleanup(self.network_patch.stop)
        self.old = self.make_pair('a' * 40)
        pair._write_manifest(self.old)

    def make_pair(self, revision):
        root = self.home / 'checkouts' / revision
        (root / '.git').mkdir(parents=True)
        (root / 'backends').mkdir()
        binaries = {name: str(root / 'target' / 'release' / name) for name in pair.BINS}
        binaries['ostadix-mcp'] = str(root / 'mcp/ostadix_lang_mcp_server/target/release/ostadix-mcp')
        for name, filename in binaries.items():
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('#!/bin/sh\n# ' + revision + ' ' + name + '\n')
            path.chmod(0o700)
        return {'revision': revision, 'root': str(root), 'backends': str(root / 'backends'),
                'binary': binaries['ostadix-mcp'], 'binaries': binaries, 'upstream': pair.UPSTREAM,
                'sha256': {name: pair._hash(path) for name, path in binaries.items()},
                'verified_at': '2026-01-01T00:00:00+00:00', 'checked_at': '2026-01-01T00:00:00+00:00'}

    def git(self, arguments, cwd=None, timeout=20):
        if arguments[:1] == ['ls-remote']:
            return self.remote_revision + '\tHEAD'
        if arguments == ['rev-parse', 'HEAD']:
            return Path(cwd).name
        if arguments == ['remote', 'get-url', 'origin']:
            return pair.UPSTREAM
        return ''

    def test_current_revision_reuses_verified_build_and_checks_upstream(self):
        self.remote_revision = self.old['revision']
        with patch.object(pair, '_git', side_effect=self.git) as git, patch.object(pair, '_health') as health:
            current = pair.update_pair()
            self.assertEqual(current['revision'], self.old['revision'])
            self.assertEqual(current['verified_at'], self.old['verified_at'])
            self.assertNotEqual(current['checked_at'], self.old['checked_at'])
            self.assertEqual(pair.load_pair(), current)
            self.assertEqual(git.call_args_list[0].args[0], ['ls-remote', pair.UPSTREAM, 'HEAD'])
            health.assert_not_called()

    def test_binary_fingerprint_mismatch_is_rejected(self):
        Path(self.old['binaries']['O']).write_text('a different executable')
        with patch.object(pair, '_git', side_effect=self.git):
            with self.assertRaisesRegex(pair.PairError, 'does not match the recorded build'):
                pair.load_pair()

    def test_repository_revision_mismatch_is_rejected(self):
        with patch.object(pair, '_git', return_value='b' * 40):
            with self.assertRaisesRegex(pair.PairError, 'repository revision changed'):
                pair.load_pair()

    def test_malformed_hash_records_are_reported_as_invalid_pair(self):
        for hashes in (None, [], 'invalid'):
            with self.subTest(hashes=hashes), patch.object(pair, '_git', side_effect=self.git):
                pair._write_manifest(dict(self.old, sha256=hashes))
                with self.assertRaisesRegex(pair.PairError, 'invalid executable hash records'):
                    pair.load_pair()

    def test_fetch_failure_preserves_last_verified_pair(self):
        before = (self.home / 'current.json').read_bytes()

        def git(arguments, cwd=None, timeout=20):
            if arguments[:1] == ['ls-remote']:
                raise pair.PairError('offline')
            return self.git(arguments, cwd, timeout)

        messages = []
        with patch.object(pair, '_git', side_effect=git), \
                patch.object(pair, '_run_build') as build:
            self.assertEqual(pair.update_pair(progress=messages.append), self.old)
        build.assert_not_called()
        self.assertTrue(any('freshness is unknown' in message for message in messages))
        self.assertEqual((self.home / 'current.json').read_bytes(), before)
        with patch.object(pair, '_git', side_effect=self.git):
            self.assertEqual(pair.load_pair(), self.old)

    def test_unavailable_upstream_and_unverifiable_cache_preserve_manifest_but_report_failure(self):
        before = (self.home / 'current.json').read_bytes()
        with patch.object(pair, '_git', side_effect=pair.PairError('offline')), \
                self.assertRaisesRegex(pair.PairError, 'offline'):
            pair.update_pair()
        self.assertEqual((self.home / 'current.json').read_bytes(), before)

    def test_failed_new_health_check_does_not_activate_new_pair(self):
        self.remote_revision = 'b' * 40
        self.make_pair(self.remote_revision)
        before = (self.home / 'current.json').read_bytes()
        with patch.object(pair, '_git', side_effect=self.git), \
             patch.object(pair, '_run_build', return_value=0), \
             patch.object(pair, '_health', side_effect=pair.PairError('smoke failed')):
            with self.assertRaisesRegex(pair.PairError, 'smoke failed'):
                pair.update_pair()
        self.assertEqual((self.home / 'current.json').read_bytes(), before)
        with patch.object(pair, '_git', side_effect=self.git):
            self.assertEqual(pair.load_pair(), self.old)

    def test_build_timeout_does_not_activate_new_pair(self):
        self.remote_revision = 'b' * 40
        self.make_pair(self.remote_revision)
        before = (self.home / 'current.json').read_bytes()
        with patch.object(pair, '_git', side_effect=self.git), \
             patch.object(pair, '_run_build', side_effect=subprocess.TimeoutExpired('cargo', 1800)):
            with self.assertRaisesRegex(pair.PairError, 'did not finish'):
                pair.update_pair()
        self.assertEqual((self.home / 'current.json').read_bytes(), before)

    def test_update_rejects_configured_source_from_another_repository(self):
        source = self.home / 'installed-source'
        source.mkdir()
        (self.home / 'source.json').write_text(json.dumps({'repository': str(source)}))
        pair._write_manifest(dict(self.old, upstream='https://github.com/lostadi/OSTADIX.git'))
        before = (self.home / 'current.json').read_bytes()

        def git(arguments, cwd=None, timeout=20):
            if arguments[:1] == ['ls-remote']:
                raise pair.PairError('network unavailable')
            if arguments == ['remote', 'get-url', 'origin'] and cwd == source:
                return 'https://github.com/lostadi/OSTADIX.git'
            return self.git(arguments, cwd, timeout)

        with patch.object(pair, '_git', side_effect=git) as commands, \
                patch.object(pair, '_run_build') as build, \
                self.assertRaisesRegex(pair.PairError, 'origin is not OSTADIX-BM'):
            pair.update_pair()
        self.assertEqual(commands.call_args_list[-1].args, (['remote', 'get-url', 'origin'], source))
        build.assert_not_called()
        self.assertEqual((self.home / 'current.json').read_bytes(), before)

    def test_online_update_succeeds_when_optional_source_directory_was_removed(self):
        (self.home / 'source.json').write_text(json.dumps({'repository': str(self.home / 'removed-source')}))
        self.remote_revision = self.old['revision']
        with patch.object(pair, '_git', side_effect=self.git), \
                patch.object(pair, '_run_build') as build:
            current = pair.update_pair()
            self.assertEqual(pair.load_pair(), current)
        build.assert_not_called()
        self.assertEqual(current['revision'], self.old['revision'])
        self.assertNotEqual(current['checked_at'], self.old['checked_at'])

    def test_unavailable_upstream_keeps_verified_pair_when_optional_source_was_removed(self):
        (self.home / 'source.json').write_text(json.dumps({'repository': str(self.home / 'removed-source')}))
        before = (self.home / 'current.json').read_bytes()

        def git(arguments, cwd=None, timeout=20):
            if arguments[:1] == ['ls-remote']:
                raise pair.PairError('network unavailable')
            return self.git(arguments, cwd, timeout)

        with patch.object(pair, '_git', side_effect=git), \
                patch.object(pair, '_run_build') as build:
            self.assertEqual(pair.update_pair(), self.old)
        build.assert_not_called()
        self.assertEqual((self.home / 'current.json').read_bytes(), before)

    def test_unavailable_upstream_can_build_from_configured_ostadix_bm_source(self):
        source = self.home / 'installed-source'
        source.mkdir()
        (self.home / 'source.json').write_text(json.dumps({'repository': str(source)}))
        self.remote_revision = 'b' * 40
        self.make_pair(self.remote_revision)
        (self.home / 'current.json').unlink()
        messages = []

        def git(arguments, cwd=None, timeout=20):
            if arguments[:1] == ['ls-remote']:
                raise pair.PairError('network unavailable')
            if arguments == ['rev-parse', 'HEAD'] and cwd == source:
                return self.remote_revision
            return self.git(arguments, cwd, timeout)

        def health(candidate):
            self.assertFalse((self.home / 'current.json').exists())
            self.assertEqual(candidate['revision'], self.remote_revision)
            return {'smoke': 'passed'}

        with patch.object(pair, '_git', side_effect=git) as commands, \
                patch.object(pair, '_run_build', return_value=0) as build, \
                patch.object(pair, '_health', side_effect=health):
            current = pair.update_pair(progress=messages.append)

        self.assertEqual(current['upstream'], 'https://github.com/lostadi/ostadix-BM.git')
        self.assertEqual(current['health'], {'smoke': 'passed'})
        self.assertEqual(build.call_count, 2)
        self.assertTrue(any(call.args[0] == ['fetch', '--depth', '1', str(source), self.remote_revision]
                            for call in commands.call_args_list))
        self.assertTrue(any('network unavailable' in message for message in messages))
        self.assertTrue((self.home / 'current.json').exists())
        with patch.object(pair, '_git', side_effect=self.git):
            self.assertEqual(pair.load_pair(), current)

    def test_unexpected_managed_checkout_origin_preserves_verified_pair(self):
        self.remote_revision = 'b' * 40
        candidate = self.make_pair(self.remote_revision)
        before = (self.home / 'current.json').read_bytes()

        def git(arguments, cwd=None, timeout=20):
            if arguments == ['remote', 'get-url', 'origin'] and cwd == Path(candidate['root']):
                return 'https://github.com/lostadi/OSTADIX.git'
            return self.git(arguments, cwd, timeout)

        with patch.object(pair, '_git', side_effect=git), \
                patch.object(pair, '_run_build') as build, \
                self.assertRaisesRegex(pair.PairError, 'unexpected origin'):
            pair.update_pair()
        build.assert_not_called()
        self.assertEqual((self.home / 'current.json').read_bytes(), before)

    def test_successful_health_check_precedes_activation_of_matched_pair(self):
        self.remote_revision = 'b' * 40
        self.make_pair(self.remote_revision)
        before = (self.home / 'current.json').read_bytes()
        old_binary = Path(self.old['binary']).read_bytes()

        def health(candidate):
            self.assertEqual((self.home / 'current.json').read_bytes(), before)
            self.assertEqual(candidate['revision'], self.remote_revision)
            return {'smoke': 'passed'}

        with patch.object(pair, '_git', side_effect=self.git), \
                patch.object(pair, '_run_build', return_value=0) as build, \
                patch.object(pair, '_health', side_effect=health):
            current = pair.update_pair()
            self.assertEqual(pair.load_pair(), current)
        self.assertEqual(build.call_count, 2)
        self.assertEqual(current['revision'], self.remote_revision)
        self.assertEqual(current['health'], {'smoke': 'passed'})
        self.assertEqual(Path(self.old['binary']).read_bytes(), old_binary)
        self.assertNotEqual((self.home / 'current.json').read_bytes(), before)

    def test_failed_manifest_replace_preserves_previous_pair_and_removes_temporary_file(self):
        before = (self.home / 'current.json').read_bytes()
        with patch.object(pair.os, 'replace', side_effect=OSError('replace failed')), \
                self.assertRaisesRegex(OSError, 'replace failed'):
            pair._write_manifest(dict(self.old, revision='b' * 40))
        self.assertEqual((self.home / 'current.json').read_bytes(), before)
        self.assertEqual(list(self.home.glob('pair-*.json')), [])

    def test_environment_always_selects_matched_binaries(self):
        with patch.dict(pair.os.environ, {'O_LANG_ROOT': '/old/repo', 'OLANG': '/old/O'}):
            env = pair.pair_environment(self.old)
        self.assertEqual(env['O_LANG_ROOT'], self.old['root'])
        self.assertEqual(env['OLANG'], self.old['binaries']['O'])
        self.assertEqual(env['OSTADIX_O_CLI_BIN'], self.old['binaries']['o-cli'])
        self.assertEqual(env['PATH'].split(':')[0], str(Path(self.old['binaries']['O']).parent))

    def test_termux_exec_workaround_is_local_to_mcp_children(self):
        preload = '/termux/libtermux-exec-ld-preload.so:/other/preload.so'
        with patch.dict(pair.os.environ, {'LD_PRELOAD': preload}):
            env = pair.pair_environment(self.old)
            self.assertEqual(pair.os.environ['LD_PRELOAD'], preload)
            self.assertEqual(env['LD_PRELOAD'], '/other/preload.so')
        with patch.dict(pair.os.environ, {'LD_PRELOAD': '/termux/libtermux-exec.so'}):
            self.assertNotIn('LD_PRELOAD', pair.pair_environment(self.old))

    def test_smoke_checks_actual_output_and_successful_job(self):
        result = {'structuredContent': {'action': 'execute', 'state': 'completed',
                                       'exit_code': 0, 'result': {'ok': True, 'type': 'number',
                                       'value': {'t': 'number', 'v': {'kind': 'int', 'v': '2'}}}}}
        pair._smoke_result(result, 'execute')
        for changes in ({'result': 'wrong', 'job_id': '2'}, {'exit_code': 2}, {'state': 'running'}):
            bad = {'structuredContent': dict(result['structuredContent'], **changes)}
            with self.assertRaises(pair.PairError):
                pair._smoke_result(bad, 'execute')
        with self.assertRaises(pair.PairError):
            pair._smoke_result({'content': [{'type': 'text', 'text': '2'}]}, 'execute')


class RepositoryNetworkTests(unittest.TestCase):
    def test_local_repository_verification_does_not_prepare_network(self):
        result = Mock(returncode=0, stdout='a' * 40 + '\n')
        for arguments in (['rev-parse', 'HEAD'], ['diff', '--exit-code', 'HEAD', '--']):
            with self.subTest(arguments=arguments), \
                    patch.object(pair, '_network_environment') as network, \
                    patch.object(pair.subprocess, 'run', return_value=result) as run:
                self.assertEqual(pair._git(arguments), 'a' * 40)
                network.assert_not_called()
                self.assertEqual(run.call_args.kwargs['env']['GIT_TERMINAL_PROMPT'], '0')

    def test_remote_repository_operations_use_prepared_network_environment(self):
        result = Mock(returncode=0, stdout='fixture\n')
        for arguments in (['ls-remote', pair.UPSTREAM, 'HEAD'], ['fetch', 'origin', 'a' * 40]):
            with self.subTest(arguments=arguments), \
                    patch.object(pair, '_network_environment', return_value={'HTTPS_PROXY': 'http://fixture:8889'}) as network, \
                    patch.object(pair.subprocess, 'run', return_value=result) as run:
                self.assertEqual(pair._git(arguments), 'fixture')
                network.assert_called_once_with()
                self.assertEqual(run.call_args.kwargs['env'], {
                    'HTTPS_PROXY': 'http://fixture:8889', 'GIT_TERMINAL_PROMPT': '0'})

    def test_network_setup_failure_is_reported_as_pair_error(self):
        for fail_at in ('environment', 'relay'):
            with self.subTest(fail_at=fail_at), \
                    patch.object(agent, 'codex_environment', return_value={}) as environment, \
                    patch.object(agent, '_ensure_codex_network') as relay, \
                    patch.object(pair.subprocess, 'run') as run:
                (environment if fail_at == 'environment' else relay).side_effect = agent.AppError('fixture unavailable')
                with self.assertRaisesRegex(pair.PairError, 'network setup failed: fixture unavailable'):
                    pair._git(['ls-remote', pair.UPSTREAM, 'HEAD'])
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
