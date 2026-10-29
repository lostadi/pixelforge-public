"""Codex discovery and child environment regressions; no cloud calls required."""
import contextlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent
from agent import AppError, codex_launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {'PATH': str(self.root), 'PREFIX': ''})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_missing_installation_is_distinct_from_inaccessible_launcher(self):
        with self.assertRaisesRegex(AppError, 'not on PATH'):
            codex_launcher()
        launcher = self.root / 'codex'
        launcher.write_text('#!/bin/sh\n')
        launcher.chmod(0o600)
        with self.assertRaisesRegex(AppError, 'cannot access or execute'):
            codex_launcher()

    def test_dangling_launcher_reports_existing_installation_problem(self):
        (self.root / 'codex').symlink_to(self.root / 'removed-installation')
        with self.assertRaisesRegex(AppError, 'launcher target'):
            codex_launcher()

    def test_working_node_launcher_survives_earlier_stale_symlink(self):
        stale = self.root / 'stale'; stale.mkdir()
        (stale / 'codex').symlink_to(self.root / 'removed-installation')
        js = self.root / 'codex.js'
        js.write_text('#!/usr/bin/env node\n'); js.chmod(0o700)
        (self.root / 'codex').symlink_to(js)
        node = self.root / 'node'
        node.write_text('fixture'); node.chmod(0o700)
        with patch.dict(os.environ, {'PATH': str(stale) + os.pathsep + str(self.root)}):
            self.assertEqual(codex_launcher(), [str(node), str(js)])

    def test_termux_package_entry_point_takes_precedence_over_personal_wrapper(self):
        wrapper = self.root / 'codex'
        wrapper.write_text('#!/bin/sh\n')
        wrapper.chmod(0o700)
        node = self.root / 'node'
        node.write_text('fixture')
        node.chmod(0o700)
        package = self.root / 'package'
        (package / 'bin').mkdir(parents=True)
        entry = package / 'bin/codex'
        entry.write_text('#!/usr/bin/env node\n')
        entry.chmod(0o700)
        prefix = '/data/data/com.termux/files/usr'

        def fixture_path(value):
            return package if value == prefix else Path(value)

        with patch.dict(os.environ, {'PREFIX': prefix}), \
                patch.object(agent, 'Path', side_effect=fixture_path):
            self.assertEqual(codex_launcher(), [str(node), str(entry)])

    def test_unavailable_package_entry_point_preserves_working_path_launcher(self):
        wrapper = self.root / 'codex'
        wrapper.write_text('#!/bin/sh\n')
        wrapper.chmod(0o700)
        package = self.root / 'package'
        (package / 'bin').mkdir(parents=True)
        prefix = '/data/data/com.termux/files/usr'

        def fixture_path(value):
            return package if value == prefix else Path(value)

        with patch.dict(os.environ, {'PREFIX': prefix}), \
                patch.object(agent, 'Path', side_effect=fixture_path):
            self.assertEqual(codex_launcher(), [str(wrapper)])
            entry = package / 'bin/codex'
            entry.write_text('#!/bin/sh\n')
            entry.chmod(0o600)
            self.assertEqual(codex_launcher(), [str(wrapper)])


class CodexEnvironmentTests(unittest.TestCase):
    PREFIX = '/data/data/com.termux/files/usr'
    PROXY_VARIABLES = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                       'http_proxy', 'https_proxy', 'all_proxy')

    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {
            'PREFIX': self.PREFIX,
            'LD_PRELOAD': 'termux-exec.so',
            'OPENAI_API_KEY': 'fixture-api-key',
            'CODEX_API_KEY': 'fixture-codex-key',
            'OPENAI_BASE_URL': 'https://unused.invalid',
        }, clear=True))
        self.bundle_exists = self.stack.enter_context(patch.object(agent.Path, 'is_file', return_value=True))
        self.bundle_readable = self.stack.enter_context(patch.object(agent.os, 'access', return_value=True))
        self.proxy = self.stack.enter_context(patch.object(agent, '_termux_proxy', return_value=None))
        self.spawn = self.stack.enter_context(patch.object(agent.subprocess, 'Popen'))

    def test_termux_defaults_apply_only_to_the_codex_child(self):
        original = dict(os.environ)

        env = agent.codex_environment()

        bundle = self.PREFIX + '/etc/tls/cert.pem'
        self.assertEqual(env['SSL_CERT_FILE'], bundle)
        self.assertEqual(env['CODEX_CA_CERTIFICATE'], bundle)
        for name in ('LD_PRELOAD', 'OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL'):
            self.assertNotIn(name, env)
        self.assertEqual(dict(os.environ), original)
        self.spawn.assert_not_called()

    def test_nontermux_preserves_native_environment_without_ca_or_proxy_defaults(self):
        os.environ['PREFIX'] = '/usr/local'

        env = agent.codex_environment()

        self.assertEqual(env['LD_PRELOAD'], 'termux-exec.so')
        self.assertNotIn('SSL_CERT_FILE', env)
        self.assertNotIn('CODEX_CA_CERTIFICATE', env)
        self.proxy.assert_not_called()
        self.spawn.assert_not_called()

    def test_explicit_certificate_overrides_are_preserved_and_fill_only_missing_peer(self):
        cases = (
            ({'SSL_CERT_FILE': '/custom/ssl.pem'}, '/custom/ssl.pem', '/custom/ssl.pem'),
            ({'CODEX_CA_CERTIFICATE': '/custom/codex.pem'}, '/custom/codex.pem', '/custom/codex.pem'),
            ({'SSL_CERT_FILE': '/custom/ssl.pem', 'CODEX_CA_CERTIFICATE': '/custom/codex.pem'},
             '/custom/ssl.pem', '/custom/codex.pem'),
        )
        for overrides, ssl, codex in cases:
            with self.subTest(overrides=overrides), patch.dict(os.environ, overrides):
                env = agent.codex_environment()
                self.assertEqual(env['SSL_CERT_FILE'], ssl)
                self.assertEqual(env['CODEX_CA_CERTIFICATE'], codex)

    def test_missing_or_unreadable_default_ca_bundle_reports_error_without_launching(self):
        for exists, readable in ((False, True), (True, False)):
            with self.subTest(exists=exists, readable=readable):
                self.bundle_exists.return_value = exists
                self.bundle_readable.return_value = readable

                with self.assertRaises(AppError) as error:
                    agent.codex_environment()

                self.assertIn('cert', str(error.exception).lower())
                self.spawn.assert_not_called()

    def test_explicit_proxy_is_preserved_without_discovering_fallback(self):
        for name in self.PROXY_VARIABLES:
            with self.subTest(name=name), patch.dict(os.environ, {name: 'http://existing.invalid:8080'}):
                env = agent.codex_environment()
                self.assertEqual(env[name], 'http://existing.invalid:8080')
                self.proxy.assert_not_called()
                self.spawn.assert_not_called()

    def test_configured_proxy_is_child_only_and_environment_build_does_not_start_it(self):
        proxy_url = 'http://127.0.0.1:8888'
        self.proxy.return_value = (Path('/fixture/codex.conf'), proxy_url)
        os.environ['NO_PROXY'] = 'internal.example'
        os.environ['no_proxy'] = 'another.example'
        original = dict(os.environ)

        env = agent.codex_environment()

        for name in self.PROXY_VARIABLES:
            self.assertEqual(env[name], proxy_url)
        for name, preserved in (('NO_PROXY', 'internal.example'), ('no_proxy', 'another.example')):
            self.assertIn(preserved, env[name].split(','))
            self.assertIn('127.0.0.1', env[name].split(','))
            self.assertIn('localhost', env[name].split(','))
        self.assertEqual(dict(os.environ), original)
        self.spawn.assert_not_called()


class CodexNetworkTests(unittest.TestCase):
    PREFIX = '/data/data/com.termux/files/usr'

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(agent.Path, 'home', return_value=self.root))
        self.config = self.root / '.config/tinyproxy/codex.conf'
        self.url = 'http://127.0.0.1:8888'
        self.env = {'PREFIX': self.PREFIX, 'HTTPS_PROXY': self.url}
        self.spawn = self.stack.enter_context(patch.object(agent.subprocess, 'Popen'))
        self.ready = self.stack.enter_context(patch.object(agent, '_proxy_ready', return_value=True))

    def configure(self, contents='Listen 127.0.0.1\nPort 8888\n'):
        self.config.parent.mkdir(parents=True, exist_ok=True)
        self.config.write_text(contents)

    def test_only_existing_loopback_configuration_is_accepted(self):
        self.assertIsNone(agent._termux_proxy())
        self.configure('# Existing Codex relay\nListen "127.0.0.1"\nPort 8888 # local port\n')
        self.assertEqual(agent._termux_proxy(), (self.config, self.url))
        for configuration in ('Port 8888\n', 'Listen 0.0.0.0\nPort 8888\n',
                              'Listen 127.0.0.1\nPort 65536\n', 'Listen 127.0.0.1\nPort invalid\n'):
            with self.subTest(configuration=configuration):
                self.configure(configuration)
                with self.assertRaises(AppError):
                    agent._termux_proxy()
        self.spawn.assert_not_called()
        self.ready.assert_not_called()

    def test_ai_cli_proxy_configuration_takes_precedence_over_legacy_proxy(self):
        self.configure()
        preferred = self.root / '.config/ai-cli/tinyproxy.conf'
        preferred.parent.mkdir(parents=True)
        preferred.write_text('Listen 127.0.0.1\nPort 8889\n')

        self.assertEqual(agent._termux_proxy(), (preferred, 'http://127.0.0.1:8889'))
        preferred.write_text('Listen 0.0.0.0\nPort 8889\n')
        with self.assertRaises(AppError):
            agent._termux_proxy()

    def test_native_or_custom_proxy_environment_does_not_manage_local_relay(self):
        self.configure('not a valid local proxy configuration')
        for env in ({'PREFIX': '/usr/local'},
                    {'PREFIX': self.PREFIX, 'HTTPS_PROXY': 'http://custom.invalid:8080'}):
            with self.subTest(env=env):
                agent._ensure_codex_network(env)
        self.spawn.assert_not_called()
        self.ready.assert_not_called()

    def test_ready_configured_relay_is_reused_without_spawning(self):
        self.configure()

        agent._ensure_codex_network(self.env)

        self.ready.assert_called_once_with(self.url)
        self.spawn.assert_not_called()

    def test_cloud_connection_starts_missing_configured_relay_with_child_environment(self):
        self.configure()
        self.ready.side_effect = [False, True]
        original = dict(os.environ)
        with patch.object(agent.shutil, 'which', return_value='/fixture/tinyproxy'):
            agent._ensure_codex_network(self.env)

        self.spawn.assert_called_once()
        args, kwargs = self.spawn.call_args
        self.assertEqual(args[0], ['/fixture/tinyproxy', '-d', '-c', str(self.config)])
        self.assertEqual(kwargs['env'], self.env)
        self.assertTrue(kwargs['start_new_session'])
        self.assertEqual(dict(os.environ), original)
        self.assertEqual((self.root / '.cache/ostadix-voice/codex-proxy.log').stat().st_mode & 0o777, 0o600)

    def test_failed_relay_start_is_reported_instead_of_continuing(self):
        self.configure()
        self.ready.return_value = False
        self.spawn.return_value.poll.return_value = 1
        with patch.object(agent.shutil, 'which', return_value='/fixture/tinyproxy'):
            with self.assertRaisesRegex(AppError, 'relay could not start'):
                agent._ensure_codex_network(self.env)
        self.spawn.return_value.terminate.assert_not_called()

    def test_rpc_prepares_network_before_starting_codex(self):
        calls = Mock()
        calls.attach_mock(self.spawn, 'spawn')
        with patch.object(agent, 'codex_environment', return_value=self.env), \
                patch.object(agent, '_ensure_codex_network') as ensure, \
                patch.object(agent.threading, 'Thread'), \
                patch.object(agent.RPC, 'call', return_value={}), \
                patch.object(agent.RPC, 'send'):
            calls.attach_mock(ensure, 'network')

            agent.RPC(['/fixture/codex', 'app-server'], self.root, Mock())

        self.assertEqual([call[0] for call in calls.mock_calls], ['network', 'spawn'])
        ensure.assert_called_once_with(self.env)
        self.assertEqual(self.spawn.call_args.kwargs['env'], self.env)

    def test_version_check_does_not_start_network_relay(self):
        result = Mock(returncode=0, stdout='codex-cli 0.153.4\n')
        with patch.object(agent, 'codex_environment', return_value=self.env), \
                patch.object(agent, 'codex_launcher', return_value=['/fixture/codex']), \
                patch.object(agent.subprocess, 'run', return_value=result), \
                patch.object(agent.Path, 'is_file', return_value=False), \
                patch.object(agent, '_ensure_codex_network') as ensure:
            command = agent.codex_command(self.root)

        self.assertEqual(command[:2], ['/fixture/codex', 'app-server'])
        ensure.assert_not_called()
        self.spawn.assert_not_called()


class VersionCompatibilityTests(unittest.TestCase):
    def command(self, version):
        result = Mock(returncode=0, stdout=version + '\n')
        with patch.object(agent, 'codex_launcher', return_value=['/fixture/codex']), \
                patch.object(agent.subprocess, 'run', return_value=result), \
                patch.object(agent, 'codex_environment', return_value={}), \
                patch.object(agent.Path, 'is_file', return_value=False):
            return agent.codex_command(Path('/workspace'))

    def test_verified_versions_keep_workspace_restrictions(self):
        for version in ('codex-cli 0.153.4', 'codex-cli 0.162.1'):
            with self.subTest(version=version):
                command = self.command(version)
                self.assertIn('sandbox_mode="read-only"', command)
                self.assertIn('features.shell_tool=false', command)
                self.assertIn('features.code_mode=false', command)
                self.assertIn('approval_policy="on-request"', command)
                self.assertIn('approvals_reviewer="user"', command)
                self.assertIn('forced_login_method="chatgpt"', command)

    def test_unknown_versions_still_require_protocol_validation(self):
        for version in ('codex-cli 0.162.2', 'codex-cli 0.163.0', ''):
            with self.subTest(version=version), self.assertRaisesRegex(AppError, 'Revalidate'):
                self.command(version)
