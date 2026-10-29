"""Approved OSTADIX MCP access through the existing persistent workspace tool.

The reserved @ostadix argv is a local protocol route, never a shell executable.
Keeping the workspace schema stable makes it usable in existing Codex threads.
"""
import json
import os
from pathlib import Path
import threading
import time

from agent import AppError, Workspace


INSTRUCTIONS = '''OSTADIX MCP is your preferred tool whenever its real capabilities fit the task.
O/Ostadix is a general-purpose polyglot language and execution/runtime system; this
phone assistant is one application of it. Do not describe it as exclusively for AI
agents, invent hardware optimizations, or repeat unverified claims from conversation
history as established capabilities. Distinguish implemented behavior, measured
evidence, and future compiler/runtime objectives. Query o_capabilities and o_guide
before explaining unfamiliar features or choosing a substantive demonstration.
For a demonstration, execute a relevant supported workflow and show its actual code,
result and limits. Arithmetic, greeting scripts and ordinary phone controls do not
demonstrate automatic lowering of an external runtime/compiler or a speed improvement.
Before choosing a nontrivial implementation, discover relevant OSTADIX capabilities and use
them if appropriate. Prefer o_execute with exactly one complete source or existing path;
use o_capabilities(query), o_guide(topic), and actual tool schemas to plan correctly.
Do not force OSTADIX for simple file reading or tasks it cannot do, and explain a capability
or connection failure before falling back. Never assume catalog presence proves execution.

OSTADIX is connected through the existing workspace tool so saved threads keep working.
Use operation command and these RESERVED argv arrays (they are handled locally, not by a shell):
  ["@ostadix", "status"]
  ["@ostadix", "tools"]
  ["@ostadix", "schema", "o_execute"]
  ["@ostadix", "call", "o_capabilities", "{\\"query\\":\\"runtime\\"}"]
  ["@ostadix", "call", "o_guide", "{\\"topic\\":\\"runtime\\"}"]
  ["@ostadix", "call", "o_execute", "JSON_OBJECT_AS_A_STRING"]
Other tools returned by tools/list are accessible with the same call route. Query schema
before using unfamiliar parameters. Non-catalog calls follow the user's action policy:
auto runs requested calls directly; ask waits for exact local approval. The recorded
proposal includes the exact repo revision and effective
arguments; cwd defaults to the selected workspace. Compilation, execution, jobs, remote
node calls and their outputs remain in this same conversation. Commands and MCP execution
have Termux account access under that policy; do not request unrelated access or bypass a denial.
Use background jobs only when necessary and report any unfinished job ID. Job handles are
session-scoped, valid only while this local MCP process lives. Monitor with o_job_status/read;
cancel with o_job_cancel. Remote effects may survive cancellation. Never replay a tool call
whose completion is unknown. A successful job submission is not execution success.
'''


SAFE_TOOLS = {'o_capabilities': {'query'}, 'o_guide': {'topic'}}
ENV_KEYS = ('PATH', 'HOME', 'PREFIX', 'TMPDIR', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH',
            'LD_PRELOAD', 'ANDROID_ROOT', 'ANDROID_DATA')


def paired_environment(pair):
    from ostadix_pair import pair_environment
    env = {key: value for key, value in os.environ.items() if key in ENV_KEYS}
    env['PATH'] = os.pathsep.join(p for p in env.get('PATH', '').split(os.pathsep) if 'codex-path' not in p)
    return pair_environment(pair, environment=env)


class OstadixTools:
    def __init__(self, pair_loader=None, client_factory=None):
        self.pair_loader = pair_loader
        self.client_factory = client_factory
        self.connections = {}
        self.lock = threading.RLock()
        self.closed = False

    def connection(self, sid, workspace, cancel_event=None):
        with self.lock:
            if self.closed or (cancel_event and cancel_event.is_set()):
                raise AppError('OSTADIX connection cancelled or closed')
            existing = self.connections.get(sid)
            if existing and existing['client'].alive:
                return existing
            if existing:
                existing['client'].close()
                del self.connections[sid]
                raise AppError('OSTADIX disconnected. Previous job handles are lost and outcomes may be incomplete. No execution was replayed; request status explicitly to reconnect.')
        # Never hold the pool lock across process startup or MCP I/O: /cancel
        # and shutdown must remain available even if initialization stalls.
        try:
            if self.pair_loader is None:
                from ostadix_pair import load_pair
                loader = load_pair
            else:
                loader = self.pair_loader
            pair = loader()
            if self.client_factory is None:
                from mcp_transport import MCPClient
                factory = MCPClient
            else:
                factory = self.client_factory
            client = factory([pair['binary']], Path(workspace), paired_environment(pair))
            try:
                tools = {}
                cursor = None
                deadline = time.monotonic() + 15
                for _ in range(20):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise AppError('OSTADIX tool discovery timed out')
                    page = client.call('tools/list', {'cursor': cursor} if cursor else {},
                                       timeout=remaining, cancel_event=cancel_event)
                    tools.update({tool['name']: tool for tool in page.get('tools', [])})
                    cursor = page.get('nextCursor')
                    if not cursor:
                        break
                if cursor or not {'o_execute', 'o_capabilities', 'o_guide'} <= tools.keys():
                    raise AppError('OSTADIX MCP lacks its required current primary tools')
                with self.lock:
                    if self.closed or (cancel_event and cancel_event.is_set()):
                        raise AppError('OSTADIX connection cancelled or closed')
                    entry = {'client': client, 'pair': pair, 'tools': tools, 'jobs': set()}
                    existing = self.connections.setdefault(sid, entry)
                if existing is not entry:
                    client.close()
                return existing
            except Exception:
                client.close()
                raise
        except Exception as error:
            raise AppError('OSTADIX MCP is unavailable: ' + str(error) + '. Use c --update-ostadix to verify/update the paired installation.') from error

    def dispatch(self, agent, args, workspace):
        argv = args.get('argv', [])
        if len(argv) < 2 or argv[0] != '@ostadix':
            raise AppError('Invalid OSTADIX bridge request')
        if agent.cancelled.is_set():
            raise AppError('Task cancelled before OSTADIX access')
        entry = self.connection(agent.sid, workspace.root, agent.cancelled)
        pair, client, tools = entry['pair'], entry['client'], entry['tools']
        verb = argv[1]
        identity = {'repository': pair['root'], 'revision': pair['revision'],
                    'upstream': pair.get('upstream'),
                    'binary': pair['binary'], 'server': client.info.get('serverInfo'),
                    'primary_tool': 'o_execute'}
        if verb == 'status' and len(argv) == 2:
            return dict(identity, connected=True, tool_count=len(tools), jobs_survive_restart=False)
        if verb == 'tools' and len(argv) == 2:
            return dict(identity, tools=[{'name': tool['name'], 'description': tool.get('description', '')} for tool in tools.values()])
        if verb == 'schema' and len(argv) == 3 and argv[2] in tools:
            return tools[argv[2]]
        if verb != 'call' or len(argv) != 4 or argv[2] not in tools:
            raise AppError('Use @ostadix status, tools, schema NAME, or call NAME JSON_OBJECT')
        name = argv[2]
        try:
            parameters = json.loads(argv[3])
        except ValueError as error:
            raise AppError('MCP arguments must be a JSON object') from error
        if not isinstance(parameters, dict):
            raise AppError('MCP arguments must be a JSON object')
        overrides = parameters.get('env') or {}
        if not isinstance(overrides, dict):
            raise AppError('MCP env must be an object')
        reserved = {'PATH', 'O_LANG_ROOT', 'O_BACKENDS_DIR', 'OLANG',
                    'OSTADIX_RUNTIME_PATH_MODE', 'LD_PRELOAD', 'LD_LIBRARY_PATH'}
        if any(key in reserved or key.startswith(('O_LANG_', 'OSTADIX_')) for key in overrides):
            raise AppError('MCP runtime environment overrides cannot replace the verified OSTADIX pair')
        if name in SAFE_TOOLS:
            if set(parameters) - SAFE_TOOLS[name] or any(not isinstance(v, str) or len(v) > 2000 for v in parameters.values()):
                raise AppError('Catalog/guide accepts only its bounded text query or topic')
        else:
            properties = tools[name].get('inputSchema', {}).get('properties', {})
            if 'cwd' in properties:
                cwd = parameters.get('cwd') or '.'
                if not isinstance(cwd, str):
                    raise AppError('MCP cwd must be a workspace directory')
                path = Path(cwd)
                if path.is_absolute():
                    try:
                        relative = path.relative_to(workspace.root)
                    except ValueError:
                        raise AppError('MCP cwd must stay inside the selected workspace')
                else:
                    relative = path
                Workspace(workspace.root).list(str(relative))
                parameters['cwd'] = str(workspace.root / relative)
            # Show the effective call, including filled-in cwd, before approval.
            proposal = {'operation': 'ostadix', 'tool': name, 'arguments': parameters,
                        'repository': pair['root'], 'revision': pair['revision']}
            sid, generation = agent.sid, agent.task_generation
            agent.wait_approval(proposal)
            with agent.lock:
                if not agent._approval_current({'session': sid, 'generation': generation}):
                    raise AppError('Task changed or cancelled before MCP execution')
            agent.store.update(agent.sid, status='executing')
        try:
            result = client.call('tools/call', {'name': name, 'arguments': parameters},
                                 timeout=args.get('timeout', 120), cancel_event=agent.cancelled)
        except Exception as error:
            if agent.cancelled.is_set():
                self.cancel_jobs(agent.sid, discover=True)
            raise AppError(str(error)) from error
        structured = result.get('structuredContent') or {}
        with self.lock:
            if isinstance(structured, dict) and structured.get('job_id'):
                if structured.get('state') == 'running':
                    entry['jobs'].add(structured['job_id'])
                elif structured.get('state') in ('completed', 'failed', 'cancelled', 'timed_out'):
                    entry['jobs'].discard(structured['job_id'])
        if agent.cancelled.is_set():
            self.cancel_jobs(agent.sid, discover=True)
            return {'mcp_tool': name, 'exit_code': 1, 'interruption': 'cancelled', 'result': result}
        # Preserve actual MCP error status; never turn a failed tool into success.
        failed = bool(result.get('isError'))
        if isinstance(structured, dict):
            failed = failed or (structured.get('exit_code') not in (None, 0)) or structured.get('state') in ('failed', 'cancelled', 'timed_out')
        return {'mcp_tool': name, 'repository_revision': pair['revision'],
                'exit_code': 1 if failed else 0,
                'execution_pending': isinstance(structured, dict) and structured.get('state') == 'running',
                'result': result}

    def status(self, sid, workspace):
        entry = self.connection(sid, workspace)
        return {'connected': True, 'primary_tool': 'o_execute',
                'repository': entry['pair']['root'], 'revision': entry['pair']['revision'],
                'upstream': entry['pair'].get('upstream'),
                'server': entry['client'].info.get('serverInfo'),
                'tool_count': len(entry['tools']), 'jobs_survive_restart': False}

    def has_jobs(self, sid):
        with self.lock:
            return bool(self.connections.get(sid, {}).get('jobs'))

    def cancel_jobs(self, sid, discover=False):
        with self.lock:
            entry = self.connections.get(sid)
        if not entry:
            return
        errors = []
        if discover:
            try:
                result = entry['client'].call('tools/call', {'name': 'o_job_list', 'arguments': {}}, timeout=3)
                if result.get('isError'):
                    raise AppError('Job discovery failed')
                with self.lock:
                    for job in (result.get('structuredContent') or {}).get('jobs', []):
                        if job.get('state') == 'running' and job.get('job_id'):
                            entry['jobs'].add(job['job_id'])
            except Exception:
                errors.append('unknown jobs (status lookup failed)')
        with self.lock:
            jobs = list(entry['jobs'])
        for job in jobs:
            try:
                result = entry['client'].call('tools/call', {'name': 'o_job_cancel', 'arguments': {'job_id': job}}, timeout=3)
                state = (result.get('structuredContent') or {}).get('state')
                if result.get('isError') or state not in ('cancelled', 'completed', 'failed', 'timed_out'):
                    errors.append(job)
                else:
                    with self.lock:
                        entry['jobs'].discard(job)
            except Exception:
                errors.append(job)
        if errors:
            raise AppError('OSTADIX job cancellation is unconfirmed for: ' + ', '.join(errors) + '. Remote effects may continue.')

    def close(self):
        with self.lock:
            self.closed = True
            entries = list(self.connections.items())
        warnings = []
        for sid, entry in entries:
            try:
                self.cancel_jobs(sid)
            except AppError as error:
                warnings.append((sid, str(error)))
            entry['client'].close()
            if getattr(entry['client'], 'shutdown_graceful', True) is False:
                warnings.append((sid, 'OSTADIX required forced shutdown; detached or remote work may remain. Inspect recorded job IDs before continuing.'))
        with self.lock:
            self.connections.clear()
        return warnings
