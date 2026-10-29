> Historical implementation notes from the earlier device integration.
> These do not describe the current default engine or establish current
> performance. Associated local runtime reports are not distributed here;
> use [the current source guide](README.md) for setup and supported entrypoints.

# Local NPU action workflow

The resident Gemma model can request workspace reads, propose exact replacements,
request a Python test script, receive real results, and draft a final answer.
The host executes approved actions. The model remains text-only and cannot grant
itself approval. No Codex call is made by this workflow.

## Use

After reopening c, start with:
/agent ask Read calc.py, fix the bug, and test calc.py.
/agent list
/agent read FULL_TASK_ID
/agent cancel FULL_TASK_ID

An edit/test proposal pauses the task. /agent read displays the exact diff or argv,
file fingerprint, and proposal digest. The terminal announces this pending state.

Approve through the existing approved command path, from ~/ostadix-voice:
python local_agent.py apply FULL_TASK_ID --digest REVIEWED_DIGEST

Or refuse the proposal:
python local_agent.py deny FULL_TASK_ID --digest REVIEWED_DIGEST

After an applied action, continue the remaining original task budget:
python local_agent.py run FULL_TASK_ID

The digest binds the exact displayed proposal; it is not authority by itself.
Only an explicit operator approval/approved host command may invoke apply.
The current terminal has no one-button approval bridge for detached tasks;
Codex can invoke the apply command through its normal local command approval route,
or the user can explicitly run it from Termux after reviewing the proposal.
Plain spoken input still routes to Codex.

For detached work:
python local_agent.py start --workspace ./example-workspace 'Your task'
python local_agent.py read FULL_TASK_ID

## Supported work

- List directories and read bounded text in the selected workspace.
- Inspect local model and optimizer status.
- Replace one exact occurrence in a previously read existing file.
- Parse Python syntax automatically after an approved edit, without executing code.
- Stage a Python test-script command for separate approval. Capture exit status,
  output, timeout and cancellation, then return that evidence to the model.
- Resume after approval and propose a revision within the remaining model budget.

An approved Python script has Termux account access; this is not an OS sandbox.
The exact command, script contents, fingerprint and working directory are shown.
Tests have a 15-second timeout and 12 KB output limit; owned process groups are
cleaned up. This is not a containment guarantee for arbitrary hostile programs.
OSTADIX syntax checking applies to O programs; Python syntax uses Python's parser.

## Integrity and persistence

Approvals are single-use. Changed files, altered proposals, denied actions and
cancelled tasks cannot be applied. A mixed final-answer/action object is rejected.
Actions record intent before execution so unknown completion is not replayed.
Task journals, proposals, host results and model backend usage persist privately
under ~/.local/state/ostadix-voice-local/evidence-agent/.

The host records edits_applied, tests_run and last_test_passed independently of
model prose. Completed means a final answer was produced, not that every requested
goal or test succeeded. Inspect host verification and errors.

Each task still permits at most four model calls of 256 output tokens each.
Approval waits do not reset that budget. Reads are 900 bytes; edits require that
the complete current file was read. Only existing small UTF-8 files are editable.
Latest evidence and updated code previews are passed back to the small-context model;
the full collected journal is retained. There is no automatic cross-task learning,
arbitrary shell tool, boot registration, local voice rerouting or self-assigned work.

## Validation

Offline tests cover staging without mutation, exact approval, stale files, denial,
cancellation, replay prevention, changed-file rereads, failed-test feedback,
syntax parsing without execution, workspace boundaries and step-budget preservation.

A live NPU trial read calc.py and proposed changing subtraction to addition.
The host applied the approved edit and verified syntax. The next model step invented
an unrelated replacement, which the host rejected. Updated file contents are now
included in post-edit feedback; that context change is regression-tested, not yet
proven to eliminate this model failure on a fresh full live repair run.

A separate live NPU task requested the actual test script. After approval it ran
successfully: three arithmetic assertions passed, exit code zero. The model received
that result and reported success, although its wording included an unnecessary
'Missing evidence' phrase. Host verification is the reliable record.

Live records:
- Edit: 759888345fb245379e0f78ce2044ce76
- Test: ce8458b2f394427790556d318f2bdefc
- Fixture: runtime/agent-edit-trial-w3zc92ow/calc.py

## Exact arithmetic

Use `/agent ask calculate: 0.1+0.2` to return `3/10` with zero model calls. This explicit prefix works without a running model. The model can also request a calculate evidence tool with an expression field. Supported operations: +, -, *, /, //, %, bounded integer powers, and parentheses. Ordinary integer and decimal literals produce exact rational results. Calls, variables and arbitrary evaluation are rejected; input length, complexity and number sizes are bounded.

See COMPUTE_OPTIMIZATION.md for hardware trials and backend limits.

## Task duration

The former four-minute overall task deadline is removed for newly launched workers. Tasks still stop at their model-step budget, on cancellation or on a failed model request. Individual model requests retain the runtime's 120-second timeout. This change does not remove platform turn limits, restart interrupted tasks or change already-running worker code.
