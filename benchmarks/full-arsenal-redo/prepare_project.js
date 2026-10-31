#!/usr/bin/env node
'use strict';
// Materialize a source-pinned, source-only Ostadix comparison into a fresh directory.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const child = require('child_process');

const dependencyFiles = [
  'compiled_model.py', 'grouped_columns.py', 'grouped_fc.py', 'dynamic_fc.py',
  'author_grouped_quant.py', 'author_dynamic_fc.py', 'gguf_quantizers.py',
  'schema_py_generated.py'
];

function fail(message) { throw new Error(message); }
function digest(data) { return crypto.createHash('sha256').update(data).digest('hex'); }
function inside(parent, candidate) {
  const relative = path.relative(parent, candidate);
  return relative === '' || (!relative.startsWith('..' + path.sep) && relative !== '..' && !path.isAbsolute(relative));
}
function readRegular(root, relative) {
  const absolute = path.join(root, relative);
  if (fs.realpathSync(absolute) !== absolute || !fs.lstatSync(absolute).isFile())
    fail('Input must be a regular file without symlink components: ' + relative);
  return fs.readFileSync(absolute);
}
function options(argv) {
  const result = {};
  const names = new Set(['source-root', 'reference-root', 'python', 'output']);
  if (argv.length !== 8) fail('Supply --source-root ROOT --reference-root ROOT --python ABSOLUTE_PYTHON --output FRESH_DIRECTORY');
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index].replace(/^--/, '');
    if (!names.has(name) || Object.hasOwn(result, name)) fail('Unknown or duplicate option: ' + argv[index]);
    if (!path.isAbsolute(argv[index + 1])) fail('Use explicit absolute paths for ' + name);
    result[name] = path.normalize(argv[index + 1]);
  }
  return result;
}
function run() {
  const args = options(process.argv.slice(2));
  const source = fs.realpathSync(args['source-root']);
  const reference = fs.realpathSync(args['reference-root']);
  const python = args.python;
  const output = args.output;
  try {
    fs.lstatSync(output);
    fail('Output already exists; choose a fresh directory');
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  if (inside(source, output) || inside(reference, output))
    fail('Materialize outside both source trees');
  if (fs.realpathSync(path.dirname(output)) !== path.dirname(output))
    fail('Output parent must exist without symlink components');
  if (!fs.statSync(python).isFile() || !(fs.statSync(python).mode & 0o111))
    fail('Selected Python must be an executable regular file');

  const shape = JSON.parse(readRegular(__dirname, 'dimensions.json').toString('utf8'));
  if (shape.schema !== 'pixelforge.completion-fixture-shape.v1' ||
      !Array.isArray(shape.dense_dimensions) || !shape.dense_dimensions.length ||
      !shape.dense_dimensions.every(row => Array.isArray(row) && row.length === 4 && row.every(Number.isInteger)))
    fail('Malformed dimension fixture');
  const sexp = '(' + shape.dense_dimensions.map(row => '(' + row.join(' ') + ')').join('\n') + ')\n';
  if (readRegular(__dirname, 'dense-plan.sexp').toString('utf8') !== sexp)
    fail('JSON and independent Lisp dimension inputs differ');

  const moduleData = new Map();
  const modulePins = {};
  for (const relative of dependencyFiles) {
    const name = 'tools/arm_npu/' + relative;
    const candidate = readRegular(source, name);
    const baseline = readRegular(reference, name);
    if (!candidate.equals(baseline)) fail('Cache/graph dependency differs between source and reference: ' + name);
    moduleData.set(relative, candidate);
    modulePins[relative] = digest(candidate);
  }
  const compiler = readRegular(source, 'tools/arm_npu/compile_grouped_columns.py').toString('utf8');
  const begin = compiler.indexOf('\ndef mark_columns_complete(');
  const end = compiler.indexOf('\ndef main(', begin + 1);
  if (begin < 0 || end < 0) fail('Expected standalone column completion helper not found');
  const helper = compiler.slice(begin + 1, end) + '\n';
  if (!helper.startsWith('def mark_columns_complete(cache, weight_dtype='))
    fail('Unexpected helper interface');
  const probe = child.spawnSync(python, ['-c',
    'import json,platform,numpy,flatbuffers; print(json.dumps(dict(python=platform.python_version(),numpy=numpy.__version__,flatbuffers=flatbuffers.__version__)))'],
    {encoding:'utf8', env:{...process.env, OPENBLAS_NUM_THREADS:'1', PYTHONDONTWRITEBYTECODE:'1'}});
  if (probe.status !== 0) fail('Selected Python cannot import NumPy/FlatBuffers: ' + probe.stderr);
  const toolchain = JSON.parse(probe.stdout);
  toolchain.python_executable_sha256 = digest(fs.readFileSync(python));
  toolchain.python_resolved_path = fs.realpathSync(python);

  // All validation above precedes creation. Partial output is retained if a
  // subsequent write fails, and the caller must choose a fresh directory.
  fs.mkdirSync(output, {mode:0o700});
  fs.mkdirSync(path.join(output, 'snapshot'), {mode:0o700});
  for (const [relative, data] of moduleData)
    fs.writeFileSync(path.join(output, 'snapshot', relative), data);
  for (const name of ['run_completion.py', 'dimensions.json', 'coverage.lisp', 'dense-plan.sexp'])
    fs.copyFileSync(path.join(__dirname, name), path.join(output, name));
  fs.writeFileSync(path.join(output, 'candidate_impl.py'), helper);

  const commandInputs = dependencyFiles.map(name => 'snapshot/' + name)
    .concat(['run_completion.py', 'candidate_impl.py', 'dimensions.json', 'coverage.lisp', 'dense-plan.sexp']).sort();
  const pins = {};
  for (const name of commandInputs) pins[name] = digest(fs.readFileSync(path.join(output, name)));
  const provenance = {
    schema:'pixelforge.completion-reproduction-inputs.v1',
    shared_cache_graph_source_equal:true,
    shared_cache_graph_source_sha256:modulePins,
    input_sha256:pins,
    toolchain,
    scope:'Synthetic fixture with observed dimensions/code sizes; dummy source and DGC0-shaped data; no executable NPU programs or model weights.',
    repetitions_per_route:3,
    staging_note:'The original private measurement declared 72 files, including 64 unused fixtures. This minimal reproduction stages the same eight imported cache/graph modules.'
  };
  fs.writeFileSync(path.join(output, 'PINNED_INPUTS.json'), JSON.stringify(provenance, null, 2) + '\n');
  const declaredInputs = commandInputs.concat(['PINNED_INPUTS.json']);
  let toml = '[project]\nname = "pixelforge-column-completion-comparison"\n';
  for (const route of ['baseline', 'candidate']) {
    const command = [python, 'run_completion.py', route, '--repetitions', '3',
      '--evidence-dir', path.join(output, 'counter-evidence')];
    toml += '\n[[routes]]\nid = ' + JSON.stringify(route) +
      '\ncommand = ' + JSON.stringify(command) +
      '\nenv = { OPENBLAS_NUM_THREADS = "1", PYTHONDONTWRITEBYTECODE = "1" }' +
      '\ninputs = ' + JSON.stringify(declaredInputs) + '\nresult_codec = "json"\n';
  }
  for (const [name, order] of [
    ['completion-forward', ['baseline', 'candidate']],
    ['completion-reverse', ['candidate', 'baseline']]
  ]) {
    toml += '\n[[route_sets]]\nprovides = ' + JSON.stringify(name) +
      '\nalternatives = ' + JSON.stringify(order) +
      '\npolicy = "benchmark_validate_and_select"\n';
  }
  fs.writeFileSync(path.join(output, 'olang.project.toml'), toml);
  console.log(JSON.stringify({ok:true, output, declared_inputs:declaredInputs.length,
    source_equivalence_verified:true, runtime:toolchain, measurements_run:false}));
}
try { run(); } catch (error) { console.error(error.message); process.exitCode = 1; }
