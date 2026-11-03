#!/usr/bin/env sh
# Build and smoke-test the installed package in a disposable directory.
set -eu
PYTHON_BIN=${PYTHON_BIN:-python3}
SOURCE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TEST_DIR=$(mktemp -d "${TMPDIR:-/tmp}/juri-wheel.XXXXXX")
trap 'rm -rf "$TEST_DIR"' EXIT HUP INT TERM
"$PYTHON_BIN" -m pip wheel "$SOURCE_DIR" --no-deps -w "$TEST_DIR/dist"
"$PYTHON_BIN" -m venv "$TEST_DIR/venv"
"$TEST_DIR/venv/bin/pip" install "$TEST_DIR"/dist/*.whl
cd "$TEST_DIR"
"$TEST_DIR/venv/bin/juri-lab" init "$TEST_DIR/workspace"
cd "$TEST_DIR/workspace"
"$TEST_DIR/venv/bin/python" -m run_live_benchmark > benchmark.log
"$TEST_DIR/venv/bin/python" -m run_llm_arm --help > /dev/null
"$TEST_DIR/venv/bin/python" -m run_dynamics --help > /dev/null
"$TEST_DIR/venv/bin/python" -m run_diagnostics_summary --help > /dev/null
"$TEST_DIR/venv/bin/python" - <<'PY'
import json
import juri_paths
from pathlib import Path
manifest = json.loads((Path(juri_paths.ROOT) / 'reproduce/manifest.json').read_text())
for job in manifest['jobs']:
    assert Path(juri_paths.module_path(job['args'][0])).is_file(), job['id']
print('Installed benchmark and all declared job entry points passed.')
PY
