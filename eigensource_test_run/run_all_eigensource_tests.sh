#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
RUNNER="$SCRIPT_DIR/run_all_eigensource_tests.py"
PYTHON_BIN="${PYTHON:-python3}"

exec "$PYTHON_BIN" "$RUNNER" "$@"
