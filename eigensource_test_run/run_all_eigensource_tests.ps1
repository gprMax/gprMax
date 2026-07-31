$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runner = Join-Path $ScriptDir "run_all_eigensource_tests.py"

if ($env:PYTHON) {
    & $env:PYTHON $Runner @args
} else {
    & python $Runner @args
}

