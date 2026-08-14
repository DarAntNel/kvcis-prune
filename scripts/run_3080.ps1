<#
  KVCIS-3 pipeline (3-class probe: fp16 / int8 / never-store) for a 10GB RTX 3080.

  Usage:
    .\run_3080.ps1              # quick pass (~10-15 min)
    .\run_3080.ps1 -Full        # larger run: better probe

  Assumes the venv lives at <repo-parent>\.venv (see ../README.md).
#>
param([switch]$Full)

# Do NOT set $ErrorActionPreference = 'Stop': PS 5.1 would abort on harmless
# native stderr output. We check $LASTEXITCODE after each step instead.
$ErrorActionPreference = "Continue"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeDir    = Join-Path (Split-Path -Parent $ScriptDir) "code"
$RepoParent = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$Py = Join-Path $RepoParent ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { Write-Error "venv python not found at $Py -- create it first (see kvcis\README_3080.md)"; exit 1 }

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TRANSFORMERS_VERBOSITY = "error"

$Model = "Qwen/Qwen2.5-1.5B-Instruct"
$Layer = 10

if ($Full) {
    $NPrompts = 500; $GenSteps = 40; $NTexts = 40
} else {
    $NPrompts = 200; $GenSteps = 30; $NTexts = 20
}

function Invoke-Step($Title, [string[]]$PyArgs) {
    Write-Host "`n=== $Title ===" -ForegroundColor Cyan
    & $Py @PyArgs
    if ($LASTEXITCODE -ne 0) { Write-Error "$Title failed (exit $LASTEXITCODE)"; exit 1 }
}

Push-Location $CodeDir
try {
    Invoke-Step "Step 1: single-prompt sanity check (trajectory + 3-class labels)" `
        @("step1_single_prompt.py", "--model", $Model, "--extraction-layer", $Layer)

    Invoke-Step "Step 2: collect two-probe training data ($NPrompts prompts, horizon $GenSteps)" `
        @("step2_collect_data.py", "--model", $Model, "--extraction-layer", $Layer,
          "--n-prompts", $NPrompts, "--generation-steps", $GenSteps,
          "--evict-window", "10", "--max-prompt-tokens", "96", "--output-dir", "../data")

    Invoke-Step "Step 3: train probe A (original KVCIS regression) + probe B (evictability)" `
        @("step3_train_probe.py", "--data-dir", "../data", "--output-dir", "../data/probe2")

    Invoke-Step "Step 4: deferred-eviction eval (KVCIS / KVCIS+Evict vs Baseline & INT8)" `
        @("step4_compression_eval.py", "--model", $Model, "--extraction-layer", $Layer,
          "--probe-path", "../data/probe2", "--n-texts", $NTexts)

    Write-Host "`nAll done. Results: ..\results\kvcis2p_results.json" -ForegroundColor Green
}
finally { Pop-Location }
