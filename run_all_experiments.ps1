param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Keep commands here so you can easily edit configs/runs.
$commands = @(
    "python run_multi_seed_experiment.py --config santander.yaml --n-runs 6",
    "python run_multi_seed_experiment.py --config home_credit.yaml --n-runs 5",
    "python run_multi_seed_experiment.py --config ieee_fraud.yaml --n-runs 5"
)

$scriptDir = Join-Path $PSScriptRoot "src\mi_fs_benchmark\scripts"
if (-not (Test-Path $scriptDir)) {
    Write-Error "Could not find scripts directory: $scriptDir"
    exit 1
}

Push-Location $scriptDir
try {
    foreach ($cmd in $commands) {
        Write-Host ""
        Write-Host ">>> Running: $cmd" -ForegroundColor Cyan
        Write-Host "    Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

        if ($DryRun) {
            Write-Host "    [DryRun] Skipped execution." -ForegroundColor Yellow
            continue
        }

        Invoke-Expression $cmd

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Command failed with exit code $LASTEXITCODE" -ForegroundColor Red
            exit $LASTEXITCODE
        }

        Write-Host "    Done:  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "All commands completed." -ForegroundColor Green

