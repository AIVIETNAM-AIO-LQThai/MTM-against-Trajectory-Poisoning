$ErrorActionPreference = "Stop"

$repo = (Get-Location).Path
$expectedBranch = "exp/rdt-source-dt-legacy-mha-compat"

$currentBranch = (git branch --show-current).Trim()

if ($currentBranch -ne $expectedBranch) {
    throw "Wrong branch: '$currentBranch'; expected '$expectedBranch'"
}

Write-Host ""
Write-Host "======================================================================"
Write-Host "SOURCE-COMPATIBLE RDT DT CLEAN VIABILITY"
Write-Host "======================================================================"
Write-Host "branch: $currentBranch"

python -m scripts.preflight_rdt_source_dt_clean_viability

if ($LASTEXITCODE -ne 0) {
    throw "clean-victim preflight failed"
}

$root = "experiments\attack_qualification\rdt_source_dt_legacy_mha_compat"
$dataset = "data\derived\rdt_source_random_state_corruption\walker2d-medium-v2\clean_ratio_0p02.hdf5"

function Get-LatestCheckpoint {
    param(
        [string]$CheckpointDir
    )

    if (-not (Test-Path $CheckpointDir)) {
        return $null
    }

    $items = Get-ChildItem `
        -Path $CheckpointDir `
        -Filter "checkpoint_step_*.pt" `
        -File |
        Sort-Object Name -Descending

    if ($items.Count -eq 0) {
        return $null
    }

    return $items[0].FullName
}

foreach ($seed in @(0, 1, 2)) {
    Write-Host ""
    Write-Host "======================================================================"
    Write-Host "CLEAN SOURCE-COMPATIBLE DT seed=$seed"
    Write-Host "======================================================================"

    $runDir = Join-Path `
        $root `
        "clean\model_seed_$seed"

    $failure = Join-Path `
        $runDir `
        "failure.json"

    if (Test-Path $failure) {
        Write-Host "Existing viability failure detected."
        Get-Content $failure
        throw "Seed $seed has failed clean viability. Corrupted training remains blocked."
    }

    $finalCheckpoint = Join-Path `
        $runDir `
        "checkpoints\checkpoint_step_100000.pt"

    $summary = Join-Path `
        $runDir `
        "summary.json"

    if (
        (Test-Path $finalCheckpoint) -and
        (Test-Path $summary)
    ) {
        Write-Host "Seed $seed already complete -> SKIP"
        continue
    }

    New-Item `
        -ItemType Directory `
        -Force `
        -Path $runDir |
        Out-Null

    $checkpointDir = Join-Path `
        $runDir `
        "checkpoints"

    $resume = Get-LatestCheckpoint `
        -CheckpointDir $checkpointDir

    $args = @(
        "-m", "scripts.train_rdt_source_dt_clean",
        "--dataset", $dataset,
        "--seed", "$seed",
        "--num-updates", "100000",
        "--batch-size", "64",
        "--learning-rate", "0.0001",
        "--weight-decay", "0.0001",
        "--warmup-steps", "10000",
        "--grad-clip-norm", "0.25",
        "--log-every", "100",
        "--checkpoint-every", "10000",
        "--output-dir", $runDir
    )

    if ($null -ne $resume) {
        Write-Host "Resume checkpoint: $resume"
        $args += @(
            "--resume",
            $resume
        )
    }
    else {
        $metrics = Join-Path `
            $runDir `
            "training_metrics.jsonl"

        $manifest = Join-Path `
            $runDir `
            "run_manifest.json"

        if (
            (Test-Path $metrics) -or
            (Test-Path $manifest)
        ) {
            throw "Partial run metadata exists without a checkpoint for seed $seed"
        }

        Write-Host "Fresh clean viability run"
    }

    & python @args

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Seed $seed failed clean viability."
        Write-Host "STOP: do not run corrupted training."
        throw "Clean source-compatible DT viability failed at seed $seed"
    }

    if (
        -not (Test-Path $finalCheckpoint)
    ) {
        throw "Seed $seed ended without a 100000-update checkpoint"
    }
}

Write-Host ""
Write-Host "======================================================================"
Write-Host "SUMMARIZE CLEAN VIABILITY"
Write-Host "======================================================================"

python -m scripts.summarize_rdt_source_dt_clean_viability

if ($LASTEXITCODE -ne 0) {
    throw "clean viability summarizer failed"
}

$viabilityPath = Join-Path `
    $root `
    "clean_viability_summary.json"

$viability = Get-Content `
    $viabilityPath `
    -Raw |
    ConvertFrom-Json

if (-not $viability.clean_viability_pass) {
    throw "Clean viability did not pass. Corrupted training is blocked."
}

Write-Host ""
Write-Host "All three clean source-compatible DT seeds passed."
Write-Host "Corrupted training may now be implemented in the next frozen step."
