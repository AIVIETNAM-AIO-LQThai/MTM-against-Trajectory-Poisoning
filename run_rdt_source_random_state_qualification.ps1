$ErrorActionPreference = "Stop"

$repo = (Get-Location).Path

$root = "experiments\attack_qualification\rdt_source_random_state_corruption"
$preflight = Join-Path $root "preflight.json"

$cleanDataset = "data\derived\rdt_source_random_state_corruption\walker2d-medium-v2\clean_ratio_0p02.hdf5"
$corruptedRoot = "data\derived\rdt_source_random_state_corruption\walker2d-medium-v2\corrupted"
$normalization = "data\metadata\rdt_source_random_state_corruption\walker2d-medium-v2\clean_ratio_0p02_normalization.npz"

Write-Host ""
Write-Host "======================================================================"
Write-Host "RDT-SOURCE RANDOM STATE CORRUPTION QUALIFICATION"
Write-Host "======================================================================"

if (-not (Test-Path $preflight)) {
    throw "Preflight missing: $preflight"
}

$pf = Get-Content $preflight -Raw | ConvertFrom-Json

if ($pf.status -ne "PREFLIGHT_PASS") {
    throw "Preflight is not PASS"
}

Write-Host "preflight: PASS"

$expectedBranch = "exp/rdt-source-random-state-corruption"
$currentBranch = (git branch --show-current).Trim()

if ($currentBranch -ne $expectedBranch) {
    throw "Wrong branch: '$currentBranch'; expected '$expectedBranch'"
}

Write-Host "branch: $currentBranch"

if (-not (Test-Path $cleanDataset)) {
    throw "Missing clean 2% dataset: $cleanDataset"
}

if (-not (Test-Path $normalization)) {
    throw "Missing clean 2% normalization: $normalization"
}

$corruptionSeeds = @(2023, 2024, 2025)
$modelSeeds = @(0, 1, 2)

foreach ($corruptionSeed in $corruptionSeeds) {
    $dataset = Join-Path $corruptedRoot "corruption_seed_$corruptionSeed.hdf5"

    if (-not (Test-Path $dataset)) {
        throw "Missing corrupted dataset: $dataset"
    }
}

Write-Host "datasets: clean 1 / 1, corrupted 3 / 3"

function Get-LatestCheckpoint {
    param([string]$CheckpointDir)

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

function Invoke-DTTraining {
    param(
        [string]$Dataset,
        [string]$Condition,
        [string]$Rho,
        [int]$AttackSeed,
        [int]$ModelSeed,
        [string]$TrainDir
    )

    $finalCheckpoint = Join-Path `
        $TrainDir `
        "checkpoints\checkpoint_step_100000.pt"

    if (Test-Path $finalCheckpoint) {
        Write-Host "Final 100k checkpoint already exists -> SKIP TRAINING"
        return
    }

    New-Item `
        -ItemType Directory `
        -Force `
        -Path $TrainDir |
        Out-Null

    $checkpointDir = Join-Path $TrainDir "checkpoints"
    $resume = Get-LatestCheckpoint -CheckpointDir $checkpointDir

    $args = @(
        "-m", "scripts.train_dt_stress",
        "--dataset", $Dataset,
        "--normalization", $normalization,
        "--expected-num-trajectories", "23",
        "--expected-num-transitions", "20147",
        "--expected-trailing-transitions", "0",
        "--condition", $Condition,
        "--rho", $Rho,
        "--attack-seed", "$AttackSeed",
        "--seed", "$ModelSeed",
        "--num-updates", "100000",
        "--batch-size", "64",
        "--learning-rate", "0.0001",
        "--weight-decay", "0.0001",
        "--warmup-steps", "10000",
        "--grad-clip-norm", "0.25",
        "--log-every", "100",
        "--checkpoint-every", "10000",
        "--output-dir", $TrainDir
    )

    if ($null -ne $resume) {
        Write-Host "Resume checkpoint: $resume"
        $args += @("--resume", $resume)
    }
    else {
        $metricsPath = Join-Path $TrainDir "training_metrics.jsonl"
        $manifestPath = Join-Path $TrainDir "run_manifest.json"
        $summaryPath = Join-Path $TrainDir "summary.json"

        if (
            (Test-Path $metricsPath) -or
            (Test-Path $manifestPath) -or
            (Test-Path $summaryPath)
        ) {
            Write-Host "Stale partial metadata without checkpoint -> RESET CELL"
            Remove-Item -Recurse -Force $TrainDir
            New-Item -ItemType Directory -Force -Path $TrainDir | Out-Null
        }

        Write-Host "Fresh training cell"
    }

    & python @args

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed: condition=$Condition attack=$AttackSeed model=$ModelSeed"
    }

    if (-not (Test-Path $finalCheckpoint)) {
        throw "Training ended without final checkpoint: $finalCheckpoint"
    }
}

foreach ($modelSeed in $modelSeeds) {
    Write-Host ""
    Write-Host "======================================================================"
    Write-Host "TRAIN CLEAN model=$modelSeed"
    Write-Host "======================================================================"

    $trainDir = Join-Path $root "clean\model_seed_$modelSeed\train"

    Invoke-DTTraining `
        -Dataset $cleanDataset `
        -Condition "rdt_source_clean_ratio_0p02" `
        -Rho "0.0" `
        -AttackSeed 1234 `
        -ModelSeed $modelSeed `
        -TrainDir $trainDir
}

foreach ($corruptionSeed in $corruptionSeeds) {
    foreach ($modelSeed in $modelSeeds) {
        Write-Host ""
        Write-Host "======================================================================"
        Write-Host "TRAIN CORRUPTED corruption=$corruptionSeed model=$modelSeed"
        Write-Host "======================================================================"

        $dataset = Join-Path $corruptedRoot "corruption_seed_$corruptionSeed.hdf5"
        $trainDir = Join-Path `
            $root `
            "corrupted\corruption_seed_$corruptionSeed\model_seed_$modelSeed\train"

        Invoke-DTTraining `
            -Dataset $dataset `
            -Condition "rdt_source_random_state_corruption_ratio_0p02" `
            -Rho "0.30" `
            -AttackSeed $corruptionSeed `
            -ModelSeed $modelSeed `
            -TrainDir $trainDir
    }
}

$trainCount = 0

foreach ($modelSeed in $modelSeeds) {
    $checkpoint = Join-Path `
        $root `
        "clean\model_seed_$modelSeed\train\checkpoints\checkpoint_step_100000.pt"

    if (Test-Path $checkpoint) {
        $trainCount++
    }
}

foreach ($corruptionSeed in $corruptionSeeds) {
    foreach ($modelSeed in $modelSeeds) {
        $checkpoint = Join-Path `
            $root `
            "corrupted\corruption_seed_$corruptionSeed\model_seed_$modelSeed\train\checkpoints\checkpoint_step_100000.pt"

        if (Test-Path $checkpoint) {
            $trainCount++
        }
    }
}

Write-Host ""
Write-Host "Final checkpoints: $trainCount / 12"

if ($trainCount -ne 12) {
    throw "Training matrix incomplete"
}

$wslRepo = (& wsl.exe -e wslpath -a $repo).Trim()

if ([string]::IsNullOrWhiteSpace($wslRepo)) {
    throw "Could not convert repo path to WSL path"
}

$evalScript = "$wslRepo/scripts/run_rdt_source_random_state_eval.sh"

Write-Host ""
Write-Host "Starting WSL evaluation matrix..."
Write-Host "repo: $wslRepo"

& wsl.exe -e bash $evalScript $wslRepo

if ($LASTEXITCODE -ne 0) {
    throw "WSL evaluation failed"
}

$evalCount = 0

foreach ($modelSeed in $modelSeeds) {
    $summary = Join-Path $root "clean\model_seed_$modelSeed\eval\summary.json"

    if (Test-Path $summary) {
        $evalCount++
    }
}

foreach ($corruptionSeed in $corruptionSeeds) {
    foreach ($modelSeed in $modelSeeds) {
        $summary = Join-Path `
            $root `
            "corrupted\corruption_seed_$corruptionSeed\model_seed_$modelSeed\eval\summary.json"

        if (Test-Path $summary) {
            $evalCount++
        }
    }
}

Write-Host ""
Write-Host "Evaluation summaries: $evalCount / 12"

if ($evalCount -ne 12) {
    throw "Evaluation matrix incomplete"
}

Write-Host ""
Write-Host "======================================================================"
Write-Host "FINAL QUALIFICATION"
Write-Host "======================================================================"

& python -m scripts.summarize_rdt_source_random_state_qualification

if ($LASTEXITCODE -ne 0) {
    throw "Qualification summarizer failed"
}

Write-Host ""
Write-Host "======================================================================"
Write-Host "QUALIFICATION PIPELINE COMPLETE"
Write-Host "======================================================================"
