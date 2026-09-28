$ErrorActionPreference = "Stop"

$repo = (Get-Location).Path
$root = "experiments\attack_qualification\a4_dt"
$preflight = Join-Path $root "preflight.json"

Write-Host ""
Write-Host "============================================================"
Write-Host "A4 VANILLA DT ACTION-SHIFT QUALIFICATION"
Write-Host "============================================================"

if (-not (Test-Path $preflight)) {
    throw "A4 preflight missing: $preflight"
}

$pf = Get-Content $preflight -Raw | ConvertFrom-Json

if ($pf.status -ne "PREFLIGHT_PASS") {
    throw "A4 preflight is not PASS"
}

Write-Host "A4 preflight: PASS"

$expectedBranch = "exp/a4-dt-action-shift-qualification"
$currentBranch = (git branch --show-current).Trim()

if ($currentBranch -ne $expectedBranch) {
    throw "Wrong branch: '$currentBranch'; expected '$expectedBranch'"
}

Write-Host "branch: $currentBranch"

$attackSeeds = @(20, 21, 22)
$modelSeeds = @(0, 1, 2)

$poisonRoot = "data\poisoned\action_cyclic_shift\walker2d-medium-v2"

foreach ($attackSeed in $attackSeeds) {
    $dataset = Join-Path $poisonRoot "attack_seed_$attackSeed.hdf5"

    if (-not (Test-Path $dataset)) {
        throw "Missing poison dataset: $dataset"
    }
}

Write-Host "A3 poison datasets: 3 / 3"

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

foreach ($attackSeed in $attackSeeds) {
    foreach ($modelSeed in $modelSeeds) {
        Write-Host ""
        Write-Host "============================================================"
        Write-Host "A4 TRAIN attack=$attackSeed model=$modelSeed"
        Write-Host "============================================================"

        $dataset = Join-Path $poisonRoot "attack_seed_$attackSeed.hdf5"

        $runRoot = Join-Path `
            $root `
            "poison\attack_seed_$attackSeed\model_seed_$modelSeed"

        $trainDir = Join-Path $runRoot "train"

        $finalCheckpoint = Join-Path `
            $trainDir `
            "checkpoints\checkpoint_step_100000.pt"

        if (Test-Path $finalCheckpoint) {
            Write-Host "Final 100k checkpoint already exists -> SKIP TRAINING"
            continue
        }

        New-Item `
            -ItemType Directory `
            -Force `
            -Path $trainDir |
            Out-Null

        $checkpointDir = Join-Path $trainDir "checkpoints"
        $resume = Get-LatestCheckpoint -CheckpointDir $checkpointDir

        $args = @(
            "-m", "scripts.train_dt_stress",
            "--dataset", $dataset,
            "--condition", "action_cyclic_shift",
            "--rho", "0.05",
            "--attack-seed", "$attackSeed",
            "--seed", "$modelSeed",
            "--num-updates", "100000",
            "--batch-size", "64",
            "--learning-rate", "0.0001",
            "--weight-decay", "0.0001",
            "--warmup-steps", "10000",
            "--grad-clip-norm", "0.25",
            "--log-every", "100",
            "--checkpoint-every", "10000",
            "--output-dir", $trainDir
        )

        if ($null -ne $resume) {
            Write-Host "Resume checkpoint: $resume"
            $args += @(
                "--resume",
                $resume
            )
        }
        else {
            $metricsPath = Join-Path $trainDir "training_metrics.jsonl"
            $manifestPath = Join-Path $trainDir "run_manifest.json"
            $summaryPath = Join-Path $trainDir "summary.json"

            if (
                (Test-Path $metricsPath) -or
                (Test-Path $manifestPath) -or
                (Test-Path $summaryPath)
            ) {
                Write-Host "Stale partial training metadata without checkpoint -> RESET CELL"
                Remove-Item -Recurse -Force $trainDir
                New-Item -ItemType Directory -Force -Path $trainDir | Out-Null
            }

            Write-Host "Fresh training cell"
        }

        & python @args

        if ($LASTEXITCODE -ne 0) {
            throw "Training failed: attack=$attackSeed model=$modelSeed"
        }

        if (-not (Test-Path $finalCheckpoint)) {
            throw "Training ended without final checkpoint: $finalCheckpoint"
        }

        Write-Host "TRAIN COMPLETE attack=$attackSeed model=$modelSeed"
    }
}

$finalCount = 0

foreach ($attackSeed in $attackSeeds) {
    foreach ($modelSeed in $modelSeeds) {
        $finalCheckpoint = Join-Path `
            $root `
            "poison\attack_seed_$attackSeed\model_seed_$modelSeed\train\checkpoints\checkpoint_step_100000.pt"

        if (Test-Path $finalCheckpoint) {
            $finalCount++
        }
    }
}

Write-Host ""
Write-Host "Poison final checkpoints: $finalCount / 9"

if ($finalCount -ne 9) {
    throw "A4 training matrix incomplete"
}

$wslRepo = (
    & wsl.exe -e wslpath -a $repo
).Trim()

if ([string]::IsNullOrWhiteSpace($wslRepo)) {
    throw "Could not convert repo path to WSL path"
}

$evalScript = "$wslRepo/scripts/run_a4_eval.sh"

Write-Host ""
Write-Host "Starting WSL evaluation matrix..."
Write-Host "repo: $wslRepo"

& wsl.exe -e bash $evalScript $wslRepo

if ($LASTEXITCODE -ne 0) {
    throw "A4 WSL evaluation failed"
}

$evalCount = 0

foreach ($attackSeed in $attackSeeds) {
    foreach ($modelSeed in $modelSeeds) {
        $summary = Join-Path `
            $root `
            "poison\attack_seed_$attackSeed\model_seed_$modelSeed\eval\summary.json"

        if (Test-Path $summary) {
            $evalCount++
        }
    }
}

Write-Host ""
Write-Host "Poison evaluation summaries: $evalCount / 9"

if ($evalCount -ne 9) {
    throw "A4 evaluation matrix incomplete"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "A4 FINAL QUALIFICATION"
Write-Host "============================================================"

& python -m scripts.summarize_a4_dt_action_shift_qualification

if ($LASTEXITCODE -ne 0) {
    throw "A4 qualification summarizer failed"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "A4 PIPELINE COMPLETE"
Write-Host "============================================================"
