param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$jobs = @(
    @{
        Condition   = "canonical"
        Rho         = "0.01"
        RhoSlug     = "001"
        DatasetRoot = "data/poisoned/csdpc/walker2d-medium-v2"
    },
    @{
        Condition   = "canonical"
        Rho         = "0.05"
        RhoSlug     = "005"
        DatasetRoot = "data/poisoned/csdpc/walker2d-medium-v2"
    },
    @{
        Condition   = "s2_overlap_r0"
        Rho         = "0.01"
        RhoSlug     = "001"
        DatasetRoot = "data/poisoned/csdpc_s2_overlap_r0_v1/walker2d-medium-v2"
    },
    @{
        Condition   = "s2_overlap_r0"
        Rho         = "0.05"
        RhoSlug     = "005"
        DatasetRoot = "data/poisoned/csdpc_s2_overlap_r0_v1/walker2d-medium-v2"
    }
)

foreach ($job in $jobs) {
    foreach ($seed in 0, 1, 2) {

        $dataset = Join-Path `
            $job.DatasetRoot `
            "rho_$($job.RhoSlug)_seed_$seed.hdf5"

        $outDir = `
            "experiments/dt_mtm_stress/walker2d_medium/" +
            "$($job.Condition)/rho_$($job.RhoSlug)/" +
            "attack_seed_$seed/train_seed_$seed"

        $summary = Join-Path $outDir "summary.json"

        Write-Host ""
        Write-Host ("=" * 78)
        Write-Host "DT+MTM | $($job.Condition) | rho=$($job.Rho) | attack=$seed | train=$seed"
        Write-Host ("=" * 78)

        if (-not (Test-Path $dataset)) {
            throw "Missing frozen dataset: $dataset"
        }

        if (Test-Path $summary) {
            $saved = Get-Content $summary -Raw | ConvertFrom-Json

            if (
                $saved.status -eq "complete" -and
                [int]$saved.final_step -eq 100000
            ) {
                Write-Host "SKIP: already complete at update 100000."
                continue
            }
        }

        $checkpointDir = Join-Path $outDir "checkpoints"
        $latestCheckpoint = $null

        if (Test-Path $checkpointDir) {
            $latestCheckpoint = Get-ChildItem `
                $checkpointDir `
                -Filter "joint_step_*.pt" `
                -File `
                -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending |
                Select-Object -First 1
        }

        $args = @(
            "-m", "scripts.train_dt_mtm_stress",
            "--dataset", $dataset,
            "--condition", $job.Condition,
            "--rho", $job.Rho,
            "--attack-seed", "$seed",
            "--seed", "$seed",
            "--device", "cuda",
            "--lambda-mtm", "1.0",
            "--num-updates", "100000",
            "--checkpoint-every", "10000",
            "--output-dir", $outDir
        )

        if ($latestCheckpoint) {
            Write-Host "RESUME: $($latestCheckpoint.FullName)"

            $args += @(
                "--resume",
                $latestCheckpoint.FullName
            )
        }
        elseif (Test-Path $outDir) {
            $files = @(
                Get-ChildItem `
                    $outDir `
                    -Recurse `
                    -File `
                    -ErrorAction SilentlyContinue
            )

            if ($files.Count -gt 0) {
                throw @"
Incomplete Group 4D run exists but has no checkpoint:

$outDir

Inspect it manually rather than overwriting it.
"@
            }
        }

        if ($DryRun) {
            Write-Host "DRY RUN:"
            Write-Host "python $($args -join ' ')"
            continue
        }

        & python @args

        if ($LASTEXITCODE -ne 0) {
            throw "Training failed: $($job.Condition), rho=$($job.Rho), seed=$seed"
        }
    }
}
