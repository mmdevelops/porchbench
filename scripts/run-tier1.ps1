# Tier 1+2 Overnight Benchmark
# 9 models x 3 suites, Tier 1 with 3 repeats, Tier 2 with 1 repeat
#
# Usage: .\scripts\run-tier1.ps1
# Expected duration: ~6-10 hours on GPU.

# ---------------------------------------------------------------------------
# GPU preflight check — abort if Ollama is running on CPU
# ---------------------------------------------------------------------------

function Test-OllamaGPU {
    Write-Host "Checking GPU status..." -ForegroundColor Cyan
    $null = ollama run gemma4:e4b "ok" --verbose 2>&1

    $logPath = "$env:LOCALAPPDATA\Ollama\server.log"
    if (-not (Test-Path $logPath)) {
        Write-Host "WARNING: Cannot find server.log at $logPath" -ForegroundColor Yellow
        return $true
    }

    $log = Get-Content $logPath -Tail 40 | Out-String
    if ($log -match 'offloaded 0/\d+ layers to GPU') {
        Write-Host "ABORT: Ollama is running on CPU (0 layers offloaded to GPU)." -ForegroundColor Red
        Write-Host "Restart Ollama and verify GPU is working before running benchmarks." -ForegroundColor Red
        Write-Host ""
        Write-Host "Fix: copy gfx1201 rocblas files from ROCm SDK into Ollama's rocm dir." -ForegroundColor Yellow
        return $false
    }
    if ($log -match 'offloaded (\d+)/(\d+) layers to GPU') {
        $offloaded = $Matches[1]
        $total = $Matches[2]
        Write-Host "GPU OK: $offloaded/$total layers offloaded to GPU" -ForegroundColor Green
        return $true
    }

    Write-Host "WARNING: Could not determine GPU status from server.log. Proceeding anyway." -ForegroundColor Yellow
    return $true
}

if (-not (Test-OllamaGPU)) {
    exit 1
}

Write-Host ""

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

$bench = "C:\Users\micmo\AppData\Roaming\Python\Python313\Scripts\ollama-bench.exe"
$out = "results"

$suites = @(
    "suites/coding-basics.yaml",
    "suites/routing-discovery.yaml",
    "suites/tool-use.yaml"
)

# Tier 1: 3 repeats — core models with strong prior signal
$tier1 = @(
    "gemma4:e4b",
    "qwen2.5-coder:14b",
    "qwen2.5-coder:14b-instruct-q8_0",
    "qwen3:8b",
    "phi4:14b",
    "deepseek-r1:8b"
)

# Tier 2: 1 repeat — scaling and architecture comparisons
$tier2 = @(
    "gemma4:e2b",
    "qwen2.5-coder:7b",
    "qwen2.5-coder:3b",
    "gemma4:26b"
)

$tier1_repeats = 3
$tier2_repeats = 1

$total_runs = ($tier1.Count * $suites.Count * $tier1_repeats) + ($tier2.Count * $suites.Count * $tier2_repeats)

Write-Host "=== Overnight Benchmark ===" -ForegroundColor Cyan
Write-Host "Tier 1: $($tier1.Count) models x $($suites.Count) suites x $tier1_repeats repeats"
Write-Host "Tier 2: $($tier2.Count) models x $($suites.Count) suites x $tier2_repeats repeat"
Write-Host "Total runs: $total_runs"
Write-Host ""

# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------

Write-Host "=== Tier 1 (3 repeats) ===" -ForegroundColor Cyan
$t1_start = Get-Date

foreach ($suite in $suites) {
    foreach ($model in $tier1) {
        Write-Host "`n>>> $model x $suite x $tier1_repeats repeats" -ForegroundColor Green
        & $bench run -s $suite -m $model --repeats $tier1_repeats -o $out
        if ($LASTEXITCODE -ne 0) {
            Write-Host "WARNING: $model failed on $suite" -ForegroundColor Yellow
        }
    }
}

$t1_elapsed = (Get-Date) - $t1_start
Write-Host "`nTier 1 done in $($t1_elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan

# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------

Write-Host "`n=== Tier 2 (1 repeat) ===" -ForegroundColor Cyan
$t2_start = Get-Date

foreach ($suite in $suites) {
    foreach ($model in $tier2) {
        Write-Host "`n>>> $model x $suite" -ForegroundColor Green
        & $bench run -s $suite -m $model -o $out
        if ($LASTEXITCODE -ne 0) {
            Write-Host "WARNING: $model failed on $suite" -ForegroundColor Yellow
        }
    }
}

$t2_elapsed = (Get-Date) - $t2_start
$total_elapsed = (Get-Date) - $t1_start

Write-Host "`nTier 2 done in $($t2_elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host "`n=== Overnight Benchmark Complete ===" -ForegroundColor Cyan
Write-Host "Total time: $($total_elapsed.ToString('hh\:mm\:ss'))"
Write-Host "Results in: $out"
