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
    "suites/cross-domain.yaml"
)

# Tool-use suite excluded from overnight run — run separately with
# tool-calling models only (gemma4:e4b, qwen3:8b, qwen2.5-coder:14b)

# Tier 1: 3 repeats — core models with strong prior signal
$tier1 = @(
    "gemma4:e4b",
    "qwen2.5-coder:14b",
    "qwen2.5-coder:14b-instruct-q8_0",
    "qwen3:8b",
    "phi4:14b",
    "deepseek-r1:8b"
)

# Tier 2 excluded from overnight run — run separately after Tier 1 analysis
# $tier2 = @(
#     "gemma4:e2b",
#     "qwen2.5-coder:7b",
#     "qwen2.5-coder:3b",
#     "gemma4:26b"
# )

$repeats = 3

$total_runs = $tier1.Count * $suites.Count * $repeats

Write-Host "=== Overnight Benchmark (Tier 1) ===" -ForegroundColor Cyan
Write-Host "Models: $($tier1.Count)"
Write-Host "Suites: $($suites.Count)"
Write-Host "Repeats: $repeats"
Write-Host "Total runs: $total_runs"
Write-Host ""

$start = Get-Date

foreach ($suite in $suites) {
    foreach ($model in $tier1) {
        Write-Host "`n>>> $model x $suite x $repeats repeats" -ForegroundColor Green
        & $bench run -s $suite -m $model --repeats $repeats -o $out
        if ($LASTEXITCODE -ne 0) {
            Write-Host "WARNING: $model failed on $suite" -ForegroundColor Yellow
        }
    }
}

$elapsed = (Get-Date) - $start

Write-Host "`n=== Overnight Benchmark Complete ===" -ForegroundColor Cyan
Write-Host "Total time: $($elapsed.ToString('hh\:mm\:ss'))"
Write-Host "Results in: $out"
