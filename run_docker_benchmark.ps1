param(
    [ValidateSet("All", "Build", "Run", "Clean")]
    [string]$Action = "All",
    [int]$Cores = 4,
    [ValidateRange(0, 2)]
    [int]$OneDnnVerbose = 0,
    [switch]$VerboseTracing,
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$env:DOCKER_BUILDKIT = 1

$DiagDir = Join-Path $ScriptDir "benchmark_diagnostics"
if (-not (Test-Path $DiagDir)) {
    New-Item -ItemType Directory -Path $DiagDir -Force | Out-Null
}

$PyTorchImg = "ml-engine-pytorch-bench:latest"
$CustomImg  = "ml-engine-custom-bench:latest"
$CpuSet     = "0-$($Cores - 1)"
$ConfigPath = Join-Path $ScriptDir "config\config.yaml"

function Get-ConfigThreadCount {
    param([int]$Fallback)
    if (-not (Test-Path $ConfigPath)) { return $Fallback }
    $content = Get-Content $ConfigPath -Raw
    if ($content -match 'num_threads:\s*(\d+)') { return [int]$Matches[1] }
    return $Fallback
}

$ThreadCount = Get-ConfigThreadCount -Fallback $Cores
if ($ThreadCount -ne $Cores) {
    Write-Warning "config num_threads=$ThreadCount differs from -Cores $Cores; using config value for OMP env."
}

# Determine oneDNN verbosity level
$VerboseLevel = $OneDnnVerbose
if ($VerboseTracing -and $VerboseLevel -eq 0) {
    $VerboseLevel = 1
}

function Stop-ExistingBenchmarkContainers {
    $ids = @(
        docker ps -q --filter "ancestor=$PyTorchImg" 2>$null
        docker ps -q --filter "ancestor=$CustomImg" 2>$null
    ) | Where-Object { $_ } | Select-Object -Unique

    if ($ids) {
        Write-Host "[!] Stopping $($ids.Count) leftover benchmark container(s)..." -ForegroundColor Yellow
        docker stop @ids | Out-Null
    }

    docker rm -f ml-engine-bench-pytorch ml-engine-bench-custom 2>$null | Out-Null
}

function Get-DockerSharedEnvArgs {
    param([int]$Threads)
    return @(
        "-e", "OMP_NUM_THREADS=$Threads",
        "-e", "OMP_THREAD_LIMIT=$Threads",
        "-e", "OMP_PROC_BIND=false",
        "-e", "OMP_DYNAMIC=false",
        "-e", "PYTHONUNBUFFERED=1"
    )
}

# Intel OpenMP / oneDNN (PyTorch CPU)
function Get-DockerPyTorchEnvArgs {
    param([int]$Threads)
    return @(
        (Get-DockerSharedEnvArgs -Threads $Threads)
        "-e", "MKL_NUM_THREADS=$Threads",
        "-e", "OPENBLAS_NUM_THREADS=$Threads",
        "-e", "KMP_DEVICE_THREAD_LIMIT=$Threads",
        "-e", "KMP_ALL_THREADS=$Threads",
        "-e", "KMP_AFFINITY=none",
        "-e", "KMP_BLOCKTIME=5"
    )
}

# libgomp (custom native AVX2 kernels)
function Get-DockerCustomEnvArgs {
    param([int]$Threads)
    return @(
        (Get-DockerSharedEnvArgs -Threads $Threads)
        "-e", "OMP_WAIT_POLICY=ACTIVE",
        "-e", "GOMP_SPINCOUNT=100000"
    )
}

function Test-DockerEndpoint {
    try {
        docker info > $null 2>&1
    }
    catch {
        Write-Warning "Docker daemon unresponsive. Re-evaluating default context..."
        docker context use default | Out-Null
    }
}

function Build-Containers {
    Write-Host ""
    Write-Host "[+] Verifying Docker context and endpoint connectivity..." -ForegroundColor Cyan
    Test-DockerEndpoint

    $CacheArg = if ($NoCache) { "--no-cache" } else { "" }

    Write-Host ""
    Write-Host "[+] Building PyTorch Isolated Image (scripts/Dockerfile.pytorch)..." -ForegroundColor Cyan
    docker build $CacheArg `
        -f scripts/Dockerfile.pytorch `
        -t $PyTorchImg .

    if ($LASTEXITCODE -ne 0) {
        Write-Error "[ERROR] PyTorch image build failed."
        exit 1
    }

    Write-Host ""
    Write-Host "[+] Building Custom Engine Isolated Image (scripts/Dockerfile.custom)..." -ForegroundColor Cyan
    docker build $CacheArg `
        -f scripts/Dockerfile.custom `
        -t $CustomImg .

    if ($LASTEXITCODE -ne 0) {
        Write-Error "[ERROR] Custom Engine image build failed."
        exit 1
    }

    Write-Host ""
    Write-Host "[OK] Both benchmark container images successfully built." -ForegroundColor Green
}

function Run-Benchmarks {
    Test-DockerEndpoint
    Stop-ExistingBenchmarkContainers

    $PyTorchEnv = Get-DockerPyTorchEnvArgs -Threads $ThreadCount
    $CustomEnv = Get-DockerCustomEnvArgs -Threads $ThreadCount

    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Yellow
    Write-Host "  DOCKER CONVERGENCE BENCHMARK ORCHESTRATOR - ISOLATED RUN" -ForegroundColor Yellow
    Write-Host "  Hardware Allocation  : $Cores Dedicated Cores (cpuset: $CpuSet)" -ForegroundColor Yellow
    Write-Host "  OpenMP Thread Count  : $ThreadCount" -ForegroundColor Yellow
    Write-Host "  PyTorch Runtime Tuning : Intel KMP (affinity=none, blocktime=5)" -ForegroundColor Yellow
    Write-Host "  Custom Runtime Tuning  : libgomp (wait=ACTIVE, spin=100k)" -ForegroundColor Yellow
    Write-Host "  oneDNN Verbose Level : $VerboseLevel" -ForegroundColor Yellow
    Write-Host "==================================================================" -ForegroundColor Yellow

    # 1. Run PyTorch in isolation
    Write-Host ""
    Write-Host "[+] Executing PyTorch Isolated Benchmark Container..." -ForegroundColor Cyan
    docker run --rm `
        --name ml-engine-bench-pytorch `
        --cpuset-cpus=$CpuSet `
        @PyTorchEnv `
        -e ONEDNN_VERBOSE=$VerboseLevel `
        -v "${DiagDir}:/workspace/benchmark_diagnostics" `
        $PyTorchImg `
        --target=pytorch

    if ($LASTEXITCODE -ne 0) {
        Write-Error "[ERROR] PyTorch container benchmark run failed."
        exit 1
    }

    # 2. Run Custom Engine in isolation
    Write-Host ""
    Write-Host "[+] Executing Custom Engine Isolated Benchmark Container..." -ForegroundColor Cyan
    docker run --rm `
        --name ml-engine-bench-custom `
        --cpuset-cpus=$CpuSet `
        @CustomEnv `
        -v "${DiagDir}:/workspace/benchmark_diagnostics" `
        $CustomImg `
        --target=custom

    if ($LASTEXITCODE -ne 0) {
        Write-Error "[ERROR] Custom Engine container benchmark run failed."
        exit 1
    }
}

function Clean-Containers {
    Write-Host ""
    Write-Host "[+] Pruning benchmark containers and dangling build stages..." -ForegroundColor Yellow
    docker image rm -f $PyTorchImg $CustomImg 2>$null
    docker builder prune -f
    Write-Host "[OK] Benchmark artifacts cleaned." -ForegroundColor Green
}

switch ($Action) {
    "Build" { Build-Containers }
    "Run"   { Run-Benchmarks }
    "Clean" { Clean-Containers }
    "All"   {
        Build-Containers
        Run-Benchmarks
    }
}