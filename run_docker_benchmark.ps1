param(
    [ValidateSet("All", "Build", "Run", "Clean", "Sweep", "Matrix", "Sample")]
    [string]$Action = "All",
    [int[]]$SampleKernels = @(1, 3, 4, 7),
    [int]$Cores = 4,
    [ValidateRange(0, 2)]
    [int]$OneDnnVerbose = 0,
    [switch]$VerboseTracing,
    [switch]$NoCache,
    [int]$KMin = 1,
    [int]$KMax = 7,
    [int]$Pad = 1
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$env:DOCKER_BUILDKIT = 1

$PyTorchImg = "ml-engine-pytorch-bench:latest"
$CustomImg  = "ml-engine-custom-bench:latest"
$CpuSet     = "0-$($Cores - 1)"
$ConfigPath = Join-Path $ScriptDir "config\config.yaml"
$RuntimeScript = Join-Path $ScriptDir "utils\runtime.py"

function Convert-DockerMountPath {
    param([string]$Path)
    return ($Path -replace '\\', '/')
}

function Get-RuntimeYamlSettings {
    if (-not (Test-Path $RuntimeScript)) {
        Write-Error "[ERROR] Missing runtime loader: $RuntimeScript"
        exit 1
    }
    $json = & python $RuntimeScript --format docker-json | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[ERROR] Failed to read config/runtime.yaml"
        exit 1
    }
    return $json
}

$RuntimeYaml = Get-RuntimeYamlSettings
$DiagDir = Join-Path $ScriptDir ($RuntimeYaml.diagnostics_dir)
$RuntimeConfigMount = @(
    "-v", "$(Convert-DockerMountPath (Join-Path $ScriptDir 'config/runtime.yaml')):/workspace/config/runtime.yaml"
)
if (-not (Test-Path $DiagDir)) {
    New-Item -ItemType Directory -Path $DiagDir -Force | Out-Null
}

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

function Write-RuntimeEnvFile {
    param(
        [int]$Threads,
        [hashtable]$Overrides = @{}
    )

    if (-not (Test-Path $RuntimeScript)) {
        Write-Error "[ERROR] Missing runtime loader: $RuntimeScript"
        exit 1
    }

    $overrideArgs = @()
    foreach ($key in $Overrides.Keys) {
        $overrideArgs += "--override", "${key}=$($Overrides[$key])"
    }

    $envFile = Join-Path $DiagDir ".runtime.env"
    $lines = & python $RuntimeScript --threads $Threads --platform linux --format docker-args @overrideArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "[ERROR] Failed to load config/runtime.yaml"
        exit 1
    }

    $lines | Set-Content -Path $envFile -Encoding ascii
    return (Convert-DockerMountPath $envFile)
}

function Invoke-DockerBenchmarkRun {
    param(
        [string]$Name,
        [string]$Image,
        [string]$EnvFile,
        [string[]]$ExtraArgs = @(),
        [string[]]$CapArgs = @(),
        [string[]]$CommandArgs
    )

    $runArgs = @(
        "run", "--rm",
        "--name", $Name,
        "--cpuset-cpus=$CpuSet",
        "--env-file=$EnvFile"
    ) + $CapArgs + $ExtraArgs + $RuntimeConfigMount + @(
        "-v", "$(Convert-DockerMountPath $DiagDir):/workspace/$($RuntimeYaml.diagnostics_dir)",
        $Image
    ) + $CommandArgs

    & docker @runArgs
}

function Get-RuntimeProfileSummary {
    param([int]$Threads)
    $json = & python $RuntimeScript --threads $Threads --platform linux --format json | ConvertFrom-Json
    $wait = if ($json.PSObject.Properties.Name -contains "OMP_WAIT_POLICY") { $json.OMP_WAIT_POLICY } else { "default" }
    $spin = if ($json.PSObject.Properties.Name -contains "GOMP_SPINCOUNT") { $json.GOMP_SPINCOUNT } else { "default" }
    return "wait=$wait, spin=$spin"
}

function Get-DockerEnvOverrides {
    $overrides = @{}
    if ($OneDnnVerbose -ne 0 -or $VerboseTracing) {
        $level = $OneDnnVerbose
        if ($VerboseTracing -and $level -eq 0) {
            $level = 1
        }
        $overrides["ONEDNN_VERBOSE"] = "$level"
    }
    return $overrides
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

function Build-CustomContainer {
    Write-Host ""
    Write-Host "[+] Verifying Docker context and endpoint connectivity..." -ForegroundColor Cyan
    Test-DockerEndpoint

    $CacheArg = if ($NoCache) { "--no-cache" } else { "" }

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
    Write-Host "[OK] Custom benchmark container image successfully built." -ForegroundColor Green
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

    Build-CustomContainer
    Write-Host ""
    Write-Host "[OK] Both benchmark container images successfully built." -ForegroundColor Green
}

function Run-KernelSweep {
    Test-DockerEndpoint
    Stop-ExistingBenchmarkContainers

    $EnvOverrides = Get-DockerEnvOverrides
    $RuntimeEnvFile = Write-RuntimeEnvFile -Threads $ThreadCount -Overrides $EnvOverrides
    $SweepLog = Join-Path $DiagDir "kernel_sweep_pad${Pad}_k${KMin}-${KMax}.log"

    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Yellow
    Write-Host "  DOCKER KERNEL SWEEP - GENERIC FALLBACK ONLY (k=${KMin}-${KMax}, pad=${Pad})" -ForegroundColor Yellow
    Write-Host "  Hardware Allocation  : $Cores Dedicated Cores (cpuset: $CpuSet)" -ForegroundColor Yellow
    Write-Host "  OpenMP Thread Count  : $ThreadCount" -ForegroundColor Yellow
    Write-Host "  Log file             : $SweepLog" -ForegroundColor Yellow
    Write-Host "==================================================================" -ForegroundColor Yellow

    Write-Host ""
    Write-Host "[+] Executing kernel sweep in Custom Engine container..." -ForegroundColor Cyan
    Invoke-DockerBenchmarkRun `
        -Name "ml-engine-bench-sweep" `
        -Image $CustomImg `
        -EnvFile $RuntimeEnvFile `
        -ExtraArgs @("--entrypoint", "python") `
        -CommandArgs @(
            "-u", "benchmarks/sweep_kernel_pad.py",
            "--k-min", "$KMin", "--k-max", "$KMax", "--pad", "$Pad",
            "--output", "/workspace/$($RuntimeYaml.diagnostics_dir)/kernel_sweep_pad${Pad}_k${KMin}-${KMax}.log"
        )

    if ($LASTEXITCODE -ne 0) {
        Write-Error "[ERROR] Kernel sweep container run failed."
        exit 1
    }

    Write-Host ""
    Write-Host "[OK] Kernel sweep complete. Log: $SweepLog" -ForegroundColor Green
}

function Run-SampleSweep {
    Test-DockerEndpoint
    Stop-ExistingBenchmarkContainers

    $EnvOverrides = Get-DockerEnvOverrides
    $RuntimeEnvFile = Write-RuntimeEnvFile -Threads $ThreadCount -Overrides $EnvOverrides
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $KernelTag = ($SampleKernels -join "-")
    $SweepLog = Join-Path $DiagDir "conv_sample_k${KernelTag}_s1-2_p1-2_$Stamp.log"

    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Yellow
    Write-Host "  DOCKER CONV SAMPLE SWEEP (kernels=$KernelTag, stride=1,2, pad=1,2)" -ForegroundColor Yellow
    Write-Host "  Hardware Allocation  : $Cores Dedicated Cores (cpuset: $CpuSet)" -ForegroundColor Yellow
    Write-Host "  OpenMP Thread Count  : $ThreadCount" -ForegroundColor Yellow
    Write-Host "  Log file             : $SweepLog" -ForegroundColor Yellow
    Write-Host "==================================================================" -ForegroundColor Yellow

    $KernelArgs = $SampleKernels | ForEach-Object { "$_" }

    $CommandArgs = @(
        "-u", "benchmarks/sweep_kernel_pad.py",
        "--kernels"
    ) + $KernelArgs + @(
        "--strides", "1", "2",
        "--pads", "1", "2",
        "--output", "/workspace/$($RuntimeYaml.diagnostics_dir)/conv_sample_k${KernelTag}_s1-2_p1-2_$Stamp.log"
    )

    Write-Host ""
    Write-Host "[+] Executing sample conv sweep in Custom Engine container..." -ForegroundColor Cyan
    Invoke-DockerBenchmarkRun `
        -Name "ml-engine-bench-sample" `
        -Image $CustomImg `
        -EnvFile $RuntimeEnvFile `
        -ExtraArgs @("--entrypoint", "python") `
        -CommandArgs $CommandArgs

    if ($LASTEXITCODE -ne 0) {
        Write-Error "[ERROR] Sample conv sweep container run failed."
        exit 1
    }

    Write-Host ""
    Write-Host "[OK] Sample conv sweep complete. Log: $SweepLog" -ForegroundColor Green
}

function Run-MatrixSweep {
    Test-DockerEndpoint
    Stop-ExistingBenchmarkContainers

    $EnvOverrides = Get-DockerEnvOverrides
    $RuntimeEnvFile = Write-RuntimeEnvFile -Threads $ThreadCount -Overrides $EnvOverrides
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $SweepLog = Join-Path $DiagDir "conv_matrix_k1-7_s1-2_p1-2_$Stamp.log"

    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Yellow
    Write-Host "  DOCKER CONV MATRIX SWEEP (k=1-7, stride=1,2, pad=1,2)" -ForegroundColor Yellow
    Write-Host "  Hardware Allocation  : $Cores Dedicated Cores (cpuset: $CpuSet)" -ForegroundColor Yellow
    Write-Host "  OpenMP Thread Count  : $ThreadCount" -ForegroundColor Yellow
    Write-Host "  Log file             : $SweepLog" -ForegroundColor Yellow
    Write-Host "==================================================================" -ForegroundColor Yellow

    Write-Host ""
    Write-Host "[+] Executing full conv matrix in Custom Engine container..." -ForegroundColor Cyan
    Invoke-DockerBenchmarkRun `
        -Name "ml-engine-bench-matrix" `
        -Image $CustomImg `
        -EnvFile $RuntimeEnvFile `
        -ExtraArgs @("--entrypoint", "python") `
        -CommandArgs @(
            "-u", "benchmarks/sweep_kernel_pad.py",
            "--full-matrix",
            "--output", "/workspace/$($RuntimeYaml.diagnostics_dir)/conv_matrix_k1-7_s1-2_p1-2_$Stamp.log"
        )

    if ($LASTEXITCODE -ne 0) {
        Write-Error "[ERROR] Conv matrix sweep container run failed."
        exit 1
    }

    Write-Host ""
    Write-Host "[OK] Conv matrix sweep complete. Log: $SweepLog" -ForegroundColor Green
}

function Run-Benchmarks {
    Test-DockerEndpoint
    Stop-ExistingBenchmarkContainers

    $EnvOverrides = Get-DockerEnvOverrides
    $RuntimeEnvFile = Write-RuntimeEnvFile -Threads $ThreadCount -Overrides $EnvOverrides

    $OnednnDisplay = $EnvOverrides["ONEDNN_VERBOSE"]
    if (-not $OnednnDisplay) {
        $OnednnDisplay = (& python $RuntimeScript --threads $ThreadCount --platform linux --format json | ConvertFrom-Json).ONEDNN_VERBOSE
    }

    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Yellow
    Write-Host "  DOCKER CONVERGENCE BENCHMARK ORCHESTRATOR - ISOLATED RUN" -ForegroundColor Yellow
    Write-Host "  Hardware Allocation  : $Cores Dedicated Cores (cpuset: $CpuSet)" -ForegroundColor Yellow
    Write-Host "  OpenMP Thread Count  : $ThreadCount" -ForegroundColor Yellow
    Write-Host "  Runtime OMP Profile  : $(Get-RuntimeProfileSummary -Threads $ThreadCount)" -ForegroundColor Yellow
    Write-Host "  oneDNN Verbose Level : $OnednnDisplay (config/runtime.yaml)" -ForegroundColor Yellow
    Write-Host "==================================================================" -ForegroundColor Yellow

    # 1. Run PyTorch in isolation
    Write-Host ""
    Write-Host "[+] Executing PyTorch Isolated Benchmark Container..." -ForegroundColor Cyan
    Invoke-DockerBenchmarkRun `
        -Name "ml-engine-bench-pytorch" `
        -Image $PyTorchImg `
        -EnvFile $RuntimeEnvFile `
        -CommandArgs @("--target=pytorch")

    if ($LASTEXITCODE -ne 0) {
        Write-Error "[ERROR] PyTorch container benchmark run failed."
        exit 1
    }

    # 2. Run Custom Engine in isolation
    Write-Host ""
    Write-Host "[+] Executing Custom Engine Isolated Benchmark Container..." -ForegroundColor Cyan
    Invoke-DockerBenchmarkRun `
        -Name "ml-engine-bench-custom" `
        -Image $CustomImg `
        -EnvFile $RuntimeEnvFile `
        -CommandArgs @("--target=custom")

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
    "Sweep" {
        Build-CustomContainer
        Run-KernelSweep
    }
    "Matrix" {
        Build-CustomContainer
        Run-MatrixSweep
    }
    "Sample" {
        Build-CustomContainer
        Run-SampleSweep
    }
    "All"   {
        Build-Containers
        Run-Benchmarks
    }
}