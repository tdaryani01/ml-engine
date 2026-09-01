# Controlled A/B: rebuild custom Docker image from current checkout, run fixed sample sweep.
param(
    [Parameter(Mandatory = $true)]
    [string]$Tag,
    [int]$Cores = 4
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ScriptDir

$CustomImg = "ml-engine-custom-bench:latest"
$CpuSet = "0-$($Cores - 1)"
$RuntimeScript = Join-Path $ScriptDir "utils\runtime.py"

function Convert-DockerMountPath {
    param([string]$Path)
    return ($Path -replace '\\', '/')
}

$RuntimeYaml = (& python $RuntimeScript --format docker-json | ConvertFrom-Json)
$DiagDir = Join-Path $ScriptDir $RuntimeYaml.diagnostics_dir
$ThreadCount = $RuntimeYaml.openmp_threads
$RuntimeConfigMount = @(
    "-v", "$(Convert-DockerMountPath (Join-Path $ScriptDir 'config/runtime.yaml')):/workspace/config/runtime.yaml"
)
if (-not (Test-Path $DiagDir)) { New-Item -ItemType Directory -Path $DiagDir | Out-Null }

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogName = "conv_ab_${Tag}_k1-3-4-7_s1-2_p1-2_$Stamp.log"
$HostLog = Join-Path $DiagDir $LogName
$ContainerLog = "/workspace/$($RuntimeYaml.diagnostics_dir)/$LogName"

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Yellow
Write-Host "  CONTROLLED A/B SAMPLE  tag=$Tag  commit=$(git rev-parse --short HEAD)" -ForegroundColor Yellow
Write-Host "  Rebuild image + run k=1,3,4,7 x s=1,2 x p=1,2" -ForegroundColor Yellow
Write-Host "  Log: $HostLog" -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Yellow

Write-Host "[+] Building custom Docker image from current tree..." -ForegroundColor Cyan
docker build -f scripts/Dockerfile.custom -t $CustomImg .
if ($LASTEXITCODE -ne 0) { throw "Docker build failed" }

$EnvFile = Join-Path $DiagDir ".runtime.env"
& python $RuntimeScript --threads $ThreadCount --platform linux --format env-file | Set-Content $EnvFile -Encoding ascii
$EnvFileMount = Convert-DockerMountPath $EnvFile

Write-Host "[+] Running sample sweep in container..." -ForegroundColor Cyan
docker run --rm `
    --name "ml-engine-ab-$Tag" `
    --cpuset-cpus=$CpuSet `
    --env-file=$EnvFileMount `
    --env "AB_TAG=$Tag" `
    --env "AB_OUTPUT=$ContainerLog" `
    @RuntimeConfigMount `
    -v "$(Convert-DockerMountPath $DiagDir):/workspace/$($RuntimeYaml.diagnostics_dir)" `
    --entrypoint python `
    $CustomImg `
    -u benchmarks/run_ab_sample.py

if ($LASTEXITCODE -ne 0) { throw "A/B sample run failed for tag=$Tag" }
Write-Host "[OK] A/B complete: $HostLog" -ForegroundColor Green
