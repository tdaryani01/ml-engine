# Build OpenBLAS (BLAS-only, USE_OPENMP=1, /openmp:llvm).
#   Artifacts: build/openblas/ (CMake tree + openblas_build.json)
#   Runtime:   bin/libopenblas.dll copied from build/openblas on every successful build
#
# Usage:
#   .\scripts\build_openblas.ps1
#   .\build_native.ps1
param(
    [string]$OpenBlasTag = "v0.3.28",
    [string]$Target = "HASWELL",
    [int]$NumThreads = 0,
    [switch]$SkipClone,
    [switch]$Full,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$ThirdParty = Join-Path $Root "third_party"
$Src = Join-Path $ThirdParty "OpenBLAS"
$ArtifactDir = Join-Path $Root "build\openblas"
$BinDir = Join-Path $Root "bin"
$OutDll = Join-Path $BinDir "libopenblas.dll"
$OutMarker = Join-Path $BinDir "openblas_build.json"
$ArtifactMarker = Join-Path $ArtifactDir "openblas_build.json"

New-Item -ItemType Directory -Force -Path $BinDir, $ThirdParty | Out-Null

if (-not $SkipClone) {
    if (-not (Test-Path $Src)) {
        Write-Host "[openblas] Cloning OpenBLAS $OpenBlasTag ..." -ForegroundColor Cyan
        git clone --depth 1 --branch $OpenBlasTag https://github.com/OpenMathLib/OpenBLAS.git $Src
    }
}

$VsPath = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $VsPath) { Write-Error "[openblas] MSVC x64 toolset not found." }
$Vcvars = Join-Path $VsPath "VC\Auxiliary\Build\vcvars64.bat"
$Cmake = Join-Path $VsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if (-not (Test-Path $Cmake)) { Write-Error "[openblas] CMake not found at $Cmake" }

$NinjaExe = (Get-Command ninja -ErrorAction SilentlyContinue).Source
if (-not $NinjaExe) {
    $wingetNinja = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet" -Recurse -Filter "ninja.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
    if ($wingetNinja) { $NinjaExe = $wingetNinja }
}
if ($NinjaExe) {
    $Generator = "Ninja"
    $GenFlag = "-G `"Ninja`""
    $env:PATH = "$(Split-Path $NinjaExe -Parent);$env:PATH"
    Write-Host "[openblas] Using ninja: $NinjaExe (parallel build)" -ForegroundColor Cyan
} else {
    Write-Host "[openblas] ninja not found; using NMake Makefiles (serial). Install: winget install Ninja-build.Ninja" -ForegroundColor Yellow
    $Generator = "NMake Makefiles"
    $GenFlag = '-G "NMake Makefiles"'
}

if ($Clean) {
    Remove-Item $ArtifactDir -Recurse -Force -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force -Path $ArtifactDir | Out-Null

$LapackFlag = if ($Full) { "-DBUILD_WITHOUT_LAPACK=OFF" } else { "-DBUILD_WITHOUT_LAPACK=ON" }
$Mode = if ($Full) { "BLAS+LAPACK" } else { "BLAS-only (SGEMM)" }
if ($NumThreads -le 0) {
    $CfgPath = Join-Path $Root "config\config.yaml"
    if (Test-Path $CfgPath) {
        $CfgRaw = Get-Content $CfgPath -Raw
        if ($CfgRaw -match 'num_threads:\s*(\d+)') {
            $NumThreads = [int]$Matches[1]
        }
    }
    if ($NumThreads -le 0) { $NumThreads = 4 }
}
Write-Host "[openblas] Configuring USE_OPENMP=1 /openmp:llvm TARGET=$Target NUM_THREADS=$NumThreads $Mode generator=$Generator" -ForegroundColor Cyan
Write-Host "[openblas] out: $ArtifactDir" -ForegroundColor DarkGray

$ConfigureBat = Join-Path $env:TEMP "ml-engine-openblas-cmake.bat"
@"
call "$Vcvars" >nul
cd /d "$ArtifactDir"
"$Cmake" $GenFlag "$Src" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
  -DUSE_OPENMP=ON ^
  -DCMAKE_C_FLAGS="/O2 /openmp:llvm" ^
  -DCMAKE_SHARED_LINKER_FLAGS="/openmp:llvm" ^
  -DBUILD_SHARED_LIBS=ON ^
  -DNOFORTRAN=ON ^
  -DBUILD_TESTING=OFF ^
  -DDYNAMIC_ARCH=OFF ^
  -DTARGET=$Target ^
  -DNUM_THREADS=$NumThreads ^
  $LapackFlag
exit /b %ERRORLEVEL%
"@ | Set-Content -Path $ConfigureBat -Encoding ASCII

$ErrorActionPreference = "Continue"
cmd /c $ConfigureBat *> $env:TEMP\ml-engine-openblas-cmake.log
$configureExit = $LASTEXITCODE
Get-Content $env:TEMP\ml-engine-openblas-cmake.log | Select-Object -Last 20
$ErrorActionPreference = "Stop"
if ($configureExit -ne 0) { Write-Error "[openblas] CMake configure failed (exit $configureExit). See $env:TEMP\ml-engine-openblas-cmake.log" }

$BuildBat = Join-Path $env:TEMP "ml-engine-openblas-build.bat"
$Jobs = [Environment]::ProcessorCount
if ($Generator -eq "Ninja") {
    $BuildCmd = "`"$Cmake`" --build . --parallel $Jobs"
} else {
    $BuildCmd = "`"$Cmake`" --build . --config Release"
}
@"
call "$Vcvars" >nul
cd /d "$ArtifactDir"
$BuildCmd
exit /b %ERRORLEVEL%
"@ | Set-Content -Path $BuildBat -Encoding ASCII

$ErrorActionPreference = "Continue"
cmd /c $BuildBat *> $env:TEMP\ml-engine-openblas-build.log
$buildExit = $LASTEXITCODE
Get-Content $env:TEMP\ml-engine-openblas-build.log | ForEach-Object {
    if ($_ -match '^\[[0-9]+/[0-9]+\]|Built target|Linking|error:|FAILED:') { Write-Host $_ }
}
$ErrorActionPreference = "Stop"
if ($buildExit -ne 0) { Write-Error "[openblas] Build failed (exit $buildExit). See $env:TEMP\ml-engine-openblas-build.log" }

function Find-BuiltOpenBlasDll {
    param([string]$SearchRoot)
    @(
        Get-ChildItem -Path $SearchRoot -Recurse -Filter "openblas.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
        Get-ChildItem -Path $SearchRoot -Recurse -Filter "libopenblas.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
    ) | Where-Object { $_ } | Select-Object -First 1
}

$BuiltDll = Find-BuiltOpenBlasDll -SearchRoot $ArtifactDir
if (-not $BuiltDll) { Write-Error "[openblas] Built DLL not found under $ArtifactDir" }

@{
    use_openmp = $true
    omp_runtime = "libomp"
    target     = $Target
    tag        = $OpenBlasTag
    num_threads = $NumThreads
    generator  = $Generator
} | ConvertTo-Json | Set-Content -Path $ArtifactMarker -Encoding UTF8

try {
    Copy-Item -Path $BuiltDll.FullName -Destination $OutDll -Force
    Copy-Item -Path $ArtifactMarker -Destination $OutMarker -Force
} catch {
    Write-Error @"
[openblas] Cannot copy to $OutDll
        Close Python / any process using libopenblas.dll, then rerun:
          .\scripts\build_openblas.ps1
"@
}

Write-Host "[openblas] Wrote $($BuiltDll.FullName) ($('{0:N0}' -f $BuiltDll.Length) bytes)" -ForegroundColor Green
Write-Host "[openblas] Copied -> $OutDll" -ForegroundColor Green
Write-Host "[openblas] Marker: $ArtifactMarker -> $OutMarker (USE_OPENMP=1 libomp)" -ForegroundColor Green

$DumpBat = Join-Path $env:TEMP "ml-engine-openblas-dump.bat"
@"
call "$Vcvars" >nul
dumpbin /exports "$OutDll" | findstr /i "openblas_set_num_threads sgemm"
dumpbin /dependents "$OutDll"
exit /b %ERRORLEVEL%
"@ | Set-Content -Path $DumpBat -Encoding ASCII
$DumpOut = cmd /c $DumpBat 2>&1 | Out-String
Write-Host $DumpOut
if ($DumpOut -match "VCOMP140") {
    Write-Error "[openblas] Built DLL still links VCOMP140 - arch.cmake patch missing or /openmp:llvm not applied."
}
if ($DumpOut -notmatch "libomp") {
    Write-Warning "[openblas] Expected libomp140.x86_64.dll dependency; verify /openmp:llvm build."
}
if ($DumpOut -notmatch "openblas_set_num_threads") {
    Write-Warning "[openblas] openblas_set_num_threads export missing"
}

Write-Host "[openblas] Done. Next: .\build_native.ps1" -ForegroundColor Green
