# Build conv_kernels.dll (MSVC x64 Native Tools).
#   Artifacts: build/native/conv_kernels.dll (+ .pdb, .obj, .lib)
#   Runtime:   bin/conv_kernels.dll copied from build/native on every successful build
#
#   .\build_native.ps1                      # default: release
#   .\build_native.ps1 release              # optimized, no PDB
#   .\build_native.ps1 release-symbols      # optimized + PDB (uProf)
#   .\build_native.ps1 debug                # /Od + debug CRT + PDB
param(
    [Parameter(Position = 0)]
    [ValidateSet("debug", "release", "release-symbols")]
    [string]$Mode = "release",

    [switch]$RunTests
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$BuildDir = Join-Path $Root "build\native"
$BinDir = Join-Path $Root "bin"
$BuiltDll = Join-Path $BuildDir "conv_kernels.dll"
$BuiltPdb = Join-Path $BuildDir "conv_kernels.pdb"
$OutDll = Join-Path $BinDir "conv_kernels.dll"
$OutPdb = Join-Path $BinDir "conv_kernels.pdb"

$SourceFiles = @(
    "src\native\conv_fallback.cpp",
    "src\native\conv_dispatcher.cpp",
    "src\native\omp_config.cpp",
    "src\native\im2col.cpp",
    "src\native\im2col_telemetry.cpp",
    "src\native\blas_dynamic.cpp",
    "src\native\conv_im2col_gemm.cpp"
)

$Selected = switch ($Mode) {
    "debug" { "Debug" }
    "release-symbols" { "ReleaseSymbols" }
    default { "Release" }
}
$EmitPdb = $Mode -in @("debug", "release-symbols")
New-Item -ItemType Directory -Force -Path $BuildDir, $BinDir | Out-Null

$VsPath = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if (-not $VsPath) {
    Write-Error "[build] MSVC x64 toolset not found (vswhere)."
}

$Vcvars = Join-Path $VsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $Vcvars)) {
    Write-Error "[build] vcvars64.bat not found at $Vcvars"
}

$CommonCl = "/nologo /arch:AVX2 /openmp:llvm /LD /I. /Isrc\native"
$SourceArg = ($SourceFiles -join " ")
$Implib = Join-Path $BuildDir "conv_kernels.lib"

switch ($Selected) {
    "Release" {
        $ClFlags = "$CommonCl /O2 /Oi /Ot /Ox /GL /Gy /Gw /fp:fast /DNDEBUG"
        $LinkFlags = "/LTCG /OPT:REF /OPT:ICF /NODEFAULTLIB:libcmtd.lib /NODEFAULTLIB:msvcrtd.lib /IMPLIB:$Implib"
        $BuildLabel = "Release"
    }
    "ReleaseSymbols" {
        # /Zi + /DEBUG:FULL — PDB for uProf function names (/GL + /LTCG need /Fd sidecar).
        $ClFlags = "$CommonCl /O2 /Oi /Ot /Ox /GL /Gy /Gw /fp:fast /DNDEBUG /Zi /Fd$BuiltPdb"
        $LinkFlags = "/LTCG /OPT:REF /OPT:ICF /DEBUG:FULL /PDB:$BuiltPdb /NODEFAULTLIB:libcmtd.lib /NODEFAULTLIB:msvcrtd.lib /IMPLIB:$Implib"
        $BuildLabel = "Release + PDB"
    }
    "Debug" {
        $ClFlags = "$CommonCl /Od /Zi /MDd /D_DEBUG /Fd$BuiltPdb"
        $LinkFlags = "/DEBUG:FULL /PDB:$BuiltPdb /IMPLIB:$Implib"
        $BuildLabel = "Debug"
    }
}

Write-Host "[build] $BuildLabel" -ForegroundColor Cyan
Write-Host "[build] out: $BuiltDll" -ForegroundColor DarkGray
Write-Host "[build] cl: $ClFlags" -ForegroundColor DarkGray
Write-Host "[build] x64 Native Tools via vcvars64.bat" -ForegroundColor Cyan

Remove-Item $BuiltDll, $BuiltPdb -Force -ErrorAction SilentlyContinue
Get-ChildItem (Join-Path $BuildDir "*.obj") -ErrorAction SilentlyContinue | Remove-Item -Force
Remove-Item (Join-Path $BuildDir "*.ilk") -Force -ErrorAction SilentlyContinue

$BuildBat = Join-Path $env:TEMP "ml-engine-build-native.bat"
@"

call "$Vcvars" >nul
cd /d "$Root"
cl.exe $ClFlags $SourceArg /Fo:build\native\ /Fe:build\native\conv_kernels.dll /link $LinkFlags
exit /b %ERRORLEVEL%

"@ | Set-Content -Path $BuildBat -Encoding ASCII

cmd /c $BuildBat
if ($LASTEXITCODE -ne 0) {
    Write-Error "[build] cl.exe failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $BuiltDll)) {
    Write-Error "[build] Expected output missing: $BuiltDll"
}

function Publish-BinArtifacts {
    param([bool]$WithPdb)
    try {
        Copy-Item -Path $BuiltDll -Destination $OutDll -Force
    } catch {
        Write-Error @"
[build] Cannot copy to $OutDll
        Close Python / any process using conv_kernels.dll, then rerun:
          .\build_native.ps1 $Mode
"@
    }
    if ($WithPdb) {
        if (-not (Test-Path $BuiltPdb)) {
            Write-Error "[build] Expected PDB missing: $BuiltPdb"
        }
        Copy-Item -Path $BuiltPdb -Destination $OutPdb -Force
    } else {
        Remove-Item $OutPdb -Force -ErrorAction SilentlyContinue
    }
}

Publish-BinArtifacts -WithPdb $EmitPdb

function Publish-OpenBlasToBin {
    $ArtifactDir = Join-Path $Root "build\openblas"
    if (-not (Test-Path $ArtifactDir)) {
        return
    }
    $built = @(
        Get-ChildItem -Path $ArtifactDir -Recurse -Filter "openblas.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
        Get-ChildItem -Path $ArtifactDir -Recurse -Filter "libopenblas.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
    ) | Where-Object { $_ } | Select-Object -First 1
    if (-not $built) {
        return
    }
    $outDll = Join-Path $BinDir "libopenblas.dll"
    try {
        Copy-Item -Path $built.FullName -Destination $outDll -Force
    } catch {
        Write-Warning "[build] OpenBLAS present in build/openblas but copy to $outDll failed (DLL locked?)"
        return
    }
    $artifactMarker = Join-Path $ArtifactDir "openblas_build.json"
    if (Test-Path $artifactMarker) {
        Copy-Item -Path $artifactMarker -Destination (Join-Path $BinDir "openblas_build.json") -Force
    }
    Write-Host "[build] Copied OpenBLAS -> $outDll" -ForegroundColor Green
}

Publish-OpenBlasToBin

Write-Host "[build] Wrote $BuiltDll" -ForegroundColor Green
Write-Host "[build] Copied -> $OutDll" -ForegroundColor Green
if (Test-Path $BuiltPdb) {
    Write-Host "[build] PDB: $BuiltPdb" -ForegroundColor Green
    if (Test-Path $OutPdb) {
        Write-Host "[build] Copied -> $OutPdb" -ForegroundColor Green
    }
}
if ($Selected -eq "Debug") {
    $dumpbin = Join-Path $VsPath "VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe"
    $dumpbinExe = (Resolve-Path $dumpbin -ErrorAction SilentlyContinue | Select-Object -First 1).Path
    if ($dumpbinExe) {
        $deps = & $dumpbinExe /DEPENDENTS $BuiltDll 2>$null | Select-String "140D|omp140d|ucrtbased"
        if ($deps) {
            Write-Host "[build] Debug CRT verified: $($deps.Line.Trim() -join ', ')" -ForegroundColor Green
        }
    }
}

if ($RunTests) {
    python (Join-Path $Root "run_tests.py")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
