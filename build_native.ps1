# Build bin/conv_kernels.dll with MSVC x64 Native Tools (user release flags).
param(
    [switch]$NoSymbols,
    [switch]$RunTests
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$BinDir = Join-Path $Root "bin"
$OutDll = Join-Path $BinDir "conv_kernels.dll"
$OutPdb = Join-Path $BinDir "conv_kernels.pdb"
$StageDll = Join-Path $BinDir "conv_kernels_stage.dll"
$StagePdb = Join-Path $BinDir "conv_kernels_stage.pdb"

New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

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

# /Zi required for static function names under /GL + /LTCG. PDB is sidecar only (no runtime cost).
# uProf gets names via PDB; source lines stay hidden unless -AnalysisType SourceDisasm.
$SymbolCl = if ($NoSymbols) { "" } else { "/Zi /Fd$StagePdb" }
$SymbolLink = if ($NoSymbols) { "" } else { "/DEBUG:FULL /PDB:$StagePdb" }

Write-Host "[build] Release$(if ($NoSymbols) { ' (no PDB)' } else { ' + link PDB' })" -ForegroundColor Cyan
Write-Host "[build] x64 Native Tools via vcvars64.bat" -ForegroundColor Cyan

Remove-Item $StageDll, $StagePdb -Force -ErrorAction SilentlyContinue

$BuildBat = Join-Path $env:TEMP "ml-engine-build-native.bat"
@"

call "$Vcvars" >nul
cd /d "$Root"
cl.exe /nologo /O2 /Oi /Ot /Ox /GL /Gy /Gw /fp:fast /arch:AVX2 /openmp:llvm /DNDEBUG /LD /I. /Isrc\native $SymbolCl src\native\conv_fallback.cpp src\native\conv_dispatcher.cpp /Fo:bin\ /Fe:bin\conv_kernels_stage.dll /link /LTCG /OPT:REF /OPT:ICF /NODEFAULTLIB:libcmtd.lib /NODEFAULTLIB:msvcrtd.lib /IMPLIB:bin\conv_kernels.lib $SymbolLink
exit /b %ERRORLEVEL%

"@ | Set-Content -Path $BuildBat -Encoding ASCII

cmd /c $BuildBat
if ($LASTEXITCODE -ne 0) {
    Write-Error "[build] cl.exe failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $StageDll)) {
    Write-Error "[build] Expected output missing: $StageDll"
}

function Install-BuildArtifact {
    param(
        [string]$FromDll,
        [string]$ToDll,
        [string]$FromPdb,
        [string]$ToPdb,
        [bool]$WithPdb
    )
    $bakDll = "$ToDll.bak"
    Remove-Item $bakDll -Force -ErrorAction SilentlyContinue
    if (Test-Path $ToDll) {
        try {
            Rename-Item -Path $ToDll -NewName (Split-Path $bakDll -Leaf) -Force
        } catch {
            Write-Error @"
[build] Cannot replace locked $ToDll
        Close Python / any process using conv_kernels.dll, then rerun:
          .\build_native.ps1
"@
        }
    }
    Move-Item -Path $FromDll -Destination $ToDll -Force
    if ($WithPdb) {
        if (-not (Test-Path $FromPdb)) {
            Write-Error "[build] Expected PDB missing: $FromPdb"
        }
        Move-Item -Path $FromPdb -Destination $ToPdb -Force
    }
    Remove-Item $bakDll -Force -ErrorAction SilentlyContinue
}

Install-BuildArtifact -FromDll $StageDll -ToDll $OutDll -FromPdb $StagePdb -ToPdb $OutPdb -WithPdb (-not $NoSymbols)

Write-Host "[build] Wrote $OutDll" -ForegroundColor Green
if (Test-Path $OutPdb) {
    Write-Host "[build] PDB: $OutPdb (paired with DLL, /Zi names only)" -ForegroundColor Green
}

if ($RunTests) {
    python (Join-Path $Root "run_tests.py")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
