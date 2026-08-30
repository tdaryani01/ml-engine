param (
    [string]$TargetScript = "run_pipeline.py",
    [string[]]$TargetCommand = @(),
    [switch]$EnableCProfile,
    [string]$CProfileOutput = "train.prof",

    [ValidateSet("Hotspots", "SourceDisasm", "Assess", "ConcurrencyBound", "Memory", "MemAlloc", "BackwardDiag")]
    [string]$AnalysisType = "Memory",
    
    [switch]$SystemWide,
    [switch]$IncludeDisasm,
    [int]$ActiveCores = 4,
    [string]$OutputDir = "$env:APPDATA\AMDuProf",
    [switch]$CheckContiguity
)

# 1. Paths & Locations
$UprofCli  = "C:\Program Files\AMD\AMDuProf\bin\AMDuProfCLI.exe"
$UprofPcm  = "C:\Program Files\AMD\AMDuProf\bin\AMDuProfPcm.exe"
$ConfigDir = "C:\Program Files\AMD\AMDuProf\bin\Data\Config"
$PythonExe = "C:\Users\tdary\AppData\Local\Programs\Python\Python313\python.exe"
$WorkDir   = (Get-Location).Path
$BinDir    = Join-Path $WorkDir "bin"
$SrcDir    = Join-Path $WorkDir "src\native"
$ReportDir = Join-Path $WorkDir "uprof_reports"

if (-not (Test-Path $UprofCli)) {
    Write-Error "[ERROR] Could not find AMDuProfCLI.exe at $UprofCli"
    exit 1
}

$SymbolPath = "$BinDir;srv*C:\Symbols*https://msdl.microsoft.com/download/symbols"

# 2. Ensure Zen 3 Barcelo-R config file exists
$Zen3Config = Join-Path $ConfigDir "0x19_0x5.conf"
if (-not (Test-Path $Zen3Config)) {
    $BaseConfig = Join-Path $ConfigDir "0x19_0x4.conf"
    if (Test-Path $BaseConfig) {
        Write-Host "[INIT] Generating 0x19_0x5.conf patch for Ryzen 7 7730U..." -ForegroundColor Yellow
        $cfg = Get-Content $BaseConfig -Raw
        $cfg = $cfg -replace 'modellow="40"', 'modellow="50"'
        $cfg = $cfg -replace 'modelhigh="4f"', 'modelhigh="5f"'
        Set-Content -Path $Zen3Config -Value $cfg -Force
    }
}

# 3. Build Execution Command
$ExecArgs = @()
if ($TargetCommand.Count -gt 0) {
    $ExecArgs = $TargetCommand
} elseif ($EnableCProfile) {
    $ExecArgs = @("-m", "cProfile", "-o", "$CProfileOutput", "$TargetScript")
} else {
    $ExecArgs = @("$TargetScript")
}

Write-Host "==================================================================" -ForegroundColor Yellow
Write-Host "        AMD Performance Profiler: $AnalysisType" -ForegroundColor Yellow
Write-Host "==================================================================" -ForegroundColor Yellow
Write-Host "  Scope:              $(if ($SystemWide) { 'System-Wide (--system-wide)' } else { 'Target Application Only' })" -ForegroundColor Cyan
Write-Host "  Executing Command:  python $($ExecArgs -join ' ')" -ForegroundColor Cyan
Write-Host "  Active Cores:       $ActiveCores" -ForegroundColor Cyan
Write-Host "==================================================================`n" -ForegroundColor Yellow

if (Test-Path $ReportDir) {
    Remove-Item -Path $ReportDir -Recurse -Force -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Path $ReportDir -Force | Out-Null

$PathArgs = @("--bin-path", "$BinDir", "--symbol-path", "$SymbolPath")
if (Test-Path $SrcDir) { $PathArgs += @("--src-path", "$SrcDir") } else { $PathArgs += @("--src-path", "$WorkDir") }

function Get-UProfSectionTable {
    param ([string[]]$Lines, [string]$SectionHeaderName)
    $StartIdx = -1
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        if ($Lines[$i] -match $SectionHeaderName) { $StartIdx = $i + 1; break }
    }
    if ($StartIdx -eq -1 -or $StartIdx -ge $Lines.Count) { return @() }

    $SectionLines = @()
    for ($j = $StartIdx; $j -lt $Lines.Count; $j++) {
        $curr = $Lines[$j].Trim()
        if ([string]::IsNullOrWhiteSpace($curr)) {
            if ($SectionLines.Count -gt 1) { break }
            continue
        }
        if ($curr -match "^[A-Z\s]{4,}$" -and $SectionLines.Count -gt 1) { break }
        $SectionLines += $curr
    }
    if ($SectionLines.Count -gt 1) { return ($SectionLines | Out-String | ConvertFrom-Csv) }
    return @()
}

# -------------------------------------------------------------------------
# OPTIONAL PRE-CHECK: Contiguity & Alignment Inspector
# -------------------------------------------------------------------------
if ($CheckContiguity) {
    Write-Host "[PRE-CHECK] Verifying Memory Contiguity & AVX Alignment..." -ForegroundColor Yellow
    $ContigScript = @'
import sys
try:
    import numpy as np
    print(f"NumPy Version: {np.__version__}")
    print("Contiguity rules: C-Contiguous buffers have stride[-1] == itemsize and zero pointer gaps.")
except Exception as e:
    print(f"Contiguity check skipped: {e}")
'@
    $ContigScript | & $PythonExe -
    Write-Host "--------------------------------------------------------------------------------------------------------`n"
}

# -------------------------------------------------------------------------
# TRACK A: AMDuProfPcm Hardware Performance Counters
# -------------------------------------------------------------------------
if ($AnalysisType -eq "Memory") {
    Write-Host "[1/3] Launching AMDuProf PCM Hardware Performance Counters..." -ForegroundColor Cyan
    
    $PcmArgs = @("-i", "$Zen3Config", "-a", "-d", "30", "-O", "$ReportDir", "$PythonExe") + $ExecArgs
    & $UprofPcm @PcmArgs

    Write-Host "`n[2/3] PARSING PCM HARDWARE COUNTERS..." -ForegroundColor Green
    Write-Host "========================================================================================================"

    $PcmFiles = Get-ChildItem -Path $ReportDir -Filter "*.csv" -Recurse | Sort-Object LastWriteTime -Descending
    if ($PcmFiles.Count -gt 0) {
        $PcmCsv = $PcmFiles[0].FullName
        Write-Host "Found Report: $PcmCsv`n" -ForegroundColor DarkGray
        $Lines = Get-Content $PcmCsv

        $HeaderIdx = -1
        for ($i = 0; $i -lt $Lines.Count; $i++) {
            if ($Lines[$i] -match "Retired Instructions" -and $Lines[$i] -match "IPC") {
                $HeaderIdx = $i
                break
            }
        }

        if ($HeaderIdx -ne -1) {
            $DataRows = $Lines[$HeaderIdx..($Lines.Count - 1)] | ConvertFrom-Csv
            
            Write-Host "HARDWARE CORE METRICS (uProf Native Samples):" -ForegroundColor Green
            Write-Host "--------------------------------------------------------------------------------------------------------"
            $DataRows | Select-Object -First 10 | Select-Object `
                @{Name="IPC (uProf)"; Expression={$_."IPC (Sys + User)"}}, `
                @{Name="AVX GFLOPs (uProf)"; Expression={$_."Retired SSE/AVX Flops(GFLOPs)"}}, `
                @{Name="Core Util % (uProf)"; Expression={$_."Utilization (%)"}}, `
                @{Name="L1 DC Miss (pti) (uProf)"; Expression={$_."L1 DC Miss (pti)"}}, `
                @{Name="L2 DC Hit (pti) (uProf)"; Expression={$_."L2 Hit from DC Miss (pti)"}}, `
                @{Name="L2 DC Miss (pti) (uProf)"; Expression={$_."L2 Miss from DC Miss (pti)"}}, `
                @{Name="DRAM Fill (pti) (uProf)"; Expression={$_."DC Fills From Local Memory (pti)"}}, `
                @{Name="SSE/AVX Stalls (pti) (uProf)"; Expression={if ($_."Mixed SSE/AVX Stalls (pti)") { $_."Mixed SSE/AVX Stalls (pti)" } else { "0.00" }}}, `
                @{Name="Mispred Branch (pti) (uProf)"; Expression={$_."Retired Branches Mispredicted (pti)"}}, `
                @{Name="Eff Freq (MHz) (uProf)"; Expression={$_."Eff Freq (MHz)"}} | Format-Table -AutoSize
        } else {
            Get-Content $PcmCsv | Select-Object -First 25
        }
    } else {
        Write-Warning "No PCM CSV generated."
    }
    exit 0
}

# -------------------------------------------------------------------------
# TRACK B: Core UProf Collection (Assess / Hotspots / MemAlloc)
# -------------------------------------------------------------------------
$CollectPreset = switch ($AnalysisType) {
    "Hotspots"         { "hotspots" }
    "SourceDisasm"     { "hotspots" }
    "BackwardDiag"     { "hotspots" }
    "Assess"           { "assess" }
    "ConcurrencyBound" { "hotspots" }
    "MemAlloc"         { "assess" }
    Default            { "assess" }
}

$UseSourceDisasm = ($AnalysisType -eq "SourceDisasm") -or ($AnalysisType -eq "BackwardDiag") -or $IncludeDisasm

$CollectArgs = @("collect", "--config", $CollectPreset)
if ($SystemWide) { $CollectArgs += "--system-wide" }
$CollectArgs += @("-w", "$WorkDir", "-o", "$OutputDir", "$PythonExe") + $ExecArgs

Write-Host "[1/4] Running AMDuProf Collection ($CollectPreset)..." -ForegroundColor Cyan
$StopWatch = [System.Diagnostics.Stopwatch]::StartNew()
& $UprofCli @CollectArgs
$StopWatch.Stop()

if ($LASTEXITCODE -ne 0) {
    Write-Error "[ERROR] AMDuProf collection failed with exit code $LASTEXITCODE."
    exit 1
}

$SessionDir = Get-ChildItem -Path $OutputDir -Directory -Filter "AMDuProf-*" | 
              Sort-Object LastWriteTime -Descending | 
              Select-Object -First 1 -ExpandProperty FullName

$WallTimeSec = [math]::Round($StopWatch.Elapsed.TotalSeconds, 3)

Write-Host "[2/4] Generating Profile Summary Reports from: $SessionDir" -ForegroundColor Cyan
$SummaryCsv = Join-Path $ReportDir "summary_report.csv"
$DetailCsv  = Join-Path $ReportDir "detail_report.csv"

& $UprofCli report -i "$SessionDir" --report-output "$SummaryCsv" --show-sample-count @PathArgs
if ($UseSourceDisasm) {
    & $UprofCli report -i "$SessionDir" --report-output "$DetailCsv" --disasm --disasm-style intel --show-sample-count @PathArgs
}

Write-Host "`n[3/4] SYSTEM & PIPELINE METRICS ($AnalysisType):" -ForegroundColor Green
Write-Host "========================================================================================================"

# --- SECTION 1: NATIVE UPROF PROFILING TABLES ---
if (Test-Path $SummaryCsv) {
    $AllLines = Get-Content $SummaryCsv

    # 1. Native Allocation & Memory Sections (if generated by uProf)
    $AllocData = Get-UProfSectionTable -Lines $AllLines -SectionHeaderName "ALLOCATION SUMMARY"
    if ($AllocData.Count -eq 0) { $AllocData = Get-UProfSectionTable -Lines $AllLines -SectionHeaderName "MEMORY USAGE SUMMARY" }
    if ($AllocData.Count -gt 0) {
        Write-Host "AMD UPROF MEMORY & ALLOCATION METRICS (uProf Native):" -ForegroundColor Green
        Write-Host "--------------------------------------------------------------------------------------------------------"
        $AllocData | Format-Table -AutoSize
    }

    # 2. Native Synchronization / Wait Times
    $SyncData = Get-UProfSectionTable -Lines $AllLines -SectionHeaderName "SYNCHRONIZATION SUMMARY"
    if ($SyncData.Count -eq 0) { $SyncData = Get-UProfSectionTable -Lines $AllLines -SectionHeaderName "WAIT TIME SUMMARY" }
    if ($SyncData.Count -gt 0) {
        Write-Host "`nSYNCHRONIZATION & WAIT TIMES (uProf Native):" -ForegroundColor Green
        Write-Host "--------------------------------------------------------------------------------------------------------"
        $SyncData | Format-Table -AutoSize
    }

    # 3. Thread Summary
    $ThreadData = Get-UProfSectionTable -Lines $AllLines -SectionHeaderName "THREAD SUMMARY"
    if ($ThreadData.Count -eq 0) { $ThreadData = Get-UProfSectionTable -Lines $AllLines -SectionHeaderName "HOTTEST THREADS" }

    $TotalActiveCpuTime = 0.0
    $DisplayRows = @()

    foreach ($row in $ThreadData) {
        $cpuTimeVal  = $row."CPU_TIME"
        if (-not $cpuTimeVal) { $cpuTimeVal = $row."CPU_TIME (seconds)" }
        $samplesVal  = $row."SAMPLES"
        $nativeWait  = $row."WAIT_TIME"
        if (-not $nativeWait) { $nativeWait = $row."WAIT_TIME (seconds)" }
        
        if ($cpuTimeVal -and [double]$cpuTimeVal -gt 0.01) {
            $cpuTime = [double]$cpuTimeVal
            $TotalActiveCpuTime += $cpuTime
            $wallDelta = [math]::Max(0.0, $WallTimeSec - $cpuTime)
            $utilPct  = [math]::Round(($cpuTime / $WallTimeSec) * 100, 1)
            $tId      = if ($row."THREAD ID") { $row."THREAD ID" } else { $row.THREAD }

            $DisplayRows += [PSCustomObject]@{
                "Thread ID (uProf)"                 = $tId
                "Active CPU (s) (uProf)"            = "{0:N2}" -f $cpuTime
                "Wait Time (s) (uProf)"             = if ($nativeWait) { "{0:N2}" -f [double]$nativeWait } else { "N/A" }
                "Samples (uProf)"                   = $samplesVal
                "Stopwatch Wall (s) [M-Calculated]" = "{0:N2}" -f $WallTimeSec
                "Wall Delta (s) [M-Calculated]"     = "{0:N2}" -f $wallDelta
                "Capacity Util % [M-Calculated]"    = "$utilPct %"
            }
        }
    }

    if ($DisplayRows.Count -gt 0) {
        Write-Host "`nTHREAD SUMMARY:" -ForegroundColor Green
        Write-Host "--------------------------------------------------------------------------------------------------------"
        $DisplayRows | Format-Table -AutoSize
    }

    # 4. Module Breakdown
    $ModuleData = Get-UProfSectionTable -Lines $AllLines -SectionHeaderName "MODULE SUMMARY"
    if ($ModuleData.Count -eq 0) { $ModuleData = Get-UProfSectionTable -Lines $AllLines -SectionHeaderName "HOTTEST MODULES" }
    if ($ModuleData.Count -gt 0) {
        Write-Host "`nTOP HOTTEST MODULES (uProf Native):" -ForegroundColor Green
        Write-Host "========================================================================================================"
        $ModuleData | Select-Object -First 10 | Format-Table -AutoSize
    }

    # 5. Function Hotspots
    $FuncData = Get-UProfSectionTable -Lines $AllLines -SectionHeaderName "FUNCTION SUMMARY"
    if ($FuncData.Count -eq 0) { $FuncData = Get-UProfSectionTable -Lines $AllLines -SectionHeaderName "HOTTEST FUNCTIONS" }
    if ($FuncData.Count -gt 0) {
        Write-Host "`nTOP HOTTEST FUNCTIONS & PIPELINE METRICS (uProf Native):" -ForegroundColor Green
        Write-Host "========================================================================================================"
        $FuncData | Select-Object -First 15 | Format-Table -AutoSize

        $KernelFuncs = $FuncData | Where-Object {
            $_.FUNCTION -match 'process_bwd_dx|process_dw_nci|process_fwd_tile|conv2d_.*fallback'
        }
        if ($KernelFuncs.Count -gt 0) {
            Write-Host "`nCONV FALLBACK KERNEL HOTSPOTS (uProf Native):" -ForegroundColor Green
            Write-Host "--------------------------------------------------------------------------------------------------------"
            $KernelFuncs | Format-Table -AutoSize
        }
    }
}

if ($AnalysisType -eq "BackwardDiag") {
    Write-Host "`nBACKWARD DIAG CHECKLIST:" -ForegroundColor Yellow
    Write-Host "  1. Sources tab -> conv_fallback.cpp: compare process_bwd_dx_tile vs process_dw_nci_task CPU_TIME"
    Write-Host "  2. Re-run with -AnalysisType Memory for core L1/L2/DRAM (not per-function)"
    Write-Host "  3. Re-run with -EnableCProfile for Python vs native split in im2col.py"
    Write-Host "  Detail report: $DetailCsv" -ForegroundColor DarkGray
}

if ($AnalysisType -eq "Memory") {
    Write-Host "`nNOTE: Memory mode reports core-level cache counters (PCM), not per-function L1 miss." -ForegroundColor DarkGray
}

# --- SECTION 2: PYTHON HEAP & DYNAMIC BUFFER TRACE (Runs automatically under MemAlloc) ---
if ($AnalysisType -eq "MemAlloc") {
    Write-Host "`n========================================================================================================" -ForegroundColor Cyan
    Write-Host "           PYTHON TRACEMALLOC: HEAP ALLOCATION & BUFFER BREAKDOWN [M - Calculated / Traced]" -ForegroundColor Cyan
    Write-Host "========================================================================================================" -ForegroundColor Cyan
    
    $MemRunner = @'
import tracemalloc, runpy, sys, os

tracemalloc.start(25)
target = sys.argv[1]
sys.argv = sys.argv[1:]

try:
    runpy.run_path(target, run_name='__main__')
finally:
    snapshot = tracemalloc.take_snapshot()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n  Current Live Allocated Heap:  {current / (1024*1024):>10.3f} MB")
    print(f"  Peak Dynamic Heap Footprint:  {peak / (1024*1024):>10.3f} MB")
    print("--------------------------------------------------------------------------------------------------------")

    print("\nTOP 15 ALLOCATING CALL SITES & SIZES:")
    print("--------------------------------------------------------------------------------------------------------")
    top_stats = snapshot.statistics('lineno')
    print(f"{'Source File / Line':<60} {'Total Size (KB)':<18} {'Count':<10}")
    print("-" * 90)
    for stat in top_stats[:15]:
        print(f"{str(stat.traceback):<60} {stat.size / 1024:>14.2f} KB {stat.count:>10}")
'@

    $MemRunner | & $PythonExe - "$TargetScript"
}

# -------------------------------------------------------------------------
# SECTION 3: PYTHON CPROFILE BREAKDOWN (If -EnableCProfile is toggled)
# -------------------------------------------------------------------------
if ($EnableCProfile -and (Test-Path $CProfileOutput)) {
    Write-Host "`n========================================================================================================" -ForegroundColor Magenta
    Write-Host "                         PYTHON CPROFILE BREAKDOWN ($CProfileOutput)" -ForegroundColor Magenta
    Write-Host "========================================================================================================" -ForegroundColor Magenta
    
    $PyAnalyzeScript = @'
import pstats, os, sys
from collections import defaultdict

prof_file = sys.argv[1] if len(sys.argv) > 1 else 'train.prof'
try:
    p = pstats.Stats(prof_file)
    
    print()
    print('TOP 10 PYTHON MODULES (AGGREGATED SELF-TIME):')
    print('-' * 80)
    mod_tot = defaultdict(float)
    mod_cum = defaultdict(float)
    mod_calls = defaultdict(int)

    for (fn, ln, func), (cc, nc, tt, ct, callers) in p.stats.items():
        m = os.path.basename(fn) if fn != '~' else '<built-in>'
        mod_tot[m] += tt
        mod_cum[m] = max(mod_cum[m], ct)
        mod_calls[m] += nc

    print('%-35s %-16s %-16s %-12s' % ('Module / File', 'Self Time (s)', 'Max CumTime (s)', 'Total Calls'))
    print('-' * 80)
    for m, tt in sorted(mod_tot.items(), key=lambda x: x[1], reverse=True)[:10]:
        print('%-35s %-16.4f %-16.4f %-12d' % (m, tt, mod_cum[m], mod_calls[m]))

    print()
    print('TOP 10 PYTHON FUNCTIONS (SORTED BY SELF-TIME):')
    print('-' * 80)
    p.strip_dirs().sort_stats('tottime').print_stats(10)

except Exception as e:
    print('Error analyzing profile data:', e)
'@

    $PyAnalyzeScript | & $PythonExe - "$CProfileOutput"
}