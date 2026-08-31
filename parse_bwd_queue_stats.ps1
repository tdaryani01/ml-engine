param(
    [Parameter(ValueFromPipeline = $true, Position = 0)]
    [string]$InputPath,

    [switch]$FromClipboard
)

$ErrorActionPreference = "Stop"

function Get-LogText {
    param([string]$Path, [switch]$Clip)
    if ($Clip) {
        return Get-Clipboard -Raw
    }
    if ($Path) {
        return Get-Content -Path $Path -Raw
    }
    if ($InputPath) {
        return Get-Content -Path $InputPath -Raw
    }
    $pipe = @($input) -join "`n"
    if ($pipe.Trim().Length -gt 0) {
        return $pipe
    }
    throw "Usage: .\parse_bwd_queue_stats.ps1 log.txt | .\parse_bwd_queue_stats.ps1 -FromClipboard"
}

function New-LayerKey {
    param($Plan)
    return "N=$($Plan.N) Cin=$($Plan.Cin) Cout=$($Plan.Cout) H=$($Plan.H) W=$($Plan.W)"
}

$text = Get-LogText -Path $InputPath -Clip:$FromClipboard
$lines = $text -split "`r?`n"

$layers = [ordered]@{}
$currentKey = $null
$current = $null

foreach ($line in $lines) {
    if ($line -match '^\[BWD_QUEUE_PLAN\] N=(\d+) Cin=(\d+) Cout=(\d+) H=(\d+) W=(\d+) \| queue_len=(\d+) dx=(\d+) dw=(\d+) chunk=(\d+) threads=(\d+)') {
        $key = "N=$($Matches[1]) Cin=$($Matches[2]) Cout=$($Matches[3]) H=$($Matches[4]) W=$($Matches[5])"
        if (-not $layers.Contains($key)) {
            $layers[$key] = [pscustomobject]@{
                Key         = $key
                N           = [int64]$Matches[1]
                Cin         = [int64]$Matches[2]
                Cout        = [int64]$Matches[3]
                H           = [int64]$Matches[4]
                W           = [int64]$Matches[5]
                QueueLen    = [int64]$Matches[6]
                DxTotal     = [int64]$Matches[7]
                DwTotal     = [int64]$Matches[8]
                Chunk       = [int64]$Matches[9]
                Threads     = [int64]$Matches[10]
                Pattern     = $null
                MaxDwChunk  = $null
                AllDwChunks = $null
                NumChunks   = $null
                Samples     = [System.Collections.Generic.List[object]]::new()
                Timing      = $null
            }
        }
        $currentKey = $key
        $current = $layers[$key]
        $current.Pattern = $null
        continue
    }

    if (-not $current) { continue }

    if ($line -match '^\[BWD_QUEUE_PLAN\] wid\[0\.\.\d+\): (.+)$') {
        $current.Pattern = $Matches[1]
        continue
    }

    if ($line -match 'omp_chunk_dw_hist.*max_dw=(\d+) all_dw_chunks=(\d+)/(\d+)') {
        $current.MaxDwChunk = [int64]$Matches[1]
        $current.AllDwChunks = [int64]$Matches[2]
        $current.NumChunks = [int64]$Matches[3]
        continue
    }

    if ($line -match '^\s+t(\d+): dx=(\d+) dw=(\d+) total=(\d+)') {
        $sample = [pscustomobject]@{
            Thread = [int]$Matches[1]
            Dx     = [int64]$Matches[2]
            Dw     = [int64]$Matches[3]
            Total  = [int64]$Matches[4]
        }
        if ($current.Samples.Count -eq 0 -or $current.Samples[-1].Count -ge $current.Threads) {
            $current.Samples.Add([System.Collections.Generic.List[object]]::new())
        }
        $current.Samples[-1].Add($sample) | Out-Null
        continue
    }

    if ($line -match '^\[BWD_QUEUE_TIMING\] tsc_ghz=([\d.]+)') {
        if (-not $current.Timing) {
            $current.Timing = [ordered]@{ TscGhz = [double]$Matches[1] }
        } else {
            $current.Timing.TscGhz = [double]$Matches[1]
        }
        continue
    }

    if ($line -match '^\[BWD_QUEUE_TIMING\] dx: n=(\d+) total_cycles=(\d+) avg_cycles=(\d+) avg_ns=([\d.]+) min_cycles=(\d+) max_cycles=(\d+)') {
        if (-not $current.Timing) { $current.Timing = [ordered]@{} }
        $current.Timing.Dx = [pscustomobject]@{
            N           = [int64]$Matches[1]
            TotalCycles = [uint64]$Matches[2]
            AvgCycles   = [uint64]$Matches[3]
            AvgNs       = [double]$Matches[4]
            MinCycles   = [uint64]$Matches[5]
            MaxCycles   = [uint64]$Matches[6]
        }
        continue
    }

    if ($line -match '^\[BWD_QUEUE_TIMING\] dw: n=(\d+) total_cycles=(\d+) avg_cycles=(\d+) avg_ns=([\d.]+) min_cycles=(\d+) max_cycles=(\d+)') {
        if (-not $current.Timing) { $current.Timing = [ordered]@{} }
        $current.Timing.Dw = [pscustomobject]@{
            N           = [int64]$Matches[1]
            TotalCycles = [uint64]$Matches[2]
            AvgCycles   = [uint64]$Matches[3]
            AvgNs       = [double]$Matches[4]
            MinCycles   = [uint64]$Matches[5]
            MaxCycles   = [uint64]$Matches[6]
        }
        continue
    }

    if ($line -match '^\[BWD_QUEUE_TIMING\] dw/dx avg_cycles_per_slot=([\d.]+)x') {
        if (-not $current.Timing) { $current.Timing = [ordered]@{} }
        $current.Timing.Ratio = [double]$Matches[1]
    }
}

if ($layers.Count -eq 0) {
    Write-Host "No [BWD_QUEUE_*] lines found." -ForegroundColor Yellow
    exit 1
}

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  BWD queue stats report" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

foreach ($layer in $layers.Values) {
    $dxPct = if ($layer.QueueLen -gt 0) { 100.0 * $layer.DxTotal / $layer.QueueLen } else { 0 }
    $dwPct = if ($layer.QueueLen -gt 0) { 100.0 * $layer.DwTotal / $layer.QueueLen } else { 0 }
    $sampleCount = $layer.Samples.Count

    Write-Host "$($layer.Key)  (samples=$sampleCount  chunk=$($layer.Chunk)  threads=$($layer.Threads))" -ForegroundColor Yellow
    Write-Host "  queue: len=$($layer.QueueLen)  dx=$($layer.DxTotal) ($([math]::Round($dxPct,1))%)  dw=$($layer.DwTotal) ($([math]::Round($dwPct,1))%)"

    if ($layer.Pattern) {
        $dCount = ([regex]::Matches($layer.Pattern, 'd')).Count
        $wCount = ([regex]::Matches($layer.Pattern, 'w')).Count
        $patLen = $layer.Pattern.Length
        Write-Host "  interleave[0..$($patLen-1)]: $($layer.Pattern)"
        Write-Host "  pattern mix: d=$dCount w=$wCount ($([math]::Round(100.0*$dCount/$patLen,1))% d / $([math]::Round(100.0*$wCount/$patLen,1))% w in window)"
    }

    if ($null -ne $layer.MaxDwChunk) {
        $allDwPct = if ($layer.NumChunks -gt 0) { 100.0 * $layer.AllDwChunks / $layer.NumChunks } else { 0 }
        Write-Host "  fixed-chunk plan (contiguous wid blocks of $($layer.Chunk)): max_dw=$($layer.MaxDwChunk)  all-dw-chunks=$($layer.AllDwChunks)/$($layer.NumChunks) ($([math]::Round($allDwPct,2))%)"
    }

    if ($sampleCount -eq 0) {
        Write-Host "  runtime: (no per-thread samples)" -ForegroundColor DarkGray
        Write-Host ""
        continue
    }

    $dxSpreads = @()
    $dwSpreads = @()
    $totalSpreads = @()
    $dwShareSpreads = @()
    $expectDwShare = if ($layer.DxTotal + $layer.DwTotal -gt 0) {
        100.0 * $layer.DwTotal / ($layer.DxTotal + $layer.DwTotal)
    } else { 0 }

    foreach ($sample in $layer.Samples) {
        $dxVals = $sample | ForEach-Object { $_.Dx }
        $dwVals = $sample | ForEach-Object { $_.Dw }
        $totVals = $sample | ForEach-Object { $_.Total }
        $dwShare = $sample | ForEach-Object {
            if ($_.Total -gt 0) { 100.0 * $_.Dw / $_.Total } else { 0 }
        }
        $dxSpreads += ($dxVals | Measure-Object -Maximum).Maximum - ($dxVals | Measure-Object -Minimum).Minimum
        $dwSpreads += ($dwVals | Measure-Object -Maximum).Maximum - ($dwVals | Measure-Object -Minimum).Minimum
        $totalSpreads += ($totVals | Measure-Object -Maximum).Maximum - ($totVals | Measure-Object -Minimum).Minimum
        $dwShareSpreads += ($dwShare | Measure-Object -Maximum).Maximum - ($dwShare | Measure-Object -Minimum).Minimum
    }

    $avg = {
        param($arr)
        if ($arr.Count -eq 0) { return 0 }
        ($arr | Measure-Object -Average).Average
    }

    Write-Host ("  runtime imbalance (avg over {0} backward passes):" -f $sampleCount)
    Write-Host ("    dx count spread (max-min per pass):     {0:N0} avg  ({1:N0} per thread if even)" -f (& $avg $dxSpreads), ($layer.DxTotal / $layer.Threads))
    Write-Host ("    dw count spread (max-min per pass):     {0:N0} avg  ({1:N0} per thread if even)" -f (& $avg $dwSpreads), ($layer.DwTotal / $layer.Threads))
    Write-Host ("    slot count spread (max-min per pass):   {0:N0} avg" -f (& $avg $totalSpreads))
    Write-Host ("    dw share spread across threads:         {0:N1} pp avg  (expect ~{1:N1}% dw slots)" -f (& $avg $dwShareSpreads), $expectDwShare)

    $last = $layer.Samples[-1]
    Write-Host "  last pass per-thread:"
    foreach ($t in ($last | Sort-Object Thread)) {
        $share = if ($t.Total -gt 0) { 100.0 * $t.Dw / $t.Total } else { 0 }
        Write-Host ("    t{0}: dx={1,5} dw={2,4} total={3,5}  ({4,4:N1}% dw slots)" -f $t.Thread, $t.Dx, $t.Dw, $t.Total, $share)
    }
    Write-Host ""
    if ($layer.Timing -and $layer.Timing.Dx -and $layer.Timing.Dw) {
        $ghz = $layer.Timing.TscGhz
        $dx = $layer.Timing.Dx
        $dw = $layer.Timing.Dw
        $ratio = if ($layer.Timing.Ratio) { $layer.Timing.Ratio } else {
            if ($dx.AvgCycles -gt 0) { $dw.AvgCycles / $dx.AvgCycles } else { 0 }
        }
        Write-Host "  per-slot timing (rdtsc around kernel, last sample):" -ForegroundColor Green
        Write-Host ("    dX tile: avg {0:N0} cycles ({1:N1} ns)  min={2:N0} max={3:N0}" -f $dx.AvgCycles, $dx.AvgNs, $dx.MinCycles, $dx.MaxCycles)
        Write-Host ("    dW task: avg {0:N0} cycles ({1:N1} ns)  min={2:N0} max={3:N0}" -f $dw.AvgCycles, $dw.AvgNs, $dw.MinCycles, $dw.MaxCycles)
        Write-Host ("    dW/dx:   {0:N2}x cycles per slot  (tsc_ghz={1:N2})" -f $ratio, $ghz)
    }
    Write-Host ""
}

Write-Host "Notes:" -ForegroundColor DarkGray
Write-Host "  - PLAN = Bresenham order + worst-case contiguous OpenMP chunks of size chunk." -ForegroundColor DarkGray
Write-Host "  - RUNTIME = actual dynamic steal; slot counts != CPU time (dW slots are heavier)." -ForegroundColor DarkGray
Write-Host "  - TIMING = __rdtsc() wrap of process_bwd_dx_tile / process_dw_nci_task only." -ForegroundColor DarkGray
Write-Host "  - Parse file:  .\parse_bwd_queue_stats.ps1 .\terminals\9.txt" -ForegroundColor DarkGray
Write-Host "  - Or pipe:     `$env:BWD_QUEUE_STATS='1'; python run_pipeline.py 2>&1 | .\parse_bwd_queue_stats.ps1" -ForegroundColor DarkGray
