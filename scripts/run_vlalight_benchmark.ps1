# Resume the full local matrix. The child checks API access before starting SUMO.
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $taskRoot
$stamp = Get-Date -Format 'yyyyMMddTHHmmss'
$jobDir = Join-Path $taskRoot "logs\benchmark_jobs\$stamp"
New-Item -ItemType Directory -Path $jobDir -Force | Out-Null
$statusPath = Join-Path $taskRoot 'logs\jev_full_job_status.json'
$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'benchmark_vlalight_jev\.py.*--method\s+jev-image'
}
if ($existing) { throw 'A Jev benchmark process is already running. Inspect its status before starting another.' }
# Windows releases this exclusive file handle after shutdown or process exit.
$jobLock = [IO.File]::Open((Join-Path $taskRoot 'logs\jev_full_job.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
$bench = $null
$taskStarted = (Get-Date).ToString('o')

function Save-TaskStatus($state, $message) {
    $counts = @{}
    foreach ($method in @('maxpressure','jev-image')) {
        try {
            $result = Get-Content -LiteralPath "results/vlalight_jev/$method.json" -Raw | ConvertFrom-Json
            $counts[$method] = @($result.runs | Where-Object { $_.status -eq 'complete' }).Count
        } catch { $counts[$method] = 0 }
    }
    $record = @{state=$state;message=$message;pid=$PID;started=$taskStarted;
        heartbeat=(Get-Date).ToString('o');completed_seeds=$counts;log_directory=$jobDir}
    if ($bench) { $record.benchmark_pid = $bench.Id }
    $temporary = "$statusPath.tmp"
    $record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

try {
    if (-not $env:TYPESAFE_API_KEY) {
        $env:TYPESAFE_API_KEY = [Environment]::GetEnvironmentVariable('TYPESAFE_API_KEY','User')
    }
    if (-not $env:TYPESAFE_API_KEY) { throw 'TYPESAFE_API_KEY is not configured.' }
    $pythonPath = (Get-Command python).Source
    Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;
public static class JevBenchmarkPower {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@
    [JevBenchmarkPower]::SetThreadExecutionState([uint32]2147483649) | Out-Null
    $bench = Start-Process -FilePath $pythonPath -ArgumentList @(
        'scripts/benchmark_vlalight_jev.py','--method','jev-image','--execute','--resume'
    ) -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput (Join-Path $jobDir 'benchmark.log') `
      -RedirectStandardError (Join-Path $jobDir 'benchmark.stderr.log')
    $unusedHandle = $bench.Handle
    Save-TaskStatus 'checking_api' 'Checking image API access before resuming unfinished seeds.'
    $tick = 0
    while (-not $bench.HasExited) {
        if (($tick % 2) -eq 0) {
            $report = Start-Process -FilePath $pythonPath -ArgumentList @('scripts/report_vlalight_jev.py') `
                -WorkingDirectory $taskRoot -WindowStyle Hidden -Wait -PassThru `
                -RedirectStandardOutput (Join-Path $jobDir 'report.log') `
                -RedirectStandardError (Join-Path $jobDir 'report.stderr.log')
        }
        $api = Get-Content -LiteralPath 'results/vlalight_jev/api_preflight.json' -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
        if ($api -and $api.status -eq 'ready' -and [DateTimeOffset]$api.checked_at -ge [DateTimeOffset]$taskStarted) {
            Save-TaskStatus 'running' 'Resuming incomplete seeds; completed seeds are reused.'
        } else {
            Save-TaskStatus 'checking_api' 'Waiting for the image API preflight; no score is inferred from this state.'
        }
        Start-Sleep -Seconds 30
        $bench.Refresh()
        $tick += 1
    }
    $bench.WaitForExit()
    & $pythonPath scripts/report_vlalight_jev.py
    if ($LASTEXITCODE -ne 0) { throw 'Final report generation failed.' }
    $api = Get-Content -LiteralPath 'results/vlalight_jev/api_preflight.json' -Raw | ConvertFrom-Json
    if ($bench.ExitCode -ne 0 -and $api.status -eq 'blocked_api') {
        Save-TaskStatus 'blocked_api' $api.error
    } else {
        $result = Get-Content -LiteralPath 'results/vlalight_jev/jev-image.json' -Raw | ConvertFrom-Json
        $baseline = Get-Content -LiteralPath 'results/vlalight_jev/maxpressure.json' -Raw | ConvertFrom-Json
        $expected = $result.plan.cases.Count * $result.plan.seeds.Count
        $allComplete = $bench.ExitCode -eq 0 -and $result.runs.Count -eq $expected -and $baseline.runs.Count -eq $expected
        foreach ($run in @($result.runs) + @($baseline.runs)) {
            if ($run.status -ne 'complete') { $allComplete = $false }
        }
        $branch = (& git branch --show-current).Trim()
        if ($allComplete -and $branch -eq 'local/jev-vlalight-benchmark') {
            $paths = @('results/vlalight_jev/README.md','results/vlalight_jev/jev-image.json',
                       'results/vlalight_jev/plan_jev-image.json','results/vlalight_jev/maxpressure.json',
                       'results/vlalight_jev/api_preflight.json')
            & git add -- @paths
            if ($LASTEXITCODE -ne 0) { throw 'Could not stage local results.' }
            & git commit --only -m 'Record full VLALight Jev benchmark results' -- @paths
            if ($LASTEXITCODE -ne 0) { throw 'Could not commit local results.' }
            Save-TaskStatus 'complete' 'Full matrix completed and committed locally; nothing pushed.'
        } elseif ($allComplete) {
            Save-TaskStatus 'complete_uncommitted' 'Results completed; branch changed, so the local commit was skipped.'
        } else {
            Save-TaskStatus 'incomplete' 'Some seeds failed or were interrupted. Inspect logs before resuming.'
        }
    }
} catch {
    Save-TaskStatus 'failed' $_.Exception.Message
    throw
} finally {
    if ('JevBenchmarkPower' -as [type]) {
        [JevBenchmarkPower]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    }
    $jobLock.Dispose()
}
