# run_blocksworld_hard.ps1
$ErrorActionPreference = "Continue"

$heuristicNames = @(
    "blocksworld9_10.py",
    "blocksworld16_10.py",
    "blocksworld24_10.py",
    "blocksworld8_10.py"
)

$heuristics = @()
foreach ($name in $heuristicNames) {
    $path = "archive-heuristics\2025-03-10-deepseek-r1\$name"
    if (Test-Path $path) {
        $heuristics += (Resolve-Path $path).Path
    } else {
        Write-Host "WARNING: not found $path"
    }
}

Write-Host "Heuristics: $($heuristics.Count)"


$problems = @()
$problems += Get-ChildItem benchmarks\ipc2023-learning\testing\blocksworld\hard-p0[1-5].pddl |
    ForEach-Object { $_.FullName }

Write-Host "Problems: $($problems.Count)"

$timeoutSec = 300
$results = @()

foreach ($h in $heuristics) {
    $hName = Split-Path $h -Leaf
    Write-Host ""
    Write-Host "=== $hName ==="

    foreach ($p in $problems) {
        $pName = Split-Path $p -Leaf
        Write-Host -NoNewline "  $pName ... "

        $job = Start-Job -ScriptBlock {
            param($hPath, $pPath, $wd)
            Set-Location $wd
            uv run src/pyperplan/pyperplan.py -s gbfs_early_goal_test -H $hPath $pPath 2>&1
        } -ArgumentList $h, $p, $PWD.Path

        $finished = Wait-Job $job -Timeout $timeoutSec

        if ($finished) {
            $out = Receive-Job $job
            $solved = if ($out -match "Plan length") { 1 } else { 0 }
            $nodesLine = ($out | Select-String "Nodes expanded").ToString()
            $nodes = if ($nodesLine -match '(\d+)\s+Nodes expanded') { $matches[1] } else { "" }
            $timeLine = ($out | Select-String "Search time").ToString()
            $time = if ($timeLine -match 'Search time:\s*([\d.]+)') { $matches[1] } else { "" }
            $planLine = ($out | Select-String "Plan length").ToString()
            $plan = if ($planLine -match 'Plan length:\s*(\d+)') { $matches[1] } else { "" }
            Write-Host "solved=$solved nodes=$nodes time=$time"
        } else {
            Stop-Job $job
            $solved = 0
            $nodes = "TIMEOUT"
            $time  = "TIMEOUT"
            $plan  = ""
            Write-Host "TIMEOUT (>${timeoutSec}s)"
        }
        Remove-Job $job -Force

        $results += [PSCustomObject]@{
            Heuristic = $hName
            Problem   = $pName
            Solved    = $solved
            Nodes     = $nodes
            Time      = $time
            PlanLen   = $plan
        }
    }
}

$results | Export-Csv -Path results_blocksworld_hard.csv -NoTypeInformation -Encoding UTF8
Write-Host ""
Write-Host "DONE. Rows: $($results.Count). File: results_blocksworld_hard.csv"