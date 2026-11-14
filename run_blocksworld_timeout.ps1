# run_blocksworld_timeout.ps1
$ErrorActionPreference = "Continue"

$heuristics = @()
$heuristics += Get-ChildItem archive-heuristics\2025-03-10-deepseek-r1\blocksworld*_10.py |
    ForEach-Object { $_.FullName }
$heuristics += (Resolve-Path archive-heuristics\selected-heuristics\deepseek-r1\blocksworld9_10.py).Path

Write-Host "Heuristics: $($heuristics.Count)"

$problems = @()
$problems += Get-ChildItem benchmarks\ipc2023-learning\testing\blocksworld\easy-p0[1-5].pddl |
    ForEach-Object { $_.FullName }
$problems += Get-ChildItem benchmarks\ipc2023-learning\testing\blocksworld\medium-p0[1-5].pddl |
    ForEach-Object { $_.FullName }

Write-Host "Problems: $($problems.Count)"

$timeoutSec = 60
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
            $nodes = (($out | Select-String "Nodes expanded") -replace ".*Nodes expanded: ", "").ToString().Trim()
            $time  = (($out | Select-String "Search time")   -replace ".*Search time: ",   "").ToString().Trim()
            $plan  = (($out | Select-String "Plan length")   -replace ".*Plan length: ",   "").ToString().Trim()
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

$results | Export-Csv -Path results_blocksworld_timeout.csv -NoTypeInformation -Encoding UTF8
Write-Host ""
Write-Host "DONE. Rows: $($results.Count). File: results_blocksworld_timeout.csv"