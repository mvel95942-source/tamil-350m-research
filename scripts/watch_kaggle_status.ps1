param(
    [string]$Kernel = "velmurugan04/tamil-pretrain",
    [int]$IntervalSec = 300
)
# Wait for a Kaggle kernel to leave RUNNING/QUEUED, then exit. The harness resumes the
# agent when this background command completes; the agent then verifies the checkpoint.
while ($true) {
    $status = & kaggle kernels status $Kernel 2>&1 | Out-String
    Write-Output "$(Get-Date -Format o) $($status.Trim())"
    if ($status -match "COMPLETE|ERROR|CANCEL") { break }
    Start-Sleep -Seconds $IntervalSec
}
