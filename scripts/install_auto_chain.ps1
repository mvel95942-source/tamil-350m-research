# Install the unattended Tamil training chain.
# - keeps the machine awake on AC (including lid close)
# - registers a scheduled task that runs scripts/auto_chain.py every 30 minutes
# - starts it immediately
#
# Run: powershell -ExecutionPolicy Bypass -File scripts\install_auto_chain.ps1

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$script = Join-Path $root "scripts\auto_chain.py"
$taskName = "MerzalTamilAutoChain"

Write-Host "Repo: $root"

# ---- keep the orchestrator alive on AC
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0 2>&1 | Out-Null
powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0 2>&1 | Out-Null
powercfg /setactive SCHEME_CURRENT 2>&1 | Out-Null
powercfg /change standby-timeout-ac 0 2>&1 | Out-Null
powercfg /change hibernate-timeout-ac 0 2>&1 | Out-Null
powercfg /change disk-timeout-ac 0 2>&1 | Out-Null
Write-Host "Power: lid-close=do-nothing, AC sleep/hibernate/disk timeouts=never"

# ---- locate pythonw so the task has no console window
$python = (Get-Command python -ErrorAction Stop).Source
$pythonw = Join-Path (Split-Path $python) "pythonw.exe"
if (-not (Test-Path $pythonw)) { $pythonw = $python }
Write-Host "Python: $pythonw"

# ---- register (and replace) the scheduled task
$registered = $false
try {
    $action = New-ScheduledTaskAction -Execute $pythonw -Argument "`"$script`"" -WorkingDirectory $root
    $t1 = New-ScheduledTaskTrigger -AtLogOn
    $t2 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes 30) `
        -RepetitionDuration (New-TimeSpan -Days 3650)
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours 0) `
        -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $taskName -Action $action `
        -Trigger @($t1, $t2) -Settings $settings -Force `
        -Description "Polls Kaggle and pushes tamil pretrain v5, then SFT, then downloads and evaluates." | Out-Null
    $registered = $true
    Write-Host "Scheduled task registered via Register-ScheduledTask."
} catch {
    Write-Host "Register-ScheduledTask failed: $($_.Exception.Message)"
}

if (-not $registered) {
    $tr = "`"$pythonw`" `"$script`""
    schtasks /create /tn $taskName /tr $tr /sc minute /mo 30 /f 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $registered = $true
        Write-Host "Scheduled task registered via schtasks."
    } else {
        Write-Host "schtasks failed with exit code $LASTEXITCODE."
    }
}

# ---- start now
if ($registered) {
    try {
        Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
        Write-Host "Started task $taskName."
    } catch {
        Write-Host "Start-ScheduledTask failed: $($_.Exception.Message)"
        Start-Process -FilePath $pythonw -ArgumentList "`"$script`"" -WorkingDirectory $root -WindowStyle Hidden
        Write-Host "Started auto_chain.py directly."
    }
}

Write-Host "State:  $root\artifacts\auto_chain_state.json"
Write-Host "Log:    $root\artifacts\auto_chain.log"
Write-Host "Status: $root\artifacts\auto_chain_status.txt"
