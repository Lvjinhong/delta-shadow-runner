[CmdletBinding()]
param(
    [ValidateSet("Setup", "Sample", "Calibrate", "Evaluate", "TestTarget", "Benchmark", "DryRun", "Armed", "ControlledE2E")]
    [string]$Mode = "DryRun",

    [string]$Config = "configs/controlled-window.json",

    [string]$Artifacts,

    [string]$WindowTitle = "三角洲行动",

    [ValidateSet("dxcam", "mss")]
    [string]$Backend = "dxcam",

    [ValidateSet("calibration", "validation", "blind")]
    [string]$Split = "calibration",

    [string]$Dataset,

    [string]$Labels,

    [string]$ProfilePath,

    [string]$RunId,

    [ValidateRange(1, 86400)]
    [double]$Duration = 120,

    [ValidateRange(2, 5)]
    [int]$SampleFps = 5,

    [ValidateRange(0, 60)]
    [double]$StartDelay = 5,

    [ValidateRange(0.001, 1000000)]
    [double]$DistanceTolerance = 25,

    [switch]$ConfirmArmed
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$UvVersion = "0.11.28"

function Get-UvExecutable {
    $command = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links\uv.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    return $null
}

function Install-Uv {
    Write-Host "[Delta Vision] Installing uv $UvVersion..."
    if (Get-Command winget.exe -ErrorAction SilentlyContinue) {
        & winget.exe install --id astral-sh.uv -e --version $UvVersion `
            --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            throw "winget 安装 uv 失败，退出码: $LASTEXITCODE"
        }
        return
    }

    $installerPath = Join-Path $env:TEMP "uv-$UvVersion-install.ps1"
    $installerUrl = "https://astral.sh/uv/0.11.28/install.ps1"
    Invoke-WebRequest -UseBasicParsing -Uri $installerUrl -OutFile $installerPath
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installerPath
    if ($LASTEXITCODE -ne 0) {
        throw "Astral 官方安装脚本执行失败，退出码: $LASTEXITCODE"
    }
}

function Initialize-PythonEnvironment {
    $uvPath = Get-UvExecutable
    if (-not $uvPath) {
        Install-Uv
        $uvPath = Get-UvExecutable
    }
    if (-not $uvPath) {
        throw "uv 安装后仍无法定位 uv.exe。请重新打开终端后再试。"
    }

    & $uvPath python install 3.12
    if ($LASTEXITCODE -ne 0) {
        throw "安装 Python 3.12 失败，退出码: $LASTEXITCODE"
    }
    & $uvPath sync --frozen --python 3.12
    if ($LASTEXITCODE -ne 0) {
        throw "按 uv.lock 同步依赖失败，退出码: $LASTEXITCODE"
    }
    return $uvPath
}

function Enter-WorkerLock {
    $mutex = New-Object System.Threading.Mutex($false, "Local\DeltaVisionWorker")
    if (-not $mutex.WaitOne(0)) {
        $mutex.Dispose()
        throw "已有一个 Delta Vision Worker 正在运行。"
    }
    return $mutex
}

function Assert-ArmedConfirmation {
    if (-not $ConfirmArmed.IsPresent) {
        throw "armed 模式必须显式传入 -ConfirmArmed。F12 是急停键。"
    }
}

function Wait-ControlledTargetArrival {
    param(
        [Parameter(Mandatory = $true)]
        [string]$GroundTruthPath,

        [int]$TimeoutMs = 2000
    )

    $deadline = [DateTime]::UtcNow.AddMilliseconds($TimeoutMs)
    do {
        if (Test-Path -LiteralPath $GroundTruthPath) {
            try {
                $sawStart = $false
                $latestTrialArrived = $false
                foreach ($line in @(Get-Content -LiteralPath $GroundTruthPath -ErrorAction Stop)) {
                    if ([string]::IsNullOrWhiteSpace($line)) {
                        continue
                    }
                    $event = $line | ConvertFrom-Json -ErrorAction Stop
                    if ($event.event -eq "start") {
                        $sawStart = $true
                        $latestTrialArrived = $false
                        continue
                    }
                    if ($sawStart -and $event.payload.arrived -eq $true) {
                        $latestTrialArrived = $true
                    }
                }
                if ($latestTrialArrived) {
                    return $true
                }
            }
            catch {
                # 目标窗口可能刚好在追加一行；在超时前继续读取完整 JSONL。
            }
        }
        if ([DateTime]::UtcNow -lt $deadline) {
            Start-Sleep -Milliseconds 100
        }
    } while ([DateTime]::UtcNow -lt $deadline)
    return $false
}

function Stop-ControlledTargetProcess {
    param(
        [System.Diagnostics.Process]$TargetProcess
    )

    if ($null -eq $TargetProcess) {
        return $true
    }
    try {
        $TargetProcess.Refresh()
        if ($TargetProcess.HasExited) {
            return $true
        }
        $taskkillOutput = & taskkill.exe /PID $TargetProcess.Id /T /F 2>&1
        $taskkillExitCode = $LASTEXITCODE
        if ($taskkillExitCode -ne 0) {
            $TargetProcess.Refresh()
            if ($TargetProcess.HasExited) {
                return $true
            }
            Write-Warning "受控目标清理失败，taskkill 退出码: $taskkillExitCode；输出: $taskkillOutput"
            return $false
        }
        $exited = $TargetProcess.WaitForExit(2000)
        $TargetProcess.Refresh()
        if (-not $exited -or -not $TargetProcess.HasExited) {
            Write-Warning "受控目标在 taskkill 成功后 2 秒内仍未退出。"
            return $false
        }
        return $true
    }
    catch {
        Write-Warning "复核受控目标退出状态失败: $($_.Exception.Message)"
        return $false
    }
}

$uv = Initialize-PythonEnvironment
if ($Mode -eq "Setup") {
    Write-Host "[Delta Vision] Python 3.12 and locked dependencies are ready."
    exit 0
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
if (-not $Artifacts) {
    $Artifacts = Join-Path $PSScriptRoot "artifacts\runs\$timestamp"
}

if ($Mode -eq "Sample") {
    if (-not $Dataset) {
        $Dataset = Join-Path $PSScriptRoot "artifacts\datasets\$timestamp-$Split"
    }
    if (-not $RunId) {
        $RunId = "route-$timestamp-$Split"
    }
    $sampleArguments = @(
        "run", "python", "-m", "delta_vision.sample_frames",
        "--window-title", $WindowTitle,
        "--backend", $Backend,
        "--output", $Dataset,
        "--run-id", $RunId,
        "--split", $Split,
        "--duration", $Duration,
        "--fps", $SampleFps,
        "--start-delay", $StartDelay
    )
    & $uv @sampleArguments
    exit $LASTEXITCODE
}

if ($Mode -eq "Calibrate") {
    if (-not $Dataset -or -not $Labels -or -not $ProfilePath) {
        throw "Calibrate 必须提供 -Dataset、-Labels 和 -ProfilePath（输出目录）。"
    }
    & $uv run python -m delta_vision.calibrate_templates `
        --dataset $Dataset --labels $Labels --output $ProfilePath
    exit $LASTEXITCODE
}

if ($Mode -eq "Evaluate") {
    if (-not $Dataset -or -not $Labels -or -not $ProfilePath) {
        throw "Evaluate 必须提供 -Dataset、-Labels 和 -ProfilePath（templates.json）。"
    }
    & $uv run python -m delta_vision.evaluate_templates `
        --profile $ProfilePath --dataset $Dataset --labels $Labels `
        --output $Artifacts --split $Split --distance-tolerance $DistanceTolerance
    exit $LASTEXITCODE
}

$configPath = Join-Path $PSScriptRoot $Config
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "找不到 Worker 配置: $configPath"
}

New-Item -ItemType Directory -Path $Artifacts -Force | Out-Null

if ($Mode -eq "TestTarget") {
    & $uv run python -m delta_vision.controlled_target `
        --artifacts (Join-Path $Artifacts "target")
    exit $LASTEXITCODE
}

if ($Mode -eq "Benchmark") {
    $benchmarkConfig = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $benchmarkArguments = @(
        "run", "python", "-m", "delta_vision.benchmark",
        "--window-title", $benchmarkConfig.target_window_title,
        "--backend", $benchmarkConfig.capture_backend,
        "--duration", "60",
        "--artifacts", (Join-Path $Artifacts "capture-benchmark")
    )
    & $uv @benchmarkArguments
    exit $LASTEXITCODE
}

if ($Mode -eq "DryRun") {
    & $uv run python -m delta_vision.worker `
        --config $configPath --artifacts (Join-Path $Artifacts "worker")
    exit $LASTEXITCODE
}

Assert-ArmedConfirmation
$workerMutex = Enter-WorkerLock

if ($Mode -eq "Armed") {
    try {
        & $uv run python -m delta_vision.worker `
            --config $configPath --artifacts (Join-Path $Artifacts "worker") "--armed"
        $workerExitCode = $LASTEXITCODE
    }
    finally {
        $workerMutex.ReleaseMutex()
        $workerMutex.Dispose()
    }
    exit $workerExitCode
}

$targetProcess = $null
$workerExitCode = 1
$cleanupSucceeded = $true
$groundTruthPath = Join-Path $Artifacts "target\target-ground-truth.jsonl"
try {
    $targetArguments = @(
        "run", "python", "-m", "delta_vision.controlled_target",
        "--artifacts", (Join-Path $Artifacts "target")
    )
    $targetProcess = Start-Process -FilePath $uv -ArgumentList $targetArguments -PassThru
    Start-Sleep -Seconds 2
    & $uv run python -m delta_vision.worker `
        --config $configPath --artifacts (Join-Path $Artifacts "worker") "--armed"
    $workerExitCode = $LASTEXITCODE
    if ($workerExitCode -eq 0 -and -not (Wait-ControlledTargetArrival `
        -GroundTruthPath $groundTruthPath)) {
        Write-Warning "Worker 报告到达，但受控目标 ground truth 未确认到达。"
        $workerExitCode = 3
    }
}
finally {
    try {
        $cleanupSucceeded = Stop-ControlledTargetProcess -TargetProcess $targetProcess
    }
    finally {
        $workerMutex.ReleaseMutex()
        $workerMutex.Dispose()
    }
}
if (-not $cleanupSucceeded) {
    $workerExitCode = 4
}
exit $workerExitCode
