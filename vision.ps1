[CmdletBinding()]
param(
    [ValidateSet("Setup", "Sample", "Calibrate", "Evaluate", "SessionSample", "TestTarget", "Benchmark", "DryRun", "Armed", "SessionArmed", "ControlledE2E", "Preflight")]
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

    [ValidateSet("ncc", "orb", "sift")]
    [string]$FeatureBackend = "ncc",

    [ValidateRange(32, 50000)]
    [int]$MaximumFeatures = 3000,

    [ValidatePattern("^[A-Za-z0-9._-]+$")]
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
            --accept-package-agreements --accept-source-agreements | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "winget 安装 uv 失败，退出码: $LASTEXITCODE"
        }
        return
    }

    $installerPath = Join-Path $env:TEMP "uv-$UvVersion-install.ps1"
    $installerUrl = "https://astral.sh/uv/0.11.28/install.ps1"
    Invoke-WebRequest -UseBasicParsing -Uri $installerUrl -OutFile $installerPath
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $installerPath | Out-Host
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

    & $uvPath python install 3.12 | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "安装 Python 3.12 失败，退出码: $LASTEXITCODE"
    }
    & $uvPath sync --frozen --python 3.12 | Out-Host
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

        [Parameter(Mandatory = $true)]
        [string]$RunId,

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
                    if ($event.run_id -ne $RunId) {
                        continue
                    }
                    if ($event.event -eq "start") {
                        $sawStart = $true
                        $latestTrialArrived = $false
                        continue
                    }
                    if (
                        $sawStart `
                        -and $event.event -eq "position" `
                        -and $event.payload.arrived -eq $true
                    ) {
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

function Invoke-ControlledE2E {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Uv,

        [Parameter(Mandatory = $true)]
        [string]$ConfigPath,

        [Parameter(Mandatory = $true)]
        [string]$ArtifactsPath,

        [Parameter(Mandatory = $true)]
        [string]$RunId
    )

    $targetProcess = $null
    $workerExitCode = 1
    $cleanupSucceeded = $true
    $groundTruthPath = Join-Path $ArtifactsPath "target\target-ground-truth.jsonl"
    try {
        $targetArtifactsPath = Join-Path $ArtifactsPath "target"
        # Windows PowerShell 5.1 会把 ArgumentList 数组拼成字符串；显式保留路径引号。
        $targetArtifactsArgument = '"' + $targetArtifactsPath + '"'
        $runIdArgument = '"' + $RunId + '"'
        $targetArgumentLine = "run python -m delta_vision.controlled_target --artifacts $targetArtifactsArgument --run-id $runIdArgument"
        $targetProcess = Start-Process -FilePath $Uv -ArgumentList $targetArgumentLine -PassThru
        Start-Sleep -Seconds 2
        & $Uv run python -m delta_vision.worker `
            --config $ConfigPath --artifacts (Join-Path $ArtifactsPath "worker") `
            --run-id $RunId "--armed" `
            | Out-Host
        $workerExitCode = $LASTEXITCODE
        if ($workerExitCode -eq 0 -and -not (Wait-ControlledTargetArrival `
            -GroundTruthPath $groundTruthPath -RunId $RunId)) {
            Write-Warning "Worker 报告到达，但受控目标 ground truth 未确认到达。"
            $workerExitCode = 3
        }
    }
    finally {
        $cleanupSucceeded = Stop-ControlledTargetProcess -TargetProcess $targetProcess
    }
    if (-not $cleanupSucceeded) {
        $workerExitCode = 4
    }
    return $workerExitCode
}

$uv = Initialize-PythonEnvironment
if ($Mode -eq "Setup") {
    Write-Host "[Delta Vision] Python 3.12 and locked dependencies are ready."
    exit 0
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$effectiveRunId = if ($RunId) { $RunId } else { [Guid]::NewGuid().ToString("N") }
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
        --dataset $Dataset --labels $Labels --output $ProfilePath `
        --feature-backend $FeatureBackend --maximum-features $MaximumFeatures
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

if ($Mode -eq "SessionSample") {
    Assert-ArmedConfirmation
    if (-not $ProfilePath) {
        throw "SessionSample 必须提供 -ProfilePath（菜单 menu.json）。"
    }
    $workerMutex = Enter-WorkerLock
    try {
        & $uv run python -m delta_vision.session_sample `
            --menu-profile $ProfilePath `
            --window-title $WindowTitle `
            --backend $Backend `
            --artifacts $Artifacts `
            --run-id $effectiveRunId `
            --split $Split `
            --duration $Duration `
            --fps $SampleFps `
            "--armed"
        $workerExitCode = $LASTEXITCODE
    }
    finally {
        $workerMutex.ReleaseMutex()
        $workerMutex.Dispose()
    }
    exit $workerExitCode
}

$configPath = Join-Path $PSScriptRoot $Config
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "找不到 Worker 配置: $configPath"
}

if ($Mode -eq "SessionArmed") {
    Assert-ArmedConfirmation
    $workerMutex = Enter-WorkerLock
    try {
        & $uv run python -m delta_vision.game_session `
            --config $configPath --artifacts $Artifacts `
            --run-id $effectiveRunId "--armed"
        $workerExitCode = $LASTEXITCODE
    }
    finally {
        $workerMutex.ReleaseMutex()
        $workerMutex.Dispose()
    }
    exit $workerExitCode
}

New-Item -ItemType Directory -Path $Artifacts -Force | Out-Null

if ($Mode -eq "TestTarget") {
    & $uv run python -m delta_vision.controlled_target `
        --artifacts (Join-Path $Artifacts "target") --run-id $effectiveRunId
    exit $LASTEXITCODE
}

if ($Mode -eq "Benchmark") {
    $benchmarkArguments = @(
        "run", "python", "-m", "delta_vision.benchmark",
        "--config", $configPath,
        "--duration", "60",
        "--artifacts", (Join-Path $Artifacts "capture-benchmark"),
        "--run-id", $effectiveRunId
    )
    & $uv @benchmarkArguments
    exit $LASTEXITCODE
}

if ($Mode -eq "DryRun") {
    $workerMutex = Enter-WorkerLock
    try {
        & $uv run python -m delta_vision.worker `
            --config $configPath --artifacts (Join-Path $Artifacts "worker") `
            --run-id $effectiveRunId
        $workerExitCode = $LASTEXITCODE
    }
    finally {
        $workerMutex.ReleaseMutex()
        $workerMutex.Dispose()
    }
    exit $workerExitCode
}

Assert-ArmedConfirmation
$workerMutex = Enter-WorkerLock

if ($Mode -eq "Armed") {
    try {
        & $uv run python -m delta_vision.worker `
            --config $configPath --artifacts (Join-Path $Artifacts "worker") `
            --run-id $effectiveRunId "--armed"
        $workerExitCode = $LASTEXITCODE
    }
    finally {
        $workerMutex.ReleaseMutex()
        $workerMutex.Dispose()
    }
    exit $workerExitCode
}

if ($Mode -eq "ControlledE2E") {
    try {
        $workerExitCode = Invoke-ControlledE2E `
            -Uv $uv -ConfigPath $configPath -ArtifactsPath $Artifacts `
            -RunId $effectiveRunId
    }
    finally {
        $workerMutex.ReleaseMutex()
        $workerMutex.Dispose()
    }
    exit $workerExitCode
}

$controlledArtifacts = Join-Path $Artifacts "controlled-e2e"
$benchmarkArtifacts = Join-Path $Artifacts "capture-benchmark"
$controlledConfigPath = Join-Path $PSScriptRoot "configs\controlled-window.json"
try {
    & $uv run python -m delta_vision.worker `
        --config $configPath "--validate-only"
    $configExitCode = $LASTEXITCODE

    $controlledExitCode = Invoke-ControlledE2E `
        -Uv $uv -ConfigPath $controlledConfigPath -ArtifactsPath $controlledArtifacts `
        -RunId $effectiveRunId

    Write-Host "[Delta Vision] 受控 E2E 已结束，请在 5 秒内切回三角洲行动窗口。"
    Start-Sleep -Seconds 5
    $benchmarkArguments = @(
        "run", "python", "-m", "delta_vision.benchmark",
        "--config", $configPath,
        "--duration", "60",
        "--artifacts", $benchmarkArtifacts,
        "--run-id", $effectiveRunId
    )
    & $uv @benchmarkArguments
    $benchmarkExitCode = $LASTEXITCODE

    & $uv run python -m delta_vision.preflight `
        --run-id $effectiveRunId `
        --config $configPath `
        --capture-metrics (Join-Path $benchmarkArtifacts "capture-metrics.json") `
        --capture-gate (Join-Path $benchmarkArtifacts "capture-gate.json") `
        --worker-events (Join-Path $controlledArtifacts "worker\events.jsonl") `
        --ground-truth (Join-Path $controlledArtifacts "target\target-ground-truth.jsonl") `
        --config-exit-code $configExitCode `
        --controlled-exit-code $controlledExitCode `
        --benchmark-exit-code $benchmarkExitCode `
        --output (Join-Path $Artifacts "preflight-report.json")
    $reportExitCode = $LASTEXITCODE

    if (
        $configExitCode -ne 0 `
        -or $controlledExitCode -ne 0 `
        -or $benchmarkExitCode -ne 0 `
        -or $reportExitCode -ne 0
    ) {
        $preflightExitCode = 2
    }
    else {
        $preflightExitCode = 0
    }
}
finally {
    $workerMutex.ReleaseMutex()
    $workerMutex.Dispose()
}
exit $preflightExitCode
