[CmdletBinding()]
param(
    [ValidateSet("Setup", "TestTarget", "Benchmark", "DryRun", "Armed", "ControlledE2E")]
    [string]$Mode = "DryRun",

    [string]$Config = "configs/controlled-window.json",

    [string]$Artifacts,

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

$uv = Initialize-PythonEnvironment
if ($Mode -eq "Setup") {
    Write-Host "[Delta Vision] Python 3.12 and locked dependencies are ready."
    exit 0
}

$configPath = Join-Path $PSScriptRoot $Config
if (-not (Test-Path -LiteralPath $configPath)) {
    throw "找不到 Worker 配置: $configPath"
}

if (-not $Artifacts) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $Artifacts = Join-Path $PSScriptRoot "artifacts\runs\$timestamp"
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
}
finally {
    if ($targetProcess -and -not $targetProcess.HasExited) {
        & taskkill.exe /PID $targetProcess.Id /T /F | Out-Null
    }
    $workerMutex.ReleaseMutex()
    $workerMutex.Dispose()
}
exit $workerExitCode
