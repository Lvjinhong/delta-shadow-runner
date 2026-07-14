param(
    [ValidateSet("dev", "test", "build", "start")]
    [string]$Mode = "dev",

    [switch]$ForceInstall
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Test-DependenciesHealthy {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ModulesDirectory,

        [Parameter(Mandatory = $true)]
        [string]$StampFile,

        [Parameter(Mandatory = $true)]
        [string]$LockHash
    )

    if (-not (Test-Path $ModulesDirectory)) {
        return $false
    }

    if (-not (Test-Path $StampFile)) {
        return $false
    }

    $installedLockHash = (Get-Content -LiteralPath $StampFile -Raw).Trim().ToLowerInvariant()
    if ($installedLockHash -ne $LockHash) {
        return $false
    }

    $null = & npm.cmd ls --all --silent
    if ($LASTEXITCODE -ne 0) {
        return $false
    }

    return $true
}

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "未找到 Node.js。请安装 Node.js 24 或更高版本后重试。"
}

if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "未找到 npm。请检查 Node.js 安装和 PATH。"
}

$nodeVersion = & node --version
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$nodeMajorVersion = [int]($nodeVersion.TrimStart("v").Split(".")[0])
if ($nodeMajorVersion -lt 24) {
    throw "当前 Node.js 主版本为 $nodeMajorVersion，项目要求 Node.js 24 或更高版本。"
}

$lockFile = Join-Path $PSScriptRoot "package-lock.json"
$modulesDirectory = Join-Path $PSScriptRoot "node_modules"
$stampFile = Join-Path $PSScriptRoot "node_modules/.shadow-runner-lock.sha256"

if (-not (Test-Path $lockFile)) {
    throw "缺少 package-lock.json，无法执行可复现的依赖安装。"
}

$lockHash = (Get-FileHash -LiteralPath $lockFile -Algorithm SHA256).Hash.ToLowerInvariant()
$requiresInstall = $ForceInstall.IsPresent

if (-not $requiresInstall) {
    $requiresInstall = -not (Test-DependenciesHealthy `
        -ModulesDirectory $modulesDirectory `
        -StampFile $stampFile `
        -LockHash $lockHash)
}

if ($requiresInstall) {
    & npm.cmd ci
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    # 只有完整安装成功后才落戳，避免失败安装在下次启动时被误判为可用。
    Set-Content -LiteralPath $stampFile -Value $lockHash -Encoding ASCII -NoNewline
}

& npm.cmd run $Mode
exit $LASTEXITCODE
