param(
    [ValidateSet("dev", "test", "build", "start")]
    [string]$Mode = "dev"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "未找到 Node.js。请安装 Node.js 24 或更高版本后重试。"
}

if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "未找到 npm。请检查 Node.js 安装和 PATH。"
}

$nodeMajorVersion = [int]((node --version).TrimStart("v").Split(".")[0])
if ($nodeMajorVersion -lt 24) {
    throw "当前 Node.js 主版本为 $nodeMajorVersion，项目要求 Node.js 24 或更高版本。"
}

if (-not (Test-Path "node_modules")) {
    if (Test-Path "package-lock.json") {
        & npm.cmd ci
    } else {
        & npm.cmd install
    }

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

& npm.cmd run $Mode
exit $LASTEXITCODE
