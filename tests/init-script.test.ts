import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

const projectRoot = resolve(import.meta.dirname, "..");
const script = readFileSync(resolve(projectRoot, "init.ps1"), "utf8");

function indexOfOrThrow(fragment: string, fromIndex = 0): number {
  const index = script.indexOf(fragment, fromIndex);
  if (index === -1) {
    throw new Error(`init.ps1 缺少契约片段: ${fragment}`);
  }
  return index;
}

describe("Windows bootstrap contract", () => {
  it("保留 Mode、Node 24 下限，并提供可选 ForceInstall", () => {
    expect(script).toMatch(/\[ValidateSet\("dev", "test", "build", "start"\)\]/);
    expect(script).toMatch(/\[string\]\$Mode\s*=\s*"dev"/);
    expect(script).toMatch(/\[switch\]\$ForceInstall/);
    expect(script).toMatch(/\$nodeMajorVersion\s+-lt\s+24/);
  });

  it("强制使用 package-lock 和 npm ci，不允许回退到 npm install", () => {
    expect(script).toContain('$lockFile = Join-Path $PSScriptRoot "package-lock.json"');
    expect(script).toMatch(/if\s*\(-not\s*\(Test-Path\s+\$lockFile\)\)\s*\{[\s\S]*?throw/);
    expect(script).toContain("& npm.cmd ci");
    expect(script).not.toContain("& npm.cmd install");
  });

  it("以 package-lock 的 SHA256 作为 node_modules 内的安装戳", () => {
    expect(script).toContain(
      '$stampFile = Join-Path $PSScriptRoot "node_modules/.shadow-runner-lock.sha256"',
    );
    expect(script).toMatch(/Get-FileHash\s+-LiteralPath\s+\$lockFile\s+-Algorithm\s+SHA256/);
    expect(script).toMatch(/\.Hash\.ToLowerInvariant\(\)/);
  });

  it("仅当目录、戳哈希和 npm ls 完整性全部有效时跳过 npm ci", () => {
    expect(script).toMatch(/\$requiresInstall\s*=\s*\$ForceInstall\.IsPresent/);
    expect(script).toMatch(/Test-Path\s+\$modulesDirectory/i);
    expect(script).toMatch(/Test-Path\s+\$stampFile/i);
    expect(script).toMatch(/Get-Content\s+-LiteralPath\s+\$stampFile\s+-Raw/i);
    expect(script).toMatch(/\$installedLockHash\s+-ne\s+\$lockHash/i);
    expect(script).toContain("& npm.cmd ls --all --silent");
    expect(script).toMatch(/\$LASTEXITCODE\s+-ne\s+0[\s\S]*?return\s+\$false/);
    expect(script).toMatch(
      /\$requiresInstall\s*=\s*-not\s*\(Test-DependenciesHealthy[\s\S]*?-LockHash\s+\$lockHash\)/,
    );
    expect(script).toMatch(/if\s*\(\$requiresInstall\)\s*\{[\s\S]*?& npm\.cmd ci/);
  });

  it("npm ci 成功后才写戳，且把失败码原样传播", () => {
    const nodeRunIndex = indexOfOrThrow("$nodeVersion = & node --version");
    const nodeExitCheckIndex = indexOfOrThrow("if ($LASTEXITCODE -ne 0)", nodeRunIndex);
    const nodeParseIndex = indexOfOrThrow("$nodeMajorVersion = [int]($nodeVersion");
    const ciIndex = indexOfOrThrow("& npm.cmd ci");
    const ciExitCheckIndex = indexOfOrThrow("if ($LASTEXITCODE -ne 0)", ciIndex);
    const stampWriteIndex = indexOfOrThrow("Set-Content -LiteralPath $stampFile");
    const modeRunIndex = indexOfOrThrow("& npm.cmd run $Mode");
    const finalExitIndex = script.lastIndexOf("exit $LASTEXITCODE");

    expect(nodeExitCheckIndex).toBeGreaterThan(nodeRunIndex);
    expect(script.slice(nodeExitCheckIndex, nodeParseIndex)).toMatch(/exit\s+\$LASTEXITCODE/);
    expect(ciExitCheckIndex).toBeGreaterThan(ciIndex);
    expect(script.slice(ciExitCheckIndex, stampWriteIndex)).toMatch(/exit\s+\$LASTEXITCODE/);
    expect(stampWriteIndex).toBeGreaterThan(ciExitCheckIndex);
    expect(modeRunIndex).toBeGreaterThan(stampWriteIndex);
    expect(finalExitIndex).toBeGreaterThan(modeRunIndex);
  });
});
