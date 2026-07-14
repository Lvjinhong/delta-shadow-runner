import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

const projectRoot = resolve(import.meta.dirname, "../..");
const styles = readFileSync(resolve(projectRoot, "src/web/styles.css"), "utf8");
const mapStyles = readFileSync(resolve(projectRoot, "src/web/map.css"), "utf8");
const appSource = readFileSync(resolve(projectRoot, "src/web/App.tsx"), "utf8");

function selectorBlock(source: string, selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = source.match(new RegExp(`${escaped}\\s*\\{([^}]*)\\}`));
  if (!match?.[1]) {
    throw new Error(`未找到 CSS selector: ${selector}`);
  }
  return match[1];
}

function property(block: string, name: string): string {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = block.match(new RegExp(`${escaped}\\s*:\\s*([^;]+)`));
  if (!match?.[1]) {
    throw new Error(`未找到 CSS property: ${name}`);
  }
  return match[1].trim();
}

function parseHex(value: string): readonly [number, number, number] {
  const match = value.match(/^#([0-9a-f]{6})$/i);
  if (!match?.[1]) {
    throw new Error(`不是六位十六进制颜色: ${value}`);
  }
  return [
    Number.parseInt(match[1].slice(0, 2), 16),
    Number.parseInt(match[1].slice(2, 4), 16),
    Number.parseInt(match[1].slice(4, 6), 16),
  ];
}

function luminance(value: string): number {
  const channels = parseHex(value).map((channel) => {
    const normalized = channel / 255;
    return normalized <= 0.04045
      ? normalized / 12.92
      : ((normalized + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * (channels[0] ?? 0) + 0.7152 * (channels[1] ?? 0) + 0.0722 * (channels[2] ?? 0);
}

function contrast(foreground: string, background: string): number {
  const lighter = Math.max(luminance(foreground), luminance(background));
  const darker = Math.min(luminance(foreground), luminance(background));
  return (lighter + 0.05) / (darker + 0.05);
}

function rootColor(name: string): string {
  return property(selectorBlock(styles, ":root"), name);
}

function resolveColor(value: string): string {
  const variable = value.match(/^var\((--[a-z-]+)\)$/)?.[1];
  return variable ? rootColor(variable) : value;
}

function minHeight(source: string, selector: string): number {
  const value = property(selectorBlock(source, selector), "min-height");
  const match = value.match(/^(\d+)px$/);
  if (!match?.[1]) {
    throw new Error(`${selector} 的 min-height 不是 px: ${value}`);
  }
  return Number(match[1]);
}

describe("dashboard visual accessibility contract", () => {
  it("小字、footer 与地图图形达到明确对比度阈值", () => {
    const coal = rootColor("--coal");
    const inkDim = rootColor("--ink-dim");
    const footer = resolveColor(property(selectorBlock(styles, ".system-footer"), "color"));
    const mapBackground = property(selectorBlock(mapStyles, ".map-stage"), "background");
    const baseEdge = property(selectorBlock(mapStyles, ".route-edge--base"), "stroke");
    const edgeCost = property(selectorBlock(mapStyles, ".edge-cost"), "fill");

    expect(contrast(inkDim, coal)).toBeGreaterThanOrEqual(4.5);
    expect(contrast(footer, coal)).toBeGreaterThanOrEqual(4.5);
    expect(contrast(baseEdge, mapBackground)).toBeGreaterThanOrEqual(3);
    expect(contrast(edgeCost, mapBackground)).toBeGreaterThanOrEqual(4.5);
  });

  it("主要 button、summary 与错误关闭按钮提供至少 44px 命中区域", () => {
    expect(minHeight(styles, ".control-button")).toBeGreaterThanOrEqual(44);
    expect(minHeight(styles, ".collapsible-panel > summary")).toBeGreaterThanOrEqual(44);
    expect(minHeight(styles, ".error-banner button")).toBeGreaterThanOrEqual(44);
    expect(
      Number(property(selectorBlock(styles, ".error-banner button"), "min-width").replace("px", "")),
    ).toBeGreaterThanOrEqual(44);
  });

  it.each([
    [styles, ".error-banner span"],
    [styles, ".event span"],
    [styles, ".decision-readout > strong"],
    [mapStyles, ".node-label"],
  ] as const)("%s 中 %s 可在自身容器内安全换行", (source, selector) => {
    const block = selectorBlock(source, selector);
    expect(property(block, "min-width")).toBe("0");
    expect(property(block, "overflow-wrap")).toBe("anywhere");
  });

  it("保留键盘 focus 和 reduced-motion 保护", () => {
    expect(styles).toMatch(/button:focus-visible[\s\S]*outline:/);
    expect(styles).toContain("@media (prefers-reduced-motion: reduce)");
  });
});

describe("dashboard event accessibility structure", () => {
  it("事件列表本身不 live，仅使用 sr-only status 播报最新事件", () => {
    expect(appSource).toContain('<ol className="event-list">');
    expect(appSource).not.toContain('<ol className="event-list" aria-live="polite">');
    expect(appSource).toMatch(
      /className="sr-only event-announcer"\s+role="status"\s+aria-live="polite"/,
    );
  });

  it("事件 key 不依赖反转后的 index，runId 提供完整 title", () => {
    expect(appSource).not.toMatch(/key=\{`\$\{event\.tick\}[^`]*\$\{index\}/);
    expect(appSource).toContain("title={snapshot.runId}");
  });
});
