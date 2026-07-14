import { expect, test, type Locator, type Page } from "@playwright/test";

interface BrowserDiagnostics {
  readonly consoleErrors: string[];
  readonly pageErrors: string[];
  readonly failedRequests: string[];
  readonly badResponses: string[];
  readonly statuses: string[];
  readonly progress: number[];
  readonly currentNodes: string[];
}

function recordBrowserDiagnostics(page: Page): BrowserDiagnostics {
  const diagnostics: BrowserDiagnostics = {
    consoleErrors: [],
    pageErrors: [],
    failedRequests: [],
    badResponses: [],
    statuses: [],
    progress: [],
    currentNodes: [],
  };

  page.on("console", (message) => {
    if (message.type() === "error") {
      diagnostics.consoleErrors.push(message.text());
    }
  });
  page.on("pageerror", (error) => diagnostics.pageErrors.push(error.message));
  page.on("requestfailed", (request) => {
    diagnostics.failedRequests.push(
      `${request.method()} ${request.url()} ${request.failure()?.errorText ?? "unknown"}`,
    );
  });
  page.on("response", (response) => {
    const monitoredTypes = new Set([
      "document",
      "script",
      "stylesheet",
      "fetch",
      "xhr",
    ]);
    if (response.status() >= 400 && monitoredTypes.has(response.request().resourceType())) {
      diagnostics.badResponses.push(
        `${response.status()} ${response.request().method()} ${response.url()}`,
      );
    }
  });
  page.on("websocket", (socket) => {
    socket.on("framereceived", ({ payload }) => {
      if (typeof payload !== "string") {
        return;
      }
      try {
        const message = JSON.parse(payload) as {
          readonly type?: unknown;
          readonly data?: {
            readonly snapshot?: {
              readonly status?: unknown;
              readonly currentNodeId?: unknown;
              readonly metrics?: { readonly routeProgress?: unknown };
            };
          };
        };
        if (message.type !== "snapshot" || !message.data?.snapshot) {
          return;
        }
        const { snapshot } = message.data;
        if (typeof snapshot.status === "string") {
          diagnostics.statuses.push(snapshot.status);
        }
        if (typeof snapshot.currentNodeId === "string") {
          diagnostics.currentNodes.push(snapshot.currentNodeId);
        }
        if (typeof snapshot.metrics?.routeProgress === "number") {
          diagnostics.progress.push(snapshot.metrics.routeProgress);
        }
      } catch {
        // 非 JSON 帧由页面协议校验负责；这里只记录有效遥测快照。
      }
    });
  });

  return diagnostics;
}

async function activateWithKeyboard(
  page: Page,
  button: Locator,
  endpoint: string,
): Promise<void> {
  await button.focus();
  await expect(button).toBeFocused();
  const responsePromise = page.waitForResponse(
    (response) =>
      response.url().endsWith(endpoint) &&
      response.request().method() === "POST",
  );
  await page.keyboard.press("Enter");
  const response = await responsePromise;
  expect(response.status()).toBe(200);
}

function metric(page: Page, label: string): Locator {
  return page.locator(".metric-cell", { hasText: label }).locator("strong");
}

async function expectDashboardReady(page: Page): Promise<void> {
  await expect(page).toHaveTitle("Shadow Runner Lab // 仿真控制台");
  await expect(
    page.getByRole("heading", { level: 1, name: "SHADOW RUNNER LAB" }),
  ).toBeVisible();
  await expect(page.getByText("遥测在线", { exact: true })).toBeVisible();
  await expect(page.getByText("WS LIVE", { exact: true })).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);
}

async function expectNoHorizontalOverflow(page: Page): Promise<void> {
  const overflow = await page.evaluate(() => ({
    document: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    body: document.body.scrollWidth - document.body.clientWidth,
  }));
  expect(overflow).toEqual({ document: 0, body: 0 });
}

test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ request }) => {
  const response = await request.post("/api/control/reset", { data: {} });
  expect(response.status()).toBe(200);
});

test("真实控制流完成暂停、恢复、撤离与重置闭环", async ({ page }, testInfo) => {
  const diagnostics = recordBrowserDiagnostics(page);
  await page.goto("/");
  await expectDashboardReady(page);

  const safetyNote = page.getByRole("note");
  await expect(safetyNote).toContainText("SIMULATION");
  await expect(safetyNote).toContainText("CPU ONLY");
  await expect(safetyNote).toContainText("NO INPUT DEVICE");
  await expect(safetyNote).toContainText("不读取游戏进程，不发送键鼠输入");
  await expect(page.getByRole("img", { name: /训练地图/ })).toHaveAccessibleName(
    /当前位置 SPAWN A，目标 SPAWN A/,
  );

  const startButton = page.getByRole("button", { name: /启动 \/ 继续/ });
  const pauseButton = page.getByRole("button", { name: /暂停任务/ });
  const resetButton = page.getByRole("button", { name: /重置模拟/ });
  const injectButton = page.getByRole("button", { name: /注入卡住/ });

  await expect(page.getByText("待机", { exact: true })).toBeVisible();
  await expect(startButton).toBeEnabled();
  await expect(resetButton).toBeEnabled();
  await expect(pauseButton).toBeDisabled();
  await expect(injectButton).toBeDisabled();

  await activateWithKeyboard(page, startButton, "/api/control/start");
  await expect(pauseButton).toBeEnabled();
  await activateWithKeyboard(page, pauseButton, "/api/control/pause");
  await expect(page.getByText("已暂停", { exact: true })).toBeVisible();
  await expect(page.getByRole("region", { name: "运行事件" })).toContainText(
    "任务已暂停",
  );

  await activateWithKeyboard(page, startButton, "/api/control/start");
  await expect(injectButton).toBeEnabled();
  await activateWithKeyboard(page, injectButton, "/api/control/inject-stuck");

  await expect
    .poll(() => diagnostics.statuses.includes("recovering"), {
      message: "WebSocket 遥测应真实经过 recovering 状态",
    })
    .toBe(true);
  await expect(metric(page, "恢复次数")).toHaveText("1");
  await expect(page.getByRole("region", { name: "运行事件" })).toContainText(
    "检测到卡住，执行回退恢复",
  );

  await expect(page.getByText("已撤离", { exact: true })).toBeVisible();
  await expect(page.getByRole("img", { name: /训练地图/ })).toHaveAccessibleName(
    /当前位置 EXTRACT，目标 EXTRACT/,
  );
  await expect(metric(page, "路线进度")).toHaveText("100%");
  await expect(page.getByRole("region", { name: "运行事件" })).toContainText(
    "已到达撤离点",
  );
  await expect(startButton).toBeDisabled();
  await expect(pauseButton).toBeDisabled();
  await expect(injectButton).toBeDisabled();
  await expect(resetButton).toBeEnabled();

  expect(new Set(diagnostics.currentNodes).size).toBeGreaterThanOrEqual(3);
  expect(diagnostics.progress.some((value) => value > 0 && value < 1)).toBe(true);
  for (let index = 1; index < diagnostics.progress.length; index += 1) {
    expect(diagnostics.progress[index]).toBeGreaterThanOrEqual(
      diagnostics.progress[index - 1] ?? 0,
    );
  }

  await page.screenshot({
    path: testInfo.outputPath("dashboard-extracted.png"),
    fullPage: true,
  });

  await activateWithKeyboard(page, resetButton, "/api/control/reset");
  await expect(page.getByText("待机", { exact: true })).toBeVisible();
  await expect(page.getByRole("img", { name: /训练地图/ })).toHaveAccessibleName(
    /当前位置 SPAWN A，目标 SPAWN A/,
  );
  await expect(metric(page, "路线进度")).toHaveText("0%");
  await expect(metric(page, "恢复次数")).toHaveText("0");
  await expect(metric(page, "非法观测")).toHaveText("0");

  expect(diagnostics.consoleErrors).toEqual([]);
  expect(diagnostics.pageErrors).toEqual([]);
  expect(diagnostics.failedRequests).toEqual([]);
  expect(diagnostics.badResponses).toEqual([]);
});

const viewports = [
  { name: "desktop", width: 1440, height: 900, columns: 3 },
  { name: "tablet", width: 1024, height: 768, columns: 2 },
  { name: "mobile", width: 390, height: 844, columns: 1 },
] as const;

for (const viewport of viewports) {
  test(`${viewport.name} 布局无横向溢出且关键控件可用`, async ({ page }, testInfo) => {
    const diagnostics = recordBrowserDiagnostics(page);
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await page.goto("/");
    await expectDashboardReady(page);
    await expectNoHorizontalOverflow(page);

    const gridColumns = await page.locator(".dashboard-grid").evaluate((element) =>
      getComputedStyle(element).gridTemplateColumns.split(" ").filter(Boolean).length,
    );
    expect(gridColumns).toBe(viewport.columns);

    const routeMap = page.getByRole("img", { name: /训练地图/ });
    await expect(routeMap).toBeVisible();
    await expect(routeMap).toHaveAccessibleName(/当前位置 SPAWN A，目标 SPAWN A/);
    const meter = page.getByRole("meter", { name: "定位置信度" });
    await expect(meter).toHaveAttribute("aria-valuemin", "0");
    await expect(meter).toHaveAttribute("aria-valuemax", "100");

    const buttons = page.getByRole("button");
    for (let index = 0; index < (await buttons.count()); index += 1) {
      const button = buttons.nth(index);
      if (!(await button.isVisible())) {
        continue;
      }
      const box = await button.boundingBox();
      expect(box, `第 ${index + 1} 个按钮应有布局框`).not.toBeNull();
      expect(box?.width ?? 0).toBeGreaterThanOrEqual(44);
      expect(box?.height ?? 0).toBeGreaterThanOrEqual(44);
    }

    const ids = await page.locator("[id]").evaluateAll((elements) =>
      elements.map((element) => element.id),
    );
    expect(new Set(ids).size).toBe(ids.length);

    if (viewport.name === "mobile") {
      await expect(page.locator(".map-coordinates")).toBeHidden();
      const metricColumns = await page.locator(".metric-grid--rail").evaluate(
        (element) => getComputedStyle(element).gridTemplateColumns.split(" ").filter(Boolean).length,
      );
      expect(metricColumns).toBe(1);
    }

    const pulseAnimationMs = await page.locator(".position-pulse").evaluate((element) => {
      const duration = getComputedStyle(element).animationDuration;
      return duration.endsWith("ms")
        ? Number.parseFloat(duration)
        : Number.parseFloat(duration) * 1_000;
    });
    expect(pulseAnimationMs).toBeLessThanOrEqual(0.01);

    await page.screenshot({
      path: testInfo.outputPath(`dashboard-${viewport.name}.png`),
      fullPage: true,
    });
    expect(diagnostics.consoleErrors).toEqual([]);
    expect(diagnostics.pageErrors).toEqual([]);
    expect(diagnostics.failedRequests).toEqual([]);
    expect(diagnostics.badResponses).toEqual([]);
  });
}
