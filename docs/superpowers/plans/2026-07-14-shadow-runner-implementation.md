# Shadow Runner Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Windows 上交付可运行的固定路线智能体 Web Demo，包含模拟观测、A* 规划、恢复状态机、REST/WebSocket 遥测与管理界面。

**Architecture:** 单仓库 TypeScript 项目。`src/core` 保持纯函数和可测试状态；`src/server` 承载运行时与网络接口；`src/web` 使用 React/Vite 展示快照和发送控制命令。ActionIntent 只连接模拟器，不连接在线游戏输入。

**Tech Stack:** Node.js 24、TypeScript、Express、ws、React、Vite、Vitest、Supertest、Playwright。

---

## 文件结构

```text
package.json                     依赖和统一脚本
tsconfig.json                   Node/Core 编译配置
vite.config.ts                  Web 构建与开发代理
vitest.config.ts                单元测试配置
feature-list.json               长任务验收清单
Codex-progress.txt              会话进度
init.ps1                        Windows 启动入口
src/core/types.ts               公共领域类型
src/core/graph.ts               A* 路径规划
src/core/scenario.ts            固定地图和事件脚本
src/core/engine.ts              状态融合、决策和恢复
src/server/runtime.ts           定时运行与广播
src/server/app.ts               REST/WS 应用
src/server/index.ts             服务启动入口
src/web/index.html              Vite 入口
src/web/main.tsx                React 启动
src/web/App.tsx                 控制台布局和交互
src/web/api.ts                  REST/WS 客户端
src/web/styles.css              响应式战术界面
tests/core/*.test.ts            核心测试
tests/server/*.test.ts          API 测试
tests/e2e/dashboard.spec.ts     E2E 测试（验证门后执行）
```

### Task 1: 初始化项目与测试基线

**Files:** `package.json`, `tsconfig.json`, `vite.config.ts`, `vitest.config.ts`, `feature-list.json`, `Codex-progress.txt`, `init.ps1`

- [ ] 创建 npm 项目和统一脚本：`dev`、`build`、`test`、`start`、`test:e2e`。
- [ ] 安装运行依赖 `express cors ws react react-dom zod` 和开发依赖 `typescript tsx vite vitest supertest concurrently @playwright/test`。
- [ ] 运行 `npm test`，预期在尚无测试时成功退出。
- [ ] 初始化 Git 并提交 `chore: initialize shadow runner demo`。

### Task 2: TDD 实现 A* 路径规划

**Files:** `src/core/types.ts`, `src/core/graph.ts`, `tests/core/graph.test.ts`

- [ ] 先写失败测试：

```ts
it("选择到撤离点的最低代价路径", () => {
  expect(findShortestPath(graph, "spawn-a", "extract")).toEqual([
    "spawn-a", "relay", "warehouse", "extract"
  ]);
});
```

- [ ] 运行 `npm test -- tests/core/graph.test.ts`，预期因 `findShortestPath` 不存在而失败。
- [ ] 实现 `findShortestPath(graph, start, target): string[]`，未知节点或不可达时抛出带节点名的错误。
- [ ] 重跑测试并提交 `feat: add route planner`。

### Task 3: TDD 实现模拟场景与恢复引擎

**Files:** `src/core/scenario.ts`, `src/core/engine.ts`, `tests/core/engine.test.ts`

- [ ] 先写状态转换测试：启动后从 `localizing` 进入 `navigating`；注入卡住后进入 `recovering`；恢复完成后继续原路线。
- [ ] 运行 `npm test -- tests/core/engine.test.ts`，确认红灯。
- [ ] 实现不可变 `RunnerEngine.step(observation)`，每步返回新的 `EngineSnapshot`。
- [ ] ActionIntent 只允许：

```ts
type ActionIntent =
  | { type: "move"; targetNodeId: string; ttlMs: number }
  | { type: "relocalize"; ttlMs: number }
  | { type: "recover"; strategy: "backtrack"; ttlMs: number }
  | { type: "stop"; reason: string };
```

- [ ] 重跑核心测试并提交 `feat: add simulation engine`。

### Task 4: TDD 实现服务端运行时与接口

**Files:** `src/server/runtime.ts`, `src/server/app.ts`, `src/server/index.ts`, `tests/server/app.test.ts`

- [ ] 先写 API 测试：健康检查 200；快照结构合法；`start/pause/reset/inject-stuck` 成功；未知命令 400。
- [ ] 运行 `npm test -- tests/server/app.test.ts`，确认红灯。
- [ ] 实现 500ms CPU 定时循环和订阅接口，避免后台 GPU 使用。
- [ ] 实现 REST 路由和 `/ws` WebSocket 广播；关闭时清理 timer。
- [ ] 重跑测试并提交 `feat: expose runner telemetry api`。

### Task 5: 实现 Web 管理端

**Files:** `src/web/index.html`, `src/web/main.tsx`, `src/web/api.ts`, `src/web/App.tsx`, `src/web/styles.css`

- [ ] 实现 REST 控制和 WebSocket 自动重连，连接状态显式展示。
- [ ] 用 SVG 渲染节点、规划路线、当前位置和目标位置。
- [ ] 实现开始、暂停、重置、注入卡住四个按钮；请求失败展示错误。
- [ ] 实现状态卡、置信度、路径进度、恢复次数和事件时间线。
- [ ] 运行 `npm run build`，预期 TypeScript 和 Vite 均成功。
- [ ] 提交 `feat: add runner control dashboard`。

### Task 6: Windows 集成与非 E2E 验证

**Files:** `init.ps1`, `feature-list.json`, `Codex-progress.txt`

- [ ] 在 `win-dev` 安装依赖并运行 `npm test`。
- [ ] 运行 `npm run build`。
- [ ] 启动服务，使用 HTTP 请求验证 `/api/health`、`/api/snapshot` 和控制 API。
- [ ] 更新 feature list 中被真实验证的 `passes` 字段和进度文件。
- [ ] 在开始 Playwright E2E 前向用户同步，并停止执行。

### Task 7: 独立 E2E 验证门

**Files:** `tests/e2e/dashboard.spec.ts`, `playwright.config.ts`

- [ ] 用户确认后安装 Playwright Chromium。
- [ ] 独立 evaluator 执行：页面加载、开始任务、路线推进、注入卡住、进入恢复、最终撤离。
- [ ] 硬门槛：任一步骤失败则整体失败；找出问题是成功，不是失败。
- [ ] 修复后完整重跑 `npm test && npm run build && npm run test:e2e`。
- [ ] 更新 feature list、进度记录并提交 `test: verify shadow runner demo`。

## 自审

- 设计中的控制、路线、遥测、恢复和安全边界均有对应任务。
- 无 `TBD` 或未定义接口。
- `EngineSnapshot`、`ActionIntent` 和控制命令在核心、服务端、前端使用同一命名。
- E2E 明确位于用户同步门之后。
