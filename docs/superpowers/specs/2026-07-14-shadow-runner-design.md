# Shadow Runner Lab 设计说明

## 1. 目标

在 Windows 开发机上交付一个可运行的 TypeScript Web Demo，用于演示固定路线智能体的完整数据流：模拟观测、状态融合、路线规划、异常恢复、遥测展示和人工控制。

该 Demo 只连接模拟器、离线回放或 mock 数据源。它不读取在线游戏进程、不发送真实键鼠输入、不包含反作弊绕过能力。

## 2. 验收范围

1. 用户可以在 Web 控制台启动、暂停、重置模拟任务。
2. 页面实时显示路线、当前位置、目标、运行状态、置信度和最近事件。
3. 核心引擎能够从多个出生点计算到撤离点的最短路径。
4. 模拟器会注入偏航或卡住事件，行为树能够进入恢复状态并继续执行。
5. 浏览器通过 WebSocket 接收遥测；REST API 提供快照和控制命令。
6. 单元测试覆盖路径规划、状态转换、恢复和 API 输入校验。
7. E2E 验证开始前必须先向用户同步。

## 3. 架构

```text
React/Vite Web
  | REST + WebSocket
Node/Express Runtime
  | EngineSnapshot / ControlCommand
SimulationSource -> Belief Engine -> A* Planner -> Behavior State -> ActionIntent
```

### 核心边界

- `src/core`：纯 TypeScript，无网络和 UI 依赖，可独立测试。
- `src/server`：管理运行时、REST、WebSocket 和生命周期。
- `src/web`：只消费快照和发出控制命令，不包含决策逻辑。
- `ActionIntent`：只表示 `move/relocalize/recover/stop` 意图；Demo 没有真实输入执行器。

## 4. 数据模型

```ts
type RunnerStatus = "idle" | "localizing" | "navigating" | "recovering" | "extracted" | "paused";

interface EngineSnapshot {
  runId: string;
  status: RunnerStatus;
  tick: number;
  currentNodeId: string;
  targetNodeId: string;
  route: string[];
  confidence: number;
  action: ActionIntent;
  metrics: RuntimeMetrics;
  events: RunnerEvent[];
}
```

所有状态由服务端生成。控制 API 只接受 `start`、`pause`、`reset` 和 `inject-stuck`，非法命令返回 400。

## 5. UI 设计

视觉方向为深色战术仪表盘，但避免复刻游戏素材：煤黑背景、琥珀色路径、青绿色安全状态、红色告警。桌面优先，同时保证 1280px 宽度无横向滚动。

页面分为：顶部状态栏、左侧地图、右侧控制和关键指标、底部事件时间线。地图使用 SVG，节点和路线来自服务端快照。

## 6. 错误处理

- WebSocket 断开时显示“遥测断开”，并以指数退避重连。
- REST 控制失败时显示可读错误，不乐观修改服务端状态。
- 引擎遇到空路线或未知节点时进入安全停止状态并记录事件。
- 运行时关闭时清理定时器和 WebSocket 连接。

## 7. 测试策略

1. Vitest：A* 最短路径、无路径、状态机、卡住恢复、控制命令校验。
2. 构建验证：`npm run build` 同时编译服务端与前端。
3. API 冒烟：启动服务后请求 `/api/health` 和 `/api/snapshot`。
4. Playwright E2E：页面加载、启动任务、注入卡住、观察恢复、最终撤离。该步骤必须在用户确认后运行。

## 8. 非目标

- 不训练或运行游戏专用视觉模型。
- 不采集用户当前游戏画面。
- 不实现鼠标、键盘、驱动或硬件输入。
- 不估算正式服撤离率或封禁率。
