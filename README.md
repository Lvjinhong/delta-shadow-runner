# Delta External Vision Runner

Windows PC 端纯外部视觉自动化实验项目。主链路是：

```text
目标窗口客户区
→ DXcam/MSS 截图
→ OpenCV 视觉锚点与置信度
→ waypoint 定位与 A* 路线
→ 非阻塞短脉冲状态机
→ 前台窗口 + HWND + F12 安全门
→ Win32 SendInput
→ PNG/JSONL 录制与确定性回放
```

项目不读取目标进程内存，不注入 DLL，不安装驱动，不修改游戏文件，也不提供反作弊规避、自动瞄准、自动射击或敌人战斗决策。

当前可直接运行的是独立受控测试窗口，用来验证真实桌面截图、视觉定位、路线规划和标准键盘输入。`《三角洲行动》` 的实际地图路线仍需要在用户授权的受控场景中采集截图、标定 waypoint 并单独验收；受控窗口通过不等于游戏路线已经通过。

## 1. 最快开始：Windows 一键受控 E2E

环境要求：

1. Windows 10/11 x64；
2. 当前用户处于可见的交互桌面；
3. 可访问 WinGet 或 `astral.sh`；
4. 显式同意测试窗口接收标准 WASD 输入。

双击：

```text
start-controlled-e2e.cmd
```

确认后脚本会：

1. 查找 `uv`；未安装时优先通过 WinGet 安装固定版本，WinGet 不可用时使用 Astral 官方版本化安装脚本；
2. 用 `uv` 安装 Python 3.12；
3. 执行 `uv sync --frozen --python 3.12`，严格按 `uv.lock` 同步依赖；
4. 启动标题为 `Delta Vision Test Target` 的独立 Tk 测试窗口；
5. 运行真正的截图 Worker，并显式开启 `--armed`；
6. 通过截图寻找绿色锚点，沿 `start → turn → goal` 的 A* 路线发送短时 `W/D`；
7. Worker 结束后清理它启动的测试窗口进程，并保留证据。

急停键是 `F12`。切走目标窗口也会触发安全停止和按键释放。

脚本不会启动旧 Web 页面，也没有需要访问的 URL。

## 2. 分步运行

### 2.1 只安装运行环境

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\vision.ps1 -Mode Setup
```

`-ExecutionPolicy Bypass` 只作用于本次 PowerShell 进程，不修改系统持久策略。

### 2.2 只启动受控测试窗口

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\vision.ps1 -Mode TestTarget
```

窗口中的绿色圆点是 Worker 唯一使用的状态来源。测试窗口会单独写 ground truth，供独立评估器核对；Worker 不读取该文件。

### 2.3 Dry-run

先保持测试窗口打开并处于前台，再在另一个终端运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\vision.ps1 -Mode DryRun
```

Dry-run 会执行真实截图、识别、A*、状态机和录制，但只记录计划动作，不调用 `SendInput`。因为目标不会移动，状态机最终应进入有限恢复或安全停止；这属于预期行为。

### 2.4 运行 60 秒截图基准

保持测试窗口打开后运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\vision.ps1 -Mode Benchmark
```

产物包括 `capture-metrics.json`、`first-frame.png` 和 `last-frame.png`。门槛是平均 `>=20 FPS`、抓帧 P95 `<=50ms`、分辨率漂移 `0`；`None` 帧单独计数，不和尺寸漂移混在一起。

### 2.5 显式 armed

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\vision.ps1 `
  -Mode Armed `
  -Config .\configs\controlled-window.json `
  -ConfirmArmed
```

armed 模式同时要求：

1. 命令行明确传入 `-ConfirmArmed`；
2. 目标窗口标题精确匹配；
3. 启动时解析出的 HWND 与当前前台 HWND 一致；
4. `F12` 未按下；
5. 每个按键持有时间没有超过配置上限；
6. 每次 `SendInput` 返回的插入事件数等于 `1`。

任何条件不满足都会阻止新输入并尝试释放已按下的键。

## 3. 直接运行 Python 模块

完成 Setup 后，可以跳过 PowerShell 包装层：

```powershell
# 测试窗口
uv run python -m delta_vision.controlled_target `
  --artifacts artifacts\manual\target

# Worker 默认 dry-run
uv run python -m delta_vision.worker `
  --config configs\controlled-window.json `
  --artifacts artifacts\manual\worker

# 显式标准输入
uv run python -m delta_vision.worker `
  --config configs\controlled-window.json `
  --artifacts artifacts\manual\worker-armed `
  --armed
```

Worker 退出码：

| 退出码 | 含义 |
| --- | --- |
| `0` | 连续视觉帧确认到达目标 |
| `1` | 配置、窗口、采集、输入或运行异常 |
| `2` | 安全停止或在限制时间内未到达 |

## 4. 配置说明

可运行示例位于 `configs/controlled-window.json`。

### 4.1 窗口与采集

```json
{
  "target_window_title": "Delta Vision Test Target",
  "capture_backend": "dxcam",
  "emergency_virtual_key": 123,
  "max_key_hold_ms": 250,
  "loop_interval_ms": 20,
  "max_duration_seconds": 15
}
```

- `target_window_title`：必须完整匹配顶层窗口标题；
- `capture_backend`：优先 `dxcam`，兼容性回退可改为 `mss`；
- `emergency_virtual_key=123`：Win32 `VK_F12`；
- `max_key_hold_ms`：硬性卡键上限；
- `loop_interval_ms`：控制循环间隔；
- `max_duration_seconds`：单次运行总时限。

### 4.2 视觉锚点

```json
{
  "marker": {
    "bgr": [0, 255, 0],
    "tolerance": 8,
    "minimum_area": 200,
    "confidence_threshold": 0.9
  },
  "localization_radius": 18
}
```

首个受控 E2E 使用可解释的颜色连通域，而不是预训练模型。低于阈值时，检测器只保留诊断候选位置，不向导航状态机提供可执行 centroid，也不会发送新按键。

### 4.3 路线图与边动作

```json
{
  "goal_node_id": "goal",
  "nodes": {
    "start": {
      "x": 80,
      "y": 520,
      "edges": [{"target": "turn", "cost": 1}]
    },
    "turn": {
      "x": 80,
      "y": 80,
      "edges": [{"target": "goal", "cost": 1}]
    },
    "goal": {"x": 700, "y": 80, "edges": []}
  },
  "edge_actions": [
    {"source": "start", "target": "turn", "key": "w"},
    {"source": "turn", "target": "goal", "key": "d"}
  ]
}
```

节点坐标使用目标窗口客户区像素。A* 只决定最低代价节点序列；具体按键由 `edge_actions` 明确配置。缺失边动作时 Worker 会在发送输入前停止。

### 4.4 导航与恢复

```json
{
  "navigation": {
    "pulse_ms": 100,
    "min_progress_px": 4,
    "stuck_after_ms": 600,
    "localization_timeout_ms": 800,
    "max_recovery_attempts": 2,
    "recovery_keys": ["s", "a"],
    "arrival_confirmations": 2
  }
}
```

- 动作是非阻塞短脉冲；timer 只能释放到期按键，不能在没有新截图时生成新输入；
- 卡住只由锚点到下一 waypoint 的视觉距离长期不改善产生，不接受外部 `stuck=true`；
- 恢复次数有硬上限；耗尽后进入 `stopped`；
- 到达目标需要连续视觉确认，单帧不会直接成功。

## 5. 运行产物

默认目录：

```text
artifacts/runs/YYYYMMDD-HHMMSS/
├── target/
│   └── target-ground-truth.jsonl
└── worker/
    ├── events.jsonl
    └── replay/
        ├── manifest.jsonl
        └── frames/
            └── frame-XXXXXXXX.png
```

`manifest.jsonl` 保存 sequence、单调时间戳、来源、宽高和当帧导航/动作元数据。`ReplayFrameSource` 会拒绝时间倒退、路径逃逸、损坏图像和清单分辨率不一致。

不要让 Worker 读取 `target-ground-truth.jsonl`。它只供启动脚本和独立评估器在 Worker 结束后核对真实到达状态；视觉状态与 ground truth 任一未到达，受控 E2E 都会失败。

## 6. 开发与测试

```bash
uv sync --frozen --python 3.12
uv run pytest -q
uv run ruff check python python_tests
uv build
```

旧 TypeScript 模拟器仍保留为历史回归基线：

```bash
npm ci
npm test
npm run typecheck
npm run build
```

它不属于外部视觉 Worker 的完成证据。

## 7. Windows 常见问题

### 找不到 `uv.exe`

先运行 `vision.ps1 -Mode Setup`。脚本会检查 PATH、`%USERPROFILE%\.local\bin` 和 WinGet Links。官方安装说明见 [Astral uv Installation](https://docs.astral.sh/uv/getting-started/installation/)。

### `找不到窗口`

确认配置标题与目标窗口标题逐字一致。受控示例必须是：

```text
Delta Vision Test Target
```

### `前台窗口不是目标窗口` 或 `窗口句柄不是目标窗口`

把目标窗口切到前台后重新运行。安全门故意不使用模糊标题，也不会自动向后台窗口发输入。

### 截图黑屏或 DXcam 初始化失败

1. 必须从 Windows 可见交互桌面运行，不能在 SSH/服务会话中把虚拟桌面截图当真实结果；
2. 检查目标窗口没有最小化；
3. 将配置中的 `capture_backend` 临时改为 `mss`，只说明该配置的兼容性结果；
4. 保留错误日志和截图，不要由一次失败推断目标程序“根本不支持”。

### `SendInput` 返回 0 或窗口没有响应

确认 Worker 与目标窗口处于相同完整性级别，并检查目标是否真的位于前台。项目不会在标准 `SendInput` 被忽略时升级为驱动、注入或规避方案。

### 双击后旧网页仍出现

不要运行 `start-demo.cmd`。它属于旧模拟 Web Demo。外部视觉入口是 `start-controlled-e2e.cmd`，主程序没有浏览器 URL。

## 8. 《三角洲行动》适配流程

当前仓库没有把受控窗口坐标伪装成游戏可用路线。实际适配应按下面顺序进行：

1. 固定分辨率、显示模式、地图区域和一条无战斗测试路线；
2. 只录制人工行走的外部截图和动作时间戳；
3. 按整次运行切分训练/验证数据，避免相邻帧泄漏；
4. 标注视觉锚点、路线节点、交互提示和失败帧；
5. 先在 replay 上评估 precision、recall、F1 和路线序列稳定性；
6. 再以 dry-run 观察动作决策；
7. 最后在用户授权的受控场景中显式 armed，先验证一个移动键和一次小转向；
8. 只有完整保存截图、观测、决策、输入和最终状态后，才能报告该固定路线的成功率。

准确率不能从 GitHub 项目介绍、单张截图或模拟器测试推算。首个游戏 MVP 的最低目标和证据要求记录在 `plan.md` 与 `feature-list.json`。

## 9. 研究依据与边界

源码级方案调研见：

- `docs/research/2026-07-15-external-vision-options.md`

当前实现的主要取舍：

1. DXcam 作为 Windows DXGI 主采集路径，MSS 作为 GDI 回退；
2. OpenCV 只做后处理和首版可解释视觉；
3. 自己封装最小 Win32 `SendInput`，检查返回数量并维护 pressed-key registry；
4. BetterGI 等 GPL 项目只研究架构，不复制其源码；
5. 无明确 License 的仓库只参考概念，不复制代码。

在线使用可能违反目标产品条款并导致账号处罚。请只在你有权控制的环境、账号和受控场景中运行。

## 10. 旧 Web Demo

仓库保留了早期 `Shadow Runner Lab`：React/Vite 页面、Node REST/WebSocket、模拟观测和 TypeScript A*。它可以通过旧 `init.ps1` / `start-demo.cmd` 启动，但只用于历史回归和监控资产复用。

旧页面、HTTP 200、WebSocket 在线、模拟路线完成或浏览器 E2E 都不能替代以下证据：

1. Windows 真实桌面持续截图；
2. 来自截图的视觉定位；
3. 真实标准输入返回数量；
4. 前台切换与急停释放时间；
5. 独立评估器复跑受控窗口和授权游戏路线。

## 11. 仓库状态

- 重构计划：`plan.md`
- 逐项验收：`feature-list.json`
- 可续做进度：`Codex-progress.txt`
- 调研报告：`docs/research/2026-07-15-external-vision-options.md`
- 受控示例配置：`configs/controlled-window.json`

功能只有在对应证据达到阈值后才会在 `feature-list.json` 标为 `passes: true`。
