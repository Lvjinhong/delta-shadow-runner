# 《三角洲行动》外部视觉跑图脚本

先说明白：**这不是一个 Web 平台，没有网页、Dashboard 或访问 URL。**

这是一个 Windows 纯 Python 脚本，目标是按固定路线完成：

```text
游戏窗口截图
→ OpenCV NCC 模板或 ORB/SIFT 局部特征识别当前路线位置
→ waypoint + A* 选择后续路线
→ 每条路线边执行一次相对鼠标转向
→ 用有时限的 W/A/S/D/E/Shift/Space 短脉冲移动或交互
→ 根据下一帧重新定位、继续、恢复或停止
→ 保存截图、决策和实际输入事件
```

它不读取游戏进程，不扫描内存，不注入 DLL，不安装驱动，不修改游戏文件，也不包含自动瞄准、自动射击、敌人识别或反作弊规避。

## 1. 当前做到什么程度

已经实现并在本机离线验证：

1. DXcam 截图，MSS 兼容回退；
2. 人工路线截图采样；
3. 固定 ROI 的 NCC 多尺度模板匹配，以及可选的 ORB/SIFT + KNN ratio test + RANSAC 几何校验；所有后端都按 waypoint 聚合多张外观模板并在歧义时拒绝输出；
4. 路线建立前全局定位；路线建立后优先只匹配当前/下一节点，主候选失败时才诊断已知非相邻节点并立即停机；
5. 模板位置到 waypoint/A* 路线的映射；
6. 每条路线边一次相对鼠标转向，同一边只重复有界按键脉冲；
7. 初始位置 3 帧确认、节点推进 2 帧确认、失去定位后 3 帧重定位确认，以及低置信停止、卡住恢复和连续到达确认；
8. dry-run 与 Win32 `SendInput` armed 模式；
9. 前台窗口标题 + HWND、F12 急停、最大按键时长和后台释放 watchdog；
10. PNG/JSONL 回放、独立输入事件流和 blind set 准确度评估；
11. 标定 Profile → 独立 blind 评估（含负样本）→ schema v2 Worker → 动作回放的单条离线 E2E；
12. 自动测试、Ruff、sdist/wheel build 和独立控制链验收。最新的精确测试数与覆盖率以仓库当前测试输出和 `Codex-progress.txt` 为准。

尚未完成的是《三角洲行动》真实路线数据和 Windows 真机 E2E。仓库不能凭合成数据承诺游戏准确率；必须先在你的固定分辨率、固定地图、固定出生区域和固定视角上采样并评估。

## 2. Windows 准备

环境要求：

1. Windows 10/11 x64；
2. 在可见的交互桌面运行，不能用 SSH/服务会话的虚拟桌面代替；
3. 游戏使用固定分辨率、显示模式、HUD 缩放和鼠标灵敏度；
4. 路线首次适配时使用你有权控制的环境和账号。

安装依赖：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\vision.ps1 -Mode Setup
```

脚本会安装固定版本 `uv 0.11.28`、Python 3.12，并按 `uv.lock` 同步依赖。`ExecutionPolicy Bypass` 只作用于这次 PowerShell 进程。

### 2.1 一键 Windows Preflight

完成 `configs\game-route.json` 和 `profiles\route-01\templates.json` 后，先打开游戏并停在稳定、可见的路线画面，再双击：

```text
start-windows-preflight.cmd
```

它会在同一次带时间戳的运行中依次完成：

1. 安装/核对锁定的 uv、Python 和依赖；
2. 只读校验游戏路线配置、模板 Profile 和来源哈希，不打开游戏进程；
3. 启动独立受控窗口，实际验证截图 → 识别 → A* → `SendInput` → ground truth 到达；
4. 留出 5 秒切回游戏，对游戏客户区执行 60 秒截图 benchmark，不发送游戏输入；
5. 重新计算并核对所有门槛，生成 `preflight-report.json`。

截图基准任一条件不满足即失败：实测时长 `< 60s`、无帧次数非零、近全黑帧非零、平均 FPS `< 20`、P95 抓帧延迟 `> 50ms`、分辨率漂移非零、目标窗口曾离开前台。指标 schema、`frame_count / duration` 与 FPS、延迟顺序、分辨率和有限数也会重新校验，不能只改 JSON 中的 `passed`。

同一次 Preflight 使用唯一 `run_id` 绑定截图、Worker 和 ground truth；报告还会核对证据新鲜度、首末帧 SHA-256、事件单调时间、按键重放状态和最终事件顺序。受控 Worker 未到达、终态后还有输入、未观察到完整 `key_down/key_up`、ground truth 没有合法 `position/arrived`、退出码非零或任一证据不一致都会失败。

Preflight 的 `SendInput` 只发给标题和 HWND 都经过校验的独立测试窗口；游戏阶段只截图。F12 仍是急停键。DryRun、Armed、ControlledE2E 和 Preflight 不能并行启动，重复运行会被 Windows mutex 拒绝。

## 3. 端到端制作一条游戏路线

首版不要覆盖整张地图。先选择一条可重复路线，例如：同一出生区域 → 一个转角 → 一个固定交互点或撤离点。

### 3.1 采集 calibration 运行

先打开游戏并进入路线起点。运行下面的命令后有 5 秒切回游戏，然后由你人工完整走一遍路线；采样器只截图，不发送任何输入。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\vision.ps1 `
  -Mode Sample `
  -WindowTitle "三角洲行动" `
  -Backend dxcam `
  -Split calibration `
  -RunId route01-cal `
  -Dataset artifacts\datasets\route01-cal `
  -Duration 120 `
  -SampleFps 5 `
  -StartDelay 5
```

如果窗口标题不同，必须把 `-WindowTitle` 和配置中的 `target_window_title` 改为任务管理器/窗口实际显示的完整标题。

采样产物：

```text
artifacts/datasets/route01-cal/
├── run.json
├── manifest.jsonl
└── frames/
    ├── frame-00000000.png
    └── ...
```

### 3.2 为关键帧写 calibration 标签

查看 `manifest.jsonl` 和对应 PNG，选择能区分路线位置的关键帧。优先截取稳定的小地图、地标或交互提示区域，不要把整张动态画面都当模板。

新建 `labels/route01-cal.jsonl`，每行一个 JSON 对象：

```jsonl
{"run_id":"route01-cal","sequence":0,"split":"calibration","locatable":true,"template_id":"start-00","roi":{"id":"scene","x":0.70,"y":0.02,"width":0.28,"height":0.28},"route_position":[0,0],"waypoint_id":"start"}
{"run_id":"route01-cal","sequence":30,"split":"calibration","locatable":true,"template_id":"before-turn-00","roi":{"id":"scene","x":0.70,"y":0.02,"width":0.28,"height":0.28},"route_position":[80,0],"waypoint_id":"start"}
{"run_id":"route01-cal","sequence":45,"split":"calibration","locatable":true,"template_id":"turn-00","roi":{"id":"scene","x":0.70,"y":0.02,"width":0.28,"height":0.28},"route_position":[100,0],"waypoint_id":"turn"}
{"run_id":"route01-cal","sequence":90,"split":"calibration","locatable":true,"template_id":"goal-00","roi":{"id":"scene","x":0.70,"y":0.02,"width":0.28,"height":0.28},"route_position":[200,0],"waypoint_id":"goal"}
```

字段含义：

1. `sequence` 必须存在于这次运行的 `manifest.jsonl`；
2. ROI 使用 0..1 的归一化坐标，且同一个 `roi.id` 的定义必须一致；
3. `route_position` 是路线画布坐标，不是屏幕坐标，单位可自定义但必须全程一致；
4. 行进中的模板把 `waypoint_id` 设为当前路线段的起点；真正跨过节点后再切到下一 waypoint；
5. 同一地段建议从不同人工运行中选多个外观模板，但 calibration 与 blind 运行不能混用；
6. 给 Worker 使用的模板必须填写 `waypoint_id`。路线建立后的在线匹配会忽略 `waypoint_id=null` 的模板，避免无归属模板绕过路线约束；正常候选只含当前/下一节点，只有正常候选拒绝后才扫描其他有归属模板，用于识别已知偏航并安全停机，不会拿它们继续规划。

### 3.3 生成 NCC 或局部特征 Profile

首次标定建议同时生成 NCC 与 SIFT 两个独立 Profile。二者必须使用相同 calibration 数据，并在同一个 blind 数据集上对比；不要用 calibration 成绩选后端。

NCC 基线（默认）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\vision.ps1 `
  -Mode Calibrate `
  -Dataset artifacts\datasets\route01-cal `
  -Labels labels\route01-cal.jsonl `
  -ProfilePath profiles\route-01-ncc `
  -FeatureBackend ncc
```

SIFT + RANSAC 对照：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\vision.ps1 `
  -Mode Calibrate `
  -Dataset artifacts\datasets\route01-cal `
  -Labels labels\route01-cal.jsonl `
  -ProfilePath profiles\route-01-sift `
  -FeatureBackend sift `
  -MaximumFeatures 3000
```

成功后生成：

```text
profiles/route-01-sift/
├── templates.json
└── templates/
```

Profile 会绑定来源 run、sequence、帧像素、ROI、模板和清单的 SHA-256；SIFT/ORB Profile 还会记录后端、最大特征数、匹配点数、inlier 数/比例、重投影误差和投影区域门槛。运行时不会在不同后端之间直接比较 raw score，也不会在多个 waypoint 同时通过几何校验时强行选一个。旧的 Profile v1 必须重新标定。

选择建议：NCC 适合作为固定视角基线；SIFT 通常更能容忍缩放、旋转和视角变化；ORB 速度更高且不依赖 SIFT 描述子。最终选择必须由相同 blind set 的指标和完整路线 dry-run 共同决定，仓库不会预设某个后端在你的游戏画面上一定更准。

### 3.4 采集独立 blind 运行

重新进入同一路线，另走一遍，不能复制 calibration 图片或从同一运行抽相邻帧冒充 blind set：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\vision.ps1 `
  -Mode Sample `
  -WindowTitle "三角洲行动" `
  -Split blind `
  -RunId route01-blind `
  -Dataset artifacts\datasets\route01-blind `
  -Duration 120 `
  -SampleFps 5
```

为 blind 数据集的**每一帧**写标签。可定位帧示例：

```jsonl
{"run_id":"route01-blind","sequence":0,"split":"blind","locatable":true,"route_position":[0,0],"expected_waypoint_id":"start"}
```

不可可靠定位的帧必须明确写成：

```jsonl
{"run_id":"route01-blind","sequence":1,"split":"blind","locatable":false,"route_position":null,"expected_waypoint_id":null}
```

### 3.5 计算准确度

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\vision.ps1 `
  -Mode Evaluate `
  -Split blind `
  -Dataset artifacts\datasets\route01-blind `
  -Labels labels\route01-blind.jsonl `
  -ProfilePath profiles\route-01-ncc\templates.json `
  -Artifacts artifacts\evaluations\route01-blind-ncc `
  -DistanceTolerance 25
```

再把 `-ProfilePath` 与 `-Artifacts` 分别改为 `profiles\route-01-sift\templates.json` 和 `artifacts\evaluations\route01-blind-sift` 重跑一次。只有输入 blind 数据集、标签和容差完全相同，NCC/SIFT 的 A/B 结果才可比较。

输出 `metrics.json` 和 `predictions.jsonl`，包括：

1. waypoint top-1 accuracy；
2. 路线位置精确命中率和容差命中率；
3. pose emission precision、recall、F1 和 balanced accuracy；
4. false-lock rate；
5. 路线位置误差 median、P90、P95 和 max；
6. 标签覆盖率、数据集内容哈希和泄漏检查结果。

`metrics.json.observation_scope` 固定为 `unconstrained_perception`：这里测的是不使用路线真值、遍历所有有归属模板的全局感知能力，不能用标签里的期望节点给匹配器缩小候选集。Worker 在线运行时会在首次全局定位后，把当前/下一节点作为正常候选；仅当正常候选拒绝后，才扫描其他有归属模板以诊断已知偏航并停机。同一 waypoint 的多张模板会先聚合，再与其他 waypoint 计算 margin。这部分效果必须由完整路线 dry-run/armed E2E 另行统计，不能拿全局感知指标冒充在线路线成功率。

首个固定路线建议门槛是 pose F1 `>= 0.90`、false-lock `<= 0.5%`，再做多次完整路线成功率测试。这是验收目标，不是当前仓库已经测出的游戏准确率。

## 4. 配置路线和键鼠动作

复制安全模板：

```cmd
copy configs\game-route.example.json configs\game-route.json
```

编辑 `configs\game-route.json`：

```json
{
  "armed_ready": false,
  "navigation": {
    "arrival_confirmations": 3,
    "initial_waypoint_confirmations": 3,
    "waypoint_advance_confirmations": 2,
    "relocalization_confirmations": 3
  },
  "nodes": {
    "start": {"x": 0, "y": 0, "edges": [{"target": "turn", "cost": 1}]},
    "turn": {"x": 100, "y": 0, "edges": [{"target": "goal", "cost": 1}]},
    "goal": {"x": 200, "y": 0, "edges": []}
  },
  "edge_actions": [
    {
      "source": "start",
      "target": "turn",
      "key": "w",
      "mouse_dx": 0,
      "mouse_dy": 0
    },
    {
      "source": "turn",
      "target": "goal",
      "key": "w",
      "mouse_dx": 0,
      "mouse_dy": 0
    }
  ]
}
```

动作语义：

1. 进入一条新边时，`mouse_dx/mouse_dy` 只执行一次；
2. `key` 以短脉冲重复；初始 waypoint 连续确认 3 帧后才允许首次动作，下一 waypoint 连续确认 2 帧后才推进，丢失定位后必须连续确认 3 帧才恢复；
3. `mouse_dx` 是相对计数，不是角度，必须在固定游戏灵敏度和 FOV 下实测；
4. `e` 可用于“交互点 → 交互完成状态”的单独路线边；
5. 支持键为 `w/a/s/d/e/shift/space`；鼠标轴范围为 `-4096..4096`；
6. 示例全部是占位值，不能直接当作真实地图路线。

## 5. 运行

首次运行或修改分辨率、HUD、模板、路线配置后，应先让 `start-windows-preflight.cmd` 通过。Preflight 只证明 Windows 截图和受控输入链可用，不代表真实游戏路线已经准确。

双击：

```text
start-game-route.cmd
```

菜单提供：

1. `D`：dry-run，执行真实截图、识别、A*、状态机和回放，但不发送键鼠输入；
2. `A`：armed，向前台目标窗口发送标准 Win32 输入；
3. `Q`：退出。

armed 有三层显式确认：

1. `game-route.json` 的 `armed_ready` 必须由你在完成 blind 评估和 dry-run 后改为 `true`；
2. `.cmd` 会再次要求 `Y/N` 确认；
3. Worker 启动时检查窗口精确标题、启动 HWND、当前前台 HWND 和 F12。

也可以直接运行：

```powershell
# 只观察和记录
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\vision.ps1 -Mode DryRun -Config configs\game-route.json

# 显式发送标准输入
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\vision.ps1 -Mode Armed -Config configs\game-route.json -ConfirmArmed
```

armed 前请切回游戏窗口。F12 是急停键，切走窗口也会阻止新输入并释放已按下的键。

Worker 退出码：

| 退出码 | 含义 |
| --- | --- |
| `0` | 连续截图确认到达目标 |
| `1` | 配置、窗口、采集、输入或运行异常 |
| `2` | 安全停止或在时限内未到达 |

## 6. 运行证据

```text
artifacts/runs/YYYYMMDD-HHMMSS/worker/
├── events.jsonl
└── replay/
    ├── input-events.jsonl
    ├── manifest.jsonl
    └── frames/
        └── frame-XXXXXXXX.png
```

`input-events.jsonl` 独立记录每个 `mouse_move/key_down/key_up`。即使最后一次输入发生在无截图、超时或异常退出路径，也不依赖“下一张截图”才能落盘。

每次 Worker 运行都会清空该运行目录中的旧 `events.jsonl`，并给事件写入本次 `run_id`。不要手工把不同运行目录的证据拼成一份报告；Preflight 会将其判为失败。

一次 Preflight 还会生成：

```text
artifacts/runs/YYYYMMDD-HHMMSS/
├── preflight-report.json
├── capture-benchmark/
│   ├── capture-metrics.json
│   ├── capture-gate.json
│   ├── first-frame.png
│   └── last-frame.png
└── controlled-e2e/
    ├── target/target-ground-truth.jsonl
    └── worker/
        ├── events.jsonl
        └── replay/
```

`preflight-report.json` 的 `evidence` 会记录上述证据文件的路径、修改时间和 SHA-256；报告通过不等于游戏路线通过，仍需完成真实 blind set 和路线成功率验收。

## 7. 受控自检窗口

`start-controlled-e2e.cmd` 只是验证“截图 → 识别 → A* → 标准输入 → 到达”的独立测试工具，不是产品平台，也不代表游戏路线准确。

在 Windows 双击它，可先检查 DXcam/MSS、窗口守卫和 `SendInput` 是否正常。测试窗口标题是 `Delta Vision Test Target`，F12 同样急停。

## 8. 技术边界和研究依据

详细源码审计见 `docs/research/2026-07-15-external-vision-options.md`。当前组合参考并交叉核验了 DXcam、Python MSS、OpenCV、BetterGI、EDAutopilot、MaaFramework、RapidOCR 和 vis_nav_player 的 README、License 与关键源码；仓库没有复制 GPL 或无明确 License 项目的实现。

当前脚本的能力边界：

1. 是固定路线视觉闭环，不是自动探索整张地图；
2. 画面、HUD、地图版本、遮挡、分辨率、灵敏度和出生偏差都会影响准确度；
3. 标准 `SendInput` 如果被目标忽略，本项目只记录为不支持，不升级为驱动或注入；
4. 在线使用可能违反目标产品条款并导致账号处罚。

## 9. 常见问题

### 截图黑屏

1. 必须从 Windows 可见交互桌面运行；
2. 不要最小化游戏；
3. 先把 `capture_backend` 从 `dxcam` 改为 `mss` 做一次兼容性对照；
4. 查看最近一次 `capture-gate.json` 的 `black_frames`、`missing_frames`、`foreground_window_mismatch`，并直接检查 `first-frame.png`、`last-frame.png`；
5. 一次配置失败只能说明该配置失败，不能据此推断目标程序永远不支持。

### 找不到窗口

窗口标题必须逐字匹配。不要使用模糊标题，也不要指望 Worker 向后台窗口发输入。

### 模板 Profile 分辨率不一致

采样、评估和运行必须使用相同的游戏客户区分辨率。改变分辨率或 HUD 后应重新采样和标定。

### Worker 不发送输入

依次检查：是否选择 armed、`armed_ready` 是否为 `true`、是否传了 `-ConfirmArmed`、游戏是否在前台、F12 是否被按下、Worker 与游戏是否处于相同完整性级别。

## 10. 开发验证

```bash
uv sync --frozen --python 3.12
uv run ruff check python python_tests
uv run pytest -q
uv build
```

仓库是纯 Python CLI，不包含 Node、React、浏览器 Dashboard 或本地 Web 服务。
