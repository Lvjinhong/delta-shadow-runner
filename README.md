# 《三角洲行动》外部视觉跑图脚本

先说明白：**这不是一个 Web 平台，没有网页、Dashboard 或访问 URL。**

这是一个 Windows 纯 Python 脚本，目标是按固定路线完成：

```text
游戏窗口截图
→ OpenCV 模板识别当前路线位置
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
3. 固定 ROI、多尺度模板匹配、歧义拒绝和置信度输出；
4. 模板位置到 waypoint/A* 路线的映射；
5. 每条路线边一次相对鼠标转向，同一边只重复有界按键脉冲；
6. 低置信停止、重定位、卡住恢复和连续到达确认；
7. dry-run 与 Win32 `SendInput` armed 模式；
8. 前台窗口标题 + HWND、F12 急停、最大按键时长和后台释放 watchdog；
9. PNG/JSONL 回放、独立输入事件流和 blind set 准确度评估；
10. 标定 Profile → 独立 blind 评估（含负样本）→ schema v2 Worker → 动作回放的单条离线 E2E；
11. 386 项自动测试通过，总覆盖率 87.78%，独立 10-Gate 控制链验收通过。

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
5. 同一地段建议从不同人工运行中选多个外观模板，但 calibration 与 blind 运行不能混用。

### 3.3 生成模板 Profile

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\vision.ps1 `
  -Mode Calibrate `
  -Dataset artifacts\datasets\route01-cal `
  -Labels labels\route01-cal.jsonl `
  -ProfilePath profiles\route-01
```

成功后生成：

```text
profiles/route-01/
├── templates.json
└── templates/
```

Profile 会绑定来源 run、sequence、帧像素、ROI、模板和清单的 SHA-256；旧的 Profile v1 必须重新标定。

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
  -ProfilePath profiles\route-01\templates.json `
  -Artifacts artifacts\evaluations\route01-blind `
  -DistanceTolerance 25
```

输出 `metrics.json` 和 `predictions.jsonl`，包括：

1. waypoint top-1 accuracy；
2. 路线位置精确命中率和容差命中率；
3. pose emission precision、recall、F1 和 balanced accuracy；
4. false-lock rate；
5. 路线位置误差 median、P90、P95 和 max；
6. 标签覆盖率、数据集内容哈希和泄漏检查结果。

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
2. `key` 以短脉冲重复，直到画面确认进入下一 waypoint；
3. `mouse_dx` 是相对计数，不是角度，必须在固定游戏灵敏度和 FOV 下实测；
4. `e` 可用于“交互点 → 交互完成状态”的单独路线边；
5. 支持键为 `w/a/s/d/e/shift/space`；鼠标轴范围为 `-4096..4096`；
6. 示例全部是占位值，不能直接当作真实地图路线。

## 5. 运行

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
4. 一次配置失败只能说明该配置失败，不能据此推断目标程序永远不支持。

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
