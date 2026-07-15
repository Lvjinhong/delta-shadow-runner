# Windows 外部视觉自动化源码调研

更新时间：2026-07-15

## 1. 结论

首个可落地版本采用下面这条链路：

```text
交互桌面截图
  → 固定分辨率与 ROI 归一化
  → 场景门控 + 模板/局部特征匹配
  → 上次位置约束的路线节点定位
  → waypoint / A* 路线与闭环动作
  → 前台窗口安全门
  → Win32 SendInput
  → 连续画面进度检测与卡住恢复
```

具体选型：

1. 主采集使用 DXcam `0.3.0`；MSS `10.2.0` 作为兼容性和低频截图兜底。
2. OpenCV 只负责颜色转换、模板匹配、ORB/SIFT、几何验证和诊断图，不用 `VideoCapture` 抓桌面。
3. 首版定位使用固定 ROI、模板/局部特征、上一节点约束和 waypoint，不先训练 YOLO，也不做端到端 RL。
4. HUD 文本门控后续可接 RapidOCR；目标检测只在真实样本证明经典方法不够时引入。
5. 输入直接用 Python `ctypes` 封装 Windows `SendInput`，不依赖过时且多屏行为不完整的 PyDirectInput。
6. 现有 TypeScript A*、恢复状态机和监控面保留为可复用资产，但新的截图、感知和输入 Worker 使用 Python。

## 2. 屏幕采集

### 2.1 DXcam：主方案

- 固定版本：`0.3.0`，release commit [`de356cb5`](https://github.com/ra1nty/DXcam/tree/de356cb5a39f50645d495c522fabb03e984728e7)，MIT。
- 项目元数据明确支持 Windows 10/11 和 CPython 3.10～3.14；核心依赖为 `comtypes` 与 NumPy，可选 OpenCV/WinRT：[`pyproject.toml#L5-L49`](https://github.com/ra1nty/DXcam/blob/de356cb5a39f50645d495c522fabb03e984728e7/pyproject.toml#L5-L49)。
- 底层实际调用 Desktop Duplication 的 `DuplicateOutput(1)` 与 `AcquireNextFrame`，并对 access loss / system transition 返回恢复信号：[`dxgi_duplicator.py#L38-L182`](https://github.com/ra1nty/DXcam/blob/de356cb5a39f50645d495c522fabb03e984728e7/dxcam/core/dxgi_duplicator.py#L38-L182)。
- `grab()` 无新帧时允许返回 `None`；线程模式使用 ring buffer，零拷贝 view 会被后续帧覆盖：[`dxcam.py#L208-L255`](https://github.com/ra1nty/DXcam/blob/de356cb5a39f50645d495c522fabb03e984728e7/dxcam/dxcam.py#L208-L255)、[`dxcam.py#L497-L650`](https://github.com/ra1nty/DXcam/blob/de356cb5a39f50645d495c522fabb03e984728e7/dxcam/dxcam.py#L497-L650)。
- 外屏通过 `device_idx/output_idx` 选择；热插拔后必须重新枚举并核对分辨率，不能永久相信旧 index：[`dxcam/__init__.py#L89-L181`](https://github.com/ra1nty/DXcam/blob/de356cb5a39f50645d495c522fabb03e984728e7/dxcam/__init__.py#L89-L181)。
- 默认使用 DXGI；确实需要显示鼠标光标时才切 WinRT。BGRA 是最轻输出路径：[`README.md#L140-L226`](https://github.com/ra1nty/DXcam/blob/de356cb5a39f50645d495c522fabb03e984728e7/README.md#L140-L226)。

微软文档说明 Desktop Duplication 从 Windows 8 起提供 DXGI surface 形式的逐帧桌面更新，返回的图像格式固定为 `DXGI_FORMAT_B8G8R8A8_UNORM`：[Desktop Duplication API](https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/desktop-dup-api)。

DXcam README 的作者自测为 5900X+3090、240Hz 输出、5 次运行，平均 239.19 FPS；这个数字缺少完整环境锁定且来自项目作者，只能用于选型，不能当本项目 SLA：[`README.md#L246-L260`](https://github.com/ra1nty/DXcam/blob/de356cb5a39f50645d495c522fabb03e984728e7/README.md#L246-L260)。

### 2.2 MSS：兜底

- 固定版本：`10.2.0`，commit [`b7f4d62c`](https://github.com/BoboTiG/python-mss/tree/b7f4d62c6c5dc68fd60d43600f0e91012c8bc2e0)，MIT。
- Windows 实现是 GDI `BitBlt + CreateDIBSection`，不是 Desktop Duplication：[`gdi.py#L335-L392`](https://github.com/BoboTiG/python-mss/blob/b7f4d62c6c5dc68fd60d43600f0e91012c8bc2e0/src/mss/windows/gdi.py#L335-L392)。
- 能枚举虚拟桌面和各显示器坐标，并提供稳定 `unique_id`：[`gdi.py#L261-L333`](https://github.com/BoboTiG/python-mss/blob/b7f4d62c6c5dc68fd60d43600f0e91012c8bc2e0/src/mss/windows/gdi.py#L261-L333)。

MSS 适合确定性测试窗口、截图诊断和 DXGI 不可用时降级。全屏 Direct3D 或高帧率链路仍以 DXcam 为主。

## 3. 视觉定位与路线

### 3.1 BetterGI：最接近的完整参考

[BetterGI](https://github.com/babalae/better-genshin-impact/tree/0eb90304c4e4fa1f5cee2a4cbf68de6c8200ec94) commit `0eb9030`，GPL-3.0。它不能被直接复制进当前 MIT/自有实现，但可作为算法与验收参考：

- 地图特征预热、SIFT 全局/局部匹配与上一位置约束：[`SceneBaseMap.cs#L111-L200`](https://github.com/babalae/better-genshin-impact/blob/0eb90304c4e4fa1f5cee2a4cbf68de6c8200ec94/BetterGenshinImpact/GameTask/Common/Map/Maps/Base/SceneBaseMap.cs#L111-L200)。
- 当前坐标、目标角与距离：[`Navigation.cs#L40-L81`](https://github.com/babalae/better-genshin-impact/blob/0eb90304c4e4fa1f5cee2a4cbf68de6c8200ec94/BetterGenshinImpact/GameTask/AutoPathing/Navigation.cs#L40-L81)。
- 截图、定位、转向、前进和重定位闭环：[`PathExecutor.cs#L748-L906`](https://github.com/babalae/better-genshin-impact/blob/0eb90304c4e4fa1f5cee2a4cbf68de6c8200ec94/BetterGenshinImpact/GameTask/AutoPathing/PathExecutor.cs#L748-L906)。
- 卡住后的后退/旋转恢复：[`TrapEscaper.cs#L33-L121`](https://github.com/babalae/better-genshin-impact/blob/0eb90304c4e4fa1f5cee2a4cbf68de6c8200ec94/BetterGenshinImpact/GameTask/AutoPathing/TrapEscaper.cs#L33-L121)。

BetterGI 的小地图尺寸、搜索半径和置信阈值是另一款游戏的经验参数，不能直接声称适用于《三角洲行动》。当前实现只采用“局部优先、连续失败再扩大搜索、仍失败则安全停止”的结构。

### 3.2 可采用或借鉴的辅助项目

| 项目 | 证据 | 用途 | 采用边界 |
| --- | --- | --- | --- |
| [EDAutopilot-v2](https://github.com/Matrixchung/EDAutopilot-v2/tree/eaca754278e8ceb432420e53e3f4234b4950e2a8) | [`gameui.py#L173-L222`](https://github.com/Matrixchung/EDAutopilot-v2/blob/eaca754278e8ceb432420e53e3f4234b4950e2a8/gameui.py#L173-L222)、[`utils.py#L123-L151`](https://github.com/Matrixchung/EDAutopilot-v2/blob/eaca754278e8ceb432420e53e3f4234b4950e2a8/utils/utils.py#L123-L151) | HUD 双模板定位、跳变抑制、离散方向修正 | MIT，可借鉴；仓库维护停滞，不作为依赖 |
| [MaaFramework](https://github.com/MaaXYZ/MaaFramework/tree/76385c8871d8f59c1ca69cd35d8b50b611cd156a) | [`RecognitionTask.cpp#L12-L69`](https://github.com/MaaXYZ/MaaFramework/blob/76385c8871d8f59c1ca69cd35d8b50b611cd156a/source/MaaFramework/Task/RecognitionTask.cpp#L12-L69)、[`ActionTask.cpp#L13-L81`](https://github.com/MaaXYZ/MaaFramework/blob/76385c8871d8f59c1ca69cd35d8b50b611cd156a/source/MaaFramework/Task/ActionTask.cpp#L13-L81) | 识别结果驱动声明式动作 | LGPL-3.0；首版不用其完整框架 |
| [vis_nav_player](https://github.com/ai4ce/vis_nav_player/tree/90c48e3f1a4bee5de93219fec3344ed8a5c3ebb0) | [`baseline.py#L36-L136`](https://github.com/ai4ce/vis_nav_player/blob/90c48e3f1a4bee5de93219fec3344ed8a5c3ebb0/source/baseline.py#L36-L136)、[`baseline.py#L307-L400`](https://github.com/ai4ce/vis_nav_player/blob/90c48e3f1a4bee5de93219fec3344ed8a5c3ebb0/source/baseline.py#L307-L400) | RootSIFT/VLAD 地点识别、时序图和最短路 | 仓库没有 License 文件，仅作概念验证参考，不复制源码 |
| [RapidOCR](https://github.com/RapidAI/RapidOCR/tree/44e2e900eccf2ad0702030dce9e20f5c5941be39) | [`README.md#L34-L68`](https://github.com/RapidAI/RapidOCR/blob/44e2e900eccf2ad0702030dce9e20f5c5941be39/README.md#L34-L68) | HUD/交互提示的中英文 OCR | Apache-2.0；真实样本需要时再接入 |
| [ViZDoom](https://github.com/Farama-Foundation/ViZDoom/tree/748b2c69b0ac51a48629b24b80a8cc1603d12c65) | [`base_gymnasium_env.py#L225-L242`](https://github.com/Farama-Foundation/ViZDoom/blob/748b2c69b0ac51a48629b24b80a8cc1603d12c65/gymnasium_wrapper/base_gymnasium_env.py#L225-L242) | 视觉动作策略与恢复评测原型 | 使用内部 buffer，不能外推外部截图速度或准确率 |

### 3.3 首版定位策略

1. 对目标窗口或显示器建立固定分辨率 profile，所有 ROI 用归一化坐标表达。
2. 受控测试窗口先用高可分离的视觉 anchor 验证完整链路。
3. 实际固定路线由一组 `route node` 组成；每个节点保存参考帧/ROI、动作、邻接边和置信阈值。
4. 当前帧先在上次节点附近做 ORB/SIFT 或模板匹配；用 ratio test、RANSAC inlier 和时序连续性拒绝误锁。
5. 局部搜索失败时扩大候选范围；连续失败进入 `uncertain` 并释放所有按键，不盲走。
6. 路线使用 waypoint look-ahead 与 A*；首版只覆盖一个固定分辨率、一个地图区域和一条人工录制路线。
7. 连续画面位移或路线节点进度长期不变时，松开前进键，再执行受限的后退/横移/小角度转向并重新定位。

## 4. 标准输入与安全门

首选直接封装微软 [SendInput](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput) 和 [`INPUT`](https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-input)：

1. 每次检查返回的成功事件数是否等于提交数。
2. 键盘使用 scan code，down/up 必须成对；异常、急停和进程退出路径都执行 `release_all()`。
3. 输入前检查当前前台窗口标题是否符合允许名单；窗口失焦立即停止。
4. 使用 `GetAsyncKeyState` 轮询急停，不安装键盘 Hook。
5. 单次按键持有时间设置硬上限，控制循环超时也要释放。
6. `SendInput` 受 UIPI 限制，只能注入到相同或更低 integrity level；若目标拒绝 synthetic input，明确报不支持，不升级到驱动、虚拟 HID 或注入方案。

若未来需要绝对鼠标定位，多显示器必须设置 `MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK`，并使用虚拟桌面原点与宽高归一化：[MOUSEINPUT](https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-mouseinput)、[GetSystemMetrics](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getsystemmetrics)。

PyDirectInput 只作为源码对照：其主屏绝对坐标实现没有 `MOUSEEVENTF_VIRTUALDESK`，而且维护停滞：[`__init__.py#L251-L272`](https://github.com/learncodebygaming/pydirectinput/blob/a585d044aed678576fefd24e7ad0c5945ab52366/pydirectinput/__init__.py#L251-L272)、[`__init__.py#L384-L392`](https://github.com/learncodebygaming/pydirectinput/blob/a585d044aed678576fefd24e7ad0c5945ab52366/pydirectinput/__init__.py#L384-L392)。

## 5. Windows 实机基线

只读检查确认：

1. Windows 11 Pro 25H2 build 26200 x64，Ryzen 7 9800X3D，RTX 4070 Ti SUPER 16GB。
2. 交互显示模式为 2560×1440@240Hz。
3. 《三角洲行动》和 WeGame 已安装；没有启动游戏或读取目标进程。
4. DXGI、D3D11 与 Windows Graphics Capture 类型存在。
5. 当前没有真正可用的 Python、`py`、`pip` 或 `uv`；WindowsApps 的 `python.exe` 只是 0 字节别名。
6. SSH 会话位于 `WinDisc` 虚拟桌面，只看到 1024×768；`CopyFromScreen` 实测失败。因此 SSH 只能安装、同步和读日志，真实截图/输入 Worker 必须由 Windows 交互桌面启动。

关键探针均使用 publickey-only SSH，仓库、系统配置和游戏没有被修改。

## 6. 准确率验证

当前没有任何开源项目能给出可直接迁移到《三角洲行动》的准确率。BetterGI 参数、ViZDoom steps/s、OCR 文档指标和 COCO mAP 都不能外推。

实际游戏数据按以下方式建立：

1. 至少录制 30 次完整固定路线，覆盖画质、亮度、动态遮挡和 HUD 状态。
2. 按“整次运行”切分 calibration / validation / blind test；禁止把相邻帧随机分到不同集合。
3. 每秒抽 2～5 帧标注路线节点/位置与朝向；转角、门、楼梯、分叉和易卡点全量标注。
4. 10% 样本双人标注，记录标注分歧。

必须报告：

- 节点 top-1 准确率与相邻节点容忍准确率；
- 位置误差 median / P90 / P95 / max；
- 朝向误差 MAE / P95；
- pose 可用率与 false lock；
- 丢失后的重定位时间；
- P50/P95 感知和端到端延迟；
- 完整路线成功率、人工接管、错分叉、卡住次数与恢复率。

受控测试窗口的硬门槛为固定路线 `20/20`、卡住恢复 `10/10`、急停后 `200ms` 内释放全部按键。实际固定路线的首版目标为 blind test 完整成功率至少 90%、false lock 不高于 0.5%、视觉闭环至少 10Hz；这些是验收目标，不是当前已测结果。

## 7. License 与明确禁区

- 直接依赖只选 MIT/Apache-2.0 兼容项目并固定版本。
- BetterGI（GPL-3.0）、MaaFramework（LGPL-3.0）和无 License 的 `vis_nav_player` 只作研究参考，不复制其源码。
- 不读进程内存、不注入 DLL、不安装驱动/虚拟 HID、不 Hook 图形 API、不绕过 UIPI/UAC secure desktop、不实现反作弊规避。
- 不实现敌人识别、自动瞄准、自动射击或战斗决策。
- 如果标准桌面截图或 `SendInput` 被目标拒绝，项目停在可诊断的“不支持”，不升级到更高风险手段。
