# Windows 外部视觉自动化源码调研

更新时间：2026-07-16

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
6. 项目已收口为纯 Python CLI：路线规划、截图、感知、安全输入、采样、标定和评估都在同一个可回放链路中，不再保留 Web/Node 平台。

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

## 3. GitHub 公开源码审计

### 3.1 《三角洲行动》相关项目

| 项目 | 实际实现 | 对本项目的价值 | 边界 |
| --- | --- | --- | --- |
| [DeltaForceScript `4679841`](https://github.com/BugNotFoundX/DeltaForceScript/tree/46798410d6166abe9faf25810b98037929858391) | DXcam 截图、PaddleOCR 识别倒计时、PyDirectInput 点击市场购买：[`window_capture.py#L26-L47`](https://github.com/BugNotFoundX/DeltaForceScript/blob/46798410d6166abe9faf25810b98037929858391/window_capture.py#L26-L47)、[`main_gui.py#L100-L205`](https://github.com/BugNotFoundX/DeltaForceScript/blob/46798410d6166abe9faf25810b98037929858391/main_gui.py#L100-L205) | 证明已有 Delta 项目公开实现了外部截图、OCR、标准键鼠链路 | 只有市场 UI 自动化，无局内定位/导航；仓库未提供运行成功率；无 License，不复制源码 |
| [DeltaForceSS `92c33f3`](https://github.com/yi-zelin/DeltaForceSS/tree/92c33f326691cc5d19e560af5dad92cf89b17f00) | DXcam、灰度/Otsu、OCR/模糊匹配、Canny/Hough UI 分割：[`main.py#L189-L257`](https://github.com/yi-zelin/DeltaForceSS/blob/92c33f326691cc5d19e560af5dad92cf89b17f00/main.py#L189-L257)、[`main.py#L304-L358`](https://github.com/yi-zelin/DeltaForceSS/blob/92c33f326691cc5d19e560af5dad92cf89b17f00/main.py#L304-L358)、[`main.py#L468-L503`](https://github.com/yi-zelin/DeltaForceSS/blob/92c33f326691cc5d19e560af5dad92cf89b17f00/main.py#L468-L503) | 固定 ROI 预处理与 OCR 容错可借鉴 | 只有制造/购买；GPL-3.0，不直接拷贝到当前实现 |
| [GTImaster `91d6579`](https://github.com/screenpandar/GTImaster/tree/91d65794ca6a6bb934d001fb8586cd81ddf995f3) | MSS/EasyOCR，对明显异常的 OCR 数值拒绝操作：[`realtime_mode.py#L115-L267`](https://github.com/screenpandar/GTImaster/blob/91d65794ca6a6bb934d001fb8586cd81ddf995f3/core/realtime_mode.py#L115-L267) | “低置信度不执行”和结果一致性检查 | 只有市场交易；MIT |
| [delta-force-skill `9c57e13`](https://github.com/doyoulovemeforhi/delta-force-skill/tree/9c57e13f04c1a7c8e86ffe627b9e5c018b543801) | GDI BitBlt 截图、归一化 ROI 多尺度模板、SendInput/PostMessage：[`screenshot.py#L132-L203`](https://github.com/doyoulovemeforhi/delta-force-skill/blob/9c57e13f04c1a7c8e86ffe627b9e5c018b543801/scripts/screenshot.py#L132-L203)、[`recognition.py#L95-L182`](https://github.com/doyoulovemeforhi/delta-force-skill/blob/9c57e13f04c1a7c8e86ffe627b9e5c018b543801/scripts/recognition.py#L95-L182)、[`click.py#L78-L98`](https://github.com/doyoulovemeforhi/delta-force-skill/blob/9c57e13f04c1a7c8e86ffe627b9e5c018b543801/scripts/click.py#L78-L98) | 与当前 Profile 的 ROI/多尺度模板思路一致 | 基地/UI 操作，且会[枚举进程与可执行文件路径](https://github.com/doyoulovemeforhi/delta-force-skill/blob/9c57e13f04c1a7c8e86ffe627b9e5c018b543801/scripts/games/wegame_delta_force.py#L324-L331)；没有读内存，但不符合本项目“不接触进程”的更严边界；无 License |

直接命中“跑刀”的 [delta_force_auto_LootRun `fb16bfe`](https://github.com/zdu881/delta_force_auto_LootRun/tree/fb16bfe9f4b118a841a0c57f138a8a20a4c93c66) 并不是可完整审计的开源实现：README 明确说附件来自“自动精灵”，base64 解码后仍有加密：[`README.md#L3-L13`](https://github.com/zdu881/delta_force_auto_LootRun/blob/fb16bfe9f4b118a841a0c57f138a8a20a4c93c66/README.md#L3-L13)。对固定 commit 中 870,461 字节的 [`.zjs` 原文件](https://github.com/zdu881/delta_force_auto_LootRun/blob/fb16bfe9f4b118a841a0c57f138a8a20a4c93c66/%E4%B8%89%E8%A7%92%E6%B4%B2%E5%85%8D%E8%B4%B9%E5%85%A8%E8%87%AA%E5%8A%A8%E8%B7%91%E5%88%80%E5%8F%8C%E6%8E%92%E7%BB%84%E9%98%9F%E7%89%88.zjs) 做结构化读取，可解析出 29 条顶层 JSON 记录、26 个 `linkedFile` 名称和嵌套元数据；嵌套内容共有 82 条换行分隔 JSON，其中 56 条标记为 `type="加密动作"`。因此文件结构、脚本名称和元数据可见，但加密动作 payload 的实际行为无法审计，也没有可复现准确率。

该仓库链接的 [Visual-SIFT-Template-Matching-Method `1ca265b`](https://github.com/kongkong985/Visual-SIFT-Template-Matching-Method/tree/1ca265b59d433848421e909c882f109c6912dd5e) 仅包含一个小地图对大地图的 SIFT 示例：KNN + `0.75` ratio test，至少 10 个 good matches 后用 RANSAC homography 转换中心点：[`SIFT 实现#L20-L67`](https://github.com/kongkong985/Visual-SIFT-Template-Matching-Method/blob/1ca265b59d433848421e909c882f109c6912dd5e/%E5%AE%9A%E4%BD%8D%E7%9B%AE%E6%A0%87%E5%9B%BE%E5%83%8F%E5%9C%A8%E6%A8%A1%E6%9D%BF%E5%9B%BE%E5%83%8F%E4%B8%AD%E7%9A%84%E5%9D%90%E6%A0%87%E4%BD%8D%E7%BD%AE_SIFT%E6%A8%A1%E6%9D%BF%E5%8C%B9%E9%85%8D%E6%B3%95.py#L20-L67)。它没有 License，也没检查空 homography、inlier ratio 或时序跳变，只能证明一个简化的定位方向。

因此，在本次检查到的 GitHub 结果中，有《三角洲行动》的截图/OCR/UI 自动化，也有不可审计的加密跑刀附件，但没有找到可审计、可复现的“局内截图定位 → 路线跟踪 → 键鼠闭环”完整开源实现。

### 3.2 其他游戏的导航参考

[BetterGI](https://github.com/babalae/better-genshin-impact/tree/7e30466378d2d951fdb09fd9f9643adc8713d469) commit `7e304663`，GPL-3.0，是最完整的结构参考：

- SIFT 全局/局部匹配，优先搜索上一帧位置附近，失败后回退全局：[`SceneBaseMap.cs#L116-L212`](https://github.com/babalae/better-genshin-impact/blob/7e30466378d2d951fdb09fd9f9643adc8713d469/BetterGenshinImpact/GameTask/Common/Map/Maps/Base/SceneBaseMap.cs#L116-L212)。
- 持续截图、定位、计算距离和调整朝向：[`PathExecutor.cs#L748-L877`](https://github.com/babalae/better-genshin-impact/blob/7e30466378d2d951fdb09fd9f9643adc8713d469/BetterGenshinImpact/GameTask/AutoPathing/PathExecutor.cs#L748-L877)。
- 异常位置跳变拒绝、无进展检测和有限脱困：[`TrapEscaper.cs#L33-L121`](https://github.com/babalae/better-genshin-impact/blob/7e30466378d2d951fdb09fd9f9643adc8713d469/BetterGenshinImpact/GameTask/AutoPathing/TrapEscaper.cs#L33-L121)。

[AeronauticaHelper](https://github.com/SSkipr/AeronauticaHelper/tree/7bbb84a379b85a16542e43ffd0065340e15a68a6) commit `7bbb84a`，MIT，补足了闭环转向和恢复策略：用[最短有符号角差和 EWMA 平滑](https://github.com/SSkipr/AeronauticaHelper/blob/7bbb84a379b85a16542e43ffd0065340e15a68a6/AeroHelper/utils/bearing.py#L25-L73)，再以小角度平方根、大角度线性的[连续分段模型叠加油门修正](https://github.com/SSkipr/AeronauticaHelper/blob/7bbb84a379b85a16542e43ffd0065340e15a68a6/AeroHelper/automation/autosteer.py#L164-L194)控制按键时长；另有[距离停滞检测](https://github.com/SSkipr/AeronauticaHelper/blob/7bbb84a379b85a16542e43ffd0065340e15a68a6/AeroHelper/automation/autosteer.py#L257-L280)。它依赖可 OCR 的 HUD 航向/距离，不能直接解决 FPS 场景定位。

| 项目 | 主要用途 | 采用边界 |
| --- | --- | --- |
| [EDAutopilot-v2 `eaca754`](https://github.com/Matrixchung/EDAutopilot-v2/tree/eaca754278e8ceb432420e53e3f4234b4950e2a8) | HUD 双模板定位、上一帧跳变抑制、SendInput：[`gameui.py#L173-L226`](https://github.com/Matrixchung/EDAutopilot-v2/blob/eaca754278e8ceb432420e53e3f4234b4950e2a8/gameui.py#L173-L226) | MIT；同时读取游戏官方 Journal，不是纯视觉 |
| [MaaFramework `76385c8`](https://github.com/MaaXYZ/MaaFramework/tree/76385c8871d8f59c1ca69cd35d8b50b611cd156a) | 识别结果驱动声明式动作 | LGPL-3.0；首版不引入整个框架 |
| [RapidOCR `44e2e90`](https://github.com/RapidAI/RapidOCR/tree/44e2e900eccf2ad0702030dce9e20f5c5941be39) | HUD/交互提示 OCR | Apache-2.0；真实样本证明需要时再引入 |

### 3.3 当前落地策略

1. 对每个真实固定路线建立独立分辨率 Profile，ROI 用归一化坐标，模板保存 SHA-256，并绑定来源 run ID、帧序号和解码像素哈希。
2. 人工走一遍以 2～5 FPS 采样，对转角、门、楼梯、分叉和易卡点标注 route position/waypoint。
3. 当前帧先在上次节点附近搜索；候选置信度不足、发生不合理跳变或连续丢失时释放全部按键，不盲走。
4. A* 只选择 waypoint 序列；每次输入是有上限的短脉冲，下一帧画面决定是否继续、转向或停止。
5. 定位长期不前进时只允许有限次数的后退/横移/小角度转向，耗尽恢复预算后安全停止。
6. 采样时就声明 calibration/validation/blind，Profile 保存标定运行全部帧、感知 ROI、数据集像素内容、`run.json` 和 `manifest.jsonl` 的哈希；评估器拒绝 run ID、整帧或感知 ROI 重用，核对声明帧数，并强制标签 100% 覆盖该数据集。输出 waypoint top-1、距离阈值命中率、pose emission precision/recall/F1、balanced pose emission accuracy、false lock 和只在成功 pose 上统计的位置误差分位数。这些检查用于防止意外混集和子集挑选，不是防恶意篡改的签名系统；blind 独立性仍必须由采样流程保证。
7. Worker 首次定位显式遍历所有有归属模板；路线建立后由 Controller 逐帧传入不可变的 current/next 候选范围，匹配器在计算正常候选的跨 waypoint margin 前排除范围外和无归属模板，同 waypoint 的多张外观模板先聚合。只有正常候选拒绝后，才扫描其他有归属模板以诊断已知偏航并停机；无归属模板始终不参与在线决策。离线 blind evaluator 保持全局感知并标记 `unconstrained_perception`，不得使用标签真值生成候选范围。

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

本次审计到的开源项目中，没有项目给出可直接迁移到《三角洲行动》的准确率。BetterGI 参数、ViZDoom steps/s、OCR 文档指标和 COCO mAP 都不能外推。

实际游戏数据按以下方式建立：

1. 至少录制 30 次完整固定路线，覆盖画质、亮度、动态遮挡和 HUD 状态；每次采样时就写入不可混用的 dataset split。
2. 按“整次运行”切分 calibration / validation / blind test；禁止把相邻帧随机分到不同集合，也禁止复制标定图片后仅更换 run ID。
3. 每秒抽 2～5 帧标注路线节点/位置与朝向；转角、门、楼梯、分叉和易卡点全量标注。
4. 10% 样本双人标注，记录标注分歧。

必须报告：

- waypoint top-1 准确率（按 ID 比较，不用坐标距离冒充节点准确率）；
- exact position 与距离阈值命中率，以及成功 pose 子集上的位置误差 median / P90 / P95 / max；
- 朝向误差 MAE / P95；
- pose emission precision / recall / F1 / balanced accuracy（只表示“是否输出 pose”，不代替 waypoint/位置正确率）、可用率与 false lock；
- 数据帧数、标签帧数、标签覆盖率、标签文件哈希、数据集像素内容哈希、`run.json` 哈希和帧 manifest 哈希；
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
