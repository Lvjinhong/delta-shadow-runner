# 2026-07-18 开发暂停检查点

## 1. 暂停原因与边界

用户准备关闭本机并断开 Windows 主机连接，本轮仅保存当前状态，不再继续开发，也不再向 Windows 发送任何键盘或鼠标输入。

持久目标“按照最新重构方案开发实现并端到端验证”保持进行中，不标记完成。`feature-list.json` 本轮不改动，因为没有新增满足验收标准的功能。

## 2. 仓库状态

- macOS 仓库：`/Users/bytedance/.codex/tmp/delta-shadow-runner`
- GitHub：`https://github.com/Lvjinhong/delta-shadow-runner`
- Windows 仓库：`D:\Work\Work-Project\delta-shadow-runner`
- macOS/GitHub 暂停前提交：`2eb57c964cb68c17b388ac1f706dbd638375aa96`
- Windows 暂停前提交：`ff7a38fa9402f74b600d1adc62dd06dc647becfc`
- Windows 与 macOS 的运行源码一致；Windows 只缺少提交 `2eb57c9` 中的检查点文档。
- Windows 工作区仅有预期的未跟踪运行产物：`artifacts/`、`node_modules/`、`playwright-report/`、`test-results/`。恢复时必须保留，不能清理或覆盖。

## 3. 当前验证基线

2026-07-18 在 macOS 重新执行：

```text
uv sync --frozen --python 3.12
uv run pytest -q
uv run ruff check python python_tests
uv build
```

结果：

- `434 passed in 17.64s`
- branch coverage `87.64%`
- Ruff 通过
- sdist/wheel 构建通过

这些结果只证明本机离线代码基线，不代替 Windows 实机或游戏路线 E2E。

## 4. Windows 与游戏当前状态

- 2026-07-18 publickey-only SSH 健康检查通过；连接时显式禁用 password 和 keyboard-interactive 回退。
- Windows console 用户仍为 `USER-20250705MG\Administrator`。
- 计划任务 `DeltaVisionGameProbe` 状态为 `Ready`。
- 本轮只读截图探针未找到标题为 `三角洲行动  ` 的游戏窗口，任务返回码为 `1`，没有生成新数据集。
- 因此当前只能确认“探针执行时游戏窗口不存在或游戏未运行”，不能推断游戏内部页面。
- 本轮没有向游戏发送任何键鼠输入。

## 5. 已完成与未完成

已经实际验证：

1. 大厅 → 行前备战 → 战略板 → 零号大坝-常规。
2. 跳过配装并零购买出发。
3. 真实进入对局并取得一个仅持刀的出生样本。
4. 受控 E2E 既有基线为 `20/20`，F12 急停既有实测为 `94 ms`；这两项不是本轮重跑结果。
5. MSS 既有 60 秒截图基线为平均 `28.059 FPS`、P95 `47 ms`、无黑帧和缺帧；这不是本轮重跑结果。

仍未完成：

1. 大厅和地图选择的正式截图识别状态机尚未写入仓库源码。
2. 多个真实出生 run、ORB/SIFT 同集 A/B、独立 blind 指标尚未完成。
3. 零号大坝固定短路线、保险箱交互、退出或自雷闭环尚未完成。
4. 真实游戏中的低置信停机、卡住恢复和路线成功率尚未验证。
5. 当前 Windows 已无游戏窗口，无法继续 live E2E；恢复时必须从新截图重新确认状态。

## 6. 下次恢复入口

1. 先确认 macOS 仓库为本检查点提交，并运行 `git status --short`，不得覆盖用户产物。
2. Windows 上线后，先按 publickey-only 规则验证 SSH，再查看 `D:\Work\Work-Project\delta-shadow-runner` 的 HEAD 和工作区。
3. 把 Windows 同步到 macOS/GitHub 当前提交，但保留所有未跟踪运行产物。
4. 游戏启动后只截取一张当前画面；先识别大厅、登录页、结算页或未知状态，再决定动作。
5. 先把已有 2026-07-16 真实截图和审计 JSON 复制到本机 gitignored artifact 目录，完成离线 TDD。
6. 第一开发步限定为正式的截图驱动菜单状态检测器和 fail-closed 状态机：先写红灯测试，再实现；不直接把一次性坐标脚本当成产品逻辑。
7. 状态机离线验证后，再采集多个真实出生 run，比较 ORB/SIFT，并在独立 blind set 上报告 precision、recall、F1 和 false lock。
8. 只有上述门槛通过后，才继续零成本固定短路线 DryRun、Armed 和真实游戏 E2E。

## 7. 恢复命令

```bash
cd /Users/bytedance/.codex/tmp/delta-shadow-runner
git status --short
git log -1 --oneline
uv sync --frozen --python 3.12
uv run pytest -q
uv run ruff check python python_tests
uv build
```

Windows 连接必须继续显式使用 publickey-only、`KbdInteractiveAuthentication=no`、`PasswordAuthentication=no` 和严格 host key 校验。
