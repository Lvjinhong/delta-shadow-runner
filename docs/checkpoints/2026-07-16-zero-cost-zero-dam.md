# 2026-07-16 零成本零号大坝实机检查点

## 1. 本次目标

以 Windows 当前《三角洲行动》主页为起点，验证以下最短链路：

1. 进入危险行动战略板。
2. 选择“零号大坝-常规”。
3. 跳过购买与配装，零成本直接出发。
4. 采集真实出生画面，为后续截图定位和路线标定准备样本。

本次不以单次进图证明自动路线、识别准确率或可稳定跑刀。

## 2. 已验证结果

1. Windows 游戏窗口客户区仍为 `2560x1440`，窗口标题和前台 HWND 守卫通过。
2. 从大厅点击“行前备战”可进入战略板。
3. 2026-04 的公开新手流程与当前 PC UI 一致：点击地图名称后会选择地图；处理首次说明浮层后，模板定位“零号大坝”文字并点击成功。
4. 大厅右侧明确显示“零号大坝-常规”，角色未进入购买或配装页。
5. 绿色操作条右半区“出发”点击成功，零购买直接进入真实对局。
6. 第一张真实出生画面显示角色仅持刀；出生在货运集装箱道路，路牌显示左侧 `Substation 200m`、正前 `Administrative Area 188m`。
7. 进入约 17 秒、尚未开始路线移动时被其他玩家使用 SG552 击倒，随后进入“最终倒地”页面。

## 3. Windows 持久化证据

Windows 仓库：`D:\Work\Work-Project\delta-shadow-runner`

关键输入审计：

- `artifacts\runs\game-click-preparation-cta-20260716\audit.json`
- `artifacts\runs\game-click-zero-dam-label-ready-20260716\audit.json`
- `artifacts\runs\game-click-depart-zero-cost-20260716\audit.json`
- `artifacts\runs\game-open-map-m-20260716\audit.json`
- `artifacts\runs\game-death-next-space-20260716\audit.json`

关键截图数据集：

- `artifacts\datasets\game-click-zero-dam-label-ready-20260716`
- `artifacts\datasets\game-after-depart-zero-cost-20260716`
- `artifacts\datasets\game-after-open-map-m-20260716`
- `artifacts\datasets\game-after-death-next-20260716`

注意：`game-after-open-map-m-20260716` 捕获到的是已经发生的死亡结算，不是战术地图。`M` 输入审计只能证明按键成功插入，不能证明地图成功打开。

## 4. 尚未完成

1. 没有完成零号大坝固定路线，也没有路线成功率。
2. 没有得到出生点分类的独立 blind set、precision、recall 或 F1。
3. 没有验证战术地图打开、撤离点识别、保险箱交互或自雷退出。
4. 没有验证真实游戏中的卡住恢复、低置信停机或路线急停闭环。
5. “下一步”空格已发送，但在用户要求停机前没有再次确认当前页面；恢复时必须先截取新画面，不能沿用旧状态。

## 5. 恢复顺序

1. Windows 重新上线后先跑 publickey-only SSH 健康检查。
2. 只截取一张当前游戏画面，确认是在结算页、大厅还是登录页。
3. 如仍在结算页，只处理返回大厅；不要直接发送地图或移动按键。
4. 再次选择“零号大坝-常规”，继续使用零购买方案。
5. 下一局进图后优先在 2 秒内记录出生帧，并立即移动到最近掩体，避免重复出现原地采样期间被击倒。
6. 先积累多个真实出生 run，再做出生分类与固定短路线；没有 blind 指标前保持 `controlled-game-e2e=false`。

## 6. 验收状态

- ✅ 截图识别驱动的大厅 → 战略板 → 零号大坝-常规。
- ✅ 零购买 → 出发 → 真实对局出生。
- ✅ 输入与截图产物按 run 独立落盘。
- ❌ 地图熟悉路线。
- ❌ 保险箱/自雷/正常退出闭环。
- ❌ 真实路线准确率与成功率。
