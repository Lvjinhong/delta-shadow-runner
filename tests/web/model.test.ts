import { describe, expect, it } from "vitest";

import {
  buildWebSocketUrl,
  formatActionIntent,
  parseSnapshotEnvelope,
  parseTelemetryMessage,
  projectRouteMap,
  statusPresentation,
} from "../../src/web/model.js";

const telemetryData = {
  snapshot: {
    runId: "run-7",
    status: "navigating",
    tick: 12,
    currentNodeId: "relay",
    targetNodeId: "extract",
    route: ["spawn-a", "relay", "extract"],
    confidence: 0.94,
    action: { type: "move", targetNodeId: "extract", ttlMs: 750 },
    metrics: {
      recoveryCount: 1,
      invalidObservationCount: 0,
      routeProgress: 0.5,
    },
    events: [
      { tick: 12, kind: "info", message: "正在前往撤离点" },
    ],
  },
  scenario: {
    id: "test-route",
    spawnNodeIds: ["spawn-a"],
    defaultSpawnNodeId: "spawn-a",
    extractNodeId: "extract",
    localizationThreshold: 0.7,
    map: {
      nodes: [
        {
          id: "spawn-a",
          x: 8,
          y: 72,
          edges: [{ targetNodeId: "relay", cost: 1 }],
        },
        {
          id: "relay",
          x: 48,
          y: 54,
          edges: [{ targetNodeId: "extract", cost: 1 }],
        },
        { id: "extract", x: 92, y: 42, edges: [] },
      ],
    },
  },
} as const;

describe("dashboard view model", () => {
  it.each([
    ["idle", "待机", "neutral"],
    ["localizing", "定位中", "warning"],
    ["navigating", "航行中", "active"],
    ["recovering", "脱困中", "danger"],
    ["extracted", "已撤离", "success"],
    ["paused", "已暂停", "neutral"],
  ] as const)("将 %s 映射为中文状态与颜色语义", (status, label, tone) => {
    expect(statusPresentation(status)).toEqual({ label, tone });
  });

  it("将所有 ActionIntent 映射为可读文案", () => {
    expect(
      formatActionIntent({ type: "move", targetNodeId: "warehouse", ttlMs: 750 }),
    ).toBe("前往 warehouse · TTL 750ms");
    expect(formatActionIntent({ type: "relocalize", ttlMs: 1_000 })).toBe(
      "重新定位 · TTL 1000ms",
    );
    expect(
      formatActionIntent({
        type: "recover",
        strategy: "backtrack",
        ttlMs: 1_250,
      }),
    ).toBe("回退脱困 · TTL 1250ms");
    expect(formatActionIntent({ type: "stop", reason: "任务未启动" })).toBe(
      "停止 · 任务未启动",
    );
  });

  it.each([
    ["http:", "127.0.0.1:4173", "ws://127.0.0.1:4173/ws"],
    ["https:", "runner.example", "wss://runner.example/ws"],
  ] as const)("根据 %s 页面协议构造同源 WebSocket URL", (protocol, host, expected) => {
    expect(buildWebSocketUrl({ protocol, host })).toBe(expected);
  });

  it("拒绝不能承载 WebSocket 的页面协议", () => {
    expect(() =>
      buildWebSocketUrl({ protocol: "file:", host: "" }),
    ).toThrow(/页面协议/);
  });

  it("验证并解析 REST 快照 envelope", () => {
    const parsed = parseSnapshotEnvelope({
      success: true,
      data: telemetryData,
      error: null,
    });

    expect(parsed).toEqual(telemetryData);
    expect(parsed.snapshot.route).toEqual(["spawn-a", "relay", "extract"]);
  });

  it("保留服务端失败 envelope 的错误信息", () => {
    expect(() =>
      parseSnapshotEnvelope({
        success: false,
        data: null,
        error: { code: "API_DOWN", message: "模拟服务暂不可用" },
      }),
    ).toThrow("模拟服务暂不可用");
  });

  it.each([
    null,
    { success: true, data: { ...telemetryData, snapshot: { status: "flying" } }, error: null },
    {
      success: true,
      data: {
        ...telemetryData,
        snapshot: { ...telemetryData.snapshot, confidence: 2 },
      },
      error: null,
    },
    {
      success: true,
      data: {
        ...telemetryData,
        scenario: {
          ...telemetryData.scenario,
          map: {
            nodes: [
              ...telemetryData.scenario.map.nodes,
              {
                id: "orphan",
                x: 50,
                y: 50,
                edges: [{ targetNodeId: "missing", cost: 1 }],
              },
            ],
          },
        },
      },
      error: null,
    },
  ])("拒绝非法 REST 遥测数据 %#", (value) => {
    expect(() => parseSnapshotEnvelope(value)).toThrow();
  });

  it("验证 WebSocket connection 与 snapshot 消息", () => {
    expect(
      parseTelemetryMessage(
        JSON.stringify({
          type: "connection",
          data: { status: "connected", mode: "simulation" },
        }),
      ),
    ).toEqual({
      type: "connection",
      data: { status: "connected", mode: "simulation" },
    });

    const message = parseTelemetryMessage(
      JSON.stringify({ type: "snapshot", data: telemetryData }),
    );
    expect(message).toEqual({ type: "snapshot", data: telemetryData });
  });

  it.each([
    "not-json",
    JSON.stringify({ type: "connection", data: { status: "offline" } }),
    JSON.stringify({ type: "snapshot", data: { snapshot: {} } }),
    JSON.stringify({ type: "command", data: {} }),
  ])("拒绝非法 WebSocket 消息 %#", (raw) => {
    expect(() => parseTelemetryMessage(raw)).toThrow();
  });

  it("投影全部地图边，并区分已走与待走路线", () => {
    const projection = projectRouteMap(telemetryData);

    expect(projection.edges).toEqual([
      {
        id: "spawn-a->relay",
        sourceNodeId: "spawn-a",
        targetNodeId: "relay",
        x1: 8,
        y1: 72,
        x2: 48,
        y2: 54,
        cost: 1,
        phase: "walked",
      },
      {
        id: "relay->extract",
        sourceNodeId: "relay",
        targetNodeId: "extract",
        x1: 48,
        y1: 54,
        x2: 92,
        y2: 42,
        cost: 1,
        phase: "planned",
      },
    ]);
    expect(projection.nodes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "spawn-a", isSpawn: true, isOnRoute: true }),
        expect.objectContaining({ id: "relay", isCurrent: true, isOnRoute: true }),
        expect.objectContaining({ id: "extract", isTarget: true, isExtract: true }),
      ]),
    );
  });
});
