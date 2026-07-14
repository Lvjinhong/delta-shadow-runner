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
  capabilities: {
    canStart: false,
    canPause: true,
    canReset: true,
    canInjectStuck: true,
  },
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
    expect(parsed.capabilities).toEqual(telemetryData.capabilities);
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
    {
      success: true,
      data: {
        ...telemetryData,
        capabilities: {
          ...telemetryData.capabilities,
          canStart: "yes",
        },
      },
      error: null,
    },
    {
      success: true,
      data: {
        snapshot: telemetryData.snapshot,
        scenario: telemetryData.scenario,
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
        id: "edge:0:0",
        sourceNodeId: "spawn-a",
        targetNodeId: "relay",
        x1: 12,
        y1: 88,
        x2: 48.19047619047619,
        y2: expect.any(Number),
        cost: 1,
        phase: "walked",
      },
      {
        id: "edge:1:0",
        sourceNodeId: "relay",
        targetNodeId: "extract",
        x1: 48.19047619047619,
        y1: expect.any(Number),
        x2: 88,
        y2: 12,
        cost: 1,
        phase: "planned",
      },
    ]);
    expect(projection.edges[0]?.y2).toBeCloseTo(42.4);
    expect(projection.edges[1]?.y1).toBeCloseTo(42.4);
    expect(projection.nodes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "spawn-a", isSpawn: true, isOnRoute: true }),
        expect.objectContaining({ id: "relay", isCurrent: true, isOnRoute: true }),
        expect.objectContaining({ id: "extract", isTarget: true, isExtract: true }),
      ]),
    );
  });

  it("接受任意有限坐标与零代价边，并归一化到安全 viewBox", () => {
    const arbitraryTelemetry = {
      ...telemetryData,
      snapshot: {
        ...telemetryData.snapshot,
        currentNodeId: "middle",
        targetNodeId: "right",
        route: ["left", "middle", "right"],
        action: { type: "move", targetNodeId: "right", ttlMs: 750 },
      },
      scenario: {
        ...telemetryData.scenario,
        spawnNodeIds: ["left"],
        defaultSpawnNodeId: "left",
        extractNodeId: "right",
        map: {
          nodes: [
            {
              id: "left",
              x: -Number.MAX_VALUE,
              y: -42,
              edges: [
                { targetNodeId: "middle", cost: 0 },
                { targetNodeId: "middle", cost: 3 },
              ],
            },
            {
              id: "middle",
              x: 0,
              y: -42,
              edges: [{ targetNodeId: "right", cost: 0 }],
            },
            { id: "right", x: Number.MAX_VALUE, y: -42, edges: [] },
          ],
        },
      },
    } as const;

    const parsed = parseSnapshotEnvelope({
      success: true,
      data: arbitraryTelemetry,
      error: null,
    });
    const projection = projectRouteMap(parsed);

    expect(projection.nodes.map(({ x, y }) => ({ x, y }))).toEqual([
      { x: 12, y: 50 },
      { x: 50, y: 50 },
      { x: 88, y: 50 },
    ]);
    expect(projection.edges.map((edge) => edge.id)).toEqual([
      "edge:0:0",
      "edge:0:1",
      "edge:1:0",
    ]);
    expect(projection.edges.map((edge) => edge.phase)).toEqual([
      "walked",
      "walked",
      "planned",
    ]);
    expect(
      projection.nodes.every(
        (node) =>
          node.x >= 12 && node.x <= 88 && node.y >= 12 && node.y <= 88,
      ),
    ).toBe(true);
    expect(
      projection.nodes.every(
        (node) => node.x - 11 >= 0 && node.x + 11 <= 100,
      ),
    ).toBe(true);
  });

  it("边 ID 仅由源节点下标与边下标组成，节点 ID 含分隔符时仍无碰撞", () => {
    const separatorTelemetry = {
      ...telemetryData,
      snapshot: {
        ...telemetryData.snapshot,
        currentNodeId: "a",
        targetNodeId: "c",
        route: ["a", "b->c"],
        action: { type: "move", targetNodeId: "b->c", ttlMs: 750 },
      },
      scenario: {
        ...telemetryData.scenario,
        spawnNodeIds: ["a"],
        defaultSpawnNodeId: "a",
        extractNodeId: "c",
        map: {
          nodes: [
            {
              id: "a",
              x: 0,
              y: 0,
              edges: [{ targetNodeId: "b->c", cost: 1 }],
            },
            {
              id: "a->b",
              x: 1,
              y: 1,
              edges: [{ targetNodeId: "c", cost: 1 }],
            },
            { id: "b->c", x: 2, y: 2, edges: [] },
            { id: "c", x: 3, y: 3, edges: [] },
          ],
        },
      },
    } as const;

    const parsed = parseSnapshotEnvelope({
      success: true,
      data: separatorTelemetry,
      error: null,
    });
    const edgeIds = projectRouteMap(parsed).edges.map((edge) => edge.id);

    expect(edgeIds).toEqual(["edge:0:0", "edge:1:0"]);
    expect(new Set(edgeIds).size).toBe(edgeIds.length);
  });

  it("坐标仍须有限且边代价不能为负数", () => {
    const invalidCoordinate = {
      ...telemetryData,
      scenario: {
        ...telemetryData.scenario,
        map: {
          nodes: telemetryData.scenario.map.nodes.map((node, index) =>
            index === 0 ? { ...node, x: Number.POSITIVE_INFINITY } : node,
          ),
        },
      },
    };
    const invalidCost = {
      ...telemetryData,
      scenario: {
        ...telemetryData.scenario,
        map: {
          nodes: telemetryData.scenario.map.nodes.map((node, index) =>
            index === 0
              ? {
                  ...node,
                  edges: [{ targetNodeId: "relay", cost: -1 }],
                }
              : node,
          ),
        },
      },
    };

    expect(() =>
      parseSnapshotEnvelope({
        success: true,
        data: invalidCoordinate,
        error: null,
      }),
    ).toThrow(/有限数/);
    expect(() =>
      parseSnapshotEnvelope({
        success: true,
        data: invalidCost,
        error: null,
      }),
    ).toThrow(/cost/);
  });
});
