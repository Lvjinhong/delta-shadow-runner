import type {
  ActionIntent,
  EngineSnapshot,
  RunnerEvent,
  RunnerStatus,
  RuntimeMetrics,
} from "../core/types.js";

export type StatusTone = "neutral" | "warning" | "active" | "danger" | "success";

export interface StatusPresentation {
  readonly label: string;
  readonly tone: StatusTone;
}

export interface ScenarioTelemetry {
  readonly id: string;
  readonly spawnNodeIds: readonly string[];
  readonly defaultSpawnNodeId: string;
  readonly extractNodeId: string;
  readonly localizationThreshold: number;
  readonly map: {
    readonly nodes: readonly ScenarioNodeTelemetry[];
  };
}

export interface ScenarioNodeTelemetry {
  readonly id: string;
  readonly x: number;
  readonly y: number;
  readonly edges: readonly ScenarioEdgeTelemetry[];
}

export interface ScenarioEdgeTelemetry {
  readonly targetNodeId: string;
  readonly cost: number;
}

export interface TelemetryData {
  readonly snapshot: EngineSnapshot;
  readonly scenario: ScenarioTelemetry;
}

export type TelemetryMessage =
  | {
      readonly type: "connection";
      readonly data: { readonly status: "connected"; readonly mode: "simulation" };
    }
  | { readonly type: "snapshot"; readonly data: TelemetryData };

export interface ProjectedNode {
  readonly id: string;
  readonly x: number;
  readonly y: number;
  readonly isSpawn: boolean;
  readonly isExtract: boolean;
  readonly isCurrent: boolean;
  readonly isTarget: boolean;
  readonly isOnRoute: boolean;
}

export interface ProjectedEdge {
  readonly id: string;
  readonly sourceNodeId: string;
  readonly targetNodeId: string;
  readonly x1: number;
  readonly y1: number;
  readonly x2: number;
  readonly y2: number;
  readonly cost: number;
  readonly phase: "base" | "planned" | "walked";
}

export interface RouteMapProjection {
  readonly nodes: readonly ProjectedNode[];
  readonly edges: readonly ProjectedEdge[];
}

export class DashboardProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DashboardProtocolError";
  }
}

const statusMeta: Readonly<Record<RunnerStatus, StatusPresentation>> = {
  idle: { label: "待机", tone: "neutral" },
  localizing: { label: "定位中", tone: "warning" },
  navigating: { label: "航行中", tone: "active" },
  recovering: { label: "脱困中", tone: "danger" },
  extracted: { label: "已撤离", tone: "success" },
  paused: { label: "已暂停", tone: "neutral" },
};

const runnerStatuses = new Set<RunnerStatus>([
  "idle",
  "localizing",
  "navigating",
  "recovering",
  "extracted",
  "paused",
]);

const eventKinds = new Set<RunnerEvent["kind"]>(["info", "warning", "error"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new DashboardProtocolError(`${path} 必须是对象`);
  }
  return value;
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new DashboardProtocolError(`${path} 必须是非空字符串`);
  }
  return value;
}

function requireFiniteNumber(
  value: unknown,
  path: string,
  minimum = Number.NEGATIVE_INFINITY,
  maximum = Number.POSITIVE_INFINITY,
): number {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < minimum ||
    value > maximum
  ) {
    throw new DashboardProtocolError(
      `${path} 必须是 ${minimum} 到 ${maximum} 之间的有限数`,
    );
  }
  return value;
}

function requireNonNegativeInteger(value: unknown, path: string): number {
  const number = requireFiniteNumber(value, path, 0);
  if (!Number.isInteger(number)) {
    throw new DashboardProtocolError(`${path} 必须是非负整数`);
  }
  return number;
}

function requireStringArray(value: unknown, path: string): readonly string[] {
  if (!Array.isArray(value)) {
    throw new DashboardProtocolError(`${path} 必须是字符串数组`);
  }
  return value.map((item, index) => requireString(item, `${path}[${index}]`));
}

function validateAction(value: unknown): ActionIntent {
  const action = requireRecord(value, "snapshot.action");
  const type = requireString(action.type, "snapshot.action.type");

  switch (type) {
    case "move":
      return {
        type,
        targetNodeId: requireString(
          action.targetNodeId,
          "snapshot.action.targetNodeId",
        ),
        ttlMs: requireFiniteNumber(action.ttlMs, "snapshot.action.ttlMs", 1),
      };
    case "relocalize":
      return {
        type,
        ttlMs: requireFiniteNumber(action.ttlMs, "snapshot.action.ttlMs", 1),
      };
    case "recover":
      if (action.strategy !== "backtrack") {
        throw new DashboardProtocolError(
          "snapshot.action.strategy 必须是 backtrack",
        );
      }
      return {
        type,
        strategy: "backtrack",
        ttlMs: requireFiniteNumber(action.ttlMs, "snapshot.action.ttlMs", 1),
      };
    case "stop":
      return {
        type,
        reason: requireString(action.reason, "snapshot.action.reason"),
      };
    default:
      throw new DashboardProtocolError(`未知 ActionIntent 类型 "${type}"`);
  }
}

function validateMetrics(value: unknown): RuntimeMetrics {
  const metrics = requireRecord(value, "snapshot.metrics");
  return {
    recoveryCount: requireNonNegativeInteger(
      metrics.recoveryCount,
      "snapshot.metrics.recoveryCount",
    ),
    invalidObservationCount: requireNonNegativeInteger(
      metrics.invalidObservationCount,
      "snapshot.metrics.invalidObservationCount",
    ),
    routeProgress: requireFiniteNumber(
      metrics.routeProgress,
      "snapshot.metrics.routeProgress",
      0,
      1,
    ),
  };
}

function validateEvents(value: unknown): readonly RunnerEvent[] {
  if (!Array.isArray(value)) {
    throw new DashboardProtocolError("snapshot.events 必须是数组");
  }
  return value.map((item, index) => {
    const event = requireRecord(item, `snapshot.events[${index}]`);
    const kind = requireString(event.kind, `snapshot.events[${index}].kind`);
    if (!eventKinds.has(kind as RunnerEvent["kind"])) {
      throw new DashboardProtocolError(`未知事件类型 "${kind}"`);
    }
    return {
      tick: requireNonNegativeInteger(event.tick, `snapshot.events[${index}].tick`),
      kind: kind as RunnerEvent["kind"],
      message: requireString(event.message, `snapshot.events[${index}].message`),
    };
  });
}

function validateSnapshot(value: unknown): EngineSnapshot {
  const snapshot = requireRecord(value, "snapshot");
  const status = requireString(snapshot.status, "snapshot.status");
  if (!runnerStatuses.has(status as RunnerStatus)) {
    throw new DashboardProtocolError(`未知运行状态 "${status}"`);
  }

  return {
    runId: requireString(snapshot.runId, "snapshot.runId"),
    status: status as RunnerStatus,
    tick: requireNonNegativeInteger(snapshot.tick, "snapshot.tick"),
    currentNodeId: requireString(
      snapshot.currentNodeId,
      "snapshot.currentNodeId",
    ),
    targetNodeId: requireString(snapshot.targetNodeId, "snapshot.targetNodeId"),
    route: requireStringArray(snapshot.route, "snapshot.route"),
    confidence: requireFiniteNumber(
      snapshot.confidence,
      "snapshot.confidence",
      0,
      1,
    ),
    action: validateAction(snapshot.action),
    metrics: validateMetrics(snapshot.metrics),
    events: validateEvents(snapshot.events),
  };
}

function validateScenario(value: unknown): ScenarioTelemetry {
  const scenario = requireRecord(value, "scenario");
  const map = requireRecord(scenario.map, "scenario.map");
  if (!Array.isArray(map.nodes) || map.nodes.length === 0) {
    throw new DashboardProtocolError("scenario.map.nodes 必须是非空数组");
  }

  const seenNodeIds = new Set<string>();
  const nodes = map.nodes.map((item, nodeIndex): ScenarioNodeTelemetry => {
    const node = requireRecord(item, `scenario.map.nodes[${nodeIndex}]`);
    const id = requireString(node.id, `scenario.map.nodes[${nodeIndex}].id`);
    if (seenNodeIds.has(id)) {
      throw new DashboardProtocolError(`地图节点 "${id}" 重复`);
    }
    seenNodeIds.add(id);
    if (!Array.isArray(node.edges)) {
      throw new DashboardProtocolError(`地图节点 "${id}" 的 edges 必须是数组`);
    }

    return {
      id,
      x: requireFiniteNumber(node.x, `地图节点 "${id}".x`, 0, 100),
      y: requireFiniteNumber(node.y, `地图节点 "${id}".y`, 0, 100),
      edges: node.edges.map((edgeValue, edgeIndex) => {
        const edge = requireRecord(
          edgeValue,
          `地图节点 "${id}".edges[${edgeIndex}]`,
        );
        return {
          targetNodeId: requireString(
            edge.targetNodeId,
            `地图节点 "${id}".edges[${edgeIndex}].targetNodeId`,
          ),
          cost: requireFiniteNumber(
            edge.cost,
            `地图节点 "${id}".edges[${edgeIndex}].cost`,
            Number.EPSILON,
          ),
        };
      }),
    };
  });

  for (const node of nodes) {
    for (const edge of node.edges) {
      if (!seenNodeIds.has(edge.targetNodeId)) {
        throw new DashboardProtocolError(
          `地图边 ${node.id} -> ${edge.targetNodeId} 指向未知节点`,
        );
      }
    }
  }

  const spawnNodeIds = requireStringArray(
    scenario.spawnNodeIds,
    "scenario.spawnNodeIds",
  );
  const defaultSpawnNodeId = requireString(
    scenario.defaultSpawnNodeId,
    "scenario.defaultSpawnNodeId",
  );
  const extractNodeId = requireString(
    scenario.extractNodeId,
    "scenario.extractNodeId",
  );
  if (
    spawnNodeIds.length === 0 ||
    spawnNodeIds.some((nodeId) => !seenNodeIds.has(nodeId))
  ) {
    throw new DashboardProtocolError("scenario.spawnNodeIds 包含未知节点或为空");
  }
  if (!spawnNodeIds.includes(defaultSpawnNodeId)) {
    throw new DashboardProtocolError("默认出生点不在出生点列表中");
  }
  if (!seenNodeIds.has(extractNodeId)) {
    throw new DashboardProtocolError("撤离点不在地图中");
  }

  return {
    id: requireString(scenario.id, "scenario.id"),
    spawnNodeIds,
    defaultSpawnNodeId,
    extractNodeId,
    localizationThreshold: requireFiniteNumber(
      scenario.localizationThreshold,
      "scenario.localizationThreshold",
      0,
      1,
    ),
    map: { nodes },
  };
}

function validateTelemetryData(value: unknown): TelemetryData {
  const data = requireRecord(value, "telemetry.data");
  const snapshot = validateSnapshot(data.snapshot);
  const scenario = validateScenario(data.scenario);
  const nodeIds = new Set(scenario.map.nodes.map((node) => node.id));
  const referencedNodeIds = [
    snapshot.currentNodeId,
    snapshot.targetNodeId,
    ...snapshot.route,
  ];
  if (referencedNodeIds.some((nodeId) => !nodeIds.has(nodeId))) {
    throw new DashboardProtocolError("快照引用了场景中不存在的节点");
  }
  if (
    snapshot.action.type === "move" &&
    !nodeIds.has(snapshot.action.targetNodeId)
  ) {
    throw new DashboardProtocolError("移动意图引用了场景中不存在的节点");
  }
  return { snapshot, scenario };
}

export function statusPresentation(status: RunnerStatus): StatusPresentation {
  return statusMeta[status];
}

export function formatActionIntent(action: ActionIntent): string {
  switch (action.type) {
    case "move":
      return `前往 ${action.targetNodeId} · TTL ${action.ttlMs}ms`;
    case "relocalize":
      return `重新定位 · TTL ${action.ttlMs}ms`;
    case "recover":
      return `回退脱困 · TTL ${action.ttlMs}ms`;
    case "stop":
      return `停止 · ${action.reason}`;
  }
}

export function buildWebSocketUrl(origin: {
  readonly protocol: string;
  readonly host: string;
}): string {
  if (origin.protocol !== "http:" && origin.protocol !== "https:") {
    throw new DashboardProtocolError(
      `页面协议 ${origin.protocol || "(empty)"} 不能建立 WebSocket`,
    );
  }
  if (origin.host.length === 0) {
    throw new DashboardProtocolError("页面 host 不能为空");
  }
  const protocol = origin.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${origin.host}/ws`;
}

export function parseSnapshotEnvelope(value: unknown): TelemetryData {
  const envelope = requireRecord(value, "REST envelope");
  if (envelope.success === false) {
    const error = requireRecord(envelope.error, "REST envelope.error");
    throw new DashboardProtocolError(
      requireString(error.message, "REST envelope.error.message"),
    );
  }
  if (envelope.success !== true || envelope.error !== null) {
    throw new DashboardProtocolError("REST envelope 结构无效");
  }
  return validateTelemetryData(envelope.data);
}

export function parseTelemetryMessage(raw: string): TelemetryMessage {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw) as unknown;
  } catch {
    throw new DashboardProtocolError("WebSocket 消息不是有效 JSON");
  }

  const message = requireRecord(parsed, "WebSocket message");
  if (message.type === "connection") {
    const data = requireRecord(message.data, "WebSocket connection.data");
    if (data.status !== "connected" || data.mode !== "simulation") {
      throw new DashboardProtocolError("WebSocket 连接状态消息无效");
    }
    return {
      type: "connection",
      data: { status: "connected", mode: "simulation" },
    };
  }
  if (message.type === "snapshot") {
    return { type: "snapshot", data: validateTelemetryData(message.data) };
  }
  throw new DashboardProtocolError("WebSocket 消息类型无效");
}

function routeEdgePhase(
  route: readonly string[],
  currentNodeId: string,
  sourceNodeId: string,
  targetNodeId: string,
): ProjectedEdge["phase"] {
  const edgeIndex = route.findIndex(
    (nodeId, index) =>
      nodeId === sourceNodeId && route[index + 1] === targetNodeId,
  );
  if (edgeIndex < 0) {
    return "base";
  }
  const currentIndex = route.indexOf(currentNodeId);
  return currentIndex >= 0 && edgeIndex < currentIndex ? "walked" : "planned";
}

export function projectRouteMap(data: TelemetryData): RouteMapProjection {
  const { scenario, snapshot } = data;
  const nodeById = new Map(
    scenario.map.nodes.map((node) => [node.id, node] as const),
  );
  const nodes = scenario.map.nodes.map((node): ProjectedNode => ({
    id: node.id,
    x: node.x,
    y: node.y,
    isSpawn: scenario.spawnNodeIds.includes(node.id),
    isExtract: scenario.extractNodeId === node.id,
    isCurrent: snapshot.currentNodeId === node.id,
    isTarget: snapshot.targetNodeId === node.id,
    isOnRoute: snapshot.route.includes(node.id),
  }));
  const edges = scenario.map.nodes.flatMap((sourceNode) =>
    sourceNode.edges.map((edge): ProjectedEdge => {
      const targetNode = nodeById.get(edge.targetNodeId);
      if (!targetNode) {
        throw new DashboardProtocolError(
          `地图边 ${sourceNode.id} -> ${edge.targetNodeId} 指向未知节点`,
        );
      }
      return {
        id: `${sourceNode.id}->${edge.targetNodeId}`,
        sourceNodeId: sourceNode.id,
        targetNodeId: edge.targetNodeId,
        x1: sourceNode.x,
        y1: sourceNode.y,
        x2: targetNode.x,
        y2: targetNode.y,
        cost: edge.cost,
        phase: routeEdgePhase(
          snapshot.route,
          snapshot.currentNodeId,
          sourceNode.id,
          edge.targetNodeId,
        ),
      };
    }),
  );
  return { nodes, edges };
}
