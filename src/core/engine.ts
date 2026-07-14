import { findShortestPath } from "./graph.js";
import type {
  ActionIntent,
  EngineSnapshot,
  RunnerEvent,
  RunnerObservation,
  RunnerScenario,
  RuntimeMetrics,
} from "./types.js";

interface RunnerEngineOptions {
  readonly eventLimit?: number;
}

const MOVE_TTL_MS = 750;
const RELOCALIZE_TTL_MS = 1_000;
const RECOVER_TTL_MS = 1_250;
const DEFAULT_EVENT_LIMIT = 50;

function freezeMetrics(metrics: RuntimeMetrics): RuntimeMetrics {
  return Object.freeze({ ...metrics });
}

function freezeAction(action: ActionIntent): ActionIntent {
  return Object.freeze({ ...action });
}

function freezeEvents(events: readonly RunnerEvent[]): readonly RunnerEvent[] {
  return Object.freeze(events.map((event) => Object.freeze({ ...event })));
}

function freezeSnapshot(snapshot: EngineSnapshot): EngineSnapshot {
  return Object.freeze({
    ...snapshot,
    route: Object.freeze([...snapshot.route]),
    action: freezeAction(snapshot.action),
    metrics: freezeMetrics(snapshot.metrics),
    events: freezeEvents(snapshot.events),
  });
}

function calculateRouteProgress(
  route: readonly string[],
  currentNodeId: string,
): number {
  if (route.length <= 1) {
    return route[0] === currentNodeId ? 1 : 0;
  }

  const routeIndex = route.indexOf(currentNodeId);
  return routeIndex < 0 ? 0 : routeIndex / (route.length - 1);
}

function isKnownNode(scenario: RunnerScenario, nodeId: string): boolean {
  return Object.prototype.hasOwnProperty.call(scenario.graph, nodeId);
}

export class RunnerEngine {
  private readonly scenario: RunnerScenario;
  private readonly eventLimit: number;
  private runCounter = 0;
  private snapshot: EngineSnapshot;

  constructor(
    scenario: RunnerScenario,
    options: RunnerEngineOptions = {},
  ) {
    this.scenario = scenario;
    this.eventLimit = options.eventLimit ?? DEFAULT_EVENT_LIMIT;

    if (!Number.isInteger(this.eventLimit) || this.eventLimit < 1) {
      throw new Error("事件上限必须是正整数");
    }
    if (!isKnownNode(scenario, scenario.defaultSpawnNodeId)) {
      throw new Error(`场景中不存在默认出生点 "${scenario.defaultSpawnNodeId}"`);
    }
    if (!isKnownNode(scenario, scenario.extractNodeId)) {
      throw new Error(`场景中不存在撤离点 "${scenario.extractNodeId}"`);
    }

    this.snapshot = this.createInitialSnapshot("引擎已初始化");
  }

  getSnapshot(): EngineSnapshot {
    return this.snapshot;
  }

  start(): EngineSnapshot {
    this.runCounter += 1;
    const spawnNodeId = this.snapshot.currentNodeId;
    this.snapshot = freezeSnapshot({
      runId: `run-${this.runCounter}`,
      status: "localizing",
      tick: 0,
      currentNodeId: spawnNodeId,
      targetNodeId: spawnNodeId,
      route: [],
      confidence: 0,
      action: { type: "relocalize", ttlMs: RELOCALIZE_TTL_MS },
      metrics: {
        recoveryCount: 0,
        invalidObservationCount: 0,
        routeProgress: 0,
      },
      events: [{ tick: 0, kind: "info", message: "任务已启动，等待定位" }],
    });
    return this.snapshot;
  }

  pause(): EngineSnapshot {
    const tick = this.snapshot.tick + 1;
    this.snapshot = this.withEvent(
      {
        ...this.snapshot,
        status: "paused",
        tick,
        action: { type: "stop", reason: "任务已暂停" },
      },
      { tick, kind: "info", message: "任务已暂停" },
    );
    return this.snapshot;
  }

  reset(): EngineSnapshot {
    this.runCounter += 1;
    this.snapshot = this.createInitialSnapshot("任务已重置");
    return this.snapshot;
  }

  step(observation: RunnerObservation): EngineSnapshot {
    const tick = this.snapshot.tick + 1;

    if (
      this.snapshot.status === "idle" ||
      this.snapshot.status === "paused" ||
      this.snapshot.status === "extracted"
    ) {
      this.snapshot = freezeSnapshot({ ...this.snapshot, tick });
      return this.snapshot;
    }

    const invalidReason = this.validateObservation(observation);
    if (invalidReason) {
      this.snapshot = this.withEvent(
        {
          ...this.snapshot,
          status: "paused",
          tick,
          action: { type: "stop", reason: invalidReason },
          metrics: {
            ...this.snapshot.metrics,
            invalidObservationCount:
              this.snapshot.metrics.invalidObservationCount + 1,
          },
        },
        { tick, kind: "error", message: invalidReason },
      );
      return this.snapshot;
    }

    if (observation.confidence < this.scenario.localizationThreshold) {
      this.snapshot = this.withEvent(
        {
          ...this.snapshot,
          status: "localizing",
          tick,
          confidence: observation.confidence,
          action: { type: "relocalize", ttlMs: RELOCALIZE_TTL_MS },
        },
        {
          tick,
          kind: "warning",
          message: `定位置信度 ${observation.confidence.toFixed(2)} 低于阈值`,
        },
      );
      return this.snapshot;
    }

    if (observation.stuck === true) {
      this.snapshot = this.withEvent(
        {
          ...this.snapshot,
          status: "recovering",
          tick,
          currentNodeId: observation.nodeId,
          confidence: observation.confidence,
          action: {
            type: "recover",
            strategy: "backtrack",
            ttlMs: RECOVER_TTL_MS,
          },
          metrics: {
            ...this.snapshot.metrics,
            recoveryCount: this.snapshot.metrics.recoveryCount + 1,
          },
        },
        { tick, kind: "warning", message: "检测到卡住，执行回退恢复" },
      );
      return this.snapshot;
    }

    return this.navigateFromObservation(observation, tick);
  }

  private createInitialSnapshot(message: string): EngineSnapshot {
    const spawnNodeId = this.scenario.defaultSpawnNodeId;
    return freezeSnapshot({
      runId: `run-${this.runCounter}`,
      status: "idle",
      tick: 0,
      currentNodeId: spawnNodeId,
      targetNodeId: spawnNodeId,
      route: [],
      confidence: 0,
      action: { type: "stop", reason: "任务未启动" },
      metrics: {
        recoveryCount: 0,
        invalidObservationCount: 0,
        routeProgress: 0,
      },
      events: [{ tick: 0, kind: "info", message }],
    });
  }

  private validateObservation(observation: RunnerObservation): string | undefined {
    if (!isKnownNode(this.scenario, observation.nodeId)) {
      return `观测包含未知节点 "${observation.nodeId}"，已安全停止`;
    }
    if (
      !Number.isFinite(observation.confidence) ||
      observation.confidence < 0 ||
      observation.confidence > 1
    ) {
      return "观测置信度必须是 0 到 1 的有限数，已安全停止";
    }
    if (
      observation.stuck !== undefined &&
      typeof observation.stuck !== "boolean"
    ) {
      return "观测卡住标记必须是布尔值，已安全停止";
    }
    return undefined;
  }

  private navigateFromObservation(
    observation: RunnerObservation,
    tick: number,
  ): EngineSnapshot {
    let route = this.snapshot.route;
    const needsReplan =
      this.snapshot.status === "localizing" ||
      route.length === 0 ||
      !route.includes(observation.nodeId);

    if (needsReplan) {
      try {
        route = findShortestPath(
          this.scenario.graph,
          observation.nodeId,
          this.scenario.extractNodeId,
        );
      } catch (error) {
        const reason =
          error instanceof Error ? error.message : "路线规划发生未知错误";
        return this.stopForPlanningFailure(reason, tick);
      }
    }

    if (observation.nodeId === this.scenario.extractNodeId) {
      this.snapshot = this.withEvent(
        {
          ...this.snapshot,
          status: "extracted",
          tick,
          currentNodeId: observation.nodeId,
          targetNodeId: observation.nodeId,
          route,
          confidence: observation.confidence,
          action: { type: "stop", reason: "已到达撤离点" },
          metrics: { ...this.snapshot.metrics, routeProgress: 1 },
        },
        { tick, kind: "info", message: "已到达撤离点" },
      );
      return this.snapshot;
    }

    const routeIndex = route.indexOf(observation.nodeId);
    const targetNodeId = route[routeIndex + 1];
    if (!targetNodeId) {
      return this.stopForPlanningFailure(
        `节点 "${observation.nodeId}" 后没有可执行的路线目标`,
        tick,
      );
    }

    const wasRecovering = this.snapshot.status === "recovering";
    this.snapshot = this.withEvent(
      {
        ...this.snapshot,
        status: "navigating",
        tick,
        currentNodeId: observation.nodeId,
        targetNodeId,
        route,
        confidence: observation.confidence,
        action: { type: "move", targetNodeId, ttlMs: MOVE_TTL_MS },
        metrics: {
          ...this.snapshot.metrics,
          routeProgress: calculateRouteProgress(route, observation.nodeId),
        },
      },
      {
        tick,
        kind: "info",
        message: wasRecovering
          ? `恢复完成，继续前往 ${targetNodeId}`
          : `位置确认，前往 ${targetNodeId}`,
      },
    );
    return this.snapshot;
  }

  private stopForPlanningFailure(reason: string, tick: number): EngineSnapshot {
    this.snapshot = this.withEvent(
      {
        ...this.snapshot,
        status: "paused",
        tick,
        action: { type: "stop", reason },
      },
      { tick, kind: "error", message: reason },
    );
    return this.snapshot;
  }

  private withEvent(
    snapshot: EngineSnapshot,
    event: RunnerEvent,
  ): EngineSnapshot {
    const events = [...snapshot.events, event].slice(-this.eventLimit);
    return freezeSnapshot({ ...snapshot, events });
  }
}
