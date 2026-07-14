import { findShortestPath } from "./graph.js";
import type {
  ActionIntent,
  EngineSnapshot,
  RunnerEvent,
  RunnerObservation,
  RunnerScenario,
  RunnerStatus,
  RuntimeMetrics,
} from "./types.js";

interface RunnerEngineOptions {
  readonly eventLimit?: number;
}

interface ResumableState {
  readonly status: Extract<
    RunnerStatus,
    "localizing" | "navigating" | "recovering"
  >;
  readonly action: ActionIntent;
}

type ObservationValidation =
  | { readonly valid: true; readonly observation: RunnerObservation }
  | { readonly valid: false; readonly reason: string };

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

function isResumableStatus(
  status: RunnerStatus,
): status is ResumableState["status"] {
  return (
    status === "localizing" ||
    status === "navigating" ||
    status === "recovering"
  );
}

export class RunnerEngine {
  private readonly scenario: RunnerScenario;
  private readonly eventLimit: number;
  private runCounter = 0;
  private lastCapturedAt = -1;
  private resumableState: ResumableState | undefined;
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

  get canResume(): boolean {
    return (
      this.snapshot.status === "paused" && this.resumableState !== undefined
    );
  }

  start(): EngineSnapshot {
    if (this.snapshot.status === "paused" && this.resumableState) {
      const tick = this.snapshot.tick + 1;
      const resumableState = this.resumableState;
      this.resumableState = undefined;
      this.snapshot = this.withEvent(
        {
          ...this.snapshot,
          status: resumableState.status,
          tick,
          action: resumableState.action,
        },
        { tick, kind: "info", message: "任务已继续" },
      );
      return this.snapshot;
    }

    if (this.snapshot.status !== "idle") {
      return this.cloneCurrentSnapshot();
    }

    this.runCounter += 1;
    this.lastCapturedAt = -1;
    this.resumableState = undefined;
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
    if (!isResumableStatus(this.snapshot.status)) {
      return this.cloneCurrentSnapshot();
    }

    this.resumableState = {
      status: this.snapshot.status,
      action: this.snapshot.action,
    };
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
    this.lastCapturedAt = -1;
    this.resumableState = undefined;
    this.snapshot = this.createInitialSnapshot("任务已重置");
    return this.snapshot;
  }

  step(value: unknown): EngineSnapshot {
    if (
      this.snapshot.status === "idle" ||
      this.snapshot.status === "paused" ||
      this.snapshot.status === "extracted"
    ) {
      return this.cloneCurrentSnapshot();
    }

    const tick = this.snapshot.tick + 1;
    const validation = this.validateObservation(value);
    if (!validation.valid) {
      return this.stopForInvalidObservation(validation.reason, tick);
    }

    const observation = validation.observation;
    this.lastCapturedAt = observation.capturedAt;

    if (observation.confidence < this.scenario.localizationThreshold) {
      return this.requestRelocalization(
        observation.confidence,
        tick,
        `定位置信度 ${observation.confidence.toFixed(2)} 低于阈值`,
      );
    }

    if (this.isUnexpectedNavigationNode(observation.nodeId)) {
      return this.requestRelocalization(
        observation.confidence,
        tick,
        `导航观测跳到非相邻节点 "${observation.nodeId}"，请求重新定位`,
      );
    }

    if (observation.nodeId === this.scenario.extractNodeId) {
      return this.navigateFromObservation(observation, tick);
    }

    if (observation.stuck === true) {
      return this.recoverFromObservation(observation, tick);
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

  private cloneCurrentSnapshot(): EngineSnapshot {
    this.snapshot = freezeSnapshot({ ...this.snapshot });
    return this.snapshot;
  }

  private validateObservation(value: unknown): ObservationValidation {
    if (value === null || typeof value !== "object" || Array.isArray(value)) {
      return { valid: false, reason: "观测必须是对象，已安全停止" };
    }

    const candidate = value as Record<string, unknown>;
    if (typeof candidate.nodeId !== "string") {
      return { valid: false, reason: "观测节点必须是字符串，已安全停止" };
    }
    if (!isKnownNode(this.scenario, candidate.nodeId)) {
      return {
        valid: false,
        reason: `观测包含未知节点 "${candidate.nodeId}"，已安全停止`,
      };
    }
    if (
      typeof candidate.confidence !== "number" ||
      !Number.isFinite(candidate.confidence) ||
      candidate.confidence < 0 ||
      candidate.confidence > 1
    ) {
      return {
        valid: false,
        reason: "观测置信度必须是 0 到 1 的有限数，已安全停止",
      };
    }
    if (
      candidate.stuck !== undefined &&
      typeof candidate.stuck !== "boolean"
    ) {
      return {
        valid: false,
        reason: "观测卡住标记必须是布尔值，已安全停止",
      };
    }
    if (
      typeof candidate.capturedAt !== "number" ||
      !Number.isFinite(candidate.capturedAt) ||
      candidate.capturedAt < 0
    ) {
      return {
        valid: false,
        reason: "观测时间必须是非负有限数，已安全停止",
      };
    }
    if (candidate.capturedAt <= this.lastCapturedAt) {
      return {
        valid: false,
        reason: `观测时间 ${candidate.capturedAt} 未严格递增，已安全停止`,
      };
    }

    return { valid: true, observation: candidate as unknown as RunnerObservation };
  }

  private isUnexpectedNavigationNode(nodeId: string): boolean {
    if (
      this.snapshot.status !== "navigating" &&
      this.snapshot.status !== "recovering"
    ) {
      return false;
    }

    return (
      nodeId !== this.snapshot.currentNodeId &&
      nodeId !== this.snapshot.targetNodeId
    );
  }

  private requestRelocalization(
    confidence: number,
    tick: number,
    message: string,
  ): EngineSnapshot {
    this.snapshot = this.withEvent(
      {
        ...this.snapshot,
        status: "localizing",
        tick,
        confidence,
        action: { type: "relocalize", ttlMs: RELOCALIZE_TTL_MS },
      },
      { tick, kind: "warning", message },
    );
    return this.snapshot;
  }

  private recoverFromObservation(
    observation: RunnerObservation,
    tick: number,
  ): EngineSnapshot {
    let route = this.snapshot.route;
    if (this.snapshot.status === "localizing" || route.length === 0) {
      try {
        route = findShortestPath(
          this.scenario.graph,
          observation.nodeId,
          this.scenario.extractNodeId,
        );
      } catch (error) {
        return this.stopForPlanningFailure(this.errorMessage(error), tick);
      }
    }

    const routeIndex = route.indexOf(observation.nodeId);
    const targetNodeId = route[routeIndex + 1];
    if (!targetNodeId) {
      return this.stopForPlanningFailure(
        `节点 "${observation.nodeId}" 后没有可恢复的路线目标`,
        tick,
      );
    }

    const recoveryIncrement = this.snapshot.status === "recovering" ? 0 : 1;
    this.snapshot = this.withEvent(
      {
        ...this.snapshot,
        status: "recovering",
        tick,
        currentNodeId: observation.nodeId,
        targetNodeId,
        route,
        confidence: observation.confidence,
        action: {
          type: "recover",
          strategy: "backtrack",
          ttlMs: RECOVER_TTL_MS,
        },
        metrics: {
          ...this.snapshot.metrics,
          recoveryCount:
            this.snapshot.metrics.recoveryCount + recoveryIncrement,
          routeProgress: calculateRouteProgress(route, observation.nodeId),
        },
      },
      { tick, kind: "warning", message: "检测到卡住，执行回退恢复" },
    );
    return this.snapshot;
  }

  private navigateFromObservation(
    observation: RunnerObservation,
    tick: number,
  ): EngineSnapshot {
    let route = this.snapshot.route;
    if (this.snapshot.status === "localizing" || route.length === 0) {
      try {
        route = findShortestPath(
          this.scenario.graph,
          observation.nodeId,
          this.scenario.extractNodeId,
        );
      } catch (error) {
        return this.stopForPlanningFailure(this.errorMessage(error), tick);
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

  private stopForInvalidObservation(reason: string, tick: number): EngineSnapshot {
    this.resumableState = undefined;
    this.snapshot = this.withEvent(
      {
        ...this.snapshot,
        status: "paused",
        tick,
        action: { type: "stop", reason },
        metrics: {
          ...this.snapshot.metrics,
          invalidObservationCount:
            this.snapshot.metrics.invalidObservationCount + 1,
        },
      },
      { tick, kind: "error", message: reason },
    );
    return this.snapshot;
  }

  private stopForPlanningFailure(reason: string, tick: number): EngineSnapshot {
    this.resumableState = undefined;
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

  private errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : "路线规划发生未知错误";
  }

  private withEvent(
    snapshot: EngineSnapshot,
    event: RunnerEvent,
  ): EngineSnapshot {
    const events = [...snapshot.events, event].slice(-this.eventLimit);
    return freezeSnapshot({ ...snapshot, events });
  }
}
