import { describe, expect, it } from "vitest";

import { RunnerEngine } from "../../src/core/engine.js";
import {
  createDeterministicSimulationSource,
  defaultScenario,
} from "../../src/core/scenario.js";
import type { RunnerObservation } from "../../src/core/types.js";

function observation(
  nodeId: string,
  capturedAt = 0,
  options: Partial<RunnerObservation> & { readonly capturedAt?: number } = {},
): RunnerObservation {
  return {
    nodeId,
    confidence: 0.95,
    stuck: false,
    capturedAt,
    ...options,
  } as RunnerObservation;
}

describe("RunnerEngine", () => {
  it("从定位、导航、恢复到撤离形成完整闭环", () => {
    const engine = new RunnerEngine(defaultScenario);

    const localizing = engine.start();
    expect(localizing.status).toBe("localizing");
    expect(localizing.action).toEqual({ type: "relocalize", ttlMs: 1_000 });

    const navigating = engine.step(observation("spawn-a", 0));
    expect(navigating.status).toBe("navigating");
    expect(navigating.route).toEqual([
      "spawn-a",
      "relay",
      "warehouse",
      "extract",
    ]);
    expect(navigating.targetNodeId).toBe("relay");
    expect(navigating.action).toEqual({
      type: "move",
      targetNodeId: "relay",
      ttlMs: 750,
    });

    const advanced = engine.step(observation("relay", 1));
    expect(advanced.currentNodeId).toBe("relay");
    expect(advanced.targetNodeId).toBe("warehouse");

    const recovering = engine.step(
      observation("relay", 2, { stuck: true }),
    );
    expect(recovering.status).toBe("recovering");
    expect(recovering.route).toEqual(navigating.route);
    expect(recovering.targetNodeId).toBe("warehouse");
    expect(recovering.action).toEqual({
      type: "recover",
      strategy: "backtrack",
      ttlMs: 1_250,
    });
    expect(recovering.metrics.recoveryCount).toBe(1);

    const resumed = engine.step(observation("relay", 3));
    expect(resumed.status).toBe("navigating");
    expect(resumed.targetNodeId).toBe("warehouse");
    expect(resumed.action.type).toBe("move");

    engine.step(observation("warehouse", 4));
    const extracted = engine.step(observation("extract", 5));
    expect(extracted.status).toBe("extracted");
    expect(extracted.currentNodeId).toBe("extract");
    expect(extracted.targetNodeId).toBe("extract");
    expect(extracted.action).toEqual({
      type: "stop",
      reason: "已到达撤离点",
    });
    expect(extracted.metrics.routeProgress).toBe(1);
  });

  it("暂停后忽略观测，重置后回到可重复的初始状态", () => {
    const engine = new RunnerEngine(defaultScenario);
    engine.start();
    engine.step(observation("spawn-a", 0));

    const paused = engine.pause();
    const ignored = engine.step(observation("relay", 1));
    expect(paused.status).toBe("paused");
    expect(ignored.status).toBe("paused");
    expect(ignored.currentNodeId).toBe("spawn-a");
    expect(ignored.action.type).toBe("stop");

    const reset = engine.reset();
    expect(reset.status).toBe("idle");
    expect(reset.tick).toBe(0);
    expect(reset.currentNodeId).toBe("spawn-a");
    expect(reset.targetNodeId).toBe("spawn-a");
    expect(reset.route).toEqual([]);
    expect(reset.events).toHaveLength(1);
    expect(reset.runId).not.toBe(paused.runId);
  });

  it("只允许人工暂停恢复，安全停止不能被 start 猜测为可恢复", () => {
    const manuallyPaused = new RunnerEngine(defaultScenario);
    expect(manuallyPaused.canResume).toBe(false);
    manuallyPaused.start();
    manuallyPaused.step(observation("spawn-a", 0));
    manuallyPaused.pause();
    expect(manuallyPaused.canResume).toBe(true);
    expect(manuallyPaused.start().status).toBe("navigating");
    expect(manuallyPaused.canResume).toBe(false);

    const safetyStopped = new RunnerEngine(defaultScenario);
    safetyStopped.start();
    safetyStopped.step({
      nodeId: "missing",
      confidence: 0.9,
      capturedAt: 0,
    });
    expect(safetyStopped.getSnapshot().status).toBe("paused");
    expect(safetyStopped.canResume).toBe(false);
    expect(safetyStopped.start().status).toBe("paused");
  });

  it("低置信观测触发重新定位，高置信观测后重新规划", () => {
    const engine = new RunnerEngine(defaultScenario);
    engine.start();
    engine.step(observation("spawn-a", 0));

    const uncertain = engine.step(
      observation("relay", 1, { confidence: 0.4 }),
    );
    expect(uncertain.status).toBe("localizing");
    expect(uncertain.currentNodeId).toBe("spawn-a");
    expect(uncertain.action.type).toBe("relocalize");

    const relocalized = engine.step(observation("relay", 2));
    expect(relocalized.status).toBe("navigating");
    expect(relocalized.route).toEqual(["relay", "warehouse", "extract"]);
    expect(relocalized.targetNodeId).toBe("warehouse");
  });

  it.each<RunnerObservation>([
    { nodeId: "missing", confidence: 0.9, stuck: false, capturedAt: 0 },
    {
      nodeId: "spawn-a",
      confidence: Number.NaN,
      stuck: false,
      capturedAt: 0,
    },
    { nodeId: "spawn-a", confidence: 1.1, stuck: false, capturedAt: 0 },
  ])("非法观测安全停止且不抛异常: %j", (invalidObservation) => {
    const engine = new RunnerEngine(defaultScenario);
    engine.start();

    expect(() => engine.step(invalidObservation)).not.toThrow();
    const snapshot = engine.getSnapshot();
    expect(snapshot.status).toBe("paused");
    expect(snapshot.action.type).toBe("stop");
    expect(snapshot.events.at(-1)?.kind).toBe("error");
    expect(snapshot.metrics.invalidObservationCount).toBe(1);
  });

  it("不修改观测或历史快照，并为每一步返回新快照", () => {
    const engine = new RunnerEngine(defaultScenario);
    const started = engine.start();
    const input = Object.freeze(
      observation("spawn-a", 0, {
        metadata: Object.freeze({ source: "test" }),
      }),
    );
    const before = JSON.stringify(input);

    const next = engine.step(input);
    expect(next).not.toBe(started);
    expect(JSON.stringify(input)).toBe(before);
    expect(started.status).toBe("localizing");
    expect(started.route).toEqual([]);
    expect(Object.isFrozen(next)).toBe(true);
    expect(Object.isFrozen(next.route)).toBe(true);
    expect(Object.isFrozen(next.events)).toBe(true);
  });

  it("事件数量受上限约束", () => {
    const engine = new RunnerEngine(defaultScenario, { eventLimit: 3 });
    engine.start();
    engine.step(observation("spawn-a", 0, { confidence: 0.2 }));
    engine.step(observation("spawn-a", 1, { confidence: 0.3 }));
    const snapshot = engine.step(
      observation("spawn-a", 2, { confidence: 0.4 }),
    );

    expect(snapshot.events).toHaveLength(3);
    expect(snapshot.events[0]?.tick).toBeGreaterThan(0);
    expect(snapshot.events.at(-1)?.message).toContain("置信度");
  });
});

describe("DeterministicSimulationSource", () => {
  it("固定输出路线观测，可注入一次卡住并可重置", () => {
    const source = createDeterministicSimulationSource(defaultScenario);

    expect(source.next()).toEqual(observation("spawn-a", 0));
    source.injectStuck();
    expect(source.next()).toEqual(observation("relay", 1, { stuck: true }));
    expect(source.next()).toEqual(observation("relay", 2));
    source.reset();
    expect(source.next()).toEqual(observation("spawn-a", 0));
  });
});

describe("RunnerEngine 边界状态", () => {
  it.each(["warehouse", "extract"])(
    "不会从 spawn-a 直接跳到 %s",
    (unexpectedNodeId) => {
      const engine = new RunnerEngine(defaultScenario);
      engine.start();
      const navigating = engine.step(observation("spawn-a", 0));

      const unexpected = engine.step(observation(unexpectedNodeId, 1));

      expect(unexpected.status).toBe("localizing");
      expect(unexpected.currentNodeId).toBe("spawn-a");
      expect(unexpected.targetNodeId).toBe("relay");
      expect(unexpected.route).toEqual(navigating.route);
      expect(unexpected.metrics.routeProgress).toBe(0);
      expect(unexpected.action).toEqual({
        type: "relocalize",
        ttlMs: 1_000,
      });
    },
  );

  it("不会从 warehouse 倒退到 relay", () => {
    const engine = new RunnerEngine(defaultScenario);
    engine.start();
    engine.step(observation("spawn-a", 0));
    engine.step(observation("relay", 1));
    const warehouse = engine.step(observation("warehouse", 2));

    const backwards = engine.step(observation("relay", 3));

    expect(backwards.status).toBe("localizing");
    expect(backwards.currentNodeId).toBe("warehouse");
    expect(backwards.targetNodeId).toBe("extract");
    expect(backwards.metrics.routeProgress).toBe(
      warehouse.metrics.routeProgress,
    );
    expect(backwards.action.type).toBe("relocalize");
  });

  it.each([null, undefined, "spawn-a", []])(
    "非法 runtime 观测安全停止且不抛异常: %j",
    (invalidObservation) => {
      const engine = new RunnerEngine(defaultScenario);
      engine.start();

      expect(() =>
        engine.step(invalidObservation as unknown as RunnerObservation),
      ).not.toThrow();
      const snapshot = engine.getSnapshot();
      expect(snapshot.status).toBe("paused");
      expect(snapshot.action.type).toBe("stop");
      expect(snapshot.metrics.invalidObservationCount).toBe(1);
    },
  );

  it("idle、paused、extracted 忽略 step 且不增加 tick", () => {
    const idleEngine = new RunnerEngine(defaultScenario);
    const idle = idleEngine.getSnapshot();
    const ignoredIdle = idleEngine.step(observation("spawn-a", 0));
    expect(ignoredIdle).not.toBe(idle);
    expect(ignoredIdle.tick).toBe(idle.tick);

    const pausedEngine = new RunnerEngine(defaultScenario);
    pausedEngine.start();
    pausedEngine.step(observation("spawn-a", 0));
    const paused = pausedEngine.pause();
    const ignoredPaused = pausedEngine.step(observation("relay", 1));
    expect(ignoredPaused).not.toBe(paused);
    expect(ignoredPaused.tick).toBe(paused.tick);

    const extractedEngine = new RunnerEngine(defaultScenario);
    extractedEngine.start();
    extractedEngine.step(observation("spawn-a", 0));
    extractedEngine.step(observation("relay", 1));
    extractedEngine.step(observation("warehouse", 2));
    const extracted = extractedEngine.step(observation("extract", 3));
    const ignoredExtracted = extractedEngine.step(observation("extract", 4));
    expect(ignoredExtracted).not.toBe(extracted);
    expect(ignoredExtracted.tick).toBe(extracted.tick);
  });

  it("start 在活动态幂等，暂停后恢复同一 run 和进度", () => {
    const engine = new RunnerEngine(defaultScenario);
    engine.start();
    const navigating = engine.step(observation("spawn-a", 0));

    const duplicateStart = engine.start();
    expect(duplicateStart).not.toBe(navigating);
    expect(duplicateStart.runId).toBe(navigating.runId);
    expect(duplicateStart.status).toBe("navigating");
    expect(duplicateStart.route).toEqual(navigating.route);

    const paused = engine.pause();
    const resumed = engine.start();
    expect(resumed.runId).toBe(paused.runId);
    expect(resumed.status).toBe("navigating");
    expect(resumed.currentNodeId).toBe("spawn-a");
    expect(resumed.targetNodeId).toBe("relay");
    expect(resumed.route).toEqual(navigating.route);
    expect(resumed.action).toEqual(navigating.action);
  });

  it("extracted 后 pause 和 start 均保持终态", () => {
    const engine = new RunnerEngine(defaultScenario);
    engine.start();
    engine.step(observation("spawn-a", 0));
    engine.step(observation("relay", 1));
    engine.step(observation("warehouse", 2));
    const extracted = engine.step(observation("extract", 3));

    const paused = engine.pause();
    const started = engine.start();
    expect(paused.status).toBe("extracted");
    expect(started.status).toBe("extracted");
    expect(started.runId).toBe(extracted.runId);
    expect(started.metrics.routeProgress).toBe(1);
  });

  it("在当前 target 卡住时先推进，并且连续卡住只计一次恢复", () => {
    const engine = new RunnerEngine(defaultScenario);
    engine.start();
    engine.step(observation("spawn-a", 0));

    const stuckAtRelay = engine.step(
      observation("relay", 1, { stuck: true }),
    );
    expect(stuckAtRelay.status).toBe("recovering");
    expect(stuckAtRelay.currentNodeId).toBe("relay");
    expect(stuckAtRelay.targetNodeId).toBe("warehouse");
    expect(stuckAtRelay.metrics.routeProgress).toBeCloseTo(1 / 3);
    expect(stuckAtRelay.metrics.recoveryCount).toBe(1);

    const stuckAtWarehouse = engine.step(
      observation("warehouse", 2, { stuck: true }),
    );
    expect(stuckAtWarehouse.status).toBe("recovering");
    expect(stuckAtWarehouse.currentNodeId).toBe("warehouse");
    expect(stuckAtWarehouse.targetNodeId).toBe("extract");
    expect(stuckAtWarehouse.metrics.routeProgress).toBeCloseTo(2 / 3);
    expect(stuckAtWarehouse.metrics.recoveryCount).toBe(1);
  });

  it.each([-1, Number.NaN, Number.POSITIVE_INFINITY])(
    "拒绝非法 capturedAt: %s",
    (capturedAt) => {
      const engine = new RunnerEngine(defaultScenario);
      engine.start();
      const invalid = engine.step(observation("spawn-a", capturedAt));

      expect(invalid.status).toBe("paused");
      expect(invalid.currentNodeId).toBe("spawn-a");
      expect(invalid.action.type).toBe("stop");
    },
  );

  it("capturedAt 必须严格递增，重复或乱序观测不得推进", () => {
    const engine = new RunnerEngine(defaultScenario);
    engine.start();
    const navigating = engine.step(observation("spawn-a", 10));

    const duplicate = engine.step(observation("relay", 10));
    expect(duplicate.status).toBe("paused");
    expect(duplicate.currentNodeId).toBe(navigating.currentNodeId);
    expect(duplicate.targetNodeId).toBe(navigating.targetNodeId);
    expect(duplicate.metrics.routeProgress).toBe(
      navigating.metrics.routeProgress,
    );

    engine.reset();
    engine.start();
    engine.step(observation("spawn-a", 10));
    const outOfOrder = engine.step(observation("relay", 9));
    expect(outOfOrder.status).toBe("paused");
    expect(outOfOrder.currentNodeId).toBe("spawn-a");
    expect(outOfOrder.targetNodeId).toBe("relay");
  });
});
