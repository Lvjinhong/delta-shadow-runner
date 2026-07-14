import { describe, expect, it } from "vitest";

import { RunnerEngine } from "../../src/core/engine.js";
import {
  createDeterministicSimulationSource,
  defaultScenario,
} from "../../src/core/scenario.js";
import type { RunnerObservation } from "../../src/core/types.js";

function observation(
  nodeId: string,
  options: Partial<RunnerObservation> = {},
): RunnerObservation {
  return {
    nodeId,
    confidence: 0.95,
    stuck: false,
    ...options,
  };
}

describe("RunnerEngine", () => {
  it("从定位、导航、恢复到撤离形成完整闭环", () => {
    const engine = new RunnerEngine(defaultScenario);

    const localizing = engine.start();
    expect(localizing.status).toBe("localizing");
    expect(localizing.action).toEqual({ type: "relocalize", ttlMs: 1_000 });

    const navigating = engine.step(observation("spawn-a"));
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

    const advanced = engine.step(observation("relay"));
    expect(advanced.currentNodeId).toBe("relay");
    expect(advanced.targetNodeId).toBe("warehouse");

    const recovering = engine.step(
      observation("relay", { stuck: true }),
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

    const resumed = engine.step(observation("relay"));
    expect(resumed.status).toBe("navigating");
    expect(resumed.targetNodeId).toBe("warehouse");
    expect(resumed.action.type).toBe("move");

    engine.step(observation("warehouse"));
    const extracted = engine.step(observation("extract"));
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
    engine.step(observation("spawn-a"));

    const paused = engine.pause();
    const ignored = engine.step(observation("relay"));
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

  it("低置信观测触发重新定位，高置信观测后重新规划", () => {
    const engine = new RunnerEngine(defaultScenario);
    engine.start();
    engine.step(observation("spawn-a"));

    const uncertain = engine.step(
      observation("relay", { confidence: 0.4 }),
    );
    expect(uncertain.status).toBe("localizing");
    expect(uncertain.currentNodeId).toBe("spawn-a");
    expect(uncertain.action.type).toBe("relocalize");

    const relocalized = engine.step(observation("relay"));
    expect(relocalized.status).toBe("navigating");
    expect(relocalized.route).toEqual(["relay", "warehouse", "extract"]);
    expect(relocalized.targetNodeId).toBe("warehouse");
  });

  it.each<RunnerObservation>([
    { nodeId: "missing", confidence: 0.9, stuck: false },
    { nodeId: "spawn-a", confidence: Number.NaN, stuck: false },
    { nodeId: "spawn-a", confidence: 1.1, stuck: false },
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
      observation("spawn-a", { metadata: Object.freeze({ source: "test" }) }),
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
    engine.step(observation("spawn-a", { confidence: 0.2 }));
    engine.step(observation("spawn-a", { confidence: 0.3 }));
    const snapshot = engine.step(observation("spawn-a", { confidence: 0.4 }));

    expect(snapshot.events).toHaveLength(3);
    expect(snapshot.events[0]?.tick).toBeGreaterThan(0);
    expect(snapshot.events.at(-1)?.message).toContain("置信度");
  });
});

describe("DeterministicSimulationSource", () => {
  it("固定输出路线观测，可注入一次卡住并可重置", () => {
    const source = createDeterministicSimulationSource(defaultScenario);

    expect(source.next()).toEqual(observation("spawn-a"));
    source.injectStuck();
    expect(source.next()).toEqual(observation("relay", { stuck: true }));
    expect(source.next()).toEqual(observation("relay"));
    source.reset();
    expect(source.next()).toEqual(observation("spawn-a"));
  });
});
