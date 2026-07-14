import { RunnerEngine } from "../core/engine.js";
import {
  createDeterministicSimulationSource,
  defaultScenario,
} from "../core/scenario.js";
import type {
  EngineSnapshot,
  RunnerCapabilities,
  RunnerScenario,
  SimulationSource,
} from "../core/types.js";

export type ControlCommand = "start" | "pause" | "reset" | "inject-stuck";

export interface RuntimeScheduler {
  setInterval(callback: () => void, tickMs: number): unknown;
  clearInterval(handle: unknown): void;
}

export interface RunnerRuntimeOptions {
  readonly scenario?: RunnerScenario;
  readonly engine?: RunnerEngine;
  readonly source?: SimulationSource;
  readonly tickMs?: number;
  readonly scheduler?: RuntimeScheduler;
}

export type SnapshotListener = (snapshot: EngineSnapshot) => void;

const defaultScheduler: RuntimeScheduler = {
  setInterval(callback, tickMs) {
    return globalThis.setInterval(callback, tickMs);
  },
  clearInterval(handle) {
    globalThis.clearInterval(handle as ReturnType<typeof setInterval>);
  },
};

export class RunnerRuntime {
  readonly scenario: RunnerScenario;
  readonly tickMs: number;

  private readonly engine: RunnerEngine;
  private readonly source: SimulationSource;
  private readonly scheduler: RuntimeScheduler;
  private readonly listeners = new Set<SnapshotListener>();
  private timerHandle: unknown = undefined;
  private running = false;

  constructor(options: RunnerRuntimeOptions = {}) {
    this.scenario = options.scenario ?? defaultScenario;
    this.engine = options.engine ?? new RunnerEngine(this.scenario);
    this.source =
      options.source ?? createDeterministicSimulationSource(this.scenario);
    this.tickMs = options.tickMs ?? 500;
    this.scheduler = options.scheduler ?? defaultScheduler;

    if (!Number.isFinite(this.tickMs) || this.tickMs <= 0) {
      throw new Error("运行时 tickMs 必须是正有限数");
    }
  }

  get isRunning(): boolean {
    return this.running;
  }

  get subscriberCount(): number {
    return this.listeners.size;
  }

  get capabilities(): RunnerCapabilities {
    const status = this.engine.getSnapshot().status;
    const active =
      status === "localizing" ||
      status === "navigating" ||
      status === "recovering";
    return Object.freeze({
      canStart: status === "idle" || this.engine.canResume,
      canPause: active,
      canReset: true,
      canInjectStuck: active,
    });
  }

  getSnapshot(): EngineSnapshot {
    return this.engine.getSnapshot();
  }

  start(): void {
    if (this.running) {
      return;
    }

    this.timerHandle = this.scheduler.setInterval(() => {
      this.tick();
    }, this.tickMs);
    this.running = true;
  }

  stop(): void {
    if (!this.running) {
      return;
    }

    this.scheduler.clearInterval(this.timerHandle);
    this.running = false;
    this.timerHandle = undefined;
  }

  tick(): EngineSnapshot {
    const current = this.engine.getSnapshot();
    if (
      current.status !== "localizing" &&
      current.status !== "navigating" &&
      current.status !== "recovering"
    ) {
      return current;
    }

    let snapshot: EngineSnapshot;
    try {
      snapshot = this.engine.step(this.source.next());
    } catch (error) {
      // 模拟源异常也必须走引擎的安全停止路径，不能让 interval 抛出未处理异常。
      snapshot = this.engine.step({
        sourceError:
          error instanceof Error ? error.message : "模拟源发生未知错误",
      });
    }
    this.publish(snapshot);
    return snapshot;
  }

  control(command: ControlCommand): EngineSnapshot {
    let snapshot: EngineSnapshot;
    switch (command) {
      case "start":
        snapshot = this.engine.start();
        break;
      case "pause":
        snapshot = this.engine.pause();
        break;
      case "reset":
        this.source.reset();
        snapshot = this.engine.reset();
        break;
      case "inject-stuck":
        if (this.capabilities.canInjectStuck) {
          this.source.injectStuck();
        }
        snapshot = this.engine.getSnapshot();
        break;
    }

    this.publish(snapshot);
    return snapshot;
  }

  subscribe(listener: SnapshotListener): () => void {
    this.listeners.add(listener);
    let active = true;

    return () => {
      if (!active) {
        return;
      }
      active = false;
      this.listeners.delete(listener);
    };
  }

  private publish(snapshot: EngineSnapshot): void {
    for (const listener of [...this.listeners]) {
      listener(snapshot);
    }
  }
}
