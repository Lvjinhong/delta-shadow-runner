import { afterEach, describe, expect, it, vi } from "vitest";

import {
  createTelemetryClient,
  sendControl,
  type TelemetryClient,
  type TelemetryClientCallbacks,
} from "../../src/web/api.js";

const telemetryData = {
  capabilities: {
    canStart: false,
    canPause: true,
    canReset: true,
    canInjectStuck: true,
  },
  snapshot: {
    runId: "run-1",
    status: "localizing",
    tick: 0,
    currentNodeId: "spawn-a",
    targetNodeId: "spawn-a",
    route: [],
    confidence: 0,
    action: { type: "relocalize", ttlMs: 1_000 },
    metrics: {
      recoveryCount: 0,
      invalidObservationCount: 0,
      routeProgress: 0,
    },
    events: [{ tick: 0, kind: "info", message: "任务已启动" }],
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
          edges: [{ targetNodeId: "extract", cost: 1 }],
        },
        { id: "extract", x: 92, y: 42, edges: [] },
      ],
    },
  },
} as const;

function jsonResponse(value: unknown): Response {
  return new Response(JSON.stringify(value), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

type SocketEvent = "open" | "message" | "error" | "close";

class FakeSocket {
  readyState = 0;
  closeCalls = 0;
  readonly listeners = new Map<SocketEvent, Array<(event: { data?: unknown }) => void>>();

  addEventListener(
    type: SocketEvent,
    listener: (event: { data?: unknown }) => void,
  ): void {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  close(): void {
    this.closeCalls += 1;
    this.readyState = 3;
  }

  emit(type: SocketEvent, data?: unknown): void {
    if (type === "open") {
      this.readyState = 1;
    } else if (type === "close") {
      this.readyState = 3;
    }
    for (const listener of this.listeners.get(type) ?? []) {
      listener({ data });
    }
  }
}

class FakeTimerScheduler {
  private nextId = 0;
  readonly delays: number[] = [];
  readonly callbacks = new Map<number, () => void>();
  readonly cleared: number[] = [];

  setTimeout(callback: () => void, delay: number): number {
    this.nextId += 1;
    this.delays.push(delay);
    this.callbacks.set(this.nextId, callback);
    return this.nextId;
  }

  clearTimeout(handle: unknown): void {
    const id = handle as number;
    this.cleared.push(id);
    this.callbacks.delete(id);
  }

  runNext(): void {
    const entry = this.callbacks.entries().next().value as
      | [number, () => void]
      | undefined;
    if (!entry) {
      throw new Error("没有待执行的 timer");
    }
    this.callbacks.delete(entry[0]);
    entry[1]();
  }
}

interface TestEnvironment {
  readonly fetch: typeof fetch;
  readonly createWebSocket: (url: string) => FakeSocket;
  readonly setTimeout: (callback: () => void, delay: number) => unknown;
  readonly clearTimeout: (handle: unknown) => void;
  readonly location: { readonly protocol: string; readonly host: string };
}

const createClientWithEnvironment = createTelemetryClient as unknown as (
  callbacks: TelemetryClientCallbacks,
  environment: TestEnvironment,
) => TelemetryClient;

function clientHarness(): {
  readonly client: TelemetryClient;
  readonly sockets: FakeSocket[];
  readonly scheduler: FakeTimerScheduler;
  readonly statuses: string[];
  readonly errors: string[];
  readonly dataTicks: number[];
} {
  const sockets: FakeSocket[] = [];
  const scheduler = new FakeTimerScheduler();
  const statuses: string[] = [];
  const errors: string[] = [];
  const dataTicks: number[] = [];
  const environment: TestEnvironment = {
    fetch: vi.fn<typeof fetch>(),
    createWebSocket(url) {
      expect(url).toBe("ws://runner.test/ws");
      const socket = new FakeSocket();
      sockets.push(socket);
      return socket;
    },
    setTimeout: (callback, delay) => scheduler.setTimeout(callback, delay),
    clearTimeout: (handle) => scheduler.clearTimeout(handle),
    location: { protocol: "http:", host: "runner.test" },
  };
  const client = createClientWithEnvironment(
    {
      onData(data) {
        dataTicks.push(data.snapshot.tick);
      },
      onStatus(status) {
        statuses.push(status.phase);
      },
      onError(error) {
        errors.push(error.message);
      },
    },
    environment,
  );
  return { client, sockets, scheduler, statuses, errors, dataTicks };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("dashboard control client", () => {
  it("一次 POST 直接解析完整 TelemetryData，不再追加 GET", async () => {
    const fetchMock = vi.fn<typeof fetch>(async (input, init) => {
      expect(input).toBe("/api/control/start");
      expect(init?.method).toBe("POST");
      return jsonResponse({ success: true, data: telemetryData, error: null });
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(sendControl("start")).resolves.toEqual(telemetryData);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("严格拒绝缺失 capabilities 的控制响应", async () => {
    const { capabilities: _capabilities, ...incomplete } = telemetryData;
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>(async () =>
        jsonResponse({ success: true, data: incomplete, error: null }),
      ),
    );

    await expect(sendControl("start")).rejects.toThrow(/capabilities/);
  });
});

describe("dashboard telemetry client lifecycle", () => {
  it("按 500/1000/2000/4000/8000ms 重连五次后明确停止", () => {
    const harness = clientHarness();
    harness.client.connect();

    for (let attempt = 0; attempt < 5; attempt += 1) {
      harness.sockets.at(-1)?.emit("close");
      harness.scheduler.runNext();
    }
    harness.sockets.at(-1)?.emit("close");

    expect(harness.sockets).toHaveLength(6);
    expect(harness.scheduler.delays).toEqual([500, 1_000, 2_000, 4_000, 8_000]);
    expect(harness.statuses.at(-1)).toBe("disconnected");
    expect(harness.errors.at(-1)).toContain("重连次数已用尽");
    expect(harness.scheduler.callbacks.size).toBe(0);
  });

  it("dispose 清理 timer/socket，之后的事件不再触发任何回调", () => {
    const harness = clientHarness();
    harness.client.connect();
    const socket = harness.sockets[0];
    expect(socket).toBeDefined();
    socket?.emit("close");
    const callbackCounts = {
      statuses: harness.statuses.length,
      errors: harness.errors.length,
      data: harness.dataTicks.length,
    };

    harness.client.dispose();
    socket?.emit("open");
    socket?.emit(
      "message",
      JSON.stringify({ type: "snapshot", data: telemetryData }),
    );
    socket?.emit("error");
    socket?.emit("close");

    expect(harness.scheduler.callbacks.size).toBe(0);
    expect(harness.scheduler.cleared).toHaveLength(1);
    expect(socket?.closeCalls).toBe(0);
    expect(harness.statuses).toHaveLength(callbackCounts.statuses);
    expect(harness.errors).toHaveLength(callbackCounts.errors);
    expect(harness.dataTicks).toHaveLength(callbackCounts.data);
  });

  it("dispose 主动关闭仍连接的 socket 且不派发后续消息", () => {
    const harness = clientHarness();
    harness.client.connect();
    const socket = harness.sockets[0];
    socket?.emit("open");

    harness.client.dispose();
    const statusCount = harness.statuses.length;
    socket?.emit(
      "message",
      JSON.stringify({ type: "snapshot", data: telemetryData }),
    );

    expect(socket?.closeCalls).toBe(1);
    expect(harness.statuses).toHaveLength(statusCount);
    expect(harness.dataTicks).toEqual([]);
  });
});
