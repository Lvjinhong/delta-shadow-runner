import type { AddressInfo } from "node:net";

import request from "supertest";
import { afterEach, describe, expect, it } from "vitest";
import WebSocket from "ws";

import { createRunnerApp, createRunnerServer } from "../../src/server/app.js";
import {
  RunnerRuntime,
  type RuntimeScheduler,
} from "../../src/server/runtime.js";

class FakeScheduler implements RuntimeScheduler {
  private nextId = 0;
  private readonly callbacks = new Map<number, () => void>();

  setInterval(callback: () => void, _tickMs: number): unknown {
    this.nextId += 1;
    this.callbacks.set(this.nextId, callback);
    return this.nextId;
  }

  clearInterval(handle: unknown): void {
    this.callbacks.delete(handle as number);
  }

  tick(): void {
    for (const callback of [...this.callbacks.values()]) {
      callback();
    }
  }

  get activeCount(): number {
    return this.callbacks.size;
  }
}

function waitForOpen(socket: WebSocket): Promise<void> {
  return new Promise((resolve, reject) => {
    socket.once("open", () => resolve());
    socket.once("error", reject);
  });
}

function waitForClose(socket: WebSocket): Promise<void> {
  if (socket.readyState === WebSocket.CLOSED) {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    socket.once("close", () => resolve());
  });
}

function waitForMessage(
  socket: WebSocket,
  predicate: (message: Record<string, unknown>) => boolean,
): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error("等待 WebSocket 消息超时"));
    }, 1_000);

    const onMessage = (raw: WebSocket.RawData): void => {
      try {
        const message = JSON.parse(raw.toString()) as Record<string, unknown>;
        if (predicate(message)) {
          cleanup();
          resolve(message);
        }
      } catch (error) {
        cleanup();
        reject(error);
      }
    };

    const cleanup = (): void => {
      clearTimeout(timeout);
      socket.off("message", onMessage);
    };

    socket.on("message", onMessage);
  });
}

describe("RunnerRuntime", () => {
  it("以可注入的 CPU interval 推进模拟，且 start/stop 幂等并清理 timer", () => {
    const scheduler = new FakeScheduler();
    const runtime = new RunnerRuntime({ tickMs: 500, scheduler });
    const snapshots: number[] = [];
    const unsubscribe = runtime.subscribe((snapshot) => {
      snapshots.push(snapshot.tick);
    });

    runtime.start();
    runtime.start();
    expect(runtime.tickMs).toBe(500);
    expect(scheduler.activeCount).toBe(1);

    runtime.control("start");
    scheduler.tick();
    expect(runtime.getSnapshot().status).toBe("navigating");
    expect(runtime.getSnapshot().tick).toBe(1);
    expect(snapshots).toEqual([0, 1]);

    unsubscribe();
    scheduler.tick();
    expect(snapshots).toEqual([0, 1]);

    runtime.stop();
    runtime.stop();
    expect(scheduler.activeCount).toBe(0);
    const stoppedTick = runtime.getSnapshot().tick;
    scheduler.tick();
    expect(runtime.getSnapshot().tick).toBe(stoppedTick);
  });

  it("即使 scheduler 返回 undefined handle，start/stop 仍保持幂等", () => {
    let setCalls = 0;
    let clearCalls = 0;
    const scheduler: RuntimeScheduler = {
      setInterval() {
        setCalls += 1;
        return undefined;
      },
      clearInterval(handle) {
        expect(handle).toBeUndefined();
        clearCalls += 1;
      },
    };
    const runtime = new RunnerRuntime({ scheduler });

    runtime.start();
    runtime.start();
    expect(runtime.isRunning).toBe(true);
    expect(setCalls).toBe(1);

    runtime.stop();
    runtime.stop();
    expect(runtime.isRunning).toBe(false);
    expect(clearCalls).toBe(1);
  });
});

describe("Runner REST API", () => {
  it("健康检查声明 simulation 与 cpu-only 模式", async () => {
    const runtime = new RunnerRuntime();
    const response = await request(createRunnerApp(runtime)).get("/api/health");

    expect(response.status).toBe(200);
    expect(response.body).toEqual({
      success: true,
      data: {
        status: "ok",
        mode: "simulation",
        compute: "cpu-only",
        tickMs: 500,
        running: false,
      },
      error: null,
    });
  });

  it("快照使用统一 envelope 并包含固定场景地图", async () => {
    const runtime = new RunnerRuntime();
    const response = await request(createRunnerApp(runtime)).get(
      "/api/snapshot",
    );

    expect(response.status).toBe(200);
    expect(response.body.success).toBe(true);
    expect(response.body.error).toBeNull();
    expect(response.body.data.snapshot).toMatchObject({
      status: "idle",
      tick: 0,
      currentNodeId: "spawn-a",
      targetNodeId: "spawn-a",
    });
    expect(response.body.data.scenario).toMatchObject({
      id: "fixed-training-route",
      spawnNodeIds: ["spawn-a", "spawn-b"],
      defaultSpawnNodeId: "spawn-a",
      extractNodeId: "extract",
      map: {
        nodes: expect.arrayContaining([
          expect.objectContaining({ id: "spawn-a", x: 8, y: 72 }),
          expect.objectContaining({ id: "extract", x: 92, y: 42 }),
        ]),
      },
    });
  });

  it.each([
    ["start", "localizing"],
    ["pause", "paused"],
    ["reset", "idle"],
  ] as const)("控制命令 %s 返回服务端真实状态", async (command, status) => {
    const runtime = new RunnerRuntime();
    const app = createRunnerApp(runtime);
    if (command === "pause") {
      runtime.control("start");
    }

    const response = await request(app)
      .post(`/api/control/${command}`)
      .send({});

    expect(response.status).toBe(200);
    expect(response.body).toMatchObject({
      success: true,
      data: { snapshot: { status } },
      error: null,
    });
  });

  it("inject-stuck 只写入模拟源，并在下一 tick 进入恢复", async () => {
    const scheduler = new FakeScheduler();
    const runtime = new RunnerRuntime({ scheduler });
    const app = createRunnerApp(runtime);
    runtime.control("start");
    runtime.tick();
    expect(runtime.getSnapshot().status).toBe("navigating");

    const response = await request(app)
      .post("/api/control/inject-stuck")
      .send({});
    expect(response.status).toBe(200);
    expect(response.body.data.snapshot.status).toBe("navigating");

    runtime.tick();
    expect(runtime.getSnapshot().status).toBe("recovering");
    expect(runtime.getSnapshot().metrics.recoveryCount).toBe(1);
  });

  it.each([
    ["/api/control/launch", {}],
    ["/api/control/start/extra", {}],
    ["/api/control/start", { unexpected: true }],
  ])("拒绝无效控制 path/body: %s", async (path, body) => {
    const runtime = new RunnerRuntime();
    const response = await request(createRunnerApp(runtime)).post(path).send(body);

    expect(response.status).toBe(400);
    expect(response.body).toMatchObject({
      success: false,
      data: null,
      error: { code: "INVALID_CONTROL_REQUEST" },
    });
  });

  it("将无法解析的 JSON 转成统一 400 响应", async () => {
    const runtime = new RunnerRuntime();
    const response = await request(createRunnerApp(runtime))
      .post("/api/control/start")
      .set("content-type", "application/json")
      .send("{");

    expect(response.status).toBe(400);
    expect(response.body).toMatchObject({
      success: false,
      data: null,
      error: { code: "INVALID_JSON" },
    });
  });

  it.each([
    ["get", "/api/unknown", 404, "API_NOT_FOUND"],
    ["post", "/api/control", 400, "INVALID_CONTROL_REQUEST"],
  ] as const)(
    "未命中的 API 也返回 JSON envelope: %s %s",
    async (method, path, status, code) => {
      const runtime = new RunnerRuntime();
      const client = request(createRunnerApp(runtime));
      const response =
        method === "get" ? await client.get(path) : await client.post(path).send({});

      expect(response.status).toBe(status);
      expect(response.headers["content-type"]).toContain("application/json");
      expect(response.body).toMatchObject({
        success: false,
        data: null,
        error: { code },
      });
    },
  );

  it("将超限 JSON 转成统一 413 响应", async () => {
    const runtime = new RunnerRuntime();
    const response = await request(createRunnerApp(runtime))
      .post("/api/control/start")
      .send({ padding: "x".repeat(17 * 1_024) });

    expect(response.status).toBe(413);
    expect(response.body).toMatchObject({
      success: false,
      data: null,
      error: { code: "PAYLOAD_TOO_LARGE" },
    });
  });
});

describe("Runner WebSocket", () => {
  const closeTasks: Array<() => Promise<void>> = [];

  afterEach(async () => {
    await Promise.all(closeTasks.splice(0).map((close) => close()));
  });

  it("连接后发送连接状态与初始快照，REST 控制后广播新快照", async () => {
    const scheduler = new FakeScheduler();
    const runtime = new RunnerRuntime({ scheduler });
    const service = createRunnerServer(runtime);
    await service.listen(0, "127.0.0.1");
    runtime.start();
    closeTasks.push(() => service.close());

    const address = service.server.address() as AddressInfo;
    const socket = new WebSocket(`ws://127.0.0.1:${address.port}/ws`);
    closeTasks.push(async () => {
      socket.close();
      await waitForClose(socket);
    });

    const connectedPromise = waitForMessage(
      socket,
      (message) => message.type === "connection",
    );
    const initialPromise = waitForMessage(
      socket,
      (message) =>
        message.type === "snapshot" &&
        (message.data as { snapshot?: { status?: string } }).snapshot?.status ===
          "idle",
    );
    await waitForOpen(socket);
    await expect(connectedPromise).resolves.toMatchObject({
      type: "connection",
      data: { status: "connected", mode: "simulation" },
    });
    await expect(initialPromise).resolves.toMatchObject({
      type: "snapshot",
      data: {
        snapshot: { status: "idle" },
        scenario: { id: "fixed-training-route" },
      },
    });

    socket.send(JSON.stringify({ command: "start" }));
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(runtime.getSnapshot().status).toBe("idle");

    const broadcastPromise = waitForMessage(
      socket,
      (message) =>
        message.type === "snapshot" &&
        (message.data as { snapshot?: { status?: string } }).snapshot?.status ===
          "localizing",
    );
    const response = await request(service.server)
      .post("/api/control/start")
      .send({});
    expect(response.status).toBe(200);
    await expect(broadcastPromise).resolves.toMatchObject({
      type: "snapshot",
      data: { snapshot: { status: "localizing" } },
    });
  });

  it("关闭服务后断开客户端、取消订阅并清理 interval", async () => {
    const scheduler = new FakeScheduler();
    const runtime = new RunnerRuntime({ scheduler });
    const service = createRunnerServer(runtime);
    await service.listen(0, "127.0.0.1");
    runtime.start();

    const address = service.server.address() as AddressInfo;
    const socket = new WebSocket(`ws://127.0.0.1:${address.port}/ws`);
    await waitForOpen(socket);
    expect(scheduler.activeCount).toBe(1);

    const closed = waitForClose(socket);
    await service.close();
    await closed;
    expect(socket.readyState).toBe(WebSocket.CLOSED);
    expect(scheduler.activeCount).toBe(0);
    expect(runtime.subscriberCount).toBe(0);

    runtime.control("reset");
    expect(runtime.subscriberCount).toBe(0);
  });

  it("客户端 WebSocket error 被连接级监听器消费，不影响 HTTP 服务", async () => {
    const runtime = new RunnerRuntime();
    const service = createRunnerServer(runtime);
    await service.listen(0, "127.0.0.1");
    closeTasks.push(() => service.close());

    const address = service.server.address() as AddressInfo;
    const socket = new WebSocket(`ws://127.0.0.1:${address.port}/ws`);
    closeTasks.push(async () => {
      socket.close();
      await waitForClose(socket);
    });
    await waitForOpen(socket);

    const serverSocket = [...service.webSocketServer.clients][0];
    expect(serverSocket).toBeDefined();
    expect(() =>
      serverSocket?.emit("error", new Error("模拟无效协议帧")),
    ).not.toThrow();

    const health = await request(service.server).get("/api/health");
    expect(health.status).toBe(200);
  });

  it("并发 close 的每个调用都等待同一轮完整清理", async () => {
    const runtime = new RunnerRuntime();
    const service = createRunnerServer(runtime);
    await service.listen(0, "127.0.0.1");
    runtime.start();

    const firstClose = service.close();
    const secondClose = service.close();
    await secondClose;

    expect(service.server.listening).toBe(false);
    expect(runtime.isRunning).toBe(false);
    expect(runtime.subscriberCount).toBe(0);
    await firstClose;
  });

  it("close 后拒绝重新 listen，避免产生无法通过公开 API 清理的句柄", async () => {
    const runtime = new RunnerRuntime();
    const service = createRunnerServer(runtime);
    await service.listen(0, "127.0.0.1");
    await service.close();

    let listenError: unknown;
    try {
      await service.listen(0, "127.0.0.1");
    } catch (error) {
      listenError = error;
    } finally {
      // RED 实现可能错误地重新监听；测试自身必须回收该句柄。
      if (service.server.listening) {
        await new Promise<void>((resolve, reject) => {
          service.server.close((error) => {
            if (error) {
              reject(error);
              return;
            }
            resolve();
          });
        });
      }
    }

    expect(listenError).toBeInstanceOf(Error);
    expect((listenError as Error).message).toContain("已关闭");
  });
});
