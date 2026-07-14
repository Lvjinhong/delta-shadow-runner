import { createServer, type Server } from "node:http";

import cors from "cors";
import express, {
  type ErrorRequestHandler,
  type Express,
  type Response,
} from "express";
import { WebSocket, WebSocketServer } from "ws";
import { z } from "zod";

import type {
  EngineSnapshot,
  RunnerCapabilities,
  RunnerScenario,
} from "../core/types.js";
import {
  RunnerRuntime,
  type ControlCommand,
} from "./runtime.js";

interface ApiError {
  readonly code: string;
  readonly message: string;
}

interface ApiEnvelope<T> {
  readonly success: boolean;
  readonly data: T | null;
  readonly error: ApiError | null;
  readonly meta?: Readonly<Record<string, unknown>>;
}

export interface ScenarioTelemetry {
  readonly id: string;
  readonly spawnNodeIds: readonly string[];
  readonly defaultSpawnNodeId: string;
  readonly extractNodeId: string;
  readonly localizationThreshold: number;
  readonly map: {
    readonly nodes: readonly {
      readonly id: string;
      readonly x: number;
      readonly y: number;
      readonly edges: readonly {
        readonly targetNodeId: string;
        readonly cost: number;
      }[];
    }[];
  };
}

export interface TelemetryData {
  readonly snapshot: EngineSnapshot;
  readonly scenario: ScenarioTelemetry;
  readonly capabilities: RunnerCapabilities;
}

export interface RunnerAppOptions {
  readonly staticDir?: string;
}

export interface RunnerServer {
  readonly app: Express;
  readonly server: Server;
  readonly webSocketServer: WebSocketServer;
  listen(port: number, host?: string): Promise<void>;
  close(): Promise<void>;
}

const controlCommandSchema = z.enum([
  "start",
  "pause",
  "reset",
  "inject-stuck",
]);
const emptyControlBodySchema = z.object({}).strict();

function success<T>(data: T): ApiEnvelope<T> {
  return { success: true, data, error: null };
}

function failure(code: string, message: string): ApiEnvelope<never> {
  return { success: false, data: null, error: { code, message } };
}

function scenarioTelemetry(scenario: RunnerScenario): ScenarioTelemetry {
  return {
    id: scenario.id,
    spawnNodeIds: [...scenario.spawnNodeIds],
    defaultSpawnNodeId: scenario.defaultSpawnNodeId,
    extractNodeId: scenario.extractNodeId,
    localizationThreshold: scenario.localizationThreshold,
    map: {
      nodes: Object.values(scenario.graph).map((node) => ({
        id: node.id,
        x: node.x,
        y: node.y,
        edges: node.edges.map((edge) => ({ ...edge })),
      })),
    },
  };
}

function telemetryData(
  runtime: RunnerRuntime,
  snapshot = runtime.getSnapshot(),
): TelemetryData {
  return {
    snapshot,
    scenario: scenarioTelemetry(runtime.scenario),
    capabilities: runtime.capabilities,
  };
}

function sendInvalidControl(res: Response): void {
  res
    .status(400)
    .json(
      failure(
        "INVALID_CONTROL_REQUEST",
        "控制路径或请求体无效，仅允许空对象与已声明的模拟命令",
      ),
    );
}

export function createRunnerApp(
  runtime: RunnerRuntime,
  options: RunnerAppOptions = {},
): Express {
  const app = express();
  app.disable("x-powered-by");
  app.use(cors());
  app.use(express.json({ limit: "16kb", strict: true }));

  app.get("/api/health", (_req, res) => {
    res.json(
      success({
        status: "ok",
        mode: "simulation",
        compute: "cpu-only",
        tickMs: runtime.tickMs,
        running: runtime.isRunning,
      }),
    );
  });

  app.get("/api/snapshot", (_req, res) => {
    res.json(success(telemetryData(runtime)));
  });

  app.post("/api/control/:command", (req, res) => {
    const command = controlCommandSchema.safeParse(req.params.command);
    const body = emptyControlBodySchema.safeParse(req.body);
    if (!command.success || !body.success) {
      sendInvalidControl(res);
      return;
    }

    const snapshot = runtime.control(command.data as ControlCommand);
    res.json(success(telemetryData(runtime, snapshot)));
  });

  app.all(/^\/api\/control(?:\/.*)?$/, (_req, res) => {
    sendInvalidControl(res);
  });

  app.all(/^\/api(?:\/.*)?$/, (_req, res) => {
    res
      .status(404)
      .json(failure("API_NOT_FOUND", "请求的 API 路径不存在"));
  });

  if (options.staticDir) {
    app.use(express.static(options.staticDir));
  }

  const errorHandler: ErrorRequestHandler = (error, _req, res, _next) => {
    const bodyError = error as { readonly status?: unknown; readonly type?: unknown };
    if (bodyError.status === 413 || bodyError.type === "entity.too.large") {
      res
        .status(413)
        .json(failure("PAYLOAD_TOO_LARGE", "请求体超过 16KB 限制"));
      return;
    }

    if (
      error instanceof SyntaxError &&
      bodyError.status === 400 &&
      bodyError.type === "entity.parse.failed"
    ) {
      res
        .status(400)
        .json(failure("INVALID_JSON", "请求体不是有效的 JSON"));
      return;
    }

    if (
      typeof bodyError.status === "number" &&
      bodyError.status >= 400 &&
      bodyError.status < 500
    ) {
      res
        .status(bodyError.status)
        .json(failure("INVALID_REQUEST", "请求无法被服务端接受"));
      return;
    }

    res
      .status(500)
      .json(failure("INTERNAL_ERROR", "服务端处理请求时发生错误"));
  };
  app.use(errorHandler);

  return app;
}

function sendJson(socket: WebSocket, payload: unknown): void {
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
  }
}

export function createRunnerServer(
  runtime: RunnerRuntime,
  options: RunnerAppOptions = {},
): RunnerServer {
  const app = createRunnerApp(runtime, options);
  const server = createServer(app);
  const webSocketServer = new WebSocketServer({ server, path: "/ws" });
  let lifecycle: "open" | "closing" | "closed" = "open";
  let closePromise: Promise<void> | undefined;

  webSocketServer.on("connection", (socket) => {
    socket.on("error", () => {
      // ws 会负责关闭协议错误连接；监听器用于阻止 error 冒泡为进程级异常。
    });
    sendJson(socket, {
      type: "connection",
      data: { status: "connected", mode: "simulation" },
    });
    sendJson(socket, { type: "snapshot", data: telemetryData(runtime) });
  });

  const unsubscribe = runtime.subscribe((snapshot) => {
    const message = JSON.stringify({
      type: "snapshot",
      data: telemetryData(runtime, snapshot),
    });
    for (const client of webSocketServer.clients) {
      if (client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    }
  });

  return {
    app,
    server,
    webSocketServer,
    listen(port, host = "127.0.0.1") {
      if (lifecycle !== "open") {
        return Promise.reject(new Error("Runner 服务已关闭，不能重新监听"));
      }
      if (server.listening) {
        return Promise.resolve();
      }
      return new Promise((resolve, reject) => {
        const onError = (error: Error): void => {
          server.off("listening", onListening);
          reject(error);
        };
        const onListening = (): void => {
          server.off("error", onError);
          resolve();
        };
        server.once("error", onError);
        server.once("listening", onListening);
        server.listen(port, host);
      });
    },
    close() {
      if (closePromise) {
        return closePromise;
      }
      lifecycle = "closing";
      closePromise = (async () => {
        unsubscribe();
        runtime.stop();

        for (const client of webSocketServer.clients) {
          client.terminate();
        }
        await new Promise<void>((resolve, reject) => {
          webSocketServer.close((error) => {
            if (error && server.listening) {
              reject(error);
              return;
            }
            resolve();
          });
        });
        if (server.listening) {
          await new Promise<void>((resolve, reject) => {
            server.close((error) => {
              if (error) {
                reject(error);
                return;
              }
              resolve();
            });
          });
        }
      })().finally(() => {
        lifecycle = "closed";
      });
      return closePromise;
    },
  };
}
