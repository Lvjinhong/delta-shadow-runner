import {
  DashboardProtocolError,
  buildWebSocketUrl,
  parseSnapshotEnvelope,
  parseTelemetryMessage,
  type TelemetryData,
} from "./model.js";

export type ControlCommand = "start" | "pause" | "reset" | "inject-stuck";

export type TelemetryConnectionState =
  | { readonly phase: "connecting"; readonly attempt: number }
  | { readonly phase: "connected"; readonly attempt: number }
  | {
      readonly phase: "reconnecting";
      readonly attempt: number;
      readonly retryInMs: number;
    }
  | { readonly phase: "disconnected"; readonly attempt: number };

export interface TelemetryClientCallbacks {
  readonly onData: (data: TelemetryData) => void;
  readonly onStatus: (status: TelemetryConnectionState) => void;
  readonly onError: (error: Error) => void;
}

export interface TelemetryClient {
  connect(): void;
  dispose(): void;
}

export interface DashboardFetchEnvironment {
  readonly fetch: typeof fetch;
}

export interface TelemetrySocket {
  readonly readyState: number;
  addEventListener(
    type: "open" | "message" | "error" | "close",
    listener: (event: { readonly data?: unknown }) => void,
  ): void;
  close(code?: number, reason?: string): void;
}

export interface TelemetryClientEnvironment extends DashboardFetchEnvironment {
  readonly createWebSocket: (url: string) => TelemetrySocket;
  readonly setTimeout: (callback: () => void, delay: number) => unknown;
  readonly clearTimeout: (handle: unknown) => void;
  readonly location: { readonly protocol: string; readonly host: string };
}

const retryDelaysMs = [500, 1_000, 2_000, 4_000, 8_000] as const;
const protocolHandshakeTimeoutMs = 5_000;
const protocolErrorCloseCode = 1002;

function errorFromUnknown(error: unknown, fallback: string): Error {
  return error instanceof Error ? error : new Error(fallback);
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return (await response.json()) as unknown;
  } catch {
    throw new DashboardProtocolError(
      `服务端返回了无法解析的响应（HTTP ${response.status}）`,
    );
  }
}

function controlErrorMessage(value: unknown, status: number): string {
  if (value !== null && typeof value === "object" && !Array.isArray(value)) {
    const envelope = value as Record<string, unknown>;
    const error = envelope.error;
    if (error !== null && typeof error === "object" && !Array.isArray(error)) {
      const message = (error as Record<string, unknown>).message;
      if (typeof message === "string" && message.length > 0) {
        return message;
      }
    }
  }
  return `控制请求失败（HTTP ${status}）`;
}

function browserFetchEnvironment(): DashboardFetchEnvironment {
  return {
    fetch(input, init) {
      return globalThis.fetch(input, init);
    },
  };
}

function browserTelemetryEnvironment(): TelemetryClientEnvironment {
  return {
    ...browserFetchEnvironment(),
    createWebSocket(url) {
      return new WebSocket(url) as unknown as TelemetrySocket;
    },
    setTimeout(callback, delay) {
      return window.setTimeout(callback, delay);
    },
    clearTimeout(handle) {
      window.clearTimeout(handle as number);
    },
    location: window.location,
  };
}

export async function requestSnapshot(
  signal?: AbortSignal,
  environment: DashboardFetchEnvironment = browserFetchEnvironment(),
): Promise<TelemetryData> {
  const response = await environment.fetch("/api/snapshot", {
    method: "GET",
    headers: { accept: "application/json" },
    signal,
  });
  const body = await readJson(response);
  return parseSnapshotEnvelope(body);
}

export async function sendControl(
  command: ControlCommand,
  signal?: AbortSignal,
  environment: DashboardFetchEnvironment = browserFetchEnvironment(),
): Promise<TelemetryData> {
  const response = await environment.fetch(`/api/control/${command}`, {
    method: "POST",
    headers: {
      accept: "application/json",
      "content-type": "application/json",
    },
    body: "{}",
    signal,
  });
  const body = await readJson(response);
  if (!response.ok) {
    throw new Error(controlErrorMessage(body, response.status));
  }
  return parseSnapshotEnvelope(body);
}

export function createTelemetryClient(
  callbacks: TelemetryClientCallbacks,
  environment: TelemetryClientEnvironment = browserTelemetryEnvironment(),
): TelemetryClient {
  let socket: TelemetrySocket | undefined;
  let retryTimer: unknown = undefined;
  let handshakeTimer: unknown = undefined;
  let retryIndex = 0;
  let disposed = false;

  const clearRetry = (): void => {
    if (retryTimer !== undefined) {
      environment.clearTimeout(retryTimer);
      retryTimer = undefined;
    }
  };

  const clearHandshake = (): void => {
    if (handshakeTimer !== undefined) {
      environment.clearTimeout(handshakeTimer);
      handshakeTimer = undefined;
    }
  };

  const scheduleReconnect = (): void => {
    if (disposed) {
      return;
    }
    const delay = retryDelaysMs[retryIndex];
    if (delay === undefined) {
      callbacks.onStatus({ phase: "disconnected", attempt: retryIndex + 1 });
      callbacks.onError(
        new Error("遥测重连次数已用尽，请确认服务端状态后刷新页面"),
      );
      return;
    }

    retryIndex += 1;
    callbacks.onStatus({
      phase: "reconnecting",
      attempt: retryIndex,
      retryInMs: delay,
    });
    retryTimer = environment.setTimeout(() => {
      retryTimer = undefined;
      openSocket();
    }, delay);
  };

  const openSocket = (): void => {
    if (disposed) {
      return;
    }
    clearRetry();
    if (retryIndex === 0) {
      callbacks.onStatus({ phase: "connecting", attempt: 1 });
    } else {
      callbacks.onStatus({
        phase: "reconnecting",
        attempt: retryIndex + 1,
        retryInMs: 0,
      });
    }

    let nextSocket: TelemetrySocket;
    try {
      nextSocket = environment.createWebSocket(
        buildWebSocketUrl(environment.location),
      );
    } catch (error) {
      callbacks.onError(errorFromUnknown(error, "无法创建遥测连接"));
      scheduleReconnect();
      return;
    }
    socket = nextSocket;
    let protocolReady = false;
    let protocolFailed = false;

    const failProtocol = (error: Error): void => {
      if (disposed || socket !== nextSocket || protocolFailed) {
        return;
      }
      protocolFailed = true;
      clearHandshake();
      callbacks.onError(error);
      if (nextSocket.readyState < 2) {
        nextSocket.close(protocolErrorCloseCode, "telemetry protocol error");
      }
    };

    nextSocket.addEventListener("open", () => {
      if (
        disposed ||
        socket !== nextSocket ||
        protocolReady ||
        protocolFailed
      ) {
        return;
      }
      clearHandshake();
      handshakeTimer = environment.setTimeout(() => {
        handshakeTimer = undefined;
        failProtocol(new Error("遥测协议握手超时，连接将在重试后恢复"));
      }, protocolHandshakeTimeoutMs);
    });

    nextSocket.addEventListener("message", (event) => {
      if (disposed || socket !== nextSocket || protocolFailed) {
        return;
      }
      if (typeof event.data !== "string") {
        failProtocol(new Error("收到非文本遥测协议消息，连接将在重试后恢复"));
        return;
      }
      try {
        const message = parseTelemetryMessage(event.data);
        if (!protocolReady) {
          protocolReady = true;
          clearHandshake();
          retryIndex = 0;
          callbacks.onStatus({ phase: "connected", attempt: 1 });
        }
        if (message.type === "snapshot") {
          callbacks.onData(message.data);
        }
      } catch (error) {
        failProtocol(errorFromUnknown(error, "遥测消息解析失败"));
      }
    });

    nextSocket.addEventListener("error", () => {
      if (!disposed && socket === nextSocket) {
        callbacks.onError(new Error("遥测连接发生网络错误"));
      }
    });

    nextSocket.addEventListener("close", () => {
      if (socket === nextSocket) {
        clearHandshake();
        socket = undefined;
      }
      scheduleReconnect();
    });
  };

  return {
    connect() {
      if (disposed || socket || retryTimer !== undefined) {
        return;
      }
      openSocket();
    },
    dispose() {
      if (disposed) {
        return;
      }
      disposed = true;
      clearRetry();
      clearHandshake();
      const activeSocket = socket;
      socket = undefined;
      if (activeSocket && activeSocket.readyState < 2) {
        activeSocket.close(1000, "dashboard disposed");
      }
    },
  };
}
