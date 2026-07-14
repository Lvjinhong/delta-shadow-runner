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

const retryDelaysMs = [500, 1_000, 2_000, 4_000, 8_000] as const;

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

export async function requestSnapshot(signal?: AbortSignal): Promise<TelemetryData> {
  const response = await fetch("/api/snapshot", {
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
): Promise<TelemetryData> {
  const response = await fetch(`/api/control/${command}`, {
    method: "POST",
    headers: {
      accept: "application/json",
      "content-type": "application/json",
    },
    body: "{}",
    signal,
  });
  const body = await readJson(response);
  if (
    !response.ok ||
    body === null ||
    typeof body !== "object" ||
    Array.isArray(body) ||
    (body as Record<string, unknown>).success !== true
  ) {
    throw new Error(controlErrorMessage(body, response.status));
  }

  // 控制响应只携带 snapshot；随后读取完整遥测，确保地图与快照来自同一服务端契约。
  return requestSnapshot(signal);
}

export function createTelemetryClient(
  callbacks: TelemetryClientCallbacks,
): TelemetryClient {
  let socket: WebSocket | undefined;
  let retryTimer: number | undefined;
  let retryIndex = 0;
  let disposed = false;

  const clearRetry = (): void => {
    if (retryTimer !== undefined) {
      window.clearTimeout(retryTimer);
      retryTimer = undefined;
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
    retryTimer = window.setTimeout(() => {
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

    let nextSocket: WebSocket;
    try {
      nextSocket = new WebSocket(buildWebSocketUrl(window.location));
    } catch (error) {
      callbacks.onError(errorFromUnknown(error, "无法创建遥测连接"));
      scheduleReconnect();
      return;
    }
    socket = nextSocket;

    nextSocket.addEventListener("open", () => {
      if (disposed || socket !== nextSocket) {
        return;
      }
      retryIndex = 0;
      callbacks.onStatus({ phase: "connected", attempt: 1 });
    });

    nextSocket.addEventListener("message", (event) => {
      if (disposed || socket !== nextSocket) {
        return;
      }
      if (typeof event.data !== "string") {
        callbacks.onError(new Error("收到非文本遥测消息，已忽略"));
        return;
      }
      try {
        const message = parseTelemetryMessage(event.data);
        if (message.type === "snapshot") {
          callbacks.onData(message.data);
        }
      } catch (error) {
        callbacks.onError(errorFromUnknown(error, "遥测消息解析失败"));
      }
    });

    nextSocket.addEventListener("error", () => {
      if (!disposed && socket === nextSocket) {
        callbacks.onError(new Error("遥测连接发生网络错误"));
      }
    });

    nextSocket.addEventListener("close", () => {
      if (socket === nextSocket) {
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
      const activeSocket = socket;
      socket = undefined;
      if (activeSocket && activeSocket.readyState < WebSocket.CLOSING) {
        activeSocket.close(1000, "dashboard disposed");
      }
    },
  };
}
