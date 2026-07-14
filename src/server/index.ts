import { fileURLToPath } from "node:url";

import { createRunnerServer } from "./app.js";
import { RunnerRuntime } from "./runtime.js";

function readPort(value: string | undefined): number {
  if (value === undefined) {
    return 4173;
  }

  const port = Number(value);
  if (!Number.isInteger(port) || port < 1 || port > 65_535) {
    throw new Error(`PORT 必须是 1 到 65535 的整数，收到 "${value}"`);
  }
  return port;
}

async function main(): Promise<void> {
  const port = readPort(process.env.PORT);
  const host = process.env.HOST ?? "127.0.0.1";
  const staticDir = fileURLToPath(new URL("../../web", import.meta.url));
  const runtime = new RunnerRuntime();
  const service = createRunnerServer(runtime, { staticDir });
  let shuttingDown = false;

  const shutdown = async (signal: NodeJS.Signals): Promise<void> => {
    if (shuttingDown) {
      return;
    }
    shuttingDown = true;
    console.log(`[Shadow Runner Lab] 收到 ${signal}，正在关闭服务。`);
    try {
      await service.close();
    } catch (error) {
      console.error("[Shadow Runner Lab] 关闭服务失败。", error);
      process.exitCode = 1;
    }
  };

  process.once("SIGINT", () => void shutdown("SIGINT"));
  process.once("SIGTERM", () => void shutdown("SIGTERM"));

  await service.listen(port, host);
  runtime.start();
  console.log(
    `[Shadow Runner Lab] 模拟遥测服务已启动：http://${host}:${port}（CPU-only）`,
  );
}

void main().catch((error: unknown) => {
  console.error("[Shadow Runner Lab] 启动失败。", error);
  process.exitCode = 1;
});
