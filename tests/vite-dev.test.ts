import { createServer as createHttpServer, type Server as HttpServer } from "node:http";
import { readFileSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { resolve } from "node:path";

import { afterEach, describe, expect, it } from "vitest";
import {
  createServer as createViteServer,
  type ProxyOptions,
  type ViteDevServer,
} from "vite";

import viteConfig from "../vite.config.js";

const projectRoot = resolve(import.meta.dirname, "..");

let backendServer: HttpServer | undefined;
let viteServer: ViteDevServer | undefined;

interface AddressableServer {
  address(): AddressInfo | string | null;
}

function serverOrigin(server: AddressableServer): string {
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("测试服务没有可用的 TCP 地址");
  }
  return `http://127.0.0.1:${(address as AddressInfo).port}`;
}

async function listen(server: HttpServer): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve();
    });
  });
}

async function close(server: HttpServer | undefined): Promise<void> {
  if (!server?.listening) {
    return;
  }
  await new Promise<void>((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  });
}

function proxyToBackend(backendOrigin: string): Record<string, string | ProxyOptions> {
  const configuredProxy = viteConfig.server?.proxy ?? {};
  return Object.fromEntries(
    Object.entries(configuredProxy).map(([context, options]) => {
      if (!context.startsWith("/api")) {
        return [context, options];
      }
      return [
        context,
        typeof options === "string"
          ? backendOrigin
          : { ...options, target: backendOrigin },
      ];
    }),
  );
}

afterEach(async () => {
  await viteServer?.close();
  await close(backendServer);
  viteServer = undefined;
  backendServer = undefined;
});

describe("Vite 开发入口", () => {
  it("同时加载 Web API 模块并代理真正的 API 请求", async () => {
    const backendRequests: string[] = [];
    backendServer = createHttpServer((request, response) => {
      backendRequests.push(request.url ?? "");
      if (request.url === "/api/health" || request.url === "/api/snapshot") {
        response.writeHead(200, { "content-type": "application/json" });
        response.end(JSON.stringify({ success: true }));
        return;
      }
      response.writeHead(404).end();
    });
    await listen(backendServer);

    viteServer = await createViteServer({
      ...viteConfig,
      configFile: false,
      optimizeDeps: {
        ...viteConfig.optimizeDeps,
        include: [],
        noDiscovery: true,
      },
      server: {
        ...viteConfig.server,
        host: "127.0.0.1",
        port: 0,
        strictPort: false,
        proxy: proxyToBackend(serverOrigin(backendServer)),
      },
    });
    await viteServer.listen();

    const viteOrigin = serverOrigin(viteServer.httpServer!);
    const healthResponse = await fetch(`${viteOrigin}/api/health`, {
      signal: AbortSignal.timeout(5_000),
    });
    const moduleResponse = await fetch(`${viteOrigin}/api.ts`, {
      signal: AbortSignal.timeout(5_000),
    });
    const snapshotResponse = await fetch(`${viteOrigin}/api/snapshot`, {
      signal: AbortSignal.timeout(5_000),
    });

    expect(healthResponse.status).toBe(200);
    expect(moduleResponse.status).toBe(200);
    expect(await moduleResponse.text()).toContain("requestSnapshot");
    expect(snapshotResponse.status).toBe(200);
    expect(backendRequests).toEqual(["/api/health", "/api/snapshot"]);
  });
});

describe("Windows 一键启动入口", () => {
  it("只有入口模块和 API 都可用时才报告就绪", () => {
    const launcher = readFileSync(resolve(projectRoot, "start-demo.cmd"), "utf8");

    expect(launcher).toContain(
      'set "MODULE_URL=http://localhost:5173/api.ts"',
    );
    expect(launcher.match(/%MODULE_URL%/g)).toHaveLength(2);
    expect(launcher.match(/\$module\.StatusCode -eq 200/g)).toHaveLength(2);
    expect(
      launcher.match(/\$module\.Headers\['Content-Type'\] -match 'javascript'/g),
    ).toHaveLength(2);
    expect(
      launcher.match(/\$module\.Content -match 'requestSnapshot'/g),
    ).toHaveLength(2);
  });
});
