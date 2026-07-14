import { useEffect, useMemo, useRef, useState } from "react";

import {
  createTelemetryClient,
  requestSnapshot,
  sendControl,
  type ControlCommand,
  type TelemetryConnectionState,
} from "./api.js";
import {
  formatActionIntent,
  projectRouteMap,
  statusPresentation,
  type ProjectedNode,
  type TelemetryData,
} from "./model.js";

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function nodeLabel(nodeId: string): string {
  return nodeId.replaceAll("-", " ").toUpperCase();
}

function errorMessage(error: unknown): string {
  if (error instanceof DOMException && error.name === "AbortError") {
    return "";
  }
  return error instanceof Error ? error.message : "发生未知错误";
}

function connectionPresentation(state: TelemetryConnectionState): {
  readonly label: string;
  readonly tone: string;
} {
  switch (state.phase) {
    case "connecting":
      return { label: "遥测连接中", tone: "warning" };
    case "connected":
      return { label: "遥测在线", tone: "success" };
    case "reconnecting":
      return {
        label:
          state.retryInMs > 0
            ? `遥测重连 ${state.retryInMs / 1_000}s`
            : `遥测重连 #${state.attempt}`,
        tone: "warning",
      };
    case "disconnected":
      return { label: "遥测断开", tone: "danger" };
  }
}

function MetricCell(props: {
  readonly label: string;
  readonly value: string | number;
  readonly accent?: "amber" | "cyan" | "red";
}): React.JSX.Element {
  return (
    <div className={`metric-cell metric-cell--${props.accent ?? "plain"}`}>
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}

function MapNode({ node }: { readonly node: ProjectedNode }): React.JSX.Element {
  const classes = [
    "map-node",
    node.isSpawn ? "map-node--spawn" : "",
    node.isExtract ? "map-node--extract" : "",
    node.isOnRoute ? "map-node--route" : "",
    node.isTarget ? "map-node--target" : "",
    node.isCurrent ? "map-node--current" : "",
  ]
    .filter(Boolean)
    .join(" ");
  const labelTop = node.y > 82 ? -12 : 4;

  return (
    <g className={classes} transform={`translate(${node.x} ${node.y})`}>
      {node.isCurrent ? <circle className="position-pulse" r="4.6" /> : null}
      {node.isExtract ? <circle className="extract-ring" r="3.8" /> : null}
      <circle className="node-core" r={node.isCurrent ? 2.3 : 1.8} />
      {node.isSpawn ? <path className="spawn-mark" d="M -3 2.8 L 0 -3 L 3 2.8 Z" /> : null}
      {node.isTarget ? <path className="target-cross" d="M -4 0 H 4 M 0 -4 V 4" /> : null}
      <foreignObject
        className="node-label-frame"
        x="-11"
        y={labelTop}
        width="22"
        height="12"
      >
        <div className="node-label" title={node.id}>
          {nodeLabel(node.id)}
        </div>
      </foreignObject>
    </g>
  );
}

function RouteMap({ data }: { readonly data: TelemetryData }): React.JSX.Element {
  const projection = useMemo(() => projectRouteMap(data), [data]);
  const { snapshot, scenario } = data;

  return (
    <section className="map-panel panel" aria-labelledby="route-map-title">
      <div className="panel-heading map-heading">
        <div>
          <span className="eyebrow">ROUTE TOPOLOGY / {scenario.id}</span>
          <h2 id="route-map-title">固定训练路线</h2>
        </div>
        <div className="map-readout" aria-label="当前节点与目标节点">
          <span>{nodeLabel(snapshot.currentNodeId)}</span>
          <b aria-hidden="true">→</b>
          <span>{nodeLabel(snapshot.targetNodeId)}</span>
        </div>
      </div>

      <div className="map-stage">
        <svg
          className="route-map"
          viewBox="0 0 100 100"
          role="img"
          aria-label={`训练地图，当前位置 ${nodeLabel(snapshot.currentNodeId)}，目标 ${nodeLabel(snapshot.targetNodeId)}`}
        >
          <defs>
            <pattern id="minor-grid" width="5" height="5" patternUnits="userSpaceOnUse">
              <path d="M 5 0 L 0 0 0 5" className="grid-line" />
            </pattern>
            <marker
              id="route-arrow"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="3.5"
              markerHeight="3.5"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" className="arrow-head" />
            </marker>
          </defs>
          <rect width="100" height="100" className="map-grid-fill" />
          <path d="M 4 8 H 14 M 4 8 V 18 M 96 82 H 86 M 96 82 V 92" className="map-corners" />

          <g aria-hidden="true">
            {projection.edges.map((edge) => (
              <line
                key={`base-${edge.id}`}
                className="route-edge route-edge--base"
                x1={edge.x1}
                y1={edge.y1}
                x2={edge.x2}
                y2={edge.y2}
                markerEnd="url(#route-arrow)"
              />
            ))}
            {projection.edges
              .filter((edge) => edge.phase !== "base")
              .map((edge) => (
                <line
                  key={`route-${edge.id}`}
                  className={`route-edge route-edge--${edge.phase}`}
                  x1={edge.x1}
                  y1={edge.y1}
                  x2={edge.x2}
                  y2={edge.y2}
                />
              ))}
          </g>

          <g aria-hidden="true">
            {projection.edges.map((edge) => (
              <text
                key={`cost-${edge.id}`}
                className="edge-cost"
                x={(edge.x1 + edge.x2) / 2}
                y={(edge.y1 + edge.y2) / 2 - 1.5}
                textAnchor="middle"
              >
                C{edge.cost}
              </text>
            ))}
          </g>
          <g>
            {projection.nodes.map((node) => (
              <MapNode key={node.id} node={node} />
            ))}
          </g>
        </svg>

        <div className="map-coordinates" aria-hidden="true">
          <span>GRID // 100×100</span>
          <span>LOC THRESHOLD // {percent(scenario.localizationThreshold)}</span>
        </div>
      </div>

      <div className="map-legend" aria-label="地图图例">
        <span><i className="legend-line legend-line--walked" />已走</span>
        <span><i className="legend-line legend-line--planned" />规划路线</span>
        <span><i className="legend-dot legend-dot--current" />当前位置</span>
        <span><i className="legend-dot legend-dot--extract" />撤离点</span>
      </div>
    </section>
  );
}

const controlLabels: Readonly<Record<ControlCommand, string>> = {
  start: "启动 / 继续",
  pause: "暂停任务",
  reset: "重置模拟",
  "inject-stuck": "注入卡住",
};

export function App(): React.JSX.Element {
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [connection, setConnection] = useState<TelemetryConnectionState>({
    phase: "connecting",
    attempt: 1,
  });
  const [loading, setLoading] = useState(true);
  const [pendingCommand, setPendingCommand] = useState<ControlCommand | null>(null);
  const [connectionError, setConnectionError] = useState("");
  const [controlError, setControlError] = useState("");
  const commandAbortRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    let effectActive = true;
    const canUpdate = (): boolean => effectActive && mountedRef.current;
    const initialAbort = new AbortController();
    const client = createTelemetryClient({
      onData(data) {
        if (canUpdate()) {
          setTelemetry(data);
          setLoading(false);
          setConnectionError("");
        }
      },
      onStatus(status) {
        if (canUpdate()) {
          setConnection(status);
          if (status.phase === "connected") {
            setConnectionError("");
          }
        }
      },
      onError(nextError) {
        if (canUpdate()) {
          setConnectionError(nextError.message);
        }
      },
    });

    void requestSnapshot(initialAbort.signal)
      .then((data) => {
        if (canUpdate()) {
          setTelemetry(data);
          setConnectionError("");
        }
      })
      .catch((nextError: unknown) => {
        if (canUpdate()) {
          setConnectionError(errorMessage(nextError));
        }
      })
      .finally(() => {
        if (canUpdate()) {
          setLoading(false);
        }
      });
    client.connect();

    return () => {
      effectActive = false;
      mountedRef.current = false;
      initialAbort.abort();
      commandAbortRef.current?.abort();
      client.dispose();
    };
  }, []);

  const runControl = async (command: ControlCommand): Promise<void> => {
    if (!mountedRef.current || pendingCommand || commandAbortRef.current) {
      return;
    }
    const controller = new AbortController();
    commandAbortRef.current = controller;
    setPendingCommand(command);
    setControlError("");
    try {
      const data = await sendControl(command, controller.signal);
      if (mountedRef.current && commandAbortRef.current === controller) {
        setTelemetry(data);
        setConnectionError("");
      }
    } catch (nextError) {
      const message = errorMessage(nextError);
      if (
        message &&
        mountedRef.current &&
        commandAbortRef.current === controller
      ) {
        setControlError(message);
      }
    } finally {
      if (commandAbortRef.current === controller) {
        commandAbortRef.current = null;
        if (mountedRef.current) {
          setPendingCommand(null);
        }
      }
    }
  };

  const connectionMeta = connectionPresentation(connection);
  const snapshot = telemetry?.snapshot;
  const capabilities = telemetry?.capabilities;
  const statusMeta = snapshot ? statusPresentation(snapshot.status) : null;
  const controlsDisabled = loading || !snapshot || pendingCommand !== null;
  const visibleEvents = snapshot ? [...snapshot.events].slice(-12).reverse() : [];
  const latestEvent = snapshot?.events.at(-1);
  const displayedError = controlError || connectionError;

  return (
    <div className="app-shell">
      <header className="command-header">
        <div className="brand-lockup">
          <span className="brand-index">SRL / 01</span>
          <div>
            <h1>SHADOW RUNNER LAB</h1>
            <p>固定路线智能体 · 离线仿真控制台</p>
          </div>
        </div>
        <div className="header-statuses" aria-live="polite">
          <span className={`signal signal--${connectionMeta.tone}`}>
            <i aria-hidden="true" />{connectionMeta.label}
          </span>
          <span className={`signal signal--${statusMeta?.tone ?? "neutral"}`}>
            <i aria-hidden="true" />{statusMeta?.label ?? "等待快照"}
          </span>
        </div>
      </header>

      <div className="safety-strip" role="note">
        <b>SIMULATION</b><span aria-hidden="true">/</span>
        <b>CPU ONLY</b><span aria-hidden="true">/</span>
        <b>NO INPUT DEVICE</b>
        <span className="safety-copy">不读取游戏进程，不发送键鼠输入</span>
      </div>

      {displayedError ? (
        <div className="error-banner" role="alert">
          <span><b>SYS ERR</b> {displayedError}</span>
          <button
            type="button"
            onClick={() => {
              setControlError("");
              setConnectionError("");
            }}
            aria-label="关闭错误提示"
          >×</button>
        </div>
      ) : null}

      {telemetry && snapshot ? (
        <main className="dashboard-grid">
          <aside className="mission-rail">
            <details className="panel collapsible-panel" open>
              <summary>任务态势</summary>
              <div className="panel-body">
                <div className="mission-id">
                  <span>RUN IDENT</span>
                  <strong title={snapshot.runId}>{snapshot.runId}</strong>
                </div>
                <div className="confidence-block">
                  <div><span>定位置信度</span><strong>{percent(snapshot.confidence)}</strong></div>
                  <div className="meter" role="meter" aria-label="定位置信度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(snapshot.confidence * 100)}>
                    <span style={{ width: percent(snapshot.confidence) }} />
                  </div>
                </div>
                <div className="metric-grid metric-grid--rail">
                  <MetricCell label="TICK" value={snapshot.tick} accent="amber" />
                  <MetricCell label="路线进度" value={percent(snapshot.metrics.routeProgress)} accent="cyan" />
                  <MetricCell label="恢复次数" value={snapshot.metrics.recoveryCount} accent={snapshot.metrics.recoveryCount > 0 ? "red" : undefined} />
                  <MetricCell label="非法观测" value={snapshot.metrics.invalidObservationCount} accent={snapshot.metrics.invalidObservationCount > 0 ? "red" : undefined} />
                </div>
              </div>
            </details>

            <details className="panel collapsible-panel" open>
              <summary>当前决策</summary>
              <div className="panel-body decision-readout">
                <span className="eyebrow">ACTION INTENT</span>
                <strong>{formatActionIntent(snapshot.action)}</strong>
                <dl>
                  <div><dt>当前位置</dt><dd>{nodeLabel(snapshot.currentNodeId)}</dd></div>
                  <div><dt>下一目标</dt><dd>{nodeLabel(snapshot.targetNodeId)}</dd></div>
                  <div><dt>路线节点</dt><dd>{snapshot.route.length || "—"}</dd></div>
                </dl>
              </div>
            </details>
          </aside>

          <RouteMap data={telemetry} />

          <aside className="control-rail">
            <details className="panel collapsible-panel" open>
              <summary>人工控制</summary>
              <div className="panel-body control-stack" aria-busy={pendingCommand !== null}>
                <button
                  className="control-button control-button--primary"
                  type="button"
                  disabled={controlsDisabled || !capabilities?.canStart}
                  onClick={() => void runControl("start")}
                >
                  <span>01</span>{pendingCommand === "start" ? "请求中…" : controlLabels.start}
                </button>
                <button
                  className="control-button"
                  type="button"
                  disabled={controlsDisabled || !capabilities?.canPause}
                  onClick={() => void runControl("pause")}
                >
                  <span>02</span>{pendingCommand === "pause" ? "请求中…" : controlLabels.pause}
                </button>
                <button
                  className="control-button"
                  type="button"
                  disabled={controlsDisabled || !capabilities?.canReset}
                  onClick={() => void runControl("reset")}
                >
                  <span>03</span>{pendingCommand === "reset" ? "请求中…" : controlLabels.reset}
                </button>
                <button
                  className="control-button control-button--danger"
                  type="button"
                  disabled={controlsDisabled || !capabilities?.canInjectStuck}
                  onClick={() => void runControl("inject-stuck")}
                >
                  <span>04</span>{pendingCommand === "inject-stuck" ? "请求中…" : controlLabels["inject-stuck"]}
                </button>
                <p className="control-note">控制命令仅作用于确定性模拟源。注入卡住将在下一次 tick 进入恢复分支。</p>
              </div>
            </details>

            <section className="panel protocol-panel" aria-labelledby="protocol-title">
              <div className="panel-heading"><div><span className="eyebrow">LINK HEALTH</span><h2 id="protocol-title">运行边界</h2></div></div>
              <ul className="protocol-list">
                <li><span>数据源</span><b>SIMULATOR</b></li>
                <li><span>计算设备</span><b>CPU</b></li>
                <li><span>输入执行器</span><b>NONE</b></li>
                <li><span>遥测通道</span><b>{connection.phase === "connected" ? "WS LIVE" : "WS DOWN"}</b></li>
              </ul>
            </section>
          </aside>

          <section className="panel event-panel" aria-labelledby="event-title">
            <div className="panel-heading event-heading">
              <div><span className="eyebrow">EVENT BUFFER / LAST 12</span><h2 id="event-title">运行事件</h2></div>
              <span>{snapshot.events.length.toString().padStart(2, "0")} RECORDS</span>
            </div>
            <div className="sr-only event-announcer" role="status" aria-live="polite" aria-atomic="true">
              {latestEvent
                ? `最新事件，T+${latestEvent.tick}，${latestEvent.message}`
                : "暂无运行事件"}
            </div>
            <ol className="event-list">
              {visibleEvents.map((event) => (
                <li
                  key={`${snapshot.runId}:${event.tick}:${event.kind}:${event.message}`}
                  className={`event event--${event.kind}`}
                >
                  <time>T+{event.tick.toString().padStart(3, "0")}</time>
                  <i aria-hidden="true" />
                  <span>{event.message}</span>
                  <b>{event.kind.toUpperCase()}</b>
                </li>
              ))}
            </ol>
          </section>
        </main>
      ) : (
        <main className="loading-panel" aria-live="polite">
          <span className="loading-mark" aria-hidden="true" />
          <div><span className="eyebrow">BOOT SEQUENCE</span><h2>{loading ? "正在读取模拟遥测…" : "未获得有效快照"}</h2></div>
        </main>
      )}

      <footer className="system-footer">
        <span>SHADOW RUNNER LAB // LOCAL TRAINING SYSTEM</span>
        <span>REST + WEBSOCKET / 500MS TICK</span>
      </footer>
    </div>
  );
}
