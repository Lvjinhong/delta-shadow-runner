export interface RouteEdge {
  readonly targetNodeId: string;
  readonly cost: number;
}

export interface RouteNode {
  readonly id: string;
  readonly x: number;
  readonly y: number;
  readonly edges: readonly RouteEdge[];
}

export type RouteGraph = Readonly<Record<string, RouteNode>>;

export type RunnerStatus =
  | "idle"
  | "localizing"
  | "navigating"
  | "recovering"
  | "extracted"
  | "paused";

export type ActionIntent =
  | {
      readonly type: "move";
      readonly targetNodeId: string;
      readonly ttlMs: number;
    }
  | { readonly type: "relocalize"; readonly ttlMs: number }
  | {
      readonly type: "recover";
      readonly strategy: "backtrack";
      readonly ttlMs: number;
    }
  | { readonly type: "stop"; readonly reason: string };

export interface RunnerObservation {
  readonly nodeId: string;
  readonly confidence: number;
  readonly stuck?: boolean;
  readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface RunnerScenario {
  readonly id: string;
  readonly graph: RouteGraph;
  readonly spawnNodeIds: readonly string[];
  readonly defaultSpawnNodeId: string;
  readonly extractNodeId: string;
  readonly localizationThreshold: number;
}

export interface RuntimeMetrics {
  readonly recoveryCount: number;
  readonly invalidObservationCount: number;
  readonly routeProgress: number;
}

export interface RunnerEvent {
  readonly tick: number;
  readonly kind: "info" | "warning" | "error";
  readonly message: string;
}

export interface EngineSnapshot {
  readonly runId: string;
  readonly status: RunnerStatus;
  readonly tick: number;
  readonly currentNodeId: string;
  readonly targetNodeId: string;
  readonly route: readonly string[];
  readonly confidence: number;
  readonly action: ActionIntent;
  readonly metrics: RuntimeMetrics;
  readonly events: readonly RunnerEvent[];
}

export interface SimulationSource {
  next(): RunnerObservation;
  injectStuck(): void;
  reset(): void;
}
