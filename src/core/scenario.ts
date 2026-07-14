import { findShortestPath } from "./graph.js";
import type {
  RouteGraph,
  RunnerObservation,
  RunnerScenario,
  SimulationSource,
} from "./types.js";

function freezeGraph(graph: RouteGraph): RouteGraph {
  return Object.freeze(
    Object.fromEntries(
      Object.entries(graph).map(([nodeId, node]) => [
        nodeId,
        Object.freeze({
          ...node,
          edges: Object.freeze(
            node.edges.map((edge) => Object.freeze({ ...edge })),
          ),
        }),
      ]),
    ),
  );
}

export const defaultRouteGraph: RouteGraph = freezeGraph({
  "spawn-a": {
    id: "spawn-a",
    x: 8,
    y: 72,
    edges: [
      { targetNodeId: "relay", cost: 1 },
      { targetNodeId: "yard", cost: 2 },
    ],
  },
  "spawn-b": {
    id: "spawn-b",
    x: 12,
    y: 18,
    edges: [{ targetNodeId: "yard", cost: 1 }],
  },
  relay: {
    id: "relay",
    x: 35,
    y: 62,
    edges: [{ targetNodeId: "warehouse", cost: 1 }],
  },
  yard: {
    id: "yard",
    x: 38,
    y: 25,
    edges: [{ targetNodeId: "warehouse", cost: 3 }],
  },
  warehouse: {
    id: "warehouse",
    x: 68,
    y: 48,
    edges: [{ targetNodeId: "extract", cost: 1 }],
  },
  extract: {
    id: "extract",
    x: 92,
    y: 42,
    edges: [],
  },
});

export const defaultScenario: RunnerScenario = Object.freeze({
  id: "fixed-training-route",
  graph: defaultRouteGraph,
  spawnNodeIds: Object.freeze(["spawn-a", "spawn-b"]),
  defaultSpawnNodeId: "spawn-a",
  extractNodeId: "extract",
  localizationThreshold: 0.7,
});

class DeterministicSimulationSource implements SimulationSource {
  private readonly route: readonly string[];
  private routeIndex = 0;
  private stuckPending = false;

  constructor(scenario: RunnerScenario, spawnNodeId: string) {
    this.route = Object.freeze(
      findShortestPath(scenario.graph, spawnNodeId, scenario.extractNodeId),
    );
  }

  next(): RunnerObservation {
    const nodeId = this.route[this.routeIndex];
    if (!nodeId) {
      throw new Error("确定性模拟源没有可输出的路线节点");
    }

    if (this.stuckPending) {
      this.stuckPending = false;
      return Object.freeze({ nodeId, confidence: 0.95, stuck: true });
    }

    if (this.routeIndex < this.route.length - 1) {
      this.routeIndex += 1;
    }

    return Object.freeze({ nodeId, confidence: 0.95, stuck: false });
  }

  injectStuck(): void {
    this.stuckPending = true;
  }

  reset(): void {
    this.routeIndex = 0;
    this.stuckPending = false;
  }
}

export function createDeterministicSimulationSource(
  scenario: RunnerScenario,
  spawnNodeId = scenario.defaultSpawnNodeId,
): SimulationSource {
  if (!scenario.spawnNodeIds.includes(spawnNodeId)) {
    throw new Error(`节点 "${spawnNodeId}" 不是场景出生点`);
  }

  return new DeterministicSimulationSource(scenario, spawnNodeId);
}
