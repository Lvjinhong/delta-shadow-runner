import type { RouteEdge, RouteGraph, RouteNode } from "./types.js";

interface SearchScores {
  readonly traveled: ReadonlyMap<string, number>;
  readonly estimatedTotal: ReadonlyMap<string, number>;
}

function hasNode(graph: RouteGraph, nodeId: string): boolean {
  return Object.prototype.hasOwnProperty.call(graph, nodeId);
}

function requireNode(
  graph: RouteGraph,
  nodeId: string,
  role: "起点" | "终点",
): RouteNode {
  const node = hasNode(graph, nodeId) ? graph[nodeId] : undefined;

  if (!node) {
    throw new Error(`路线图中不存在${role}节点 "${nodeId}"`);
  }

  return node;
}

function validateEdge(
  graph: RouteGraph,
  sourceNodeId: string,
  edge: RouteEdge,
): void {
  if (!hasNode(graph, edge.targetNodeId)) {
    throw new Error(
      `节点 "${sourceNodeId}" 的边指向未知节点 "${edge.targetNodeId}"`,
    );
  }

  if (!Number.isFinite(edge.cost) || edge.cost < 0) {
    throw new Error(
      `节点 "${sourceNodeId}" 到 "${edge.targetNodeId}" 的代价必须是非负有限数`,
    );
  }
}

function validateGraph(graph: RouteGraph): void {
  for (const [nodeId, node] of Object.entries(graph)) {
    if (!Number.isFinite(node.x) || !Number.isFinite(node.y)) {
      throw new Error(`节点 "${nodeId}" 的坐标必须是有限数`);
    }

    for (const edge of node.edges) {
      validateEdge(graph, nodeId, edge);
    }
  }
}

function distance(from: RouteNode, to: RouteNode): number {
  return Math.hypot(to.x - from.x, to.y - from.y);
}

function calculateHeuristicScale(graph: RouteGraph): number {
  let minimumCostPerDistance = Number.POSITIVE_INFINITY;

  for (const source of Object.values(graph)) {
    for (const edge of source.edges) {
      const target = graph[edge.targetNodeId];

      // validateGraph 已检查边；此分支仅用于收窄 Record 索引结果。
      if (!target) {
        continue;
      }

      const edgeDistance = distance(source, target);
      if (!Number.isFinite(edgeDistance) || edgeDistance === 0) {
        continue;
      }

      const costPerDistance = edge.cost / edgeDistance;
      if (Number.isFinite(costPerDistance)) {
        minimumCostPerDistance = Math.min(
          minimumCostPerDistance,
          costPerDistance,
        );
      }
    }
  }

  return Number.isFinite(minimumCostPerDistance) ? minimumCostPerDistance : 0;
}

function selectLowestScore(
  openNodeIds: ReadonlySet<string>,
  estimatedTotal: ReadonlyMap<string, number>,
): string {
  let selectedNodeId: string | undefined;
  let selectedScore = Number.POSITIVE_INFINITY;

  for (const nodeId of openNodeIds) {
    const score = estimatedTotal.get(nodeId) ?? Number.POSITIVE_INFINITY;
    if (score < selectedScore) {
      selectedNodeId = nodeId;
      selectedScore = score;
    }
  }

  if (!selectedNodeId) {
    throw new Error("A* 开放集合存在节点但无法选择候选项");
  }

  return selectedNodeId;
}

function reconstructPath(
  cameFrom: ReadonlyMap<string, string>,
  startNodeId: string,
  targetNodeId: string,
): string[] {
  const reversedPath = [targetNodeId];
  let currentNodeId = targetNodeId;

  while (currentNodeId !== startNodeId) {
    const previousNodeId = cameFrom.get(currentNodeId);
    if (!previousNodeId) {
      throw new Error(
        `无法重建从节点 "${startNodeId}" 到节点 "${targetNodeId}" 的路径`,
      );
    }

    reversedPath.push(previousNodeId);
    currentNodeId = previousNodeId;
  }

  return reversedPath.reverse();
}

function estimateTotalCost(
  traveledCost: number,
  node: RouteNode,
  targetNode: RouteNode,
  heuristicScale: number,
): number {
  const heuristicCost = heuristicScale * distance(node, targetNode);
  if (!Number.isFinite(heuristicCost)) {
    return traveledCost;
  }

  const estimatedTotal = traveledCost + heuristicCost;
  return Number.isFinite(estimatedTotal) ? estimatedTotal : traveledCost;
}

function updateNeighborScores(
  scores: SearchScores,
  currentNodeId: string,
  edge: RouteEdge,
  neighborNode: RouteNode,
  targetNode: RouteNode,
  heuristicScale: number,
): SearchScores | undefined {
  const currentCost = scores.traveled.get(currentNodeId);
  if (currentCost === undefined) {
    throw new Error(`节点 "${currentNodeId}" 缺少已行进代价`);
  }

  const nextCost = currentCost + edge.cost;
  if (!Number.isFinite(nextCost)) {
    throw new Error(
      `节点 "${currentNodeId}" 到 "${edge.targetNodeId}" 的累计代价超出有限数范围`,
    );
  }

  const knownCost = scores.traveled.get(edge.targetNodeId);
  if (knownCost !== undefined && nextCost >= knownCost) {
    return undefined;
  }

  const traveled = new Map(scores.traveled);
  const estimatedTotal = new Map(scores.estimatedTotal);
  traveled.set(edge.targetNodeId, nextCost);
  estimatedTotal.set(
    edge.targetNodeId,
    estimateTotalCost(nextCost, neighborNode, targetNode, heuristicScale),
  );

  return { traveled, estimatedTotal };
}

export function findShortestPath(
  graph: RouteGraph,
  startNodeId: string,
  targetNodeId: string,
): string[] {
  validateGraph(graph);
  const startNode = requireNode(graph, startNodeId, "起点");
  const targetNode = requireNode(graph, targetNodeId, "终点");

  if (startNodeId === targetNodeId) {
    return [startNodeId];
  }

  const heuristicScale = calculateHeuristicScale(graph);
  const openNodeIds = new Set([startNodeId]);
  const cameFrom = new Map<string, string>();
  let scores: SearchScores = {
    traveled: new Map([[startNodeId, 0]]),
    estimatedTotal: new Map([
      [startNodeId, estimateTotalCost(0, startNode, targetNode, heuristicScale)],
    ]),
  };

  while (openNodeIds.size > 0) {
    const currentNodeId = selectLowestScore(openNodeIds, scores.estimatedTotal);
    if (currentNodeId === targetNodeId) {
      return reconstructPath(cameFrom, startNodeId, targetNodeId);
    }

    openNodeIds.delete(currentNodeId);
    const currentNode = graph[currentNodeId];
    if (!currentNode) {
      throw new Error(`A* 搜索中的节点 "${currentNodeId}" 已从路线图移除`);
    }

    for (const edge of currentNode.edges) {
      validateEdge(graph, currentNodeId, edge);
      const neighbor = graph[edge.targetNodeId];
      if (!neighbor) {
        continue;
      }

      const updatedScores = updateNeighborScores(
        scores,
        currentNodeId,
        edge,
        neighbor,
        targetNode,
        heuristicScale,
      );
      if (!updatedScores) {
        continue;
      }

      scores = updatedScores;
      cameFrom.set(edge.targetNodeId, currentNodeId);
      openNodeIds.add(edge.targetNodeId);
    }
  }

  throw new Error(`无法从节点 "${startNodeId}" 到达节点 "${targetNodeId}"`);
}
