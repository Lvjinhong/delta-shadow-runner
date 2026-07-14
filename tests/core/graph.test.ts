import { describe, expect, it } from "vitest";

import { findShortestPath } from "../../src/core/graph.js";
import type { RouteGraph } from "../../src/core/types.js";

const graph: RouteGraph = {
  "spawn-a": {
    id: "spawn-a",
    x: 0,
    y: 0,
    edges: [
      { targetNodeId: "relay", cost: 1 },
      { targetNodeId: "decoy", cost: 1 },
    ],
  },
  relay: {
    id: "relay",
    x: 1,
    y: 0,
    edges: [{ targetNodeId: "warehouse", cost: 1 }],
  },
  warehouse: {
    id: "warehouse",
    x: 2,
    y: 0,
    edges: [{ targetNodeId: "extract", cost: 1 }],
  },
  decoy: {
    id: "decoy",
    x: 0,
    y: 1,
    edges: [{ targetNodeId: "extract", cost: 8 }],
  },
  extract: {
    id: "extract",
    x: 3,
    y: 0,
    edges: [],
  },
  isolated: {
    id: "isolated",
    x: 9,
    y: 9,
    edges: [],
  },
};

describe("findShortestPath", () => {
  it("选择到撤离点的最低代价路径", () => {
    expect(findShortestPath(graph, "spawn-a", "extract")).toEqual([
      "spawn-a",
      "relay",
      "warehouse",
      "extract",
    ]);
  });

  it.each([
    ["missing-start", "extract"],
    ["spawn-a", "missing-target"],
  ])("拒绝未知节点 %s -> %s", (startNodeId, targetNodeId) => {
    expect(() => findShortestPath(graph, startNodeId, targetNodeId)).toThrow(
      new RegExp(startNodeId === "missing-start" ? startNodeId : targetNodeId),
    );
  });

  it("不可达时在错误中包含起点和终点", () => {
    expect(() => findShortestPath(graph, "isolated", "extract")).toThrow(
      /isolated.*extract/,
    );
  });

  it("起点等于终点时直接返回该节点", () => {
    expect(findShortestPath(graph, "relay", "relay")).toEqual(["relay"]);
  });
});
