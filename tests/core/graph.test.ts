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

  it("不会被几何距离更近但代价更高的路线误导", () => {
    const greedyTrap: RouteGraph = {
      start: {
        id: "start",
        x: 0,
        y: 0,
        edges: [
          { targetNodeId: "near", cost: 100 },
          { targetNodeId: "detour", cost: 1 },
        ],
      },
      near: {
        id: "near",
        x: 9,
        y: 0,
        edges: [{ targetNodeId: "target", cost: 1 }],
      },
      detour: {
        id: "detour",
        x: -1_000,
        y: 0,
        edges: [{ targetNodeId: "target", cost: 1 }],
      },
      target: { id: "target", x: 10, y: 0, edges: [] },
    };

    expect(findShortestPath(greedyTrap, "start", "target")).toEqual([
      "start",
      "detour",
      "target",
    ]);
  });

  it.each([
    ["nan-node", Number.NaN, 0],
    ["infinite-node", 0, Number.POSITIVE_INFINITY],
  ])("拒绝坐标不是有限数的节点 %s", (nodeId, x, y) => {
    const invalidCoordinates: RouteGraph = {
      start: {
        id: "start",
        x: 0,
        y: 0,
        edges: [{ targetNodeId: nodeId, cost: 1 }],
      },
      [nodeId]: { id: nodeId, x, y, edges: [] },
    };

    expect(() => findShortestPath(invalidCoordinates, "start", nodeId)).toThrow(
      new RegExp(nodeId),
    );
  });

  it("有限边权累计溢出时报告发生溢出的边", () => {
    const overflowingCosts: RouteGraph = {
      start: {
        id: "start",
        x: 0,
        y: 0,
        edges: [{ targetNodeId: "middle", cost: Number.MAX_VALUE }],
      },
      middle: {
        id: "middle",
        x: 1,
        y: 0,
        edges: [{ targetNodeId: "target", cost: Number.MAX_VALUE }],
      },
      target: { id: "target", x: 2, y: 0, edges: [] },
    };

    expect(() => findShortestPath(overflowingCosts, "start", "target")).toThrow(
      /middle.*target/,
    );
  });

  it("支持零权边并仍选择最低代价路径", () => {
    const zeroCostGraph: RouteGraph = {
      start: {
        id: "start",
        x: 0,
        y: 0,
        edges: [
          { targetNodeId: "target", cost: 1 },
          { targetNodeId: "free", cost: 0 },
        ],
      },
      free: {
        id: "free",
        x: 1,
        y: 0,
        edges: [{ targetNodeId: "target", cost: 0 }],
      },
      target: { id: "target", x: 2, y: 0, edges: [] },
    };

    expect(findShortestPath(zeroCostGraph, "start", "target")).toEqual([
      "start",
      "free",
      "target",
    ]);
  });

  it("重复边存在时采用代价更低的一条", () => {
    const duplicateEdges: RouteGraph = {
      start: {
        id: "start",
        x: 0,
        y: 0,
        edges: [
          { targetNodeId: "target", cost: 10 },
          { targetNodeId: "target", cost: 1 },
        ],
      },
      target: { id: "target", x: 1, y: 0, edges: [] },
    };

    expect(findShortestPath(duplicateEdges, "start", "target")).toEqual([
      "start",
      "target",
    ]);
  });

  it("拒绝指向未知节点的悬空边", () => {
    const danglingEdge: RouteGraph = {
      start: {
        id: "start",
        x: 0,
        y: 0,
        edges: [{ targetNodeId: "missing", cost: 1 }],
      },
      target: { id: "target", x: 1, y: 0, edges: [] },
    };

    expect(() => findShortestPath(danglingEdge, "start", "target")).toThrow(
      /start.*missing/,
    );
  });

  it.each([
    ["负数", -1],
    ["NaN", Number.NaN],
    ["正无穷", Number.POSITIVE_INFINITY],
    ["负无穷", Number.NEGATIVE_INFINITY],
  ])("拒绝%s边权", (_caseName, cost) => {
    const invalidCost: RouteGraph = {
      start: {
        id: "start",
        x: 0,
        y: 0,
        edges: [{ targetNodeId: "target", cost }],
      },
      target: { id: "target", x: 1, y: 0, edges: [] },
    };

    expect(() => findShortestPath(invalidCost, "start", "target")).toThrow(
      /start.*target/,
    );
  });

  it("不会修改输入图或其嵌套边", () => {
    const immutableGraph: RouteGraph = Object.freeze({
      start: Object.freeze({
        id: "start",
        x: 0,
        y: 0,
        edges: Object.freeze([
          Object.freeze({ targetNodeId: "target", cost: 1 }),
        ]),
      }),
      target: Object.freeze({
        id: "target",
        x: 1,
        y: 0,
        edges: Object.freeze([]),
      }),
    });
    const before = JSON.stringify(immutableGraph);

    expect(findShortestPath(immutableGraph, "start", "target")).toEqual([
      "start",
      "target",
    ]);
    expect(JSON.stringify(immutableGraph)).toBe(before);
  });
});
