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
