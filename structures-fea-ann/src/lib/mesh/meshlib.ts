import type { NodalDisplacement } from '$lib/types/contracts';

export type MeshSegment = {
  a: NodalDisplacement;
  b: NodalDisplacement;
};

export type MeshSlice = {
  nodes: NodalDisplacement[];
  segments: MeshSegment[];
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  maxDisp: number;
  minVm: number;
  maxVm: number;
  sliceZ: number;
};

export function extractMidThicknessSlice(allNodes: NodalDisplacement[] | null): MeshSlice | null {
  const all = allNodes ?? [];
  if (all.length === 0) return null;

  const zValues = Array.from(new Set(all.map((n) => n.zIn))).sort((a, b) => a - b);
  const midZ = (zValues[0] + zValues[zValues.length - 1]) * 0.5;
  const sliceZ = zValues.reduce((best, z) => (Math.abs(z - midZ) < Math.abs(best - midZ) ? z : best), zValues[0]);
  const nodes = all.filter((n) => Math.abs(n.zIn - sliceZ) <= 1e-9);
  if (nodes.length === 0) return null;

  const xVals = Array.from(new Set(nodes.map((n) => n.xIn))).sort((a, b) => a - b);
  const yVals = Array.from(new Set(nodes.map((n) => n.yIn))).sort((a, b) => a - b);

  const nodeMap = new Map<string, NodalDisplacement>();
  for (const n of nodes) {
    nodeMap.set(`${n.xIn.toFixed(9)}|${n.yIn.toFixed(9)}`, n);
  }

  const getNode = (x: number, y: number) => nodeMap.get(`${x.toFixed(9)}|${y.toFixed(9)}`) ?? null;
  const segments: MeshSegment[] = [];

  for (let yi = 0; yi < yVals.length; yi++) {
    for (let xi = 0; xi < xVals.length - 1; xi++) {
      const a = getNode(xVals[xi], yVals[yi]);
      const b = getNode(xVals[xi + 1], yVals[yi]);
      if (a && b) segments.push({ a, b });
    }
  }

  for (let xi = 0; xi < xVals.length; xi++) {
    for (let yi = 0; yi < yVals.length - 1; yi++) {
      const a = getNode(xVals[xi], yVals[yi]);
      const b = getNode(xVals[xi], yVals[yi + 1]);
      if (a && b) segments.push({ a, b });
    }
  }

  const maxDisp = Math.max(1e-12, ...nodes.map((n) => n.dispMagIn));
  const vmList = nodes.map((n) => n.vmPsi);
  const minVm = Math.min(...vmList);
  const maxVm = Math.max(...vmList);

  return {
    nodes,
    segments,
    xMin: xVals[0],
    xMax: xVals[xVals.length - 1],
    yMin: yVals[0],
    yMax: yVals[yVals.length - 1],
    maxDisp,
    minVm,
    maxVm,
    sliceZ
  };
}
