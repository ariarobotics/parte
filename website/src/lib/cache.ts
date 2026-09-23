// The sole cache format, written by website/cache/export.cpp.
export type Manifest = Record<string, Record<string, Array<[number, number, string, string?]>>>;
export interface PairItem {
  id: string;
  dataset: string;
  sequence: string;
  sourceScan: number;
  targetScan: number;
  distanceRange?: string;
}
export interface PairMetadata {
  source: string;
  target: string;
  transform: number[];
  runtime: [number, number];
  errors: [number, number];
  error?: string;
}

// Voxel size (m), translation limit (m), rotation limit (degrees).
const DATASETS: Record<string, [number, number, number]> = {
  "3DMatch": [0.05, 0.3, 15],
  "3DLoMatch": [0.05, 0.3, 15],
  "KITTI-10m": [0.3, 0.6, 5],
  "KITTI-LC": [0.3, 2, 5],
  "ETH": [0.1, 0.3, 15],
  "RESSO P2F": [0.05, 0.3, 15],
};

export function listPairs(manifest: Manifest): PairItem[] {
  return Object.entries(manifest).flatMap(([dataset, sequences]) =>
    Object.entries(sequences).flatMap(([sequence, pairs]) =>
      pairs.map(([sourceScan, targetScan, id, distanceRange]) =>
        ({ dataset, sequence, sourceScan, targetScan, id, distanceRange }))));
}

export function decodeScan(buffer: ArrayBuffer) {
  const header = new DataView(buffer);
  const count = header.getUint32(0, true);
  const nonplanar = header.getUint32(4, true);
  const rows = header.getUint32(8, true);
  const minimum = [0, 1, 2].map(axis => header.getFloat64(12 + axis * 8, true));
  const scale = [0, 1, 2].map(axis => header.getFloat64(36 + axis * 8, true));
  const quantized = new Uint16Array(buffer, 60, count * 3);
  const points = new Float32Array(count * 3);
  for (let i = 0; i < points.length; i++) points[i] = minimum[i % 3] + quantized[i] * scale[i % 3];
  // Expanded labels are rendering state; the file stores only the planar suffix.
  const labels = new Int16Array(count).fill(-1, 0, nonplanar);
  labels.set(new Uint8Array(buffer, 60 + count * 6, count - nonplanar), nonplanar);
  const offset = Math.ceil((60 + count * 7 - nonplanar) / 4) * 4;
  const planes = new Float32Array(buffer, offset, rows * 6);
  const groundLabels = labels.includes(0) ? [0] : [];
  const planeLabels = Array.from({ length: rows }, (_, i) => i).filter(i => i > 0 || groundLabels.length > 0);
  return { count, points, labels, planes, groundLabels, planeLabels };
}

export function decodePair(item: PairItem, pair: PairMetadata, sourceBuffer: ArrayBuffer, targetBuffer: ArrayBuffer, matches: ArrayBuffer) {
  const source = decodeScan(sourceBuffer), target = decodeScan(targetBuffer);
  const header = new DataView(matches);
  const pc = header.getUint32(0, true), ps = header.getUint32(4, true);
  const qc = header.getUint32(8, true), qs = header.getUint32(12, true);
  const pointPairs = new Uint32Array(matches, 16, pc);
  const selectedPoints = new Uint16Array(matches, 16 + pc * 4, ps);
  const planeOffset = Math.ceil((16 + pc * 4 + ps * 2) / 4) * 4;
  const planePairs = new Uint32Array(matches, planeOffset, qc);
  const selectedPlanes = new Uint16Array(matches, planeOffset + qc * 4, qs);
  const endpoints = (pairs: Uint32Array, s: Float32Array, t: Float32Array, stride: number) => {
    const positions = new Float32Array(pairs.length * 6);
    pairs.forEach((pair, i) => {
      positions.set(s.subarray((pair & 65535) * stride, (pair & 65535) * stride + 3), i * 6);
      positions.set(t.subarray((pair >>> 16) * stride, (pair >>> 16) * stride + 3), i * 6 + 3);
    });
    return positions;
  };
  const selection = (count: number, selected: Uint16Array) => {
    const mask = new Uint8Array(count);
    selected.forEach(i => { mask[i] = 1; });
    return mask;
  };
  const planeProposalPairs = Array.from(planePairs, pair => [pair & 65535, pair >>> 16]);
  const [voxelSizeMeters, successTranslationThresholdMeters, successRotationThresholdDegrees] = DATASETS[item.dataset];
  return {
    metadata: {
      ...item, transform: [...pair.transform, 0, 0, 0, 1],
      sourceCount: source.count, targetCount: target.count,
      sourcePlaneLabels: source.planeLabels, targetPlaneLabels: target.planeLabels,
      sourceGroundLabels: source.groundLabels, targetGroundLabels: target.groundLabels,
      planeProposalPairs, matchedPlanePairs: Array.from(selectedPlanes, i => planeProposalPairs[i]),
      pointMatchCount: pc, retainedPointMatchCount: ps, planeMatchCount: qc, retainedPlaneMatchCount: qs,
      voxelSizeMeters, successTranslationThresholdMeters, successRotationThresholdDegrees,
      registrationRuntimeMs: pair.runtime[0], translationErrorMeters: pair.errors[0], rotationErrorDegrees: pair.errors[1],
      completed: !pair.error, error: pair.error,
    },
    arrays: {
      source: source.points, target: target.points, sourceLabels: source.labels, targetLabels: target.labels,
      sourcePlanes: source.planes, targetPlanes: target.planes,
      pointMatches: endpoints(pointPairs, source.points, target.points, 3),
      planeMatches: endpoints(planePairs, source.planes, target.planes, 6),
      pointSelection: selection(pc, selectedPoints), planeSelection: selection(qc, selectedPlanes),
    },
  };
}
export type MethodMetadata = ReturnType<typeof decodePair>["metadata"];
