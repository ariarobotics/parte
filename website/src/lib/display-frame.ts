import * as THREE from "three";

export interface DisplayArrays {
  source: Float32Array;
  target: Float32Array;
  sourceLabels: Int16Array;
  targetLabels: Int16Array;
  sourcePlanes: Float32Array;
  targetPlanes: Float32Array;
  pointMatches: Float32Array;
  pointSelection: Uint8Array;
  planeMatches: Float32Array;
  planeSelection: Uint8Array;
}

interface DisplayMetadata {
  dataset: string;
  transform: number[];
  sourcePlaneLabels: number[];
  targetPlaneLabels: number[];
}

const matrixFromRows = (rows: number[]) => new THREE.Matrix4().fromArray(rows).transpose();
const matrixToRows = (matrix: THREE.Matrix4) => matrix.clone().transpose().toArray();

function transformPositions(points: Float32Array, matrix: THREE.Matrix4): Float32Array {
  const output = new Float32Array(points.length);
  const point = new THREE.Vector3();
  for (let offset = 0; offset < points.length; offset += 3) {
    point.fromArray(points, offset).applyMatrix4(matrix).toArray(output, offset);
  }
  return output;
}

function swapEndpoints(points: Float32Array): Float32Array {
  const output = new Float32Array(points.length);
  for (let offset = 0; offset < points.length; offset += 6) {
    output.set(points.subarray(offset + 3, offset + 6), offset);
    output.set(points.subarray(offset, offset + 3), offset + 3);
  }
  return output;
}

function stableBounds(points: Float32Array): THREE.Box3 {
  const count = points.length / 3;
  const bounds = new THREE.Box3();
  if (!count) return bounds.set(new THREE.Vector3(), new THREE.Vector3());
  for (let axis = 0; axis < 3; axis += 1) {
    const values = new Float32Array(count);
    for (let i = 0; i < count; i += 1) values[i] = points[3 * i + axis];
    values.sort();
    bounds.min.setComponent(axis, values[Math.floor((count - 1) * 0.01)]);
    bounds.max.setComponent(axis, values[Math.ceil((count - 1) * 0.99)]);
  }
  return bounds;
}

export function isIndoorDataset(dataset: string): boolean {
  return dataset === "3DMatch" || dataset === "3DLoMatch";
}

function levelIndoorTarget(
  metadata: DisplayMetadata,
  arrays: DisplayArrays,
  rotation: THREE.Matrix4,
): THREE.Matrix4 {
  const supports = new Map<number, number>();
  for (const label of arrays.targetLabels) supports.set(label, (supports.get(label) ?? 0) + 1);
  let bestScore = 0;
  let anchor: THREE.Vector3 | undefined;
  const walls: Array<{ normal: THREE.Vector3; count: number }> = [];
  metadata.targetPlaneLabels.forEach((label) => {
    const normal = new THREE.Vector3().fromArray(
      arrays.targetPlanes, label * 6 + 3,
    ).transformDirection(rotation);
    const count = supports.get(label) ?? 0;
    if (Number.isFinite(normal.y) && count > 0) walls.push({ normal: normal.clone(), count });
    const vertical = Math.abs(normal.y);
    if (!Number.isFinite(vertical) || vertical < 0.5) return;
    const score = count * vertical ** 4;
    if (score <= bestScore) return;
    bestScore = score;
    anchor = normal.y < 0 ? normal.negate() : normal;
  });
  if (!anchor) {
    for (let i = 0; i < walls.length; i += 1) {
      for (let j = i + 1; j < walls.length; j += 1) {
        const normal = walls[i].normal.clone().cross(walls[j].normal);
        const separation = normal.length();
        if (separation < 0.5) continue;
        normal.divideScalar(separation);
        const vertical = Math.abs(normal.y);
        if (vertical < 0.5) continue;
        const score = Math.sqrt(walls[i].count * walls[j].count) * vertical ** 4 * separation ** 2;
        if (score <= bestScore) continue;
        bestScore = score;
        anchor = normal.y < 0 ? normal.negate() : normal;
      }
    }
  }
  if (!anchor) return rotation;
  const leveling = new THREE.Matrix4().makeRotationFromQuaternion(
    new THREE.Quaternion().setFromUnitVectors(anchor, new THREE.Vector3(0, 1, 0)),
  );
  return leveling.multiply(rotation);
}

export function prepareDisplay<T extends DisplayMetadata>(original: T, input: DisplayArrays) {
  const metadata = { ...original };
  let arrays = { ...input };
  let transform = matrixFromRows(original.transform);
  if (original.dataset === "RESSO P2F") {
    const values = metadata as unknown as Record<string, unknown>;
    const originals = original as unknown as Record<string, unknown>;
    for (const key of Object.keys(original)) {
      if (!key.startsWith("source")) continue;
      const targetKey = `target${key.slice(6)}`;
      if (!(targetKey in originals)) continue;
      values[key] = originals[targetKey];
      values[targetKey] = originals[key];
    }
    for (const key of ["planeProposalPairs", "matchedPlanePairs"]) {
      const pairs = originals[key] as number[][] | undefined;
      if (pairs) values[key] = pairs.map(([source, target]) => [target, source]);
    }
    arrays = {
      ...input,
      source: input.target, target: input.source,
      sourceLabels: input.targetLabels, targetLabels: input.sourceLabels,
      sourcePlanes: input.targetPlanes, targetPlanes: input.sourcePlanes,
      pointMatches: swapEndpoints(input.pointMatches),
      planeMatches: swapEndpoints(input.planeMatches),
    };
    transform.invert();
  }

  const r = isIndoorDataset(original.dataset) ? [1, 0, 0, 0, -1, 0, 0, 0, -1] : [1, 0, 0, 0, 0, 1, 0, -1, 0];
  let frame = new THREE.Matrix4().set(
    r[0], r[1], r[2], 0, r[3], r[4], r[5], 0,
    r[6], r[7], r[8], 0, 0, 0, 0, 1,
  );
  const indoor = isIndoorDataset(original.dataset);
  if (indoor) frame = levelIndoorTarget(metadata, arrays, frame);
  if (indoor || original.dataset === "RESSO P2F") {
    const bounds = stableBounds(transformPositions(arrays.target, frame));
    const center = bounds.getCenter(new THREE.Vector3());
    frame.setPosition(-center.x, -bounds.min.y, -center.z);
  }

  metadata.transform = matrixToRows(frame.clone().multiply(transform).multiply(frame.clone().invert()));
  return {
    metadata,
    gridFloorY: indoor ? 0 : undefined,
    arrays: {
      ...arrays,
      source: transformPositions(arrays.source, frame),
      target: transformPositions(arrays.target, frame),
      pointMatches: transformPositions(arrays.pointMatches, frame),
      planeMatches: transformPositions(arrays.planeMatches, frame),
    },
    frame: matrixToRows(frame),
  };
}
