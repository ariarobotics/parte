import * as THREE from "three";
import { prepareDisplay, isIndoorDataset } from "./display-frame";
import { decodePair, listPairs, type Manifest, type PairItem as ManifestItem, type PairMetadata, type MethodMetadata } from "./cache";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

type ScanView = "source" | "both" | "target";
type ViewStage =
  | "input"
  | "planar-patches"
  | "plane-matching"
  | "point-matching"
  | "outlier-rejection"
  | "alignment";

interface StageDefinition {
  id: ViewStage;
  title: string;
  description: string;
}

const PARTE_STAGES: ReadonlyArray<StageDefinition> = [
  { id: "input", title: "Input", description: "Input point-cloud pair." },
  { id: "planar-patches", title: "Planar Patches", description: "Extracted planar patches." },
  { id: "plane-matching", title: "Planar Patch Matching (PCH)", description: "Planar patches matched using mutual nearest neighbors." },
  { id: "point-matching", title: "Point Matching", description: "All point matches using FPFH and mutual nearest neighbors." },
  { id: "outlier-rejection", title: "Outlier Rejection", description: "Only the plane and point correspondences retained by PARTE." },
  { id: "alignment", title: "Alignment", description: "Transformation estimated by PARTE." },
];

interface AnimationBinding {
  geometry: THREE.BufferGeometry;
  target: THREE.BufferAttribute | THREE.InterleavedBuffer;
  input: Float32Array;
  aligned: Float32Array;
  splitRole: "source" | "target" | "pair";
}

interface AnimationTrack {
  target: THREE.BufferAttribute | THREE.InterleavedBuffer;
  from: Float32Array;
  to: Float32Array;
}

interface OpacityTrack {
  object: THREE.Object3D;
  from: number;
  to: number;
  depthWrite: boolean | null;
}

interface DrawTrack {
  target: THREE.BufferAttribute | THREE.InterleavedBuffer;
  from: Float32Array;
  to: Float32Array;
  geometry: THREE.BufferGeometry;
}

interface SceneSample {
  metadata: MethodMetadata;
  sourceInput: THREE.Points;
  targetInput: THREE.Points;
  sourceFeatures: THREE.Points;
  targetFeatures: THREE.Points;
  sourcePatchMatches: THREE.Points;
  targetPatchMatches: THREE.Points;
  sourcePlaneMatches: THREE.Points;
  targetPlaneMatches: THREE.Points;
  sourceNonplanar: THREE.Points;
  targetNonplanar: THREE.Points;
  sourcePlanarFaded: THREE.Points;
  targetPlanarFaded: THREE.Points;
  pointCandidateLines: THREE.LineSegments;
  pointSelectedLines: THREE.LineSegments;
  planeAllLines: LineSegments2;
  planeSelectedLines: LineSegments2;
  objects: THREE.Object3D[];
  points: THREE.Points[];
  animationBindings: AnimationBinding[];
  cameraBounds: THREE.Box3;
  unsplitCameraBounds: THREE.Box3;
  splitCameraBounds: THREE.Box3;
  gridFloorY?: number;
  sourceSplitOffset: THREE.Vector3;
  targetSplitOffset: THREE.Vector3;
}

const SOURCE_PATCH_COLORS = [
  0xe18a20, 0xc96d1b, 0xdc9333, 0xbc7627, 0xe4a04a,
  0xc85d21, 0xd88926, 0xb86a24, 0xe19831, 0xc58131,
];
const TARGET_PATCH_COLORS = [
  0x3b78b0, 0x21639e, 0x4a83b9, 0x356fa3, 0x568cbd,
  0x1e6095, 0x447cb0, 0x31699d, 0x6294bf, 0x286da8,
];
const MATCHED_PLANE_COLORS = [
  0x8e44ad, 0x168f7a, 0xc44752, 0x6c5fba, 0xa88a20,
  0x438e4d, 0xb34782, 0x537c37, 0x9b59b6, 0x22a2a0,
  0x8f536f, 0x6d8731, 0xc05283, 0x467f68, 0x7e5aa7,
];
const SOURCE_COLOR = 0xd47a0b;
const TARGET_COLOR = 0x2c6eaa;
const GROUND_COLOR = 0x87ad98;
const RETAINED_COLOR = 0x2f8b57;
const CANDIDATE_COLOR = 0xaab2b9;
const INITIAL_POINT_SIZE = 0.30;
const INITIAL_POINT_SCALE = 1;
const CONTEXT_BACKGROUND_MIX = 0.62;
const FADED_PLANAR_OPACITY = 0.48;
const IDLE_STAGE_DURATION_MS = 5_000;

function makeUniformPoints(
  positions: Float32Array,
  color: number,
  opacity = 0.82,
): THREE.Points {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions.slice(), 3));
  geometry.computeBoundingSphere();
  const material = new THREE.PointsMaterial({
    color,
    size: INITIAL_POINT_SIZE,
    sizeAttenuation: true,
    transparent: true,
    opacity,
    depthWrite: opacity >= 0.8,
  });
  material.userData.baseOpacity = opacity;
  return new THREE.Points(geometry, material);
}

function makeVertexPoints(
  positions: Float32Array,
  colors: Float32Array,
  scale = 1.14,
  opacity = 0.92,
): THREE.Points {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions.slice(), 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();
  const material = new THREE.PointsMaterial({
    vertexColors: true,
    size: INITIAL_POINT_SIZE * scale,
    sizeAttenuation: true,
    transparent: true,
    opacity,
    depthWrite: opacity >= 0.8,
  });
  material.userData.pointSizeScale = scale;
  material.userData.baseOpacity = opacity;
  return new THREE.Points(geometry, material);
}

function writeColor(
  output: Float32Array,
  index: number,
  color: THREE.Color,
) {
  output[3 * index] = color.r;
  output[3 * index + 1] = color.g;
  output[3 * index + 2] = color.b;
}

function makeFeaturePoints(
  positions: Float32Array,
  labels: Int16Array,
  colorByLabel: Map<number, THREE.Color>,
  groundLabels: number[],
  contextBaseColor: number,
): THREE.Points {
  const ground = new Set(groundLabels);
  const contextColor = new THREE.Color(contextBaseColor)
    .lerp(new THREE.Color(0xf4f7fa), CONTEXT_BACKGROUND_MIX);
  const groundColor = new THREE.Color(GROUND_COLOR);
  const colors = new Float32Array(positions.length);
  for (let index = 0; index < labels.length; index += 1) {
    writeColor(
      colors,
      index,
      ground.has(labels[index])
        ? groundColor
        : (colorByLabel.get(labels[index]) ?? contextColor),
    );
  }
  return makeVertexPoints(positions, colors, 1.20);
}

function makePlaneColorMaps(
  sourcePlaneLabels: number[],
  targetPlaneLabels: number[],
  proposalPairs: number[][],
): {
  source: Map<number, THREE.Color>;
  target: Map<number, THREE.Color>;
  pairColors: number[];
} {
  const source = new Map<number, THREE.Color>();
  const target = new Map<number, THREE.Color>();
  const pairColors: number[] = [];
  let colorIndex = 0;
  proposalPairs.forEach(([sourceLabel, targetLabel]) => {
    const existing = source.get(sourceLabel) ?? target.get(targetLabel);
    const color = existing ?? new THREE.Color(
      MATCHED_PLANE_COLORS[colorIndex++ % MATCHED_PLANE_COLORS.length],
    );
    if (!source.has(sourceLabel)) source.set(sourceLabel, color);
    if (!target.has(targetLabel)) target.set(targetLabel, color);
    pairColors.push(color.getHex());
  });
  sourcePlaneLabels.forEach((label) => {
    if (!source.has(label)) {
      source.set(label, new THREE.Color(
        MATCHED_PLANE_COLORS[colorIndex++ % MATCHED_PLANE_COLORS.length],
      ));
    }
  });
  targetPlaneLabels.forEach((label) => {
    if (!target.has(label)) {
      target.set(label, new THREE.Color(
        MATCHED_PLANE_COLORS[colorIndex++ % MATCHED_PLANE_COLORS.length],
      ));
    }
  });
  return { source, target, pairColors };
}

function makeExtractionColorMap(
  planeLabels: number[],
  palette: number[],
): Map<number, THREE.Color> {
  return new Map(planeLabels.map((label, index) => [
    label,
    new THREE.Color(palette[index % palette.length]),
  ]));
}

function makeNonplanarPoints(
  positions: Float32Array,
  labels: Int16Array,
  planeLabels: number[],
  color: number,
): { points: THREE.Points; positions: Float32Array } {
  const planes = new Set(planeLabels);
  let count = 0;
  for (const label of labels) {
    if (!planes.has(label)) count += 1;
  }
  const filtered = new Float32Array(count * 3);
  let output = 0;
  for (let index = 0; index < labels.length; index += 1) {
    if (planes.has(labels[index])) continue;
    filtered.set(positions.subarray(3 * index, 3 * index + 3), output);
    output += 3;
  }
  return { points: makeUniformPoints(filtered, color, 0.82), positions: filtered };
}

function makeFadedPlanarPoints(
  positions: Float32Array,
  labels: Int16Array,
  planeLabels: number[],
  matchedPairs: number[][],
  matchedColors: Map<number, THREE.Color>,
  extractionColors: Map<number, THREE.Color>,
  source: boolean,
): { points: THREE.Points; positions: Float32Array } {
  const planes = new Set(planeLabels);
  const matched = new Set(matchedPairs.map((pair) => pair[source ? 0 : 1]));
  let count = 0;
  for (const label of labels) {
    if (planes.has(label)) count += 1;
  }
  const filtered = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  let output = 0;
  for (let index = 0; index < labels.length; index += 1) {
    const label = labels[index];
    if (!planes.has(label)) continue;
    filtered.set(positions.subarray(3 * index, 3 * index + 3), 3 * output);
    const color = matched.has(label)
      ? matchedColors.get(label)
      : extractionColors.get(label);
    writeColor(colors, output, color ?? new THREE.Color(GROUND_COLOR));
    output += 1;
  }
  return {
    points: makeVertexPoints(filtered, colors, 1.20, FADED_PLANAR_OPACITY),
    positions: filtered,
  };
}

function makeMatchedPlanePoints(
  positions: Float32Array,
  labels: Int16Array,
  matchedPairs: number[][],
  colorByLabel: Map<number, THREE.Color>,
  selectedPointEndpoints: Float32Array,
  source: boolean,
  unmatchedPlaneColors?: Map<number, THREE.Color>,
  pointScale = 1.14,
): THREE.Points {
  const retainedLabels = new Set(matchedPairs.map((pair) => pair[source ? 0 : 1]));
  const selectedPoints = new Set<string>();
  for (let offset = source ? 0 : 3; offset < selectedPointEndpoints.length; offset += 6) {
    selectedPoints.add(pointKey(selectedPointEndpoints, offset));
  }
  const fallback = new THREE.Color(source ? SOURCE_COLOR : TARGET_COLOR)
    .lerp(new THREE.Color(0xf4f7fa), CONTEXT_BACKGROUND_MIX);
  const selectedColor = new THREE.Color(RETAINED_COLOR);
  const colors = new Float32Array(positions.length);
  for (let index = 0; index < labels.length; index += 1) {
    const planeColor = retainedLabels.has(labels[index])
      ? colorByLabel.get(labels[index])
      : unmatchedPlaneColors?.get(labels[index]);
    writeColor(
      colors,
      index,
      selectedPoints.has(pointKey(positions, 3 * index))
        ? selectedColor
        : (planeColor ?? fallback),
    );
  }
  return makeVertexPoints(positions, colors, pointScale);
}

function pointKey(positions: Float32Array, offset: number): string {
  return `${positions[offset]}|${positions[offset + 1]}|${positions[offset + 2]}`;
}

function makeMatchLines(
  endpoints: Float32Array,
  selection: Uint8Array,
  selectedColorForIndex: (index: number) => number,
): THREE.LineSegments {
  const colors = new Float32Array(endpoints.length);
  const color = new THREE.Color();
  for (let index = 0; index < selection.length; index += 1) {
    color.setHex(selection[index] ? selectedColorForIndex(index) : CANDIDATE_COLOR);
    writeColor(colors, 2 * index, color);
    writeColor(colors, 2 * index + 1, color);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(endpoints.slice(), 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.72,
  });
  material.userData.baseOpacity = 0.72;
  const lines = new THREE.LineSegments(geometry, material);
  lines.userData.drawTarget = geometry.getAttribute("position");
  lines.userData.drawPositions = endpoints.slice();
  return lines;
}

function makeWideMatchLines(
  endpoints: Float32Array,
  selection: Uint8Array,
  selectedColorForIndex: (index: number) => number,
  width: number,
): LineSegments2 {
  const colors = new Float32Array(endpoints.length);
  const color = new THREE.Color();
  for (let index = 0; index < selection.length; index += 1) {
    color.setHex(selection[index] ? selectedColorForIndex(index) : CANDIDATE_COLOR);
    writeColor(colors, 2 * index, color);
    writeColor(colors, 2 * index + 1, color);
  }
  const geometry = new LineSegmentsGeometry();
  geometry.setPositions(endpoints.slice());
  geometry.setColors(colors);
  const material = new LineMaterial({
    transparent: true,
    opacity: 0.88,
    alphaToCoverage: true,
  });
  material.vertexColors = true;
  material.linewidth = width;
  material.depthTest = false;
  material.userData.baseOpacity = 0.88;
  const lines = new LineSegments2(geometry, material);
  lines.renderOrder = 2;
  lines.userData.drawTarget = wideLineAnimationTarget(geometry);
  lines.userData.drawPositions = endpoints.slice();
  return lines;
}

function wideLineAnimationTarget(
  geometry: LineSegmentsGeometry,
): THREE.InterleavedBuffer {
  const start = geometry.getAttribute("instanceStart") as THREE.InterleavedBufferAttribute;
  return start.data;
}

function objectMaterial(object: THREE.Object3D): THREE.Material | null {
  const material = (object as THREE.Points | THREE.LineSegments | LineSegments2).material;
  return Array.isArray(material) ? (material[0] ?? null) : (material ?? null);
}

function baseObjectOpacity(object: THREE.Object3D): number {
  const material = objectMaterial(object);
  return Number(material?.userData.baseOpacity ?? material?.opacity ?? 1);
}

function setObjectOpacity(object: THREE.Object3D, opacity: number) {
  const material = objectMaterial(object);
  if (material) material.opacity = opacity;
}

function collapsedLinePositions(positions: Float32Array): Float32Array {
  const collapsed = positions.slice();
  for (let offset = 0; offset < collapsed.length; offset += 6) {
    collapsed[offset + 3] = collapsed[offset];
    collapsed[offset + 4] = collapsed[offset + 1];
    collapsed[offset + 5] = collapsed[offset + 2];
  }
  return collapsed;
}

function filteredMatchEndpoints(
  endpoints: Float32Array,
  selection: Uint8Array,
  keepSelected: boolean,
): Float32Array {
  const output = new Float32Array(
    selection.reduce(
      (count, selected) => count + (Boolean(selected) === keepSelected ? 6 : 0),
      0,
    ),
  );
  let outputOffset = 0;
  for (let index = 0; index < selection.length; index += 1) {
    if (Boolean(selection[index]) !== keepSelected) continue;
    output.set(endpoints.subarray(6 * index, 6 * index + 6), outputOffset);
    outputOffset += 6;
  }
  return output;
}

function selectedMatchEndpoints(
  endpoints: Float32Array,
  selection: Uint8Array,
): Float32Array {
  return filteredMatchEndpoints(endpoints, selection, true);
}

function applyTransform(points: Float32Array, matrix: number[]): Float32Array {
  if (matrix.length !== 16) throw new Error("Invalid registration transform");
  const output = new Float32Array(points.length);
  for (let offset = 0; offset < points.length; offset += 3) {
    const x = points[offset];
    const y = points[offset + 1];
    const z = points[offset + 2];
    output[offset] = matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3];
    output[offset + 1] = matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7];
    output[offset + 2] = matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11];
  }
  return output;
}

function transformSourceEndpoints(
  endpoints: Float32Array,
  matrix: number[],
): Float32Array {
  const output = endpoints.slice();
  for (let offset = 0; offset < output.length; offset += 6) {
    const transformed = applyTransform(output.subarray(offset, offset + 3), matrix);
    output.set(transformed, offset);
  }
  return output;
}

function translatedPositions(
  positions: Float32Array,
  offset: THREE.Vector3,
): Float32Array {
  const output = positions.slice();
  for (let index = 0; index < output.length; index += 3) {
    output[index] += offset.x;
    output[index + 1] += offset.y;
    output[index + 2] += offset.z;
  }
  return output;
}

function splitPositions(
  positions: Float32Array,
  role: AnimationBinding["splitRole"],
  sourceOffset: THREE.Vector3,
  targetOffset: THREE.Vector3,
): Float32Array {
  if (role === "source") return translatedPositions(positions, sourceOffset);
  if (role === "target") return translatedPositions(positions, targetOffset);
  const output = positions.slice();
  for (let index = 0; index < output.length; index += 6) {
    output[index] += sourceOffset.x;
    output[index + 1] += sourceOffset.y;
    output[index + 2] += sourceOffset.z;
    output[index + 3] += targetOffset.x;
    output[index + 4] += targetOffset.y;
    output[index + 5] += targetOffset.z;
  }
  return output;
}

function splitLayout(
  source: Float32Array,
  target: Float32Array,
  voxelSize: number,
): {
  sourceOffset: THREE.Vector3;
  targetOffset: THREE.Vector3;
  bounds: THREE.Box3;
} {
  const empty = new Float32Array();
  const sourceBounds = robustBoundsFromPositions(source, empty);
  const targetBounds = robustBoundsFromPositions(target, empty);
  const sourceCenter = sourceBounds.getCenter(new THREE.Vector3());
  const targetCenter = targetBounds.getCenter(new THREE.Vector3());
  const sourceWidth = Math.max(sourceBounds.max.x - sourceBounds.min.x, voxelSize);
  const targetWidth = Math.max(targetBounds.max.x - targetBounds.min.x, voxelSize);
  const gap = Math.max(0.04 * Math.min(sourceWidth, targetWidth), 2 * voxelSize);
  const centerDistance = 0.5 * sourceWidth + gap + 0.5 * targetWidth;
  const midpoint = 0.5 * (sourceCenter.x + targetCenter.x);
  const sourceOffset = new THREE.Vector3(
    midpoint - 0.5 * centerDistance - sourceCenter.x, 0, 0,
  );
  const targetOffset = new THREE.Vector3(
    midpoint + 0.5 * centerDistance - targetCenter.x, 0, 0,
  );
  const splitSource = translatedPositions(source, sourceOffset);
  const splitTarget = translatedPositions(target, targetOffset);
  const bounds = robustBoundsFromPositions(splitSource, splitTarget);
  return { sourceOffset, targetOffset, bounds };
}

const bufferCache = new Map<string, Promise<ArrayBuffer>>();
function loadBuffer(url: URL): Promise<ArrayBuffer> {
  const existing = bufferCache.get(url.href);
  if (existing) return existing;
  const request = fetch(url).then(response => {
    if (!response.ok) throw new Error(`Failed to load ${url.pathname}`);
    return response.arrayBuffer();
  });
  if (bufferCache.size >= 12) bufferCache.delete(bufferCache.keys().next().value!);
  bufferCache.set(url.href, request);
  request.catch(() => bufferCache.delete(url.href));
  return request;
}
function disposeObject(object: THREE.Object3D) {
  const renderable = object as THREE.Points | THREE.LineSegments;
  renderable.geometry?.dispose();
  const material = renderable.material;
  if (Array.isArray(material)) material.forEach((item) => item.dispose());
  else material?.dispose();
}

function sequenceName(value: string): string {
  return value.replaceAll("_", " ");
}

function boundsFromPositions(
  source: Float32Array,
  target: Float32Array,
): THREE.Box3 {
  const bounds = new THREE.Box3();
  const point = new THREE.Vector3();
  for (const positions of [source, target]) {
    for (let offset = 0; offset < positions.length; offset += 3) {
      point.fromArray(positions, offset);
      bounds.expandByPoint(point);
    }
  }
  return bounds;
}

function robustBoundsFromPositions(
  source: Float32Array,
  target: Float32Array,
  lowerQuantile = 0.01,
  upperQuantile = 0.99,
): THREE.Box3 {
  const count = (source.length + target.length) / 3;
  const lower = new THREE.Vector3();
  const upper = new THREE.Vector3();
  for (let axis = 0; axis < 3; axis += 1) {
    const values = new Float32Array(count);
    let output = 0;
    for (const positions of [source, target]) {
      for (let offset = axis; offset < positions.length; offset += 3) {
        values[output++] = positions[offset];
      }
    }
    values.sort();
    lower.setComponent(axis, values[Math.floor((count - 1) * lowerQuantile)]);
    upper.setComponent(axis, values[Math.ceil((count - 1) * upperQuantile)]);
  }
  return new THREE.Box3(lower, upper);
}

export async function mountMethodViewer(root: HTMLElement) {
  const canvas = root.querySelector<HTMLCanvasElement>("[data-method-canvas]");
  const title = root.querySelector<HTMLElement>("[data-method-title]");
  const status = root.querySelector<HTMLElement>("[data-method-status]");
  const stats = root.querySelector<HTMLElement>("[data-method-stats]");
  const sourcePointStat = root.querySelector<HTMLElement>('[data-method-stat="source-points"]');
  const targetPointStat = root.querySelector<HTMLElement>('[data-method-stat="target-points"]');
  const pointCandidateStat = root.querySelector<HTMLElement>('[data-method-stat="point-candidates"]');
  const selectedPointStat = root.querySelector<HTMLElement>('[data-method-stat="selected-points"]');
  const planeCandidateStat = root.querySelector<HTMLElement>('[data-method-stat="plane-candidates"]');
  const selectedPlaneStat = root.querySelector<HTMLElement>('[data-method-stat="selected-planes"]');
  const placeholder = root.querySelector<HTMLElement>("[data-method-placeholder]");
  const alignmentErrors = root.querySelector<HTMLElement>("[data-alignment-errors]");
  const rotationMetric = root.querySelector<HTMLElement>('[data-alignment-metric="rotation"]');
  const translationMetric = root.querySelector<HTMLElement>('[data-alignment-metric="translation"]');
  const rotationValue = root.querySelector<HTMLElement>('[data-alignment-value="rotation"]');
  const translationValue = root.querySelector<HTMLElement>('[data-alignment-value="translation"]');
  const rotationThreshold = root.querySelector<HTMLElement>('[data-alignment-threshold="rotation"]');
  const translationThreshold = root.querySelector<HTMLElement>('[data-alignment-threshold="translation"]');
  const datasetSelect = root.querySelector<HTMLSelectElement>("[data-method-dataset]");
  const sequenceSelect = root.querySelector<HTMLSelectElement>("[data-method-sequence]");
  const rangeSelect = root.querySelector<HTMLSelectElement>("[data-method-range]");
  const rangeField = root.querySelector<HTMLElement>("[data-method-range-field]");
  const pairPicker = root.querySelector<HTMLElement>("[data-method-pair-picker]");
  const pairSelect = root.querySelector<HTMLSelectElement>("[data-method-pair]");
  const previousButton = root.querySelector<HTMLButtonElement>("[data-stage-previous]");
  const nextButton = root.querySelector<HTMLButtonElement>("[data-stage-next]");
  const stageIndexLabel = root.querySelector<HTMLElement>("[data-stage-index]");
  const stageTitle = root.querySelector<HTMLElement>("[data-stage-title]");
  const stageDescription = root.querySelector<HTMLElement>("[data-stage-description]");
  const stageProgress = Array.from(
    root.querySelectorAll<HTMLElement>("[data-stage-progress]"),
  );
  const scanViewButtons = Array.from(
    root.querySelectorAll<HTMLButtonElement>("[data-scan-view-button]"),
  );
  const resetButton = root.querySelector<HTMLButtonElement>("[data-method-reset]");
  const splitButton = root.querySelector<HTMLButtonElement>("[data-method-split]");
  const pointSizeControl = root.querySelector<HTMLInputElement>("[data-method-point-size]");
  const pointSizeOutput = root.querySelector<HTMLOutputElement>("[data-method-point-size-output]");
  if (
    !canvas || !title || !status || !stats || !sourcePointStat || !targetPointStat
    || !pointCandidateStat || !selectedPointStat || !planeCandidateStat
    || !selectedPlaneStat || !placeholder || !alignmentErrors || !rotationMetric
    || !translationMetric || !rotationValue || !translationValue
    || !rotationThreshold || !translationThreshold || !datasetSelect
    || !sequenceSelect || !rangeSelect || !rangeField || !pairPicker || !pairSelect
    || !previousButton || !nextButton || !stageIndexLabel || !stageTitle
    || !stageDescription || !splitButton
  ) {
    throw new Error("Incomplete method viewer markup");
  }

  const manifestUrl = new URL(root.dataset.manifestUrl ?? "", window.location.href);
  const manifestResponse = await fetch(manifestUrl);
  if (!manifestResponse.ok) throw new Error("Failed to load method example manifest");
  const examples = listPairs(await manifestResponse.json() as Manifest);
  const setOptions = (
    select: HTMLSelectElement,
    options: Array<{ value: string; label: string }>,
  ) => {
    select.replaceChildren(...options.map(({ value, label }) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      return option;
    }));
  };
  const datasetNames = Array.from(new Set(
    examples.map((item) => item.dataset),
  ));
  setOptions(datasetSelect, datasetNames.map((dataset) => ({
    value: dataset,
    label: dataset,
  })));

  const populateSequences = (dataset: string, preferred?: string) => {
    const sequences = Array.from(new Set(
      examples
        .filter((item) => item.dataset === dataset)
        .map((item) => item.sequence),
    ));
    setOptions(sequenceSelect, sequences.map((sequence) => ({
      value: sequence,
      label: sequenceName(sequence),
    })));
    sequenceSelect.value = preferred && sequences.includes(preferred)
      ? preferred
      : (sequences[0] ?? "");
  };

  const populatePairs = (
    dataset: string,
    sequence: string,
    distanceRange?: string,
    preferred?: string,
  ) => {
    const pairs = examples.filter(
      (item) => item.dataset === dataset
        && item.sequence === sequence
        && (!distanceRange || item.distanceRange === distanceRange),
    );
    setOptions(pairSelect, pairs.length ? pairs.map((item) => ({
      value: item.id,
      label: item.dataset === "RESSO P2F"
        ? `${item.targetScan} - ${item.sourceScan}`
        : `${item.sourceScan} - ${item.targetScan}`,
    })) : [{ value: "", label: "No matching pairs" }]);
    pairSelect.value = preferred && pairs.some((item) => item.id === preferred)
      ? preferred
      : (pairs[0]?.id ?? "");
    pairSelect.disabled = pairs.length === 0;
    return pairs;
  };

  const populateRanges = (
    dataset: string,
    sequence: string,
    preferred?: string,
  ) => {
    const ranges = Array.from(new Set(
      examples
        .filter((item) => item.dataset === dataset && item.sequence === sequence)
        .map((item) => item.distanceRange)
        .filter((value): value is string => Boolean(value)),
    ));
    ranges.sort((left, right) => parseFloat(left) - parseFloat(right));
    const visible = dataset === "KITTI-LC" && ranges.length > 0;
    rangeField.hidden = !visible;
    pairPicker.dataset.hasRange = String(visible);
    setOptions(rangeSelect, visible
      ? ranges.map((range) => ({ value: range, label: range.replace(/^(\d+)\D+(\d+)\s*m$/, "$1-$2 m") }))
      : [{ value: "", label: "N/A" }]);
    rangeSelect.value = visible && preferred && ranges.includes(preferred)
      ? preferred
      : (visible ? (ranges[0] ?? "") : "");
    rangeSelect.disabled = !visible;
    return rangeSelect.value || undefined;
  };

  const synchronizePairSelectors = (item: ManifestItem) => {
    datasetSelect.value = item.dataset;
    populateSequences(item.dataset, item.sequence);
    const distanceRange = populateRanges(
      item.dataset, item.sequence, item.distanceRange,
    );
    populatePairs(item.dataset, item.sequence, distanceRange, item.id);
  };

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 2000);
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0xf4f7fa, 1);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.zoomToCursor = true;

  const grid = new THREE.GridHelper(2, 20, 0xaebac5, 0xd7dee4);
  scene.add(grid);

  let sample: SceneSample | null = null;
  let stageIndex = 0;
  let aligned = false;
  let splitPreference = false;
  let split = false;
  let scanView: ScanView = "both";
  let animationFrame = 0;
  let animationStart = 0;
  let animationTracks: AnimationTrack[] = [];
  let stageTransitionStart = 0;
  let opacityTracks: OpacityTrack[] = [];
  let drawTracks: DrawTrack[] = [];
  let transitionOldObjects: THREE.Object3D[] = [];
  let loadVersion = 0;
  let idleStageTimer = 0;
  let idleTourVisible = false;
  let idleTourStopped = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let idleTourObserver: IntersectionObserver | null = null;
  root.dataset.split = "false";
  const activeStages = (): ReadonlyArray<StageDefinition> => PARTE_STAGES;
  const currentStage = (): StageDefinition => activeStages()[stageIndex];

  const updateAlignmentErrors = (metadata: MethodMetadata) => {
    const rotationLimit = metadata.successRotationThresholdDegrees;
    const translationLimit = metadata.successTranslationThresholdMeters;
    if (rotationLimit === undefined || translationLimit === undefined) {
      alignmentErrors.hidden = true;
      return;
    }
    const rotationPass = Number.isFinite(metadata.rotationErrorDegrees)
      && metadata.rotationErrorDegrees <= rotationLimit;
    const translationPass = Number.isFinite(metadata.translationErrorMeters)
      && metadata.translationErrorMeters <= translationLimit;
    rotationMetric.dataset.pass = String(rotationPass);
    translationMetric.dataset.pass = String(translationPass);
    rotationValue.textContent = metadata.rotationErrorDegrees.toFixed(2);
    translationValue.textContent = metadata.translationErrorMeters < 1
      ? `${(metadata.translationErrorMeters * 100).toFixed(1)} cm`
      : `${metadata.translationErrorMeters.toFixed(2)} m`;
    rotationThreshold.textContent = rotationLimit.toFixed(0);
    translationThreshold.textContent = translationLimit < 1
      ? `${(translationLimit * 100).toFixed(0)} cm`
      : `${translationLimit.toFixed(0)} m`;
    alignmentErrors.hidden = false;
  };

  const updateSamplePresentation = () => {
    if (!sample) return;
    sourcePointStat.textContent = sample.metadata.sourceCount.toLocaleString();
    targetPointStat.textContent = sample.metadata.targetCount.toLocaleString();
    pointCandidateStat.textContent = sample.metadata.pointMatchCount.toLocaleString();
    selectedPointStat.textContent = sample.metadata.retainedPointMatchCount.toLocaleString();
    planeCandidateStat.textContent = sample.metadata.planeMatchCount.toLocaleString();
    selectedPlaneStat.textContent = sample.metadata.retainedPlaneMatchCount.toLocaleString();
    updateAlignmentErrors(sample.metadata);
  };

  const resize = () => {
    const parent = canvas.parentElement;
    if (!parent) return;
    const width = Math.max(parent.clientWidth, 1);
    const height = Math.max(parent.clientHeight, 1);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  const resizeObserver = new ResizeObserver(resize);
  if (canvas.parentElement) resizeObserver.observe(canvas.parentElement);
  resize();

  const stageAllowsScanView = (index: number) => (
    activeStages()[index].id === "input"
    || activeStages()[index].id === "planar-patches"
    || activeStages()[index].id === "alignment"
  );

  const cloudsForStage = (index: number): THREE.Points[] => {
    if (!sample) return [];
    let clouds: THREE.Points[];
    switch (activeStages()[index].id) {
      case "planar-patches":
        clouds = [sample.sourceFeatures, sample.targetFeatures];
        break;
      case "plane-matching":
        clouds = [sample.sourcePatchMatches, sample.targetPatchMatches];
        break;
      case "outlier-rejection":
      case "alignment":
        clouds = [sample.sourcePlaneMatches, sample.targetPlaneMatches];
        break;
      case "point-matching":
        clouds = [
          sample.sourceNonplanar,
          sample.targetNonplanar,
          sample.sourcePlanarFaded,
          sample.targetPlanarFaded,
        ];
        break;
      default:
        clouds = [sample.sourceInput, sample.targetInput];
    }
    if (!stageAllowsScanView(index) || scanView === "both") return clouds;
    return scanView === "source" ? [clouds[0]] : [clouds[1]];
  };

  const objectsForStage = (index: number): THREE.Object3D[] => {
    if (!sample) return [];
    const objects: THREE.Object3D[] = [...cloudsForStage(index)];
    switch (activeStages()[index].id) {
      case "plane-matching":
        objects.push(sample.planeAllLines);
        break;
      case "point-matching":
        objects.push(sample.pointCandidateLines, sample.pointSelectedLines);
        break;
      case "outlier-rejection":
        objects.push(sample.pointSelectedLines, sample.planeSelectedLines);
        break;
      case "alignment":
        if (scanView === "both") {
          objects.push(sample.pointSelectedLines, sample.planeSelectedLines);
        }
        break;
    }
    return objects;
  };

  const applyStageVisibility = () => {
    if (!sample) return;
    const visible = new Set(objectsForStage(stageIndex));
    sample.objects.forEach((object) => {
      object.visible = visible.has(object);
      setObjectOpacity(object, baseObjectOpacity(object));
    });
  };

  const finishStageTransition = () => {
    opacityTracks.forEach((track) => {
      setObjectOpacity(track.object, track.to);
      const material = objectMaterial(track.object);
      if (material && track.depthWrite !== null) {
        material.depthWrite = track.depthWrite;
      }
    });
    drawTracks.forEach((track) => {
      (track.target.array as Float32Array).set(track.to);
      track.target.needsUpdate = true;
      track.geometry.computeBoundingBox();
      track.geometry.computeBoundingSphere();
    });
    stageTransitionStart = 0;
    opacityTracks = [];
    drawTracks = [];
    transitionOldObjects = [];
    applyStageVisibility();
  };

  const beginStageTransition = (oldIndex: number) => {
    if (!sample || oldIndex === stageIndex) {
      applyStageVisibility();
      return;
    }
    const activeSample = sample;
    const oldObjects = objectsForStage(oldIndex);
    const nextObjects = objectsForStage(stageIndex);
    const oldSet = new Set(oldObjects);
    const nextSet = new Set(nextObjects);
    activeSample.objects.forEach((object) => { object.visible = false; });
    [...oldObjects, ...nextObjects].forEach((object) => { object.visible = true; });
    transitionOldObjects = oldObjects.filter((object) => !nextSet.has(object));
    opacityTracks = [];
    const addOpacityTrack = (object: THREE.Object3D, from: number, to: number) => {
      const material = objectMaterial(object);
      const depthWrite = material?.depthWrite ?? null;
      if (material) material.depthWrite = false;
      opacityTracks.push({ object, from, to, depthWrite });
    };
    transitionOldObjects.forEach((object) => {
      const opacity = baseObjectOpacity(object);
      setObjectOpacity(object, opacity);
      addOpacityTrack(object, opacity, 0);
    });
    nextObjects.filter((object) => !oldSet.has(object)).forEach((object) => {
      const opacity = baseObjectOpacity(object);
      setObjectOpacity(object, 0);
      addOpacityTrack(object, 0, opacity);
    });
    drawTracks = [];
    nextObjects.filter((object) => !oldSet.has(object)).forEach((object) => {
      if (object.userData.fadeOnly) return;
      const target = object.userData.drawTarget as
        | THREE.BufferAttribute | THREE.InterleavedBuffer | undefined;
      const binding = activeSample.animationBindings.find(
        (candidate) => candidate.geometry === (
          object as THREE.LineSegments | LineSegments2
        ).geometry,
      );
      const basePositions = binding
        ? (aligned ? binding.aligned : binding.input)
        : object.userData.drawPositions as Float32Array | undefined;
      const positions = binding && split
        ? splitPositions(
            basePositions as Float32Array,
            binding.splitRole,
            activeSample.sourceSplitOffset,
            activeSample.targetSplitOffset,
          )
        : basePositions;
      if (!target || !positions) return;
      const from = collapsedLinePositions(positions);
      (target.array as Float32Array).set(from);
      target.needsUpdate = true;
      drawTracks.push({ target, from, to: positions, geometry: (
        object as THREE.LineSegments | LineSegments2
      ).geometry });
    });
    stageTransitionStart = performance.now();
  };

  const resetCamera = () => {
    if (!sample) return;
    const bounds = sample.cameraBounds;
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z, 0.5) * 0.5;
    controls.minPolarAngle = 0;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.screenSpacePanning = false;
    controls.minDistance = 0;
    controls.target.copy(center);
    camera.up.set(0, 1, 0);
    const distanceScale = sample.metadata.dataset === "ETH" ? 1.65 : 1;
    camera.position.set(
      center.x + radius * 0.94 * distanceScale,
      center.y + radius * 0.66 * distanceScale,
      center.z + radius * 1.12 * distanceScale,
    );
    if (isIndoorDataset(sample.metadata.dataset) || sample.metadata.dataset === "RESSO P2F") {
      const direction = camera.position.clone().sub(center).normalize();
      const right = new THREE.Vector3().crossVectors(camera.up, direction).normalize();
      const up = new THREE.Vector3().crossVectors(direction, right);
      const verticalTan = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / 1.2;
      const horizontalTan = verticalTan * camera.aspect;
      let distance = 0;
      for (const x of [-0.5, 0.5]) for (const y of [-0.5, 0.5]) for (const z of [-0.5, 0.5]) {
        const corner = new THREE.Vector3(x * size.x, y * size.y, z * size.z);
        distance = Math.max(distance, corner.dot(direction) + Math.max(
          Math.abs(corner.dot(right)) / horizontalTan,
          Math.abs(corner.dot(up)) / verticalTan,
        ));
      }
      camera.position.copy(center).addScaledVector(direction, Math.max(distance, 0.5));
    }
    camera.near = Math.max(radius / 500, 0.02);
    camera.far = Math.max(camera.position.distanceTo(center) + radius * 8, 50);
    camera.updateProjectionMatrix();
    grid.scale.setScalar(Math.max(radius, 1));
    grid.position.set(center.x, sample.gridFloorY ?? bounds.min.y - radius * 0.025, center.z);
    controls.update();
  };

  const updatePointSize = () => {
    const scale = Number(pointSizeControl?.value ?? INITIAL_POINT_SCALE);
    let framingCompensation = 1;
    if (sample && split) {
      const splitSize = sample.splitCameraBounds.getSize(new THREE.Vector3());
      const unsplitSize = sample.unsplitCameraBounds.getSize(new THREE.Vector3());
      const splitSpan = Math.max(splitSize.x, splitSize.y, splitSize.z);
      const unsplitSpan = Math.max(unsplitSize.x, unsplitSize.y, unsplitSize.z, 1e-6);
      framingCompensation = Math.min(Math.max(splitSpan / unsplitSpan, 1), 1.8);
    }
    const size = scale
      * (sample?.metadata.voxelSizeMeters ?? INITIAL_POINT_SIZE)
      * framingCompensation;
    (sample?.points ?? []).forEach((points) => {
      const material = points.material as THREE.PointsMaterial;
      material.size = size * Number(material.userData.pointSizeScale ?? 1);
    });
    if (pointSizeOutput) pointSizeOutput.value = `${scale.toFixed(2)}x`;
  };

  const setScanView = (view: ScanView) => {
    finishStageTransition();
    scanView = view;
    root.dataset.scanView = view;
    scanViewButtons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.scanViewButton === view));
    });
    applyStageVisibility();
  };

  const bindingPositions = (
    binding: AnimationBinding,
    nextAligned: boolean,
    nextSplit: boolean,
  ): Float32Array => {
    const positions = nextAligned ? binding.aligned : binding.input;
    if (!nextSplit || !sample) return positions;
    return splitPositions(
      positions,
      binding.splitRole,
      sample.sourceSplitOffset,
      sample.targetSplitOffset,
    );
  };

  const applyGeometryState = (animate = true) => {
    if (!sample) return;
    sample.cameraBounds = split
      ? sample.splitCameraBounds
      : sample.unsplitCameraBounds;
    if (animate) {
      animationTracks = sample.animationBindings.map((binding) => ({
        target: binding.target,
        from: new Float32Array(binding.target.array as ArrayLike<number>),
        to: bindingPositions(binding, aligned, split),
      }));
      animationStart = performance.now();
      return;
    }
    sample.animationBindings.forEach((binding) => {
      const output = binding.target.array as Float32Array;
      output.set(bindingPositions(binding, aligned, split));
      binding.target.needsUpdate = true;
      binding.geometry.computeBoundingBox();
      binding.geometry.computeBoundingSphere();
    });
    animationStart = 0;
    animationTracks = [];
  };

  const setSplit = (nextSplit: boolean, animate = true) => {
    if (currentStage().id === "alignment") return;
    finishStageTransition();
    splitPreference = nextSplit;
    split = nextSplit;
    root.dataset.split = String(split);
    splitButton.setAttribute("aria-pressed", String(split));
    applyGeometryState(animate);
    updatePointSize();
  };

  const updateStageReadout = () => {
    const stages = activeStages();
    const stage = stages[stageIndex];
    root.dataset.stage = stage.id;
    stageIndexLabel.textContent = `Stage ${stageIndex + 1} of ${stages.length}`;
    stageTitle.textContent = stage.title;
    stageDescription.textContent = stage.description;
    previousButton.disabled = !sample || stageIndex === 0;
    nextButton.disabled = !sample || stageIndex === stages.length - 1;
    splitButton.disabled = !sample || stage.id === "alignment";
    stageProgress.forEach((marker, index) => {
      marker.hidden = index >= stages.length;
      marker.dataset.current = String(index === stageIndex);
      marker.dataset.complete = String(index < stageIndex);
    });
    const progress = stageProgress[0]?.parentElement;
    if (progress) progress.style.gridTemplateColumns = `repeat(${stages.length}, 1fr)`;
  };

  const setStage = (nextIndex: number, animate = true) => {
    finishStageTransition();
    const oldIndex = stageIndex;
    const stages = activeStages();
    const bounded = Math.max(0, Math.min(nextIndex, stages.length - 1));
    const enteringAlignment = bounded === stages.length - 1;
    stageIndex = bounded;
    const nextSplit = enteringAlignment ? false : splitPreference;
    const geometryChanged = enteringAlignment !== aligned || nextSplit !== split;
    aligned = enteringAlignment;
    split = nextSplit;
    root.dataset.split = String(split);
    splitButton.setAttribute("aria-pressed", String(split));
    updateStageReadout();
    if (enteringAlignment) updateSamplePresentation();
    if (geometryChanged || !animate) applyGeometryState(animate);
    updatePointSize();
    if (animate) beginStageTransition(oldIndex);
    else applyStageVisibility();
  };

  const clearIdleStageTimer = () => {
    window.clearTimeout(idleStageTimer);
    idleStageTimer = 0;
  };

  const scheduleIdleStage = () => {
    clearIdleStageTimer();
    if (idleTourStopped || !idleTourVisible || document.hidden || !sample) return;
    root.dataset.idleTour = "running";
    idleStageTimer = window.setTimeout(() => {
      if (idleTourStopped || !idleTourVisible || document.hidden || !sample) return;
      setStage((stageIndex + 1) % activeStages().length);
      scheduleIdleStage();
    }, IDLE_STAGE_DURATION_MS);
  };

  const stopIdleTour = () => {
    if (idleTourStopped) return;
    idleTourStopped = true;
    root.dataset.idleTour = "stopped";
    clearIdleStageTimer();
    idleTourObserver?.disconnect();
  };

  const onIdleTourInteraction = () => stopIdleTour();
  const onVisibilityChange = () => {
    if (document.hidden) clearIdleStageTimer();
    else scheduleIdleStage();
  };
  const idleTourEvents = [
    "pointerdown", "wheel", "keydown", "input", "change", "focusin",
  ] as const;

  const interactiveControls: Array<HTMLButtonElement | HTMLInputElement> = [
    previousButton,
    nextButton,
    ...scanViewButtons,
    ...(resetButton ? [resetButton] : []),
    splitButton,
    ...(pointSizeControl ? [pointSizeControl] : []),
  ];

  const setControlsDisabled = (disabled: boolean) => {
    datasetSelect.disabled = disabled;
    sequenceSelect.disabled = disabled;
    rangeSelect.disabled = disabled || rangeField.hasAttribute("hidden");
    pairSelect.disabled = disabled || !pairSelect.value;
    interactiveControls.forEach((control) => { control.disabled = disabled; });
    if (!disabled) updateStageReadout();
  };

  const loadSample = async (id: string) => {
    const version = ++loadVersion;
    const item = examples.find((entry) => entry.id === id);
    if (!item) throw new Error(`Unknown method example: ${id}`);
    setControlsDisabled(true);
    placeholder.hidden = false;
    alignmentErrors.hidden = true;
    status.hidden = false;
    stats.hidden = true;
    status.textContent = `Loading ${item.dataset} cached pair...`;

    const metadataResponse = await fetch(new URL(`pairs/${item.id}/metadata.json`, manifestUrl));
    if (!metadataResponse.ok) throw new Error(`Failed to load ${item.dataset} metadata`);
    const pair = await metadataResponse.json() as PairMetadata;
    const [sourceBuffer, targetBuffer, pairBuffer] = await Promise.all([
      loadBuffer(new URL(`scans/${pair.source}.bin`, manifestUrl)),
      loadBuffer(new URL(`scans/${pair.target}.bin`, manifestUrl)),
      loadBuffer(new URL(`pairs/${item.id}/matches.bin`, manifestUrl)),
    ]);
    if (version !== loadVersion) return;
    const decoded = decodePair(item, pair, sourceBuffer, targetBuffer, pairBuffer);
    const view = prepareDisplay(decoded.metadata, decoded.arrays);
    const { metadata, arrays } = view;

    finishStageTransition();
    sample?.objects.forEach((object) => {
      scene.remove(object);
      disposeObject(object);
    });
    animationStart = 0;
    animationTracks = [];
    stageTransitionStart = 0;
    opacityTracks = [];
    drawTracks = [];
    transitionOldObjects = [];

    const sourceInput = makeUniformPoints(arrays.source, SOURCE_COLOR, 0.70);
    const targetInput = makeUniformPoints(arrays.target, TARGET_COLOR, 0.70);
    const sourceExtractionColors = makeExtractionColorMap(
      metadata.sourcePlaneLabels, SOURCE_PATCH_COLORS,
    );
    const targetExtractionColors = makeExtractionColorMap(
      metadata.targetPlaneLabels, TARGET_PATCH_COLORS,
    );
    metadata.sourceGroundLabels.forEach((label) => {
      sourceExtractionColors.set(label, new THREE.Color(GROUND_COLOR));
    });
    metadata.targetGroundLabels.forEach((label) => {
      targetExtractionColors.set(label, new THREE.Color(GROUND_COLOR));
    });
    const patchColors = makePlaneColorMaps(
      metadata.sourcePlaneLabels,
      metadata.targetPlaneLabels,
      metadata.planeProposalPairs,
    );
    const sourceFeatures = makeFeaturePoints(
      arrays.source, arrays.sourceLabels,
      sourceExtractionColors, metadata.sourceGroundLabels, SOURCE_COLOR,
    );
    const targetFeatures = makeFeaturePoints(
      arrays.target, arrays.targetLabels,
      targetExtractionColors, metadata.targetGroundLabels, TARGET_COLOR,
    );
    const sourcePatchMatches = makeMatchedPlanePoints(
      arrays.source, arrays.sourceLabels, metadata.planeProposalPairs,
      patchColors.source, new Float32Array(), true, sourceExtractionColors, 1.20,
    );
    const targetPatchMatches = makeMatchedPlanePoints(
      arrays.target, arrays.targetLabels, metadata.planeProposalPairs,
      patchColors.target, new Float32Array(), false, targetExtractionColors, 1.20,
    );
    const sourceNonplanarData = makeNonplanarPoints(
      arrays.source, arrays.sourceLabels, metadata.sourcePlaneLabels, SOURCE_COLOR,
    );
    const targetNonplanarData = makeNonplanarPoints(
      arrays.target, arrays.targetLabels, metadata.targetPlaneLabels, TARGET_COLOR,
    );
    const sourceNonplanar = sourceNonplanarData.points;
    const targetNonplanar = targetNonplanarData.points;
    const sourcePlanarFadedData = makeFadedPlanarPoints(
      arrays.source, arrays.sourceLabels, metadata.sourcePlaneLabels,
      metadata.planeProposalPairs, patchColors.source, sourceExtractionColors, true,
    );
    const targetPlanarFadedData = makeFadedPlanarPoints(
      arrays.target, arrays.targetLabels, metadata.targetPlaneLabels,
      metadata.planeProposalPairs, patchColors.target, targetExtractionColors, false,
    );
    const sourcePlanarFaded = sourcePlanarFadedData.points;
    const targetPlanarFaded = targetPlanarFadedData.points;

    const selectedPointEndpoints = selectedMatchEndpoints(
      arrays.pointMatches, arrays.pointSelection,
    );
    const candidatePointEndpoints = filteredMatchEndpoints(
      arrays.pointMatches, arrays.pointSelection, false,
    );
    const sourcePlaneMatches = makeMatchedPlanePoints(
      arrays.source, arrays.sourceLabels, metadata.matchedPlanePairs,
      patchColors.source, selectedPointEndpoints, true,
    );
    const targetPlaneMatches = makeMatchedPlanePoints(
      arrays.target, arrays.targetLabels, metadata.matchedPlanePairs,
      patchColors.target, selectedPointEndpoints, false,
    );
    const pointCandidateLines = makeMatchLines(
      candidatePointEndpoints,
      new Uint8Array(candidatePointEndpoints.length / 6),
      () => RETAINED_COLOR,
    );
    const pointSelectedLines = makeMatchLines(
      selectedPointEndpoints,
      new Uint8Array(selectedPointEndpoints.length / 6).fill(1),
      () => RETAINED_COLOR,
    );

    const planeAllLines = makeWideMatchLines(
      arrays.planeMatches,
      new Uint8Array(arrays.planeSelection.length).fill(1),
      (index) => patchColors.pairColors[index],
      5,
    );
    const selectedPlaneEndpoints = selectedMatchEndpoints(
      arrays.planeMatches, arrays.planeSelection,
    );
    const selectedPlaneColors = patchColors.pairColors.filter(
      (_, index) => arrays.planeSelection[index],
    );
    const planeSelectedLines = makeWideMatchLines(
      selectedPlaneEndpoints,
      new Uint8Array(selectedPlaneEndpoints.length / 6).fill(1),
      (index) => selectedPlaneColors[index],
      6,
    );
    planeSelectedLines.userData.fadeOnly = true;

    const objects: THREE.Object3D[] = [
      sourceInput, targetInput, sourceFeatures, targetFeatures,
      sourcePatchMatches, targetPatchMatches,
      sourcePlaneMatches, targetPlaneMatches, sourceNonplanar, targetNonplanar,
      sourcePlanarFaded, targetPlanarFaded,
      pointCandidateLines, pointSelectedLines, planeAllLines, planeSelectedLines,


    ];

    const sourceAligned = applyTransform(arrays.source, metadata.transform);
    const robustCameraBounds = metadata.dataset === "ETH" || isIndoorDataset(metadata.dataset);
    const unsplitCameraBounds = robustCameraBounds
      ? robustBoundsFromPositions(arrays.source, arrays.target)
      : boundsFromPositions(arrays.source, arrays.target);
    unsplitCameraBounds.union(robustCameraBounds
      ? robustBoundsFromPositions(sourceAligned, arrays.target)
      : boundsFromPositions(sourceAligned, arrays.target));
    const layout = splitLayout(
      arrays.source,
      arrays.target,
      metadata.voxelSizeMeters,
    );
    objects.forEach((object) => scene.add(object));

    const sourceNonplanarAligned = applyTransform(
      sourceNonplanarData.positions, metadata.transform,
    );
    const sourcePlanarFadedAligned = applyTransform(
      sourcePlanarFadedData.positions, metadata.transform,
    );
    const pointCandidateAligned = transformSourceEndpoints(
      candidatePointEndpoints, metadata.transform,
    );
    const pointSelectedAligned = transformSourceEndpoints(selectedPointEndpoints, metadata.transform);
    const planeAllAligned = transformSourceEndpoints(arrays.planeMatches, metadata.transform);
    const planeSelectedAligned = transformSourceEndpoints(selectedPlaneEndpoints, metadata.transform);
    const sourceObjects = [
      sourceInput, sourceFeatures, sourcePatchMatches, sourcePlaneMatches,
    ];
    const targetObjects = [
      targetInput, targetFeatures, targetPatchMatches, targetPlaneMatches,
    ];
    const animationBindings: AnimationBinding[] = [
      ...sourceObjects.map((object) => ({
        geometry: object.geometry,
        target: object.geometry.getAttribute("position") as THREE.BufferAttribute,
        input: arrays.source,
        aligned: sourceAligned,
        splitRole: "source" as const,
      })),
      ...targetObjects.map((object) => ({
        geometry: object.geometry,
        target: object.geometry.getAttribute("position") as THREE.BufferAttribute,
        input: arrays.target,
        aligned: arrays.target,
        splitRole: "target" as const,
      })),
      {
        geometry: sourceNonplanar.geometry,
        target: sourceNonplanar.geometry.getAttribute("position") as THREE.BufferAttribute,
        input: sourceNonplanarData.positions,
        aligned: sourceNonplanarAligned,
        splitRole: "source",
      },
      {
        geometry: sourcePlanarFaded.geometry,
        target: sourcePlanarFaded.geometry.getAttribute("position") as THREE.BufferAttribute,
        input: sourcePlanarFadedData.positions,
        aligned: sourcePlanarFadedAligned,
        splitRole: "source",
      },
      {
        geometry: targetNonplanar.geometry,
        target: targetNonplanar.geometry.getAttribute("position") as THREE.BufferAttribute,
        input: targetNonplanarData.positions,
        aligned: targetNonplanarData.positions,
        splitRole: "target",
      },
      {
        geometry: targetPlanarFaded.geometry,
        target: targetPlanarFaded.geometry.getAttribute("position") as THREE.BufferAttribute,
        input: targetPlanarFadedData.positions,
        aligned: targetPlanarFadedData.positions,
        splitRole: "target",
      },
      {
        geometry: pointCandidateLines.geometry,
        target: pointCandidateLines.geometry.getAttribute("position") as THREE.BufferAttribute,
        input: candidatePointEndpoints,
        aligned: pointCandidateAligned,
        splitRole: "pair",
      },
      {
        geometry: pointSelectedLines.geometry,
        target: pointSelectedLines.geometry.getAttribute("position") as THREE.BufferAttribute,
        input: selectedPointEndpoints,
        aligned: pointSelectedAligned,
        splitRole: "pair",
      },
      {
        geometry: planeAllLines.geometry,
        target: wideLineAnimationTarget(planeAllLines.geometry),
        input: arrays.planeMatches,
        aligned: planeAllAligned,
        splitRole: "pair",
      },
      {
        geometry: planeSelectedLines.geometry,
        target: wideLineAnimationTarget(planeSelectedLines.geometry),
        input: selectedPlaneEndpoints,
        aligned: planeSelectedAligned,
        splitRole: "pair",
      },


    ];

    sample = {
      metadata,
      sourceInput,
      targetInput,
      sourceFeatures,
      targetFeatures,
      sourcePatchMatches,
      targetPatchMatches,
      sourcePlaneMatches,
      targetPlaneMatches,
      sourceNonplanar,
      targetNonplanar,
      sourcePlanarFaded,
      targetPlanarFaded,
      pointCandidateLines,
      pointSelectedLines,
      planeAllLines,
      planeSelectedLines,
      objects,
      points: [
        sourceInput, targetInput, sourceFeatures, targetFeatures,
        sourcePatchMatches, targetPatchMatches,
        sourcePlaneMatches, targetPlaneMatches, sourceNonplanar, targetNonplanar,
        sourcePlanarFaded, targetPlanarFaded,


      ],
      animationBindings,
      cameraBounds: split ? layout.bounds : unsplitCameraBounds,
      unsplitCameraBounds,
      splitCameraBounds: layout.bounds,
      gridFloorY: view.gridFloorY,
      sourceSplitOffset: layout.sourceOffset,
      targetSplitOffset: layout.targetOffset,
    };

    const sequence = sequenceName(metadata.sequence);
    title.textContent = `${metadata.dataset}/${sequence} scans ${metadata.sourceScan} - ${metadata.targetScan}`;
    updateSamplePresentation();
    status.hidden = metadata.completed;
    status.textContent = metadata.completed ? "" : `Registration failed: ${metadata.error || "no estimate"}`;
    stats.hidden = false;
    synchronizePairSelectors(item);
    setStage(stageIndex, false);
    if (pointSizeControl) {
      pointSizeControl.value = metadata.dataset === "ETH" ? "1.5" : "1";
    }
    updatePointSize();
    resetCamera();
    placeholder.hidden = true;
    setControlsDisabled(false);
  };

  const render = () => {
    animationFrame = requestAnimationFrame(render);
    if (animationStart && animationTracks.length) {
      const alpha = Math.min((performance.now() - animationStart) / 760, 1);
      const eased = 1 - Math.pow(1 - alpha, 3);
      animationTracks.forEach((track) => {
        const output = track.target.array as Float32Array;
        for (let index = 0; index < output.length; index += 1) {
          output[index] = track.from[index] + (track.to[index] - track.from[index]) * eased;
        }
        track.target.needsUpdate = true;
      });
      if (alpha === 1) {
        sample?.animationBindings.forEach((binding) => {
          binding.geometry.computeBoundingBox();
          binding.geometry.computeBoundingSphere();
        });
        animationStart = 0;
        animationTracks = [];
      }
    }
    if (stageTransitionStart && (opacityTracks.length || drawTracks.length)) {
      const alpha = Math.min((performance.now() - stageTransitionStart) / 620, 1);
      const eased = alpha * alpha * (3 - 2 * alpha);
      opacityTracks.forEach((track) => {
        setObjectOpacity(track.object, track.from + (track.to - track.from) * eased);
      });
      const drawAlpha = Math.max(0, Math.min((alpha - 0.18) / 0.82, 1));
      const drawEased = 1 - Math.pow(1 - drawAlpha, 3);
      drawTracks.forEach((track) => {
        const output = track.target.array as Float32Array;
        for (let index = 0; index < output.length; index += 1) {
          output[index] = track.from[index]
            + (track.to[index] - track.from[index]) * drawEased;
        }
        track.target.needsUpdate = true;
      });
      if (alpha === 1) finishStageTransition();
    }
    controls.update();
    renderer.render(scene, camera);
  };
  render();

  previousButton.addEventListener("click", () => setStage(stageIndex - 1));
  nextButton.addEventListener("click", () => setStage(stageIndex + 1));
  scanViewButtons.forEach((button) => {
    button.addEventListener("click", () => {
      setScanView(button.dataset.scanViewButton as ScanView);
    });
  });
  const reportLoadError = (error: unknown) => {
      status.textContent = "The selected cached example could not be loaded.";
      status.hidden = false;
      stats.hidden = true;
      placeholder.hidden = true;
      setControlsDisabled(false);
      console.error(error);
  };
  datasetSelect.addEventListener("change", () => {
    populateSequences(datasetSelect.value);
    const distanceRange = populateRanges(datasetSelect.value, sequenceSelect.value);
    populatePairs(datasetSelect.value, sequenceSelect.value, distanceRange);
    if (pairSelect.value) void loadSample(pairSelect.value).catch(reportLoadError);
  });
  sequenceSelect.addEventListener("change", () => {
    const distanceRange = populateRanges(datasetSelect.value, sequenceSelect.value);
    populatePairs(datasetSelect.value, sequenceSelect.value, distanceRange);
    if (pairSelect.value) void loadSample(pairSelect.value).catch(reportLoadError);
  });
  rangeSelect.addEventListener("change", () => {
    populatePairs(datasetSelect.value, sequenceSelect.value, rangeSelect.value);
    if (pairSelect.value) void loadSample(pairSelect.value).catch(reportLoadError);
  });
  pairSelect.addEventListener("change", () => {
    if (pairSelect.value) void loadSample(pairSelect.value).catch(reportLoadError);
  });
  resetButton?.addEventListener("click", resetCamera);
  splitButton.addEventListener("click", () => setSplit(!split));
  pointSizeControl?.addEventListener("input", updatePointSize);
  const initialId = examples.find((item) => item.dataset === "KITTI-LC" && item.sequence === "02" && item.sourceScan === 908 && item.targetScan === 4193)?.id
    ?? examples[0].id;

  idleTourEvents.forEach((eventName) => {
    root.addEventListener(eventName, onIdleTourInteraction, { capture: true });
  });
  document.addEventListener("visibilitychange", onVisibilityChange);
  if (!idleTourStopped) {
    idleTourObserver = new IntersectionObserver((entries) => {
      idleTourVisible = entries.some(
        (entry) => entry.isIntersecting && entry.intersectionRatio >= 0.25,
      );
      scheduleIdleStage();
    }, { threshold: [0, 0.25] });
    idleTourObserver.observe(root);
  } else {
    root.dataset.idleTour = "stopped";
  }
  await loadSample(initialId);
  scheduleIdleStage();

  return {
    dispose() {
      ++loadVersion;
      clearIdleStageTimer();
      idleTourObserver?.disconnect();
      document.removeEventListener("visibilitychange", onVisibilityChange);
      idleTourEvents.forEach((eventName) => {
        root.removeEventListener(eventName, onIdleTourInteraction, { capture: true });
      });
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      controls.dispose();
      sample?.objects.forEach((object) => disposeObject(object));
      grid.geometry.dispose();
      const gridMaterial = grid.material;
      if (Array.isArray(gridMaterial)) gridMaterial.forEach((material) => material.dispose());
      else gridMaterial.dispose();
      renderer.dispose();
    },
  };
}
