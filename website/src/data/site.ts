const ASSET_BASE_URL = "https://pub-c634e388f4154ed6a42ed6fb800cf184.r2.dev";

export const paper = {
  shortName: "PARTE",
  title:
    "PARTE: Plane-Assisted Robust Transformation Estimation for Point Cloud Registration",
  summary:
    "A global registration method that treats planar structure as complementary evidence, combining point and plane correspondences in confidence-weighted graph-based outlier rejection.",
  links: {
    paper: "https://arxiv.org/abs/2609.25375",
    code: "https://github.com/ariarobotics/parte",
    datasets:
      `${ASSET_BASE_URL}/datasets/parte_processed_datasets_2026-08-21_v3.zip`,
    datasetsChecksum:
      "2b0c106b5ce9e5cc6e3ec920f0baccd04fe73ccdff967a461442e1dfac076fc2",
    video:
      `${ASSET_BASE_URL}/media/parte_video.mp4`,
  },
} as const;

export const benchmarks = [
  {
    dataset: "3DMatch", sensor: "RGB-D", scenes: "8", pairs: "1,623",
    voxel: "5 cm", points: "4,838", planes: "4.6",
    criterion: { rotationDegrees: 15, translation: 30, translationUnit: "cm" },
  },
  {
    dataset: "3DLoMatch", sensor: "RGB-D", scenes: "8", pairs: "1,781",
    voxel: "5 cm", points: "4,821", planes: "4.6",
    criterion: { rotationDegrees: 15, translation: 30, translationUnit: "cm" },
  },
  {
    dataset: "KITTI-10m", sensor: "LiDAR", scenes: "3", pairs: "556",
    voxel: "30 cm", points: "18,238", planes: "5.1",
    criterion: { rotationDegrees: 5, translation: 0.6, translationUnit: "m" },
  },
  {
    dataset: "KITTI-LC", sensor: "LiDAR", scenes: "5", pairs: "3,325",
    voxel: "30 cm", points: "18,354", planes: "7.9",
    criterion: { rotationDegrees: 5, translation: 2, translationUnit: "m" },
  },
  {
    dataset: "RESSO P2F", sensor: "TLS + RGB-D", scenes: "7", pairs: "99",
    voxel: "5 cm", points: "10,143", planes: "7.4",
    criterion: { rotationDegrees: 15, translation: 30, translationUnit: "cm" },
  },
  {
    dataset: "ETH", sensor: "LiDAR", scenes: "4", pairs: "713",
    voxel: "10 cm", points: "30,234", planes: "2.0",
    criterion: { rotationDegrees: 15, translation: 30, translationUnit: "cm" },
  },
] as const;

export const resultHighlights = [
  {
    context: "Indoor RGB-D",
    value: "89.6 / 55.8%",
    label: "3DMatch / 3DLoMatch",
    note: "Second-highest success rate on both benchmarks.",
  },
  {
    context: "Outdoor odometry",
    value: "99.8%",
    label: "KITTI-10m",
    note: "Tied for the highest overall success rate.",
  },
  {
    context: "Partial-to-full",
    value: "59.6%",
    label: "RESSO P2F",
    note: "Highest overall success rate.",
  },
  {
    context: "Additional LiDAR",
    value: "99.6%",
    label: "ETH",
    note: "Highest overall success rate.",
  },
] as const;

export const comparisonDatasets = [
  "3DMatch",
  "3DLoMatch",
  "KITTI-10m",
  "KITTI-LC",
  "RESSO P2F",
  "ETH",
] as const;

export const registrationComparisons = [
  {
    method: "RANSAC", iterationsPower: 5, family: "Baseline",
    results: [
      { success: "85.6", runtime: "70.8" },
      { success: "45.8", runtime: "70.3" },
      { success: "91.5", runtime: "416.7" },
      { success: "42.6", runtime: "376.9" },
      { success: "34.3", runtime: "358.1" },
      { success: "63.3", runtime: "892.9" },
    ],
  },
  {
    method: "KISS-Matcher", family: "Native pipeline",
    results: [
      { success: "75.5", runtime: "110.8" },
      { success: "26.8", runtime: "26.4" },
      { success: "97.1", runtime: "98.0" },
      { success: "52.1", runtime: "101.1" },
      { success: "42.4", runtime: "213.5" },
      { success: "89.8", runtime: "226.7" },
    ],
  },
  {
    method: "CLIPPER+", family: "Graph-based",
    results: [
      { success: "85.9", runtime: "117.5" },
      { success: "44.7", runtime: "130.1" },
      { success: "98.7", runtime: "974.1" },
      { success: "65.4", runtime: "2574.9" },
      { success: "58.6", runtime: "707.9", rank: "second" },
      { success: "94.5", runtime: "4266.4", rank: "second" },
    ],
  },
  {
    method: "TEASER++", family: "Graph-based",
    results: [
      { success: "86.6", runtime: "77.7" },
      { success: "48.0", runtime: "64.5" },
      { success: "98.9", runtime: "519.8" },
      { success: "70.4", runtime: "494.7" },
      { success: "56.6", runtime: "385.2" },
      { success: "94.0", runtime: "1078.3" },
    ],
  },
  {
    method: "MAC", family: "Graph-based",
    results: [
      { success: "88.0", runtime: "112.4" },
      { success: "52.4", runtime: "85.3" },
      { success: "98.4", runtime: "1908.7" },
      { success: "66.2", runtime: "1721.4" },
      { success: "57.6", runtime: "549.9" },
      { success: "87.2", runtime: "7521.2" },
    ],
  },
  {
    method: "VBReg", family: "Learning-based",
    results: [
      { success: "89.4", runtime: "102.8" },
      { success: "53.5", runtime: "94.0" },
      { success: "97.5", runtime: "496.4" },
      { success: "62.9", runtime: "465.1" },
      { success: "57.6", runtime: "443.7" },
      { success: "90.5", runtime: "1015.4" },
    ],
  },
  {
    method: "PREDATOR", family: "Learning-based",
    results: [
      { success: "92.1", runtime: "95.7", rank: "best" },
      { success: "61.0", runtime: "82.0", rank: "best" },
      { success: "99.8", runtime: "553.3", rank: "best" },
      { success: "28.8", runtime: "366.3" },
      { success: "10.1", runtime: "392.3" },
      { success: "64.0", runtime: "1657.8" },
    ],
  },
  {
    method: "G3Reg", family: "Plane-based",
    results: [
      { success: "4.8", runtime: "35.1" },
      { success: "0.0", runtime: "42.9" },
      { success: "99.1", runtime: "88.0", rank: "second" },
      { success: "86.2", runtime: "96.5", rank: "best" },
      { success: null, runtime: null },
      { success: "32.7", runtime: "161.9" },
    ],
  },
  {
    method: "PARTE", family: "Ours", ours: true,
    results: [
      { success: "89.6", runtime: "59.4", rank: "second" },
      { success: "55.8", runtime: "56.8", rank: "second" },
      { success: "99.8", runtime: "178.0", rank: "best" },
      { success: "85.8", runtime: "127.1", rank: "second" },
      { success: "59.6", runtime: "315.8", rank: "best" },
      { success: "99.6", runtime: "437.6", rank: "best" },
    ],
  },
] as const;
