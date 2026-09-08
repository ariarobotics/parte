import sys
from types import SimpleNamespace

import numpy as np
import open3d as o3d

sys.path.insert(0, "build/python")
import parte


VOXEL_SIZE = 0.05

def prepare(path):
    print(f"Preparing {path}")
    cloud = o3d.io.read_point_cloud(path)
    points = np.asarray(cloud.points, dtype=np.float32)

    normal_neighbors = parte.compute_neighborhoods(points, 2 * VOXEL_SIZE, 30)
    fpfh_neighbors = parte.compute_neighborhoods(points, 5 * VOXEL_SIZE, 100)
    normals = parte.compute_normals(points, normal_neighbors)
    supports = parte.segment_planes(
        points, normals, normal_neighbors,
        VOXEL_SIZE, np.sin(np.deg2rad(10)), 100,
    )
    planes = np.asarray([parte.compute_plane(points, support) for support in supports], dtype=np.float32).reshape(-1, 6)

    pch = [
        parte.compute_pch(
            points, normals, plane[3:], plane[:3],
            20 * VOXEL_SIZE, 20 * VOXEL_SIZE,
        )
        for plane in planes
    ]
    groups = parte.group_planes(
        planes, 2 * VOXEL_SIZE, np.cos(np.deg2rad(5))
    )

    plane_mask = np.zeros(len(points), dtype=bool)
    for support in supports:
        plane_mask[support] = True
    point_indices = np.flatnonzero(~plane_mask)
    fpfh = parte.compute_fpfh(
        points, normals, fpfh_neighbors, point_indices
    )

    print(f"  {len(points)} points, {len(planes)} planes, {len(point_indices)} FPFH features")
    return SimpleNamespace(
        points=points,
        planes=planes,
        pch=pch,
        groups=groups,
        point_indices=point_indices,
        fpfh=fpfh,
    )


def matched(source, target, matches):
    if not matches:
        return source[:0], target[:0]
    indices = np.asarray(matches)
    return source[indices[:, 0]], target[indices[:, 1]]


def register(source, target):
    print("Matching FPFH features and plane histograms")
    point_matches = parte.mutual_correspondences(source.fpfh, target.fpfh)
    plane_matches, plane_confidences = parte.pch_matching(
        source.pch, target.pch, source.groups, target.groups
    )

    source_points, target_points = matched(
        source.points[source.point_indices],
        target.points[target.point_indices],
        point_matches,
    )
    source_planes, target_planes = matched(
        source.planes, target.planes, plane_matches
    )

    print("Constructing the consistency graph and finding its maximum clique")
    edges = parte.consistent_correspondences(
        source_points, target_points,
        source_planes, target_planes,
        VOXEL_SIZE, np.deg2rad(5),
    )
    weights = [1 + int(10 * np.clip(value, 0, 1)) for value in plane_confidences]
    weights += [1] * len(point_matches)
    clique = parte.maximum_weight_clique(len(weights), edges, weights)
    plane_count = len(plane_matches)
    selected_planes = [node for node in clique if node < plane_count]
    selected_points = [node - plane_count for node in clique if node >= plane_count]
    transformation = parte.compute_transformation(
        source_points[selected_points], target_points[selected_points],
        source_planes[selected_planes], target_planes[selected_planes],
        [weights[node] for node in selected_planes],
    )

    print(f"  {len(point_matches)} point and {len(plane_matches)} plane matches")
    print(f"  {len(edges)} consistency edges")
    print(f"  selected {len(selected_points)} point and {len(selected_planes)} plane matches")
    return transformation


def make_cloud(points, color):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.paint_uniform_color(color)
    return cloud


def main():
    source = prepare("demo/source.ply")
    target = prepare("demo/target.ply")
    transformation = register(source, target)

    print("Source-to-target transformation:")
    print(transformation)

    print("Showing input clouds; close the window to show the final alignment")
    o3d.visualization.draw_geometries([
        make_cloud(source.points, [0.92, 0.48, 0.12]),
        make_cloud(target.points, [0.10, 0.47, 0.78]),
    ], window_name="PARTE input")

    aligned = source.points @ transformation[:3, :3].T + transformation[:3, 3]
    o3d.visualization.draw_geometries([
        make_cloud(aligned, [0.92, 0.48, 0.12]),
        make_cloud(target.points, [0.10, 0.47, 0.78]),
    ], window_name="PARTE alignment")


if __name__ == "__main__":
    main()
