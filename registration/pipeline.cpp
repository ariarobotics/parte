#include "registration/pipeline.h"

#include "common/logger.h"
#include "common/preprocessing.h"

#include "registration/matching.h"
#include "registration/outliers.h"
#include "registration/transformation.h"
#include "segmentation/segmentation.h"
#include "segmentation/tgs.h"

#include <cmath>
#include <numbers>
#include <tuple>


namespace parte
{

Scalar radians(Scalar degrees)
{
  return degrees * std::numbers::pi_v<Scalar> / 180.0f;
}

ProcessedCloud process_cloud(std::vector<Eigen::Vector3d> points, const Parameters &p)
{
  std::vector<Point> casted_points;
  casted_points.reserve(points.size());
  for(const auto &point : points) {
    casted_points.push_back(point.cast<Scalar>());
  }
  ProcessedCloud result;
  result.points = std::move(casted_points);

  Indices ground_support;
  std::vector<bool> is_ground(result.points.size(), false);
  if(p.ground_segmentation) {
    is_ground = logger.time("Ground segmentation", [&] {
      return segmentation::segment_ground(result.points);
    });

    for(Index point = 0; point < result.points.size(); ++point) {
      if(is_ground[point]) {
        ground_support.push_back(point);
      }
    }
  }

  logger.start("Computing neighborhoods");
  auto fpfh_neighborhoods = compute_neighborhoods(
    result.points, p.fpfh_radius * p.voxel_size, p.fpfh_k
  );
  auto neighborhoods = filter_neighborhoods(
    result.points, fpfh_neighborhoods, p.normal_radius * p.voxel_size, p.normal_k
  );
  logger.stop("Computing neighborhoods");


  result.normals = logger.time("Computing normals", [&] {
    return compute_normals(result.points, neighborhoods);
  });

  logger.start("Plane segmentation");
  result.plane_supports = segmentation::segment_planes(
    result.points, result.normals, neighborhoods,
    p.segmentation_thickness * p.voxel_size, std::sin(radians(p.segmentation_dispersion)),
    p.segmentation_min_support, is_ground
  );
  logger.stop("Plane segmentation");

  if(ground_support.size() >= p.segmentation_min_support) {
    result.plane_supports.push_back(std::move(ground_support));
  }

  result.planes.reserve(result.plane_supports.size());
  std::vector<bool> is_plane_point(result.points.size());
  for(const auto &support : result.plane_supports) {
    result.planes.push_back(compute_plane(result.points, support));
    for(Index point : support) {
      is_plane_point[point] = true;
    }
  }

  logger.start("Computing PCH");
  result.pch.resize(result.planes.size());
  #pragma omp parallel for schedule(static)
  for(Index plane = 0; plane < result.planes.size(); ++plane) {
    const auto &[center, normal] = result.planes[plane];
    result.pch[plane] = registration::compute_pch(
      result.points, result.normals, normal, center,
      p.pch_range * p.voxel_size, p.pch_2l_radius * p.voxel_size
    );
  }
  logger.stop();

  result.non_planar.reserve(result.points.size());
  for(Index point = 0; point < result.points.size(); ++point) {
    if(!is_plane_point[point]) {
      result.non_planar.push_back(point);
    }
  }

  logger.start("Computing FPFH");
  result.fpfh = registration::compute_fpfh(
    result.points, result.normals, fpfh_neighborhoods, result.non_planar
  );
  logger.stop();
  return result;
}


RegistrationResult register_clouds(
  const ProcessedCloud &source,
  const ProcessedCloud &target,
  const Parameters &p
)
{
  RegistrationResult result;
  result.point_matches = logger.time("Point matching", [&] {
    return registration::mutual_correspondences<registration::FPFH>(source.fpfh, target.fpfh);
  });

  logger.start("Plane matching");
  std::vector<Scalar> plane_confidences;
  if(!source.pch.empty() && !target.pch.empty()) {
    auto source_groups = registration::group_planes(
      source.planes, p.pch_grouping_distance * p.voxel_size, std::cos(radians(p.pch_grouping_angle))
    );
    auto target_groups = registration::group_planes(
      target.planes, p.pch_grouping_distance * p.voxel_size, std::cos(radians(p.pch_grouping_angle))
    );
    std::tie(result.plane_matches, plane_confidences) = registration::pch_matching(
      source.pch, target.pch, std::move(source_groups), std::move(target_groups)
    );
  }
  logger.stop("Plane matching");

  logger.start("Graph construction");
  std::vector<Point> source_points, target_points;
  for(const auto &[s, t] : result.point_matches) {
    source_points.push_back(source.points[source.non_planar[s]]);
    target_points.push_back(target.points[target.non_planar[t]]);
  }

  std::vector<Plane> source_planes, target_planes;
  for(const auto &[s, t] : result.plane_matches) {
    source_planes.push_back(source.planes[s]);
    target_planes.push_back(target.planes[t]);
  }
  const std::size_t node_count = result.plane_matches.size() + result.point_matches.size();
  auto edges = registration::consistent_correspondences(
    source_points, target_points, source_planes, target_planes,
    p.outlier_rejection_distance * p.voxel_size, radians(p.outlier_rejection_angle)
  );
  std::vector<Index> weights(node_count, 1);
  for(std::size_t plane = 0; plane < plane_confidences.size(); ++plane) {
    weights[plane] = 1 + std::floor(10 * plane_confidences[plane]);
  }
  logger.stop("Graph construction");

  auto clique = logger.time("CLIPPER", [&] {
    return registration::maximum_weight_clique(node_count, edges, weights);
  });

  const Index plane_count = result.plane_matches.size();
  for(Index node : clique) {
    if(node < plane_count) {
      result.selected_planes.push_back(node);
    } else {
      result.selected_points.push_back(node - plane_count);
    }
  }
  auto [s_src_pts, s_tgt_pts] = select(result.selected_points, source_points, target_points);
  auto [s_src_planes, s_tgt_planes, clique_weights] = select(result.selected_planes, source_planes, target_planes, weights);
  std::vector<Scalar> selected_plane_weights(clique_weights.begin(), clique_weights.end());
  result.transformation = registration::compute_transformation(
    s_src_pts, s_tgt_pts,
    s_src_planes, s_tgt_planes,
    selected_plane_weights
  );
  return result;
}

}
