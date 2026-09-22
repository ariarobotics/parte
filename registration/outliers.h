#pragma once

#include <cstddef>
#include <vector>
#include <span>
#include <Eigen/Dense>

#include "common/types.h"


namespace parte::registration
{

std::vector<Correspondence> consistent_correspondences(
  std::span<const Point> source_points, std::span<const Point> target_points,
  std::span<const Plane> source_planes, std::span<const Plane> target_planes,
  double max_distance, double max_angle
);

std::vector<Index> maximum_weight_clique(
  std::size_t node_count,
  std::span<const Correspondence> edges,
  std::span<const Index> weights
);

}
