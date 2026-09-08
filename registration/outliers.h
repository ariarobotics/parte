#pragma once

#include <cstddef>
#include <utility>
#include <vector>
#include <span>
#include <Eigen/Dense>

#include "common/types.h"
#include "pmc_types.h"


namespace parte::registration
{

using wpmc_edge = std::pair<node_t, node_t>;

std::vector<wpmc_edge> consistent_correspondences(
  std::span<const Point> source_points, std::span<const Point> target_points,
  std::span<const Plane> source_planes, std::span<const Plane> target_planes,
  double max_distance, double max_angle
);

std::vector<Index> maximum_weight_clique(
  std::size_t node_count,
  std::span<const wpmc_edge> edges,
  std::span<const node_t> weights
);

}
