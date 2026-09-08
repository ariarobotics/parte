#pragma once

#include <span>
#include <Eigen/Dense>

#include "common/types.h"

namespace parte::registration
{

ScalarMatrix<4, 4> compute_transformation(
  std::span<const Point> source_points, std::span<const Point> target_points,
  std::span<const Plane> source_planes, std::span<const Plane> target_planes,
  std::span<const Scalar> plane_weights
);

}
