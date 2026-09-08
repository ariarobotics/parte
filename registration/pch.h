#pragma once

#include "common/types.h"

#include <Eigen/Dense>
#include <span>

namespace parte::registration
{

constexpr Index PCH_D_BINS = 16;
constexpr Index PCH_A_BINS = 12;

using PCH = Eigen::Matrix<float, PCH_D_BINS, PCH_A_BINS>;

std::pair<PCH, PCH> compute_pch(
  std::span<const Point> points,
  std::span<const Normal> normals,
  Normal p_normal,
  Point p_point,
  Scalar d_range,
  Scalar d_radius
);


}
