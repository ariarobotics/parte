#pragma once

#include "common/types.h"

#include <Eigen/Dense>
#include <vector>
#include <span>


namespace parte::registration
{

using FPFH = Eigen::Matrix<Scalar, 33, 1>;

std::vector<FPFH> compute_fpfh(
  std::span<const Point> points,
  std::span<const Normal> normals,
  std::span<const Neighbors> neighborhoods,
  std::span<const Index> indices
);


namespace detail
{

std::vector<FPFH> compute_spfh(
  std::span<const Point> points,
  std::span<const Normal> normals,
  std::span<const Neighbors> neighborhoods,
  std::span<const Index> indices
);

ScalarVector<3> compute_pair_features(
  const Point &p1, const Normal &n1,
  const Point &p2, const Normal &n2
);

}

}
