#pragma once

#include "common/types.h"

#include <span>

namespace parte::segmentation
{

std::vector<Indices> segment_planes(
  std::span<const Point> points,
  std::span<const Normal> normals,
  std::span<const Neighbors> neighborhoods,
  Scalar max_thickness, Scalar max_dispersion,
  size_t min_support
);


namespace detail
{

struct PlaneModel
{
  Index n_points = 0;
  Point centroid = Point::Zero();
  ScalarMatrix<3, 3> C = ScalarMatrix<3, 3>::Zero();
  ScalarMatrix<3, 3> Q = ScalarMatrix<3, 3>::Zero();

  inline static PlaneModel merge(const PlaneModel &a, const PlaneModel &b)
  {
    PlaneModel result;
    result.n_points = a.n_points + b.n_points;
    Scalar n_a = a.n_points, n_b = b.n_points;
    auto w_avg = [n_a, n_b](const auto &x, const auto &y) {
      return (x * n_a + y * n_b) / (n_a + n_b);
    };

    result.centroid = w_avg(a.centroid, b.centroid);
    result.Q = w_avg(a.Q, b.Q);

    auto delta = a.centroid - b.centroid;
    result.C = w_avg(a.C, b.C) + (n_a * n_b) / ((n_a + n_b) * (n_a + n_b)) * (delta * delta.transpose());
    return result;
  }
};

bool feasible_normal(
  Eigen::Ref<const ScalarMatrix<3, 3>> A,
  Eigen::Ref<const ScalarMatrix<3, 3>> B
);

}

}
