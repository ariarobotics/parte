#include "registration/fpfh.h"
#include "common/types.h"


#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <numbers>


namespace parte::registration
{

std::vector<FPFH> compute_fpfh(
  std::span<const Point> points,
  std::span<const Normal> normals,
  std::span<const Neighbors> neighborhoods,
  std::span<const Index> indices
)
{
  size_t n_indices = indices.size();
  std::vector<Index> indices_map(points.size(), -1);
  for(size_t i = 0; i < n_indices; ++i) {
    indices_map[indices[i]] = i;
  }

  std::vector<Index> extended_indices(indices.begin(), indices.end());
  for(auto idx : indices) {
    if(extended_indices.size() == points.size()) {
      break;
    }

    for(auto n : neighborhoods[idx]) {
      if(indices_map[n] < 0) {
        extended_indices.push_back(n);
        indices_map[n] = n_indices++;
      }
    }
  }

  n_indices = indices.size();
  auto spfh = detail::compute_spfh(points, normals, neighborhoods, extended_indices);
  std::vector<FPFH> feature(n_indices, FPFH::Zero());

  #pragma omp parallel for schedule(dynamic, 64)
  for(size_t i = 0; i < n_indices; ++i) {
    const size_t idx = indices[i];

    auto &neighbors = neighborhoods[idx];
    if(neighbors.size() <= 1) {
      feature[i] = spfh[i];
      continue;
    }

    auto &output = feature[i];
    for(size_t k = 1; k < neighbors.size(); ++k) {
      auto neighbor = indices_map[neighbors[k]];
      Scalar weight = 1.0f / (points[idx] - points[neighbors[k]]).squaredNorm();
      output += spfh[neighbor] * weight;
    }

    for(int histogram = 0; histogram < 3; ++histogram) {
      auto block = output.segment(histogram * 11, 11);
      block /= block.sum();
    }

    output += spfh[i];
  }

  return feature;
}


namespace detail
{

std::vector<FPFH> compute_spfh(
  std::span<const Point> points,
  std::span<const Normal> normals,
  std::span<const Neighbors> neighborhoods,
  std::span<const Index> indices
)
{
  size_t n_indices = indices.size();
  std::vector<FPFH> feature(n_indices, FPFH::Zero());

  #pragma omp parallel for schedule(dynamic, 64)
  for(size_t i = 0; i < n_indices; ++i) {
    auto idx = indices[i];
    auto &neighbors = neighborhoods[idx];

    float hist_incr = 1.0 / (neighbors.size() - 1);
    for(size_t k = 1; k < neighbors.size(); ++k) {
      size_t neighbor_idx = neighbors[k];
      auto pf = compute_pair_features(
        points[idx], normals[idx],
        points[neighbor_idx], normals[neighbor_idx]
      );

      int h_index = int(11.0 * (pf(0) / std::numbers::pi + 1.0) * 0.5);
      h_index = std::clamp(h_index, 0, 10);
      feature[i](h_index) += hist_incr;

      h_index = int(11.0 * (pf(1) + 1.0) * 0.5);
      h_index = std::clamp(h_index, 0, 10);
      feature[i](h_index + 11) += hist_incr;

      h_index = int(11.0 * (pf(2) + 1.0) * 0.5);
      h_index = std::clamp(h_index, 0, 10);
      feature[i](h_index + 22) += hist_incr;
    }
  }

  return feature;
}


ScalarVector<3> compute_pair_features(
  const Point &p1, const Normal &n1,
  const Point &p2, const Normal &n2
)
{
  auto u = p2 - p1;
  Scalar dist = n1.dot(u);
  Scalar angle = n1.dot(n2);

  auto v_raw = u.cross(n1);
  Scalar v_norm = v_raw.norm();

  ScalarVector<3> result;
  result(0) = std::atan2(
    std::fma(-dist, angle, u.dot(n2)),
    angle * v_norm
  );

  result(1) = v_raw.dot(n2) / v_norm;
  result(2) = dist / u.norm();
  return result;
}

}

}
