#include "registration/outliers.h"

#include "clipperplus/clipperplus_clique.h"

#include <algorithm>
#include <cmath>


namespace parte::registration
{

std::vector<Correspondence> consistent_correspondences(
  std::span<const Point> source_points, std::span<const Point> target_points,
  std::span<const Plane> source_planes, std::span<const Plane> target_planes,
  double max_distance, double max_angle
)
{
  std::vector<Correspondence> edges;
  Index n_points = source_points.size(), n_planes = source_planes.size();
  double sin_angle = std::sin(max_angle);
  double one_minus_cos_angle = 1.0 - std::cos(max_angle);

  for(Index i = 0; i < n_planes; ++i) {
    auto [p_center_i, p_normal_i] = source_planes[i];
    auto [q_center_i, q_normal_i] = target_planes[i];

    for(Index j = i + 1; j < n_planes; ++j) {
      auto p_angle = std::acos(std::clamp(
        p_normal_i.dot(source_planes[j].second), -1.0f, 1.0f
      ));
      auto q_angle = std::acos(std::clamp(
        q_normal_i.dot(target_planes[j].second), -1.0f, 1.0f
      ));

      if(std::abs(p_angle - q_angle) < 2.0 * max_angle) {
        edges.push_back({i, j});
      }
    }

    for(Index j = 0; j < n_points; ++j) {
      auto p_delta = source_points[j] - p_center_i;
      auto q_delta = target_points[j] - q_center_i;

      auto p_distance = p_normal_i.dot(p_delta);
      auto q_distance = q_normal_i.dot(q_delta);

      auto p_lateral_distance = (p_delta - p_distance * p_normal_i).norm();
      auto q_lateral_distance = (q_delta - q_distance * q_normal_i).norm();
      auto p_bound = sin_angle * p_lateral_distance + one_minus_cos_angle * std::abs(p_distance);
      auto q_bound = sin_angle * q_lateral_distance + one_minus_cos_angle * std::abs(q_distance);

      auto bound = max_distance + std::min(p_bound, q_bound);
      if(std::abs(p_distance - q_distance) < bound) {
        edges.push_back({i, j + n_planes});
      }
    }
  }

  std::vector<std::vector<Correspondence>> rows(n_points);
  #pragma omp parallel for schedule(dynamic, 16)
  for(Index i = 0; i < n_points; ++i) {
    for(Index j = i + 1; j < n_points; ++j) {
      auto p_dist = (source_points[i] - source_points[j]).norm();
      auto q_dist = (target_points[i] - target_points[j]).norm();
      if(std::abs(p_dist - q_dist) < 2.0 * max_distance) {
        rows[i].push_back({i + n_planes, j + n_planes});
      }
    }
  }

  std::size_t size = edges.size();
  for(const auto &row : rows) {
    size += row.size();
  }
  edges.reserve(size);
  for(const auto &row : rows) {
    edges.insert(edges.end(), row.begin(), row.end());
  }

  return edges;
}

std::vector<Index> maximum_weight_clique(
  std::size_t node_count,
  std::span<const Correspondence> edges,
  std::span<const Index> weights
)
{
  if(node_count == 0 || edges.empty()) {
    return {};
  }

  std::vector<std::size_t> offsets(node_count + 1);
  for(std::size_t node = 0; node < node_count; ++node) {
    offsets[node + 1] = offsets[node] + weights[node];
  }

  Eigen::MatrixXd adjacency = Eigen::MatrixXd::Zero(
    offsets.back(), offsets.back()
  );
  for(std::size_t node = 0; node < node_count; ++node) {
    adjacency.block(offsets[node], offsets[node], weights[node], weights[node]).setOnes();
  }
  adjacency.diagonal().setZero();

  for(auto [first, second] : edges) {
    adjacency.block(offsets[first], offsets[second], weights[first], weights[second]).setOnes();
    adjacency.block(offsets[second], offsets[first], weights[second], weights[first]).setOnes();
  }

  clipperplus::Graph graph(std::move(adjacency));
  auto expanded_clique = clipperplus::find_clique(graph).first;

  std::vector<Index> clique;
  clique.reserve(expanded_clique.size());
  for(auto expanded_node : expanded_clique) {
    auto original = std::upper_bound(offsets.begin(), offsets.end(), expanded_node);
    clique.push_back(original - offsets.begin() - 1);
  }
  std::sort(clique.begin(), clique.end());
  clique.erase(std::unique(clique.begin(), clique.end()), clique.end());
  return clique;
}

}
