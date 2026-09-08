#include "registration/outliers.h"

#include "pmc_graph.h"
#include "pmcx_maxclique.h"

#include <algorithm>
#include <cmath>
#include <numeric>


namespace parte::registration
{

std::vector<wpmc_edge> consistent_correspondences(
  std::span<const Point> source_points, std::span<const Point> target_points,
  std::span<const Plane> source_planes, std::span<const Plane> target_planes,
  double max_distance, double max_angle
)
{
  std::vector<wpmc_edge> edges;
  node_t n_points = source_points.size(), n_planes = source_planes.size();
  double sin_angle = std::sin(max_angle);
  double one_minus_cos_angle = 1.0 - std::cos(max_angle);

  for(node_t i = 0; i < n_planes; ++i) {
    auto [p_center_i, p_normal_i] = source_planes[i];
    auto [q_center_i, q_normal_i] = target_planes[i];

    for(node_t j = i + 1; j < n_planes; ++j) {
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

    for(node_t j = 0; j < n_points; ++j) {
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

  std::vector<std::vector<wpmc_edge>> rows(n_points);
  #pragma omp parallel for schedule(dynamic, 16)
  for(node_t i = 0; i < n_points; ++i) {
    for(node_t j = i + 1; j < n_points; ++j) {
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
  std::span<const wpmc_edge> edges,
  std::span<const node_t> weights
)
{
  if(node_count == 0) {
    return {};
  }
  if(edges.empty()) {
    auto best = std::max_element(weights.begin(), weights.end());
    return {static_cast<Index>(best - weights.begin())};
  }

  std::vector<offset_t> vertices(node_count + 1, 0);
  for(auto [u, v] : edges) {
    vertices[u + 1]++;
    vertices[v + 1]++;
  }
  std::partial_sum(vertices.begin(), vertices.end(), vertices.begin());

  std::vector<node_t> adjacency(vertices[node_count]);
  std::vector<offset_t> insertion(vertices.begin(), vertices.end() - 1);
  for(auto [u, v] : edges) {
    adjacency[insertion[u]++] = v;
    adjacency[insertion[v]++] = u;
  }

  pmc::pmc_graph graph(
    std::move(vertices), std::move(adjacency),
    std::vector<node_t>(weights.begin(), weights.end())
  );
  graph.create_adj();
  graph.compute_cores();

  pmc::pmcx_maxclique solver(graph, 0);
  auto [clique_weight, clique] = solver.search();
  auto best = std::max_element(weights.begin(), weights.end());
  if(*best > clique_weight) {
    clique = {static_cast<node_t>(best - weights.begin())};
  }
  std::sort(clique.begin(), clique.end());
  return std::vector<Index>(clique.begin(), clique.end());
}

}
