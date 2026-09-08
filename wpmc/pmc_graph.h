#pragma once

#include "pmc_types.h"

#include <algorithm>
#include <cstdint>
#include <ranges>
#include <vector>

namespace pmc
{

class pmc_graph
{
  using word_t = std::uint64_t;
  static constexpr node_t word_bits = sizeof(word_t) * 8;

public:
  pmc_graph(std::vector<offset_t> vs, std::vector<node_t> es, std::vector<node_t> ws)
      : edges(std::move(es)), vertices(std::move(vs)), weights(std::move(ws))
  {
    max_degree = 0;
    for(node_t vertex = 0; vertex < num_vertices(); ++vertex) {
      const node_t deg = vertices[vertex + 1] - vertices[vertex];
      max_degree = std::max(max_degree, deg);
    }

    weighted_degree_.resize(num_vertices());
    for(node_t vertex = 0; vertex < num_vertices(); ++vertex) {
      weighted_degree_[vertex] = weights[vertex] - 1;
      for(offset_t edge = vertices[vertex]; edge < vertices[vertex + 1]; ++edge) {
        weighted_degree_[vertex] += weights[edges[edge]];
      }
    }
  }

  void create_adj()
  {
    const node_t size = num_vertices();
    const node_t n_words = (size + word_bits - 1) / word_bits;
    adj.assign(size, std::vector<word_t>(n_words));

    for(node_t v = 0; v < size; ++v) {
      for(offset_t edge = vertices[v]; edge < vertices[v + 1]; ++edge) {
        const node_t u = edges[edge];
        adj[v][u / word_bits] |= (word_t{1} << (u % word_bits));
      }
    }
  }

  node_t weight(const node_t vertex) const noexcept
  {
    return weights[vertex];
  }

  bool is_edge(const node_t u, const node_t v) const noexcept
  {
    return (adj[u][v / word_bits] & (word_t{1} << (v % word_bits))) != 0;
  }

  auto neighbors(const node_t vertex) const noexcept
  {
    return std::ranges::subrange{
      edges.begin() + vertices[vertex],
      edges.begin() + vertices[vertex + 1]};
  }

  node_t num_vertices() const noexcept
  {
    return vertices.size() - 1;
  }

  node_t get_max_degree() const noexcept
  {
    return max_degree;
  }

  void compute_cores();
  node_t max_core() const noexcept
  {
    return max_core_;
  }

  node_t kcore(const node_t vertex) const noexcept
  {
    return kcore_[vertex];
  }

  const std::vector<node_t> &kcore_orders() const noexcept
  {
    return kcore_order_;
  }

  std::vector<node_t> edges;
  std::vector<offset_t> vertices;
  std::vector<std::vector<word_t>> adj;

  bool neigh_cores_bound(
    std::vector<node_t> &candidates, node_t &candidates_weight,
    std::vector<node_t> &indicator,
    std::vector<node_t> &certain_picks,
    node_t max_clique_size
  ) const;

private:
  node_t max_degree = 0;
  node_t max_core_ = 0;
  std::vector<node_t> kcore_;
  std::vector<node_t> kcore_order_;

  std::vector<node_t> weights;
  std::vector<node_t> weighted_degree_;
};


}  // namespace pmc
