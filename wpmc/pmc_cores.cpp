#include "pmc_graph.h"
#include <algorithm>
#include <limits>
#include <numeric>

using namespace std;
using namespace pmc;

template <bool TrackBound, typename ForeachNeighbor, typename WeightsFn>
bool compute_core(
  std::vector<node_t> &kcore,  // in: wdegree; out: kcore
  std::vector<node_t> &order,  // out: kcore order
  node_t lower_bound,          // early exit if |v : lb - 1 <= kcore(v)| < lb
  ForeachNeighbor &&foreach_neighbor,
  WeightsFn &&weights
)
{
  const node_t minimum_core = TrackBound ? static_cast<node_t>(lower_bound - 1) : node_t{0};
  node_t max_degree = 0, possible_outputs = 0;
  for(node_t node = 0; node < kcore.size(); ++node) {
    max_degree = std::max(max_degree, kcore[node]);
    if constexpr(TrackBound) {
      if(kcore[node] >= minimum_core) {
        possible_outputs += weights(node);
      }
    }
  }

  if constexpr(TrackBound) {
    if(possible_outputs < lower_bound) {
      return false;
    }
  }

  std::vector<node_t> bin(max_degree + 1, 0);
  const node_t n_nodes = static_cast<node_t>(kcore.size());
  for(node_t vertex = 0; vertex < n_nodes; ++vertex) {  // bin(d) = |v : deg(v) = d|
    ++bin[kcore[vertex]];
  }

  // bin(d) = |v : deg(v) < d|
  std::exclusive_scan(bin.begin(), bin.end(), bin.begin(), 0);
  order.resize(n_nodes);
  std::vector<node_t> pos(n_nodes, 0);
  for(node_t vertex = 0; vertex < n_nodes; ++vertex) {
    // pos(kcore(k)) = k & kcore(pos(v)) = v
    pos[vertex] = bin[kcore[vertex]];
    order[pos[vertex]] = vertex;
    ++bin[kcore[vertex]];
  }

  // bins(d) = |v : deg(v) < d|
  for(node_t degree = max_degree; degree > 0; --degree) {
    bin[degree] = bin[degree - 1];
  }
  bin[0] = 0;

  for(node_t index = 0; index < n_nodes; ++index) {
    const node_t v = order[index];  // vertex with smallest current degree
    const node_t core_v = kcore[v];
    const node_t weight_v = weights(v);
    foreach_neighbor(v, [&](auto u) {
      node_t core_u = kcore[u];
      if(core_u <= core_v) {
        return;
      }

      const node_t target_core =
        core_u - std::min<node_t>(weight_v, core_u - core_v);
      do {
        const node_t u_position = pos[u];
        const node_t first_position = bin[core_u];
        const node_t first_vertex = order[first_position];
        if(u != first_vertex) {
          pos[u] = first_position;
          order[u_position] = first_vertex;
          pos[first_vertex] = u_position;
          order[first_position] = u;
        }

        if constexpr(TrackBound) {
          if(core_u == minimum_core) {
            possible_outputs -= weights(u);
          }
        }

        ++bin[core_u];
        --core_u;
      } while(core_u > target_core);
      kcore[u] = core_u;
    });


    if constexpr(TrackBound) {
      if(possible_outputs < lower_bound) {
        return false;
      }
    }
  }

  std::reverse(order.begin(), order.end());
  return true;
}


void pmc_graph::compute_cores()
{
  kcore_ = weighted_degree_;
  if(kcore_.empty()) {
    kcore_order_.clear();
    max_core_ = 0;
    return;
  }

  compute_core<false>(kcore_, kcore_order_, 0, [&](const node_t vertex, auto &&foreach_neighbor) {
    for(const node_t neighbor : neighbors(vertex)) {
      foreach_neighbor(neighbor);
    } }, [&](const node_t vertex) { return weight(vertex); });

  max_core_ = kcore_[kcore_order_.front()];
}


bool pmc_graph::neigh_cores_bound(
  std::vector<node_t> &candidates, node_t &candidates_weight,
  std::vector<node_t> &indicator,  // indicator[i] = node_t::max()
  std::vector<node_t> &certain_picks,
  const node_t lower_bound
) const
{
  certain_picks.clear();
  if(lower_bound == 0) {
    return true;
  }

  const node_t n_candidates = candidates.size();
  for(node_t v = 0; v < n_candidates; ++v) {
    indicator[candidates[v]] = v;
  }

  std::vector<node_t> degree(n_candidates, 0);
  for(node_t vertex = 0; vertex < n_candidates; ++vertex) {
    unsigned sum = weight(candidates[vertex]) - 1;
    for(const node_t neighbor : neighbors(candidates[vertex])) {
      sum += weight(neighbor) * unsigned(indicator[neighbor] != std::numeric_limits<node_t>::max());
    }
    degree[vertex] = sum;
  }

  std::vector<node_t> universal_vertices;
  for(node_t vertex = 0; vertex < n_candidates; ++vertex) {
    if(degree[vertex] == candidates_weight - 1) {
      universal_vertices.push_back(vertex);
    }
  }

  std::vector<node_t> kcore_order;
  node_t inactive = 0;
  auto completed = compute_core<true>(degree, kcore_order, lower_bound, [&](const node_t vertex, auto &&foreach_neighbor) {
    while(inactive < n_candidates && degree[kcore_order[inactive]] <= degree[vertex]) {
      indicator[candidates[kcore_order[inactive]]] = std::numeric_limits<node_t>::max();
      ++inactive;
    }
    for(const node_t neighbor : neighbors(candidates[vertex])) {
      if(indicator[neighbor] != std::numeric_limits<node_t>::max()) {
        foreach_neighbor(indicator[neighbor]);
      }
    } }, [&](const node_t vertex) { return weight(candidates[vertex]); });

  if(!completed) {
    for(const node_t id : candidates) {
      indicator[id] = std::numeric_limits<node_t>::max();
    }
    candidates.clear();
    return false;
  }


  const node_t minimum_core = static_cast<node_t>(lower_bound - 1);
  for(node_t vertex = 0; vertex < n_candidates; ++vertex) {
    if(degree[vertex] < minimum_core) {
      candidates_weight -= weight(candidates[vertex]);
    }
  }

  std::erase_if(kcore_order, [&](const node_t vertex) {
    return degree[vertex] < minimum_core;
  });

  node_t certain_weight = 0;
  for(const node_t vertex : universal_vertices) {
    if(degree[vertex] >= minimum_core) {
      certain_picks.push_back(candidates[vertex]);
      certain_weight += weight(candidates[vertex]);
    }
  }

  std::erase_if(kcore_order, [&](const node_t vertex) {
    return std::binary_search(
      universal_vertices.begin(), universal_vertices.end(), vertex
    );
  });
  candidates_weight -= certain_weight;

  for(const node_t id : candidates) {
    indicator[id] = std::numeric_limits<node_t>::max();
  }

  for(auto &node : kcore_order) {
    node = candidates[node];
  }

  candidates = std::move(kcore_order);
  return completed;
}
