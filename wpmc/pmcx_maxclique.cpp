#include "pmc_neigh_coloring.h"
#include "pmcx_maxclique.h"

#include <limits>
#include <utility>

std::pair<node_t, std::vector<node_t>> pmc::pmcx_maxclique::search()
{
  const node_t n_nodes = graph.num_vertices();
  const auto &ordered = graph.kcore_orders();
  std::vector<node_t> vertex_rank(n_nodes);
  for(node_t rank = 0; rank < n_nodes; ++rank) {
    vertex_rank[ordered[rank]] = rank;
  }

  const node_t max_deg = graph.get_max_degree();
  #pragma omp parallel
  {
    std::vector<node_t> candidates;
    candidates.reserve(max_deg + 1);

    std::vector<node_t> clique;
    clique.reserve(max_deg + 1);

    std::vector<node_t> certain_picks;
    certain_picks.reserve(max_deg + 1);

    std::vector<std::vector<node_t>> colors(max_deg + 3);
    std::vector<node_t> candidate_scratch;
    candidate_scratch.reserve(max_deg + 1);
    std::vector<node_t> bound_scratch;
    bound_scratch.reserve(max_deg + 1);
    std::vector<offset_t> color_positions;

    std::vector<node_t> indicator(graph.num_vertices(), std::numeric_limits<node_t>::max());
    #pragma omp for schedule(dynamic)
    for(node_t index = 0; index < n_nodes; ++index) {
      if(found_upper_bound.load(std::memory_order_relaxed)) {
        continue;
      }

      const node_t vertex = ordered[index];
      if(graph.kcore(vertex) < max_clique_weight) {
        continue;
      }

      clique.push_back(vertex);
      node_t candidates_weight = 0;
      for(const node_t neighbor : graph.neighbors(vertex)) {
        if(vertex_rank[neighbor] > index && graph.kcore(neighbor) >= max_clique_weight) {
          candidates.push_back(neighbor);
          candidates_weight += graph.weight(neighbor);
        }
      }

      const node_t weight_v = graph.weight(vertex);
      if(candidates_weight + graph.weight(vertex) > max_clique_weight) {
        const node_t required_weight = std::max<int>(max_clique_weight - weight_v, 0) + 1;
        if(graph.neigh_cores_bound(
             candidates,
             candidates_weight,
             indicator,
             certain_picks,
             required_weight
           )) {
          node_t clique_weight = graph.weight(vertex);
          for(const node_t certain_pick : certain_picks) {
            clique.push_back(certain_pick);
            clique_weight += graph.weight(certain_pick);
          }

          if(candidates.empty()) {
            #pragma omp critical(update_mc)
            {
              if(clique_weight > max_clique_weight) {
                max_clique = clique;
                max_clique_weight = clique_weight;

                if(max_clique_weight >= graph.max_core() + 1) {
                  found_upper_bound.store(true, std::memory_order_relaxed);
                }
              }
            }
          } else {
            branch(
              candidates,
              candidates_weight,
              clique,
              clique_weight,
              colors,
              candidate_scratch,
              bound_scratch,
              color_positions
            );
          }
        }
      }

      candidates.clear();
      clique.clear();
      certain_picks.clear();
    }
  }

  return {max_clique_weight, std::move(max_clique)};
}


void pmc::pmcx_maxclique::branch(
  std::vector<node_t> &candidates, node_t candidates_weight,
  std::vector<node_t> &clique, node_t clique_weight,
  std::vector<std::vector<node_t>> &colors,
  std::vector<node_t> &candidate_scratch,
  std::vector<node_t> &bound_scratch,
  std::vector<offset_t> &color_positions
)
{
  if(found_upper_bound.load(std::memory_order_relaxed)) {
    return;
  }

  if(clique_weight + candidates_weight <= max_clique_weight) {
    return;
  }

  const node_t entry_clique_size = clique.size();
  std::size_t clique_prefix_size = 0;
  node_t clique_prefix_weight = 0;
  std::vector<node_t> bounds = greedy_coloring(
    graph,
    candidates,
    candidates_weight,
    colors,
    candidate_scratch,
    bound_scratch,
    color_positions,
    clique_prefix_size,
    clique_prefix_weight
  );

  if(clique_prefix_size > 0
    && clique_weight + clique_prefix_weight > max_clique_weight) {
    #pragma omp critical(update_mc)
    {
      if(clique_weight + clique_prefix_weight > max_clique_weight) {
        max_clique = clique;
        max_clique.insert(
          max_clique.end(),
          candidates.begin(),
          candidates.begin() + clique_prefix_size
        );
        max_clique_weight = clique_weight + clique_prefix_weight;

        if(max_clique_weight >= graph.max_core() + 1) {
          found_upper_bound.store(true, std::memory_order_relaxed);
        }
      }
    }
  }

  while(!candidates.empty()) {
    const node_t color_bound = bounds.back();
    if(clique_weight + color_bound <= max_clique_weight) {
      break;
    }

    const node_t vertex = candidates.back();

    clique.push_back(vertex);
    clique_weight += graph.weight(vertex);

    std::vector<node_t> remaining;
    remaining.reserve(candidates.size());

    node_t remaining_weight = 0;
    for(std::size_t index = 0; index < candidates.size(); ++index) {
      const node_t candidate = candidates[index];
      if(graph.is_edge(vertex, candidate) && graph.kcore(candidate) >= max_clique_weight) {
        remaining.push_back(candidate);
        remaining_weight += graph.weight(candidate);
      }
    }

    if(remaining.size() == candidates.size() - 1) {
      candidates_weight -= graph.weight(vertex);
      candidates.pop_back();
      bounds.pop_back();
      continue;
    }

    if(!remaining.empty()) {
      branch(
        remaining,
        remaining_weight,
        clique,
        clique_weight,
        colors,
        candidate_scratch,
        bound_scratch,
        color_positions
      );
    } else if(clique_weight > max_clique_weight) {
      #pragma omp critical(update_mc)
      {
        if(clique_weight > max_clique_weight) {
          max_clique_weight = clique_weight;
          max_clique = clique;

          if(max_clique_weight >= graph.max_core() + 1) {
            found_upper_bound.store(true, std::memory_order_relaxed);
          }
        }
      }
    }

    clique.pop_back();
    clique_weight -= graph.weight(vertex);
    candidates_weight -= graph.weight(vertex);
    candidates.pop_back();
    bounds.pop_back();
  }

  clique.resize(entry_clique_size);
}
