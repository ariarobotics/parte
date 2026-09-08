#pragma once

#include "pmc_graph.h"
#include "pmc_types.h"

#include <atomic>
#include <vector>

namespace pmc
{

class pmcx_maxclique
{
public:
  pmcx_maxclique(pmc_graph &input_graph, node_t lower_bound)
      : graph(input_graph)
  {
    max_clique_weight = lower_bound;
    max_clique.reserve(graph.get_max_degree() + 1);
  }

  std::pair<node_t, std::vector<node_t>> search();

private:
  void branch(
    std::vector<node_t> &candidates, node_t candidates_weight,
    std::vector<node_t> &clique, node_t clique_weight,
    std::vector<std::vector<node_t>> &colors,
    std::vector<node_t> &candidate_scratch,
    std::vector<node_t> &bound_scratch,
    std::vector<offset_t> &color_positions
  );

  const pmc_graph &graph;
  std::atomic<node_t> max_clique_weight;
  std::atomic<bool> found_upper_bound{false};
  std::vector<node_t> max_clique;
};

}  // namespace pmc
