#pragma once

#include "pmc_graph.h"
#include "pmc_types.h"

#include <vector>

namespace pmc
{

inline std::vector<node_t> greedy_coloring(
  const pmc_graph &graph,
  std::vector<node_t> &candidates,
  const node_t candidates_weight,
  std::vector<std::vector<node_t>> &colors,
  std::vector<node_t> &candidate_scratch,
  std::vector<node_t> &bound_scratch,
  std::vector<offset_t> &color_positions,
  std::size_t &clique_prefix_size,
  node_t &clique_prefix_weight
)
{
  if(colors.size() <= candidates_weight) {
    colors.resize(static_cast<std::size_t>(candidates_weight) + 1);
  }
  for(std::size_t color = 1; color <= candidates_weight; ++color) {
    colors[color].clear();
  }

  std::vector<node_t> bounds(candidates.size());

  // Assign one distinct color for each unit of vertex weight.
  for(std::size_t index = 0; index < candidates.size(); ++index) {
    const node_t vertex = candidates[index];
    const node_t required_colors = graph.weight(vertex);
    node_t assigned_colors = 0;
    node_t color = 1;
    node_t highest_color = 0;

    while(assigned_colors < required_colors) {
      bool conflict = false;
      for(const node_t colored_vertex : colors[color]) {
        if(graph.is_edge(vertex, colored_vertex)) {
          conflict = true;
          break;
        }
      }

      if(!conflict) {
        colors[color].push_back(vertex);
        highest_color = color;
        ++assigned_colors;
      }

      ++color;
    }

    bounds[index] = highest_color;
  }

  color_positions.assign(
    static_cast<std::size_t>(candidates_weight) + 1,
    0
  );
  for(const node_t bound : bounds) {
    ++color_positions[bound];
  }

  offset_t next_position = 0;
  for(offset_t &position : color_positions) {
    const offset_t count = position;
    position = next_position;
    next_position += count;
  }

  candidate_scratch.resize(candidates.size());
  bound_scratch.resize(candidates.size());
  for(std::size_t index = 0; index < candidates.size(); ++index) {
    const offset_t position = color_positions[bounds[index]]++;
    candidate_scratch[position] = candidates[index];
    bound_scratch[position] = bounds[index];
  }

  clique_prefix_size = 0;
  clique_prefix_weight = 0;
  node_t prefix_weight = 0;
  for(std::size_t index = 0; index < candidates.size(); ++index) {
    candidates[index] = candidate_scratch[index];
    bounds[index] = bound_scratch[index];
    prefix_weight += graph.weight(candidates[index]);
    if(bounds[index] == prefix_weight) {
      clique_prefix_size = index + 1;
      clique_prefix_weight = prefix_weight;
    }
  }

  return bounds;
}

}  // namespace pmc
