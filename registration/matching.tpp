#pragma once

#include "registration/matching.h"

#include <vector>
#include <span>

#include <Eigen/Dense>
#include "common/nanoflann.hpp"

#include "common/types.h"
#include "common/nanoflann_adaptors.h"

namespace parte::registration
{


template <typename Feature>
std::vector<Correspondence> mutual_correspondences(
  std::span<const Feature> source,
  std::span<const Feature> target
)
{
  const size_t n_source = source.size();
  const size_t n_target = target.size();
  if(n_source == 0 || n_target == 0) {
    return {};
  }

  const KDTree<Feature, Index> target_tree(target, 32), source_tree(source, 32);
  std::vector<Index> forward_matches(n_source);
  std::vector<Scalar> forward_distances(n_source);
  #pragma omp parallel for schedule(dynamic, 64)
  for(auto i : source_tree.order()) {
    nanoflann::KNNResultSet<Scalar, Index> result_set(1);
    result_set.init(&forward_matches[i], &forward_distances[i]);

    auto query = source[i].data();
    target_tree.findNeighbors(result_set, query);
  }

  std::vector<Index> candidates_target;
  candidates_target.reserve(n_source);
  std::vector<Index> candidates_source(n_target, -1);
  std::vector<Scalar> candidates_bound(n_target, std::numeric_limits<Scalar>::max());
  for(Index i = 0; i < n_source; ++i) {
    int target_index = forward_matches[i];
    if(candidates_source[target_index] < 0) {
      candidates_target.push_back(target_index);
    }

    if(candidates_bound[target_index] > forward_distances[i]) {
      candidates_bound[target_index] = forward_distances[i];
      candidates_source[target_index] = i;
    }
  }

  #pragma omp parallel for schedule(dynamic, 64)
  for(auto candidate : candidates_target) {
    FindFirstResultSet<Scalar, Index> result(candidates_bound[candidate]);
    if(source_tree.findNeighbors(result, target[candidate].data())) {
      candidates_source[candidate] = -1;
    }
  }

  std::vector<Correspondence> correspondences;
  correspondences.reserve(candidates_target.size());
  for(const auto candidate : candidates_target) {
    if(candidates_source[candidate] >= 0) {
      correspondences.emplace_back(candidates_source[candidate], candidate);
    }
  }
  return correspondences;
}

}
