#include "registration/matching.h"

#include <algorithm>


namespace parte::registration
{

std::pair<std::vector<Correspondence>, std::vector<Scalar>> pch_matching(
  std::span<const std::pair<PCH, PCH>> source,
  std::span<const std::pair<PCH, PCH>> target,
  std::vector<std::vector<Index>> source_groups,
  std::vector<std::vector<Index>> target_groups
)
{
  std::vector<PCH> source_pchs(source_groups.size(), PCH::Zero());
  for(Index i = 0; i < source_pchs.size(); ++i) {
    for(auto member : source_groups[i]) {
      source_pchs[i] += source[member].first / source_groups[i].size();
    }
  }

  std::vector<PCH> target_pchs(target_groups.size(), PCH::Zero());
  for(Index i = 0; i < target_pchs.size(); ++i) {
    for(auto member : target_groups[i]) {
      target_pchs[i] += target[member].first / target_groups[i].size();
    }
  }

  auto [group_matches, group_confidences] = detail::mutual_confidence(
    source_pchs, target_pchs
  );

  std::vector<Correspondence> matches;
  std::vector<Scalar> confidences;
  for(Index i = 0; i < group_matches.size(); ++i) {
    auto [source_group, target_group] = group_matches[i];
    if(source_groups[source_group].size() == 1 && target_groups[target_group].size() == 1) {
      matches.push_back({source_groups[source_group][0], target_groups[target_group][0]});
      confidences.push_back(group_confidences[i]);
      continue;
    }

    std::vector<PCH> source_restricted;
    std::vector<PCH> target_restricted;
    for(auto member : source_groups[source_group]) {
      source_restricted.push_back(source[member].second);
    }
    for(auto member : target_groups[target_group]) {
      target_restricted.push_back(target[member].second);
    }

    auto [member_matches, member_confidences] = detail::mutual_confidence(
      source_restricted, target_restricted
    );
    for(Index j = 0; j < member_matches.size(); ++j) {
      matches.push_back({
        source_groups[source_group][member_matches[j].first],
        target_groups[target_group][member_matches[j].second],
      });
      confidences.push_back(std::min(group_confidences[i], member_confidences[j]));
    }
  }

  return {matches, confidences};
}

std::vector<std::vector<Index>> group_planes(
  std::span<const Plane> planes,
  Scalar offset_threshold,
  Scalar angle_threshold
)
{
  const Index n_planes = planes.size();
  std::vector<Index> merged_to(n_planes);
  for(Index i = 0; i < n_planes; ++i) {
    merged_to[i] = i;
  }

  for(Index i = 0; i < n_planes; ++i) {
    for(Index j = i + 1; j < n_planes; ++j) {
      auto [center_i, normal_i] = planes[i];
      auto [center_j, normal_j] = planes[j];
      Scalar offset = (center_i - center_j).dot(normal_i);
      Scalar offset_rev = (center_i - center_j).dot(normal_j);
      offset = std::max(std::abs(offset), std::abs(offset_rev));

      Scalar angle = normal_i.dot(normal_j);
      if((offset < offset_threshold) && (angle > angle_threshold)) {
        merged_to[j] = i;
      }
    }
  }

  std::vector<std::vector<Index>> groups(n_planes);
  for(Index i = 0; i < n_planes; ++i) {
    groups[merged_to[i]].push_back(i);
  }
  Index n_groups = 0;
  for(Index i = 0; i < n_planes; ++i) {
    if(!groups[i].empty()) {
      if(n_groups != i) {
        groups[n_groups] = std::move(groups[i]);
      }
      ++n_groups;
    }
  }
  groups.resize(n_groups);
  return groups;
}


namespace detail
{

std::pair<std::vector<Correspondence>, std::vector<Scalar>> mutual_confidence(
  std::span<const PCH> source,
  std::span<const PCH> target
)
{
  const Index n_source = source.size(), n_target = target.size();
  if(n_source == 0 || n_target == 0) {
    return {{}, {}};
  }
  ScalarMatrix<Eigen::Dynamic, Eigen::Dynamic> distance_matrix(n_source, n_target);

  for(Index i = 0; i < n_source; ++i) {
    for(Index j = 0; j < n_target; ++j) {
      auto sum = (source[i] + target[j]).array();
      auto diff = (source[i] - target[j]).array().square();
      distance_matrix(i, j) = (sum > 0).select(diff / sum, 0).sum();
    }
  }

  std::vector<Scalar> confidences;
  std::vector<Correspondence> matches;
  for(Index i = 0; i < n_source; ++i) {
    Index j, k;
    Scalar d = distance_matrix.row(i).minCoeff(&j);
    distance_matrix.col(j).minCoeff(&k);
    if(k != i) {
      continue;
    }

    distance_matrix(i, j) = 2.0;  // maximum possible error
    Scalar d_source = distance_matrix.row(i).minCoeff();
    Scalar d_target = distance_matrix.col(j).minCoeff();
    distance_matrix(i, j) = d;


    matches.push_back({i, j});
    confidences.push_back(
      d_source == 0 || d_target == 0 ? 0 : std::sqrt((1 - d / d_source) * (1 - d / d_target))
    );
  }

  return {matches, confidences};
}


}

}
