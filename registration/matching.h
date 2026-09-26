#pragma once

#include <vector>
#include <span>

#include "common/types.h"
#include "registration/pch.h"

namespace parte::registration
{

template <typename Feature>
std::vector<Correspondence> mutual_correspondences(
  std::span<const Feature> source,
  std::span<const Feature> target
);


std::pair<std::vector<Correspondence>, std::vector<Scalar>> pch_matching(
  std::span<const std::pair<PCH, PCH>> source,
  std::span<const std::pair<PCH, PCH>> target,
  std::vector<std::vector<Index>> source_groups,
  std::vector<std::vector<Index>> target_groups
);


std::vector<std::vector<Index>> group_planes(
  std::span<const Plane> planes,
  Scalar offset_threshold,
  Scalar angle_threshold
);


namespace detail
{
std::pair<std::vector<Correspondence>, std::vector<Scalar>> mutual_confidence(
  std::span<const PCH> source,
  std::span<const PCH> target
);


}

}

#include "matching.tpp"
