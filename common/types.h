#pragma once

#include <Eigen/Dense>
#include <vector>
#include <span>
#include <ranges>
#include <tuple>

namespace parte
{

using Index = int;
using Scalar = float;

template <Index Dimension>
using ScalarVector = Eigen::Matrix<Scalar, Dimension, 1>;

template <Index N, Index M>
using ScalarMatrix = Eigen::Matrix<Scalar, N, M>;

using Point = ScalarVector<3>;
using Normal = ScalarVector<3>;

using Plane = std::pair<Point, Normal>;

using Correspondence = std::pair<Index, Index>;

using Indices = std::vector<Index>;
using Neighbors = std::vector<Index>;

template <typename... Data>
auto select(std::span<const Index> indices, const Data &...data)
{
  std::tuple<std::vector<std::ranges::range_value_t<Data>>...> output;
  std::apply([&](auto &...selected) {
    (selected.reserve(indices.size()), ...);
    for(Index i : indices) {
      (selected.push_back(data[i]), ...);
    }
  }, output);
  return output;
}

}
