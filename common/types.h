#pragma once

#include <Eigen/Dense>
#include <vector>
#include <span>

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

template <typename dtype>
std::vector<dtype> select(std::span<const dtype> data, std::span<const Index> indices)
{
  std::vector<dtype> output;
  output.reserve(indices.size());
  for(Index i : indices) {
    output.push_back(data[i]);
  }
  return output;
}

}
