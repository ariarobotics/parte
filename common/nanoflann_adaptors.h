#pragma once

#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

#include "common/nanoflann.hpp"
#include <Eigen/Core>

namespace parte
{

// Returns the first point found within the specified radius
template <typename DistanceType_, typename IndexType_>
class FindFirstResultSet
{
public:
  using DistanceType = DistanceType_;
  using IndexType = IndexType_;

public:
  explicit FindFirstResultSet(DistanceType radius_squared) : radius_squared_(radius_squared)
  {}

  DistanceType worstDist() const noexcept
  {
    return radius_squared_;
  }

  bool addPoint(DistanceType distance, IndexType index) noexcept
  {
    found_ = true;
    return false;
  }

  bool full() const noexcept
  {
    return found_;
  }

  void sort() noexcept
  {}

private:
  DistanceType radius_squared_;
  bool found_ = false;
};


// Accepts std::vector<>, std::array<> etc.. of Eigen vectors
template <typename Vector>
class EigenVectorSpanAdaptor
{
public:
  using Scalar = typename Vector::Scalar;
  static constexpr int Dimension = Vector::SizeAtCompileTime;

  static_assert(Vector::ColsAtCompileTime == 1);
  static_assert(Vector::RowsAtCompileTime != Eigen::Dynamic);

  explicit EigenVectorSpanAdaptor(std::span<const Vector> values) noexcept : values_(values)
  {}

  std::size_t kdtree_get_point_count() const noexcept
  {
    return values_.size();
  }

  Scalar kdtree_get_pt(std::size_t index, std::size_t dimension) const noexcept
  {
    return values_[index](static_cast<Eigen::Index>(dimension));
  }

  template <typename BoundingBox>
  bool kdtree_get_bbox(BoundingBox &) const noexcept
  {
    return false;
  }

private:
  std::span<const Vector> values_;
};


// Forwards search results with their original indices.
template <typename ResultSet, typename IndexType_>
struct RKNNResultSet
{
  using DistanceType = typename ResultSet::DistanceType;
  using IndexType = typename ResultSet::IndexType;

  ResultSet &result;
  const std::vector<IndexType_> &indices;

  DistanceType worstDist() const
  {
    return result.worstDist();
  }

  bool full() const
  {
    return result.full();
  }

  void sort()
  {
    result.sort();
  }

  bool addPoint(DistanceType distance, IndexType index)
  {
    return result.addPoint(distance, indices[index]);
  }
};


// Stores vectors in leaf order while returning their original indices.
template <typename Feature, typename IndexType_>
class KDTree
{
public:
  using Scalar = typename Feature::Scalar;
  using IndexType = IndexType_;
  using Dataset = EigenVectorSpanAdaptor<Feature>;
  using Distance = typename nanoflann::metric_L2::template traits<Scalar, Dataset, IndexType>::distance_t;
  using Tree = nanoflann::KDTreeSingleIndexAdaptor<Distance, Dataset, Dataset::Dimension, IndexType>;

  explicit KDTree(std::span<const Feature> features, std::size_t leaf_size = 10)
      : dataset_(features), tree_(Dataset::Dimension, dataset_, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_size))
  {
    indices_ = tree_.vAcc_;
    features_.reserve(features.size());
    for(IndexType index : indices_) {
      features_.push_back(features[index]);
    }

    dataset_ = Dataset(features_);
    std::iota(tree_.vAcc_.begin(), tree_.vAcc_.end(), 0);
  }

  std::span<const IndexType> order() const
  {
    return indices_;
  }

  template <typename ResultSet>
  bool findNeighbors(ResultSet &result, const Scalar *query) const
  {
    RKNNResultSet<ResultSet, IndexType> result_set{result, indices_};
    return tree_.findNeighbors(result_set, query);
  }

private:
  Dataset dataset_;
  Tree tree_;
  std::vector<Feature> features_;
  std::vector<IndexType> indices_;
};

}
