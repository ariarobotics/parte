#include "common/preprocessing.h"
#include "common/nanoflann.hpp"
#include "common/nanoflann_adaptors.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <Eigen/Eigenvalues>
#include <omp.h>

namespace parte
{

std::vector<Point> morton_reorder(std::span<const Point> points, Scalar voxel_size)
{
  std::vector<uint64_t> indices = detail::morton_order(points, voxel_size);
  std::vector<Point> reordered;
  reordered.reserve(points.size());
  for(uint64_t i : indices) {
    reordered.push_back(points[i]);
  }
  return reordered;
}


Normal compute_normal(
  const ScalarMatrix<3, 3> &covariance,
  const Point &mean
)
{
  Eigen::SelfAdjointEigenSolver<ScalarMatrix<3, 3>> solver;
  solver.computeDirect(covariance);

  Normal normal = solver.eigenvectors().col(0);
  if(mean.dot(normal) > 0) {
    normal = -normal;
  }
  return normal;
}


Normal compute_normal(
  std::span<const Point> points,
  std::span<const Index> indices
)
{
  auto [covariance, mean] = compute_covariance(points, indices);
  return compute_normal(covariance, mean);
}


Plane compute_plane(
  std::span<const Point> points,
  std::span<const Index> indices
)
{
  auto [covariance, mean] = compute_covariance(points, indices);
  return {mean, compute_normal(covariance, mean)};
}


std::pair<ScalarMatrix<3, 3>, Point> compute_covariance(
  std::span<const Point> points,
  std::span<const Index> indices
)
{
  ScalarMatrix<3, 3> covariance = ScalarMatrix<3, 3>::Zero();
  Point mean = Point::Zero();
  for(Index idx : indices) {
    mean += points[idx] / indices.size();
  }

  for(Index idx : indices) {
    Point p = points[idx] - mean;
    covariance += (p * p.transpose()) / indices.size();
  }
  return {covariance, mean};
}

std::vector<Neighbors> compute_neighborhoods(std::span<const Point> points, Scalar radius, Index knn)
{
  std::vector<Neighbors> neighborhoods(points.size());
  KDTree<Point, Index> tree(points);

  #pragma omp parallel for
  for(Index i = 0; i < points.size(); ++i) {
    Index capacity = std::min<Index>(knn, points.size());
    std::vector<Index> indices(capacity);
    std::vector<Scalar> distances(indices.size());
    nanoflann::RKNNResultSet<Scalar, Index> neighbors(capacity, radius * radius);
    neighbors.init(indices.data(), distances.data());
    tree.findNeighbors(neighbors, points[i].data());
    neighborhoods[i].assign(indices.begin(), indices.begin() + neighbors.size());
  }
  return neighborhoods;
}


std::vector<Neighbors> filter_neighborhoods(
  std::span<const Point> points,
  std::span<const Neighbors> neighborhoods,
  Scalar radius, Index knn
)
{
  const Scalar radius_squared = radius * radius;
  std::vector<Neighbors> result(points.size());
  #pragma omp parallel for schedule(static)
  for(Index point = 0; point < points.size(); ++point) {
    result[point].reserve(knn);
    for(Index neighbor : neighborhoods[point]) {
      if(result[point].size() == knn || (points[point] - points[neighbor]).squaredNorm() > radius_squared) {
        break;
      }
      result[point].push_back(neighbor);
    }
  }
  return result;
}


std::vector<Normal> compute_normals(
  std::span<const Point> points,
  std::span<const Neighbors> neighborhoods
)
{
  std::vector<Normal> normals(points.size(), Normal::UnitZ());
  #pragma omp parallel for
  for(Index point = 0; point < points.size(); ++point) {
    if(neighborhoods[point].size() >= 3) {
      normals[point] = compute_normal(points, neighborhoods[point]);
    }
    if(normals[point].dot(points[point]) > 0.0f) {
      normals[point] = -normals[point];
    }
  }
  return normals;
}


namespace detail
{

std::vector<uint64_t> morton_order(std::span<const Point> points, Scalar voxel_size)
{
  Point min_bound = points[0];
  for(const auto &p : points) {
    min_bound = min_bound.cwiseMin(p);
  }

  const Scalar inv_voxel = 1.0 / voxel_size;
  auto voxel_of = [&](const Point &p) -> Eigen::Vector3i {
    Point q = (p - min_bound) * inv_voxel;
    return Eigen::Vector3i(
      std::floor(q.x()),
      std::floor(q.y()),
      std::floor(q.z())
    );
  };

  // Using voxel indices we can construct a key for each point then using spatial sort ( Morton code ) to reorder
  std::vector<std::pair<uint64_t, uint64_t>> morton_codes;
  morton_codes.reserve(points.size());
  for(size_t i = 0; i < points.size(); ++i) {
    const auto &point = points[i];
    auto voxel_index = voxel_of(point);
    uint64_t morton_code = detail::morton3_21(
      static_cast<uint32_t>(voxel_index(0)),
      static_cast<uint32_t>(voxel_index(1)),
      static_cast<uint32_t>(voxel_index(2))
    );
    morton_codes.push_back({morton_code, i});
  }

  std::sort(morton_codes.begin(), morton_codes.end());
  std::vector<uint64_t> indices(morton_codes.size());
  for(uint64_t i = 0; i < morton_codes.size(); ++i) {
    indices[i] = morton_codes[i].second;
  }
  return indices;
}

}
}
