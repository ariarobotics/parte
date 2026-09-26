#pragma once

#include <cstdint>
#include <vector>
#include <span>

#include <Eigen/Dense>

#include "common/types.h"


namespace parte
{


// Returns the normal vector with mean(p)^T n < 0 (reoriented toward the origin)
Normal compute_normal(
  const ScalarMatrix<3, 3> &covariance,
  const Point &mean
);

Normal compute_normal(std::span<const Point> points, std::span<const Index> indices);

Plane compute_plane(std::span<const Point> points, std::span<const Index> indices);

// Returns sum p_i * p_i^T - mean(p) * mean(p)^T over the given points and indices alongside mean(p)
std::pair<ScalarMatrix<3, 3>, Point> compute_covariance(std::span<const Point> points, std::span<const Index> indices);

// Computes the ordering and applies to the point cloud
std::vector<Point> morton_reorder(std::span<const Point> points, Scalar voxel_size);

// Computes neighbors of the each point
std::vector<Neighbors> compute_neighborhoods(std::span<const Point> points, Scalar radius, Index knn);

// Reuses the existing neighbors for a smaller radius and neighbor count
std::vector<Neighbors> filter_neighborhoods(
  std::span<const Point> points,
  std::span<const Neighbors> neighborhoods,
  Scalar radius, Index knn
);

std::vector<Normal> compute_normals(
  std::span<const Point> points,
  std::span<const Neighbors> neighborhoods
);


namespace detail
{

// Returns the indices that would sort the point cloud in Morton order given a voxel size
// Morton order is a space-filling curve and tend to keep spatially close points also close in the array
std::vector<uint64_t> morton_order(std::span<const Point> points, Scalar voxel_size);


// This function extracts the first 21 bits of a 32-bit integer and spreads them out as follows
// (b0 b1 b2 ... b20) -> (b0 0 0 b1 0 0 b2 0 0 ... b20 0 0) and hence spread_by_3(5 or 00101) = 000 000 001 000 001
static inline uint64_t spread_by_3(uint32_t a)
{
  uint64_t x = a & 0x1fffff;                    // only 21 bits
  x = (x | (x << 32)) & 0x001f00000000ffffULL;  // 0000 0000 0001 1111 0000 0000 0000 0000 0000 0000 0000 0000 1111 1111 1111 1111
  x = (x | (x << 16)) & 0x001f0000ff0000ffULL;  // 0000 0000 0001 1111 0000 0000 0000 0000 1111 1111 0000 0000 0000 0000 1111 1111
  x = (x | (x << 8)) & 0x100f00f00f00f00fULL;   // 0001 0000 0000 1111 0000 0000 1111 0000 0000 1111 0000 0000 1111 0000 0000 1111
  x = (x | (x << 4)) & 0x10c30c30c30c30c3ULL;   // 0001 0000 1100 0011 0000 1100 0011 0000 1100 0011 0000 1100 0011 0000 1100 0011
  x = (x | (x << 2)) & 0x1249249249249249ULL;   // 0001 0010 0100 1001 0010 0100 1001 0010 0100 1001 0010 0100 1001 0010 0100 1001
  return x;
}

// This function takes 3 21-bit integers and interleaves their bits to produce a 63-bit Morton code
// For example if x = (x3 x2 x1 x0), y = (y3 y2 y1 y0), z = (z3 z2 z1 z0)
// Spread x: (x3 0 0 x2 0 0 x1 0 0 x0 0 0)
// Spread y: (0 y3 0 0 y2 0 0 y1 0 0 y0 0)
// Spread z: (0 0 z3 0 0 z2 0 0 z1 0 0 z0)
// Mort xyz: (x3 y3 z3 x2 y2 z2 x1 y1 z1 x0 y0 z0)
static inline uint64_t morton3_21(uint32_t x, uint32_t y, uint32_t z)
{
  return spread_by_3(x) | (spread_by_3(y) << 1) | (spread_by_3(z) << 2);
}

}

}
