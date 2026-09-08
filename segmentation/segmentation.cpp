#include "segmentation.h"
#include "union_find.h"

#include <algorithm>
#include <limits>
#include <numeric>

#include <omp.h>

namespace parte::segmentation
{


std::vector<Indices> segment_planes(
  std::span<const Point> points,
  std::span<const Normal> normals,
  std::span<const Neighbors> neighborhoods,
  Scalar max_thickness, Scalar max_dispersion,
  size_t min_support
)
{
  std::vector<uint8_t> stability_mask(points.size(), 1);
  #pragma omp parallel num_threads(omp_get_max_threads())
  {
    ScalarMatrix<3, 3> Q;
    Eigen::SelfAdjointEigenSolver<ScalarMatrix<3, 3>> solver;

    #pragma omp for
    for(size_t i = 0; i < points.size(); ++i) {
      Q.setZero();
      for(Index j : neighborhoods[i]) {
        Q += normals[j] * normals[j].transpose();
      }
      Q /= neighborhoods[i].size();

      solver.computeDirect(Q);
      if(solver.eigenvalues()(2) < 1 - max_dispersion * max_dispersion) {
        stability_mask[i] = 0;
      }
    }
  }

  std::vector<Index> candidates, point_index(points.size(), -1);
  size_t n_candidates = 0;
  candidates.reserve(points.size());
  for(Index idx = 0; idx < points.size(); ++idx) {
    if(stability_mask[idx]) {
      candidates.push_back(idx);
      point_index[idx] = n_candidates++;
    }
  }

  std::vector<detail::PlaneModel> planes(n_candidates);
  for(size_t i = 0; i < n_candidates; ++i) {
    Index idx = candidates[i];
    planes[i].n_points = 1;
    planes[i].centroid = points[idx];
    planes[i].Q = normals[idx] * normals[idx].transpose();
    planes[i].C = ScalarMatrix<3, 3>::Zero();
  }

  const auto I = ScalarMatrix<3, 3>::Identity();
  UnionFind<Index> uf(n_candidates);
  std::vector<Neighbors> edges(points.size()), next_edges(points.size());
  std::span<const Neighbors> current_edges = neighborhoods;
  size_t applied = 0;
  do {
    applied = 0;
    for(Index u = 0; u < n_candidates; ++u) {
      const Index idx = candidates[u];
      for(Index neighbor : current_edges[idx]) {
        if(!stability_mask[neighbor]) {
          continue;
        }

        auto ru = uf.find(u), rv = uf.find(point_index[neighbor]);
        if(ru == rv) {
          continue;
        }

        const auto &pu = planes[ru], &pv = planes[rv];
        auto merged = detail::PlaneModel::merge(pu, pv);

        auto A = (1 - max_dispersion * max_dispersion) * I - merged.Q;
        auto B = merged.C - I * max_thickness * max_thickness;
        if(!detail::feasible_normal(A, B)) {
          const Index first = candidates[ru], second = candidates[rv];
          next_edges[std::min(first, second)].push_back(
            std::max(first, second)
          );
          continue;
        }
        auto kept = uf.join(ru, rv);
        planes[kept] = std::move(merged);
        applied++;
      }
    }

    edges.swap(next_edges);
    for(auto &neighbors : edges) {
      std::sort(neighbors.begin(), neighbors.end());
      neighbors.erase(
        std::unique(neighbors.begin(), neighbors.end()), neighbors.end()
      );
    }
    for(auto &neighbors : next_edges) {
      neighbors.clear();
    }
    current_edges = edges;
  } while(applied > 0);

  size_t n_planes = 0;
  std::vector<Indices> results;
  results.reserve(n_candidates / min_support);
  std::vector<Index> plane_indices(n_candidates, -1);
  for(Index i = 0; i < n_candidates; ++i) {
    if(uf.size(i) < min_support) {
      continue;
    }

    Index root = uf.find(i);
    if(plane_indices[root] == -1) {
      plane_indices[root] = n_planes++;
      results.emplace_back();
      results.back().reserve(uf.size(root));
    }
    results[plane_indices[root]].push_back(candidates[i]);
  }

  return results;
}


namespace detail
{

bool feasible_normal(
  Eigen::Ref<const ScalarMatrix<3, 3>> A,
  Eigen::Ref<const ScalarMatrix<3, 3>> B
)
{
  const Scalar Tolerance = std::numeric_limits<Scalar>::epsilon() * 256.0;

  // max min u^T((1 - w) * A + w * B) u <= 0 <=> Feasible normal exists
  auto feasible = [&](const ScalarVector<3> &u) {
    return u.dot(A * u) <= Tolerance && u.dot(B * u) <= Tolerance;
  };

  Eigen::SelfAdjointEigenSolver<ScalarMatrix<3, 3>> solver;
  solver.computeDirect(A);
  if(solver.eigenvalues()(0) > Tolerance) {
    return false;
  }

  if(feasible(solver.eigenvectors().col(0))) {
    return true;
  }

  auto difference = B - A;
  double weight_low = 0.0, weight_high = 1.0;
  for(int iteration = 0; iteration < 64; ++iteration) {
    const double weight = std::midpoint(weight_low, weight_high);
    solver.computeDirect(A + weight * difference);
    if(solver.eigenvalues()(0) > Tolerance) {
      return false;
    }

    auto u = solver.eigenvectors().col(0);
    if(feasible(u)) {
      return true;
    }

    const double slope = u.dot(difference * u);
    if(slope > Tolerance) {
      weight_low = weight;
      continue;
    }

    if(slope < -Tolerance) {
      weight_high = weight;
      continue;
    }

    // A degenerate minimum eigenspace can make the slope non-unique.
    // This is extremely rare in practice and hard to handle; we treat it as infeasible.
    return false;
  }

  solver.computeDirect(B);
  if(solver.eigenvalues()(0) > Tolerance) {
    return false;
  }

  if(feasible(solver.eigenvectors().col(0))) {
    return true;
  }

  return false;
}

}

}
