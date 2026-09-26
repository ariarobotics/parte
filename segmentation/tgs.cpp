// An adaptation of G3Reg's Travel implementation. See TGS_LICENSE.

#include "segmentation/tgs.h"
#include "common/preprocessing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <queue>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace parte::segmentation
{

using GridIndex = std::pair<Index, Index>;
constexpr std::array<GridIndex, 8> NeighborOffsets{{
  {-1, -1}, {-1, 0}, 
  {-1, 1}, {0, -1}, 
  {0, 1}, {1, -1}, 
  {1, 0}, {1, 1}
}};

struct Node
{
  std::vector<Index> indices;
  bool ground = false;

  Normal normal = Normal::UnitZ();
  Point mean = Point::Zero();
  Scalar d = 0.0f;
  Scalar weight = 0.0f;
};

class TravelGroundSegmenter
{
public:
  TravelGroundSegmenter(
    std::span<const Point> points,
    const TgsParameters &parameters
  ) : points_(points), p_(parameters) {}

  std::vector<bool> run()
  {
    if(points_.empty()) {
      return {};
    }

    min_bound_ = points_[0];
    Point max_bound = points_[0];
    for(const Point &point : points_) {
      min_bound_ = min_bound_.cwiseMin(point);
      max_bound = max_bound.cwiseMax(point);
    }
    GridIndex max_index = grid_index(max_bound);
    rows_ = max_index.first + 1;
    cols_ = max_index.second + 1;
    nodes_.resize(rows_ * cols_);

    for(Index i = 0; i < points_.size(); ++i) {
      GridIndex grid = grid_index(points_[i]);
      nodes_[index(grid)].indices.push_back(i);
    }
    model_nodes();
    search_traversable_nodes();
    return classify_points();
  }

private:
  std::span<const Point> points_;
  const TgsParameters &p_;
  Index rows_ = 0;
  Index cols_ = 0;
  Point min_bound_ = Point::Zero();
  std::vector<Node> nodes_;

  GridIndex grid_index(const Point &point) const
  {
    return {
      Index((point.x() - min_bound_.x()) / p_.grid_resolution),
      Index((point.y() - min_bound_.y()) / p_.grid_resolution)
    };
  }

  Index index(const GridIndex &grid_index) const
  {
    return grid_index.first * cols_ + grid_index.second;
  }

  Node fit_plane(std::vector<Index> &indices)
  {
    Node output;
    auto [covariance, mean] = compute_covariance(points_, indices);
    Eigen::SelfAdjointEigenSolver<ScalarMatrix<3, 3>> solver;
    solver.computeDirect(covariance);
    output.mean = mean;
    output.normal = solver.eigenvectors().col(0);
    if(output.normal.z() < 0.0f) {
      output.normal *= -1.0f;
    }
    output.d = -output.normal.dot(output.mean);
    Point eigenvalues = solver.eigenvalues().reverse();
    output.weight = (eigenvalues[0] + eigenvalues[1]) * eigenvalues[1] / (eigenvalues[0] * eigenvalues[2] + 0.001f);
    return output;
  }

  void model_nodes()
  {
    for(Node &current : nodes_) {
      if(current.indices.size() < p_.min_points_per_node) {
        continue;
      }

      std::vector<Index> sorted = std::move(current.indices);
      std::sort(sorted.begin(), sorted.end(), [&](Index left, Index right) {
        return points_[left].z() < points_[right].z();
      });

      Scalar lpr_height = 0.0f;
      for(Index i = 0; i < p_.low_point_representatives; ++i) {
        lpr_height += points_[sorted[i]].z() / p_.low_point_representatives;
      }

      std::vector<Index> seeds;
      seeds.reserve(sorted.size());
      for(Index index : sorted) {
        Scalar z = points_[index].z();
        if(z < lpr_height + p_.seed_threshold && z >= lpr_height - p_.seed_threshold) {
          seeds.push_back(index);
        }
      }

      if(seeds.size() < 3) {
        continue;
      }
      Node model = fit_plane(seeds);
      for(Index iteration = 1; iteration < p_.iterations; ++iteration) {
        seeds.clear();
        Scalar threshold = p_.distance_threshold - model.d;
        for(Index index : sorted) {
          if(model.normal.dot(points_[index]) < threshold)
            seeds.push_back(index);
        }
        if(seeds.size() < 3) {
          break;
        }
        model = fit_plane(seeds);
      }

      if(seeds.size() < 3) {
        continue;
      }
      model.indices = std::move(sorted);
      model.ground = model.normal.z() >= p_.normal_z_threshold;
      current = std::move(model);
    }
  }

  void search_traversable_nodes()
  {
    GridIndex dominant{-1, -1};
    GridIndex ego = grid_index(Point::Zero());
    Index radius = Index(std::ceil(16.0f / p_.grid_resolution));
    Scalar best_weight = -1.0f;
    for(Index row = std::max(0, ego.first - radius); row < std::min(rows_, ego.first + radius); ++row) {
      for(Index col = std::max(0, ego.second - radius); col < std::min(cols_, ego.second + radius); ++col) {
        const Node &candidate = nodes_[index({row, col})];
        if(candidate.ground && candidate.weight > best_weight) {
          best_weight = candidate.weight;
          dominant = {row, col};
        }
      }
    }
    if(dominant.first < 0) {
      return;
    }

    std::queue<GridIndex> queue;
    std::vector<bool> visited(nodes_.size());
    visited[index(dominant)] = true;
    queue.push(dominant);

    while(true) {
      while(!queue.empty()) {
        const GridIndex current_index = queue.front();
        queue.pop();
        const Node &current = nodes_[index(current_index)];
        for(auto [row_offset, col_offset] : NeighborOffsets) {
          GridIndex neighbor_index{
            current_index.first + row_offset,
            current_index.second + col_offset
          };
          if(
            neighbor_index.first < 0 || neighbor_index.first >= rows_ || 
            neighbor_index.second < 0 || neighbor_index.second >= cols_
          ) {
            continue;
          }

          Index offset = index(neighbor_index);
          Node &neighbor = nodes_[offset];
          if(visited[offset] || !neighbor.ground) {
            continue;
          }

          visited[offset] = true;
          if(std::abs(current.mean.z() - neighbor.mean.z()) > p_.local_height_threshold) {
            neighbor.ground = false;
            continue;
          }
          queue.push(neighbor_index);
        }
      }

      bool seeded_component = false;
      for(Index row = 0; row < rows_ && !seeded_component; ++row) {
        for(Index col = 0; col < cols_; ++col) {
          GridIndex grid{row, col};
          Index offset = index(grid);
          Node &candidate = nodes_[offset];
          if(!visited[offset] && candidate.ground) {
            visited[offset] = true;
            queue.push(grid);
            seeded_component = true;
            break;
          }
        }
      }
      if(!seeded_component) {
        break;
      }
    }
  }

  std::vector<bool> classify_points() const
  {
    std::vector<bool> ground(points_.size(), false);
    for(const Node &current : nodes_) {
      if(!current.ground) {
        continue;
      }

      Scalar ground_limit = p_.distance_threshold - current.d;
      for(Index index : current.indices) {
        Scalar projection = current.normal.dot(points_[index]);
        if(projection < ground_limit) {
          ground[index] = true;
        }
      }
    }

    return ground;
  }
};

std::vector<bool> segment_ground(
  std::span<const Point> points, const TgsParameters &parameters
)
{
  return TravelGroundSegmenter(points, parameters).run();
}

}  // namespace parte::segmentation
