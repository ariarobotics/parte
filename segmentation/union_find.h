#pragma once


#include <vector>
#include <numeric>


template <typename dtype>
class UnionFind
{
private:
  std::vector<dtype> parent_;
  std::vector<dtype> rank_;
  std::vector<dtype> size_;

public:
  const std::vector<dtype> &parent() const noexcept
  {
    return parent_;
  }

  explicit UnionFind(std::size_t n) : parent_(n), rank_(n, 0), size_(n, 1)
  {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  dtype find(dtype x) noexcept
  {
    // Using path halving instead of full compression: https://en.wikipedia.org/wiki/Disjoint-set_data_structure#cite_note-Tarjan1984-4
    while(parent_[x] != x) {
      parent_[x] = parent_[parent_[x]];
      x = parent_[x];
    }
    return x;
  }

  dtype size(dtype x) noexcept
  {
    return size_[find(x)];
  }

  dtype join(dtype root_x, dtype root_y) noexcept
  {
    if(root_x == root_y) {
      return root_x;
    }

    if(rank_[root_x] < rank_[root_y]) {
      parent_[root_x] = root_y;
      size_[root_y] += size_[root_x];
      return root_y;
    } else {
      parent_[root_y] = root_x;
      size_[root_x] += size_[root_y];
      if(rank_[root_x] == rank_[root_y]) {
        rank_[root_x]++;
      }
      return root_x;
    }
  }
};
