#pragma once

#include "common/types.h"

#include <span>


namespace parte::segmentation
{

struct TgsParameters
{
  Scalar grid_resolution = 4.0f;
  Index iterations = 3;
  Index low_point_representatives = 5;
  Index min_points_per_node = 10;
  Scalar seed_threshold = 0.5f;
  Scalar distance_threshold = 0.125f;
  Scalar normal_z_threshold = 0.940f;
  Scalar local_height_threshold = 1.0f;
};

std::vector<Index> segment_ground(
  std::span<const Point> points,
  const TgsParameters &parameters = {}
);

}
