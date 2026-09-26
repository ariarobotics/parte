#pragma once

#include "common/types.h"
#include "registration/fpfh.h"
#include "registration/pch.h"

#include <cstddef>
#include <utility>
#include <vector>
#include <cassert>


namespace parte
{

struct Parameters
{
  //! NOTE: Other distance parameters are multiplied by voxel_size.
  Scalar voxel_size;
  explicit Parameters(Scalar voxel_size_)
  {
    assert(voxel_size_ > 0);
    voxel_size = voxel_size_;
  }


  // Normal Estimation parameters.
  std::size_t normal_k = 30;
  Scalar normal_radius = 2.0f;

  // FPFH parameters.
  std::size_t fpfh_k = 100;
  Scalar fpfh_radius = 5.0f;


  // Segmentation parameters.
  bool ground_segmentation = false;
  Scalar segmentation_thickness = 1.0f;
  Scalar segmentation_dispersion = 10.0f;  // deg
  Index segmentation_min_support = 100;

  // PCH parameters.
  Scalar pch_range = 20.0f;
  Scalar pch_2l_radius = 20.0f;
  Scalar pch_grouping_distance = 2.0f;
  Scalar pch_grouping_angle = 5.0f;  // deg

  // Outlier rejection
  Scalar outlier_rejection_distance = 1.0f;
  Scalar outlier_rejection_angle = 5.0f;  // deg
};

struct ProcessedCloud
{
  std::vector<Point> points;
  std::vector<Normal> normals;


  // Segmentation results
  std::vector<Indices> plane_supports;
  std::vector<Plane> planes;
  Indices non_planar;

  // Descriptors
  std::vector<std::pair<registration::PCH, registration::PCH>> pch;
  std::vector<registration::FPFH> fpfh;
};

ProcessedCloud process_cloud(std::vector<Eigen::Vector3d> points, const Parameters &parameters);

struct RegistrationResult
{
  ScalarMatrix<4, 4> transformation;  // Source-to-target
  std::vector<Correspondence> point_matches, plane_matches;
  Indices selected_points, selected_planes;
};

RegistrationResult register_clouds(
  const ProcessedCloud &source,
  const ProcessedCloud &target,
  const Parameters &parameters
);

}
