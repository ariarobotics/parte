#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "common/preprocessing.h"
#include "registration/fpfh.h"
#include "registration/matching.h"
#include "registration/outliers.h"
#include "registration/pch.h"
#include "registration/transformation.h"
#include "segmentation/segmentation.h"
#include "segmentation/tgs.h"


#include <stdexcept>
#include <utility>
#include <vector>


namespace nb = nanobind;
using namespace nb::literals;


namespace
{

using parte::Correspondence;
using parte::Index;
using parte::Neighbors;
using parte::Normal;
using parte::Plane;
using parte::Point;
using parte::Scalar;
using parte::registration::FPFH;
using PCHPair = std::pair<parte::registration::PCH, parte::registration::PCH>;

template <Index Columns>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Columns, Eigen::RowMajor>;

using PointMatrix = Matrix<3>;
using PlaneMatrix = Matrix<6>;
using FpfhMatrix = Matrix<33>;
using PlaneVector = parte::ScalarVector<6>;


std::vector<Point> points_from_matrix(const PointMatrix &matrix)
{
  std::vector<Point> points(matrix.rows());
  for(Index i = 0; i < matrix.rows(); ++i) {
    points[i] = matrix.row(i).transpose();
  }
  return points;
}


std::vector<Plane> planes_from_matrix(const PlaneMatrix &matrix)
{
  std::vector<Plane> planes(matrix.rows());
  for(Index i = 0; i < matrix.rows(); ++i) {
    planes[i] = {
      matrix.row(i).head<3>().transpose(),
      matrix.row(i).tail<3>().transpose()};
  }
  return planes;
}


std::vector<FPFH> fpfh_from_matrix(const FpfhMatrix &matrix)
{
  std::vector<FPFH> features(matrix.rows());
  for(Index i = 0; i < matrix.rows(); ++i) {
    features[i] = matrix.row(i).transpose();
  }
  return features;
}


FpfhMatrix fpfh_to_matrix(const std::vector<FPFH> &features)
{
  FpfhMatrix matrix(features.size(), 33);
  for(Index i = 0; i < matrix.rows(); ++i) {
    matrix.row(i) = features[i].transpose();
  }
  return matrix;
}


PointMatrix points_to_matrix(const std::vector<Point> &points)
{
  PointMatrix matrix(points.size(), 3);
  for(Index i = 0; i < matrix.rows(); ++i) {
    matrix.row(i) = points[i].transpose();
  }
  return matrix;
}


void require_equal(std::size_t first, std::size_t second, const char *message)
{
  if(first != second) {
    throw std::invalid_argument(message);
  }
}

}


NB_MODULE(parte, module)
{
  using parte::segmentation::TgsParameters;

  module.doc() = "Python bindings for PARTE's Open3D-independent algorithms";

  nb::class_<TgsParameters>(module, "TgsParameters")
    .def(nb::init<>())
    .def_rw("grid_resolution", &TgsParameters::grid_resolution)
    .def_rw("iterations", &TgsParameters::iterations)
    .def_rw("low_point_representatives", &TgsParameters::low_point_representatives)
    .def_rw("min_points_per_node", &TgsParameters::min_points_per_node)
    .def_rw("seed_threshold", &TgsParameters::seed_threshold)
    .def_rw("distance_threshold", &TgsParameters::distance_threshold)
    .def_rw("normal_z_threshold", &TgsParameters::normal_z_threshold)
    .def_rw("local_height_threshold", &TgsParameters::local_height_threshold);

  module.def(
    "compute_neighborhoods",
    [](const PointMatrix &point_matrix, Scalar radius, Index knn) {
      auto points = points_from_matrix(point_matrix);
      return parte::compute_neighborhoods(points, radius, knn);
    },
    "points"_a, "radius"_a, "knn"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "compute_normals",
    [](const PointMatrix &point_matrix,
      const std::vector<Neighbors> &neighborhoods) {
      require_equal(point_matrix.rows(), neighborhoods.size(),
        "points and neighborhoods must have the same length");
      auto points = points_from_matrix(point_matrix);
      return points_to_matrix(parte::compute_normals(points, neighborhoods));
    },
    "points"_a, "neighborhoods"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "compute_plane",
    [](const PointMatrix &point_matrix, const std::vector<Index> &indices) {
      auto points = points_from_matrix(point_matrix);
      auto [center, normal] = parte::compute_plane(points, indices);
      PlaneVector plane;
      plane << center, normal;
      return plane;
    },
    "points"_a, "indices"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "group_planes",
    [](const PlaneMatrix &plane_matrix,
      Scalar offset_threshold,
      Scalar angle_threshold) {
      auto planes = planes_from_matrix(plane_matrix);
      return parte::registration::detail::group_planes(
        planes, offset_threshold, angle_threshold
      );
    },
    "planes"_a, "offset_threshold"_a, "angle_threshold"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "segment_planes",
    [](const PointMatrix &point_matrix,
      const PointMatrix &normal_matrix,
      const std::vector<Neighbors> &neighborhoods,
      Scalar max_thickness,
      Scalar max_dispersion,
      std::size_t min_support) {
      require_equal(point_matrix.rows(), normal_matrix.rows(),
        "points and normals must have the same length");
      require_equal(point_matrix.rows(), neighborhoods.size(),
        "points and neighborhoods must have the same length");
      auto points = points_from_matrix(point_matrix);
      auto normals = points_from_matrix(normal_matrix);
      return parte::segmentation::segment_planes(
        points, normals, neighborhoods,
        max_thickness, max_dispersion, min_support
      );
    },
    "points"_a, "normals"_a, "neighborhoods"_a,
    "max_thickness"_a, "max_dispersion"_a, "min_support"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "segment_ground",
    [](const PointMatrix &point_matrix, const TgsParameters &parameters) {
      auto points = points_from_matrix(point_matrix);
      return parte::segmentation::segment_ground(points, parameters);
    },
    "points"_a, "parameters"_a = TgsParameters{},
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "compute_fpfh",
    [](const PointMatrix &point_matrix,
      const PointMatrix &normal_matrix,
      const std::vector<Neighbors> &neighborhoods,
      const std::vector<Index> &indices) {
      require_equal(point_matrix.rows(), normal_matrix.rows(),
        "points and normals must have the same length");
      require_equal(point_matrix.rows(), neighborhoods.size(),
        "points and neighborhoods must have the same length");
      auto points = points_from_matrix(point_matrix);
      auto normals = points_from_matrix(normal_matrix);
      return fpfh_to_matrix(parte::registration::compute_fpfh(
        points, normals, neighborhoods, indices
      ));
    },
    "points"_a, "normals"_a, "neighborhoods"_a, "indices"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "mutual_correspondences",
    [](const FpfhMatrix &source_matrix, const FpfhMatrix &target_matrix) {
      auto source = fpfh_from_matrix(source_matrix);
      auto target = fpfh_from_matrix(target_matrix);
      return parte::registration::mutual_correspondences<FPFH>(source, target);
    },
    "source"_a, "target"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "pch_matching",
    [](const std::vector<PCHPair> &source,
      const std::vector<PCHPair> &target,
      std::vector<std::vector<Index>> source_groups,
      std::vector<std::vector<Index>> target_groups) {
      return parte::registration::pch_matching(
        source, target, std::move(source_groups), std::move(target_groups)
      );
    },
    "source"_a, "target"_a, "source_groups"_a, "target_groups"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "consistent_correspondences",
    [](const PointMatrix &source_point_matrix,
      const PointMatrix &target_point_matrix,
      const PlaneMatrix &source_plane_matrix,
      const PlaneMatrix &target_plane_matrix,
      double max_distance,
      double max_angle) {
      require_equal(source_point_matrix.rows(), target_point_matrix.rows(),
        "source_points and target_points must have the same length");
      require_equal(source_plane_matrix.rows(), target_plane_matrix.rows(),
        "source_planes and target_planes must have the same length");
      auto source_points = points_from_matrix(source_point_matrix);
      auto target_points = points_from_matrix(target_point_matrix);
      auto source_planes = planes_from_matrix(source_plane_matrix);
      auto target_planes = planes_from_matrix(target_plane_matrix);
      return parte::registration::consistent_correspondences(
        source_points, target_points, source_planes, target_planes,
        max_distance, max_angle
      );
    },
    "source_points"_a, "target_points"_a,
    "source_planes"_a, "target_planes"_a,
    "max_distance"_a, "max_angle"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "maximum_weight_clique",
    [](std::size_t node_count,
      const std::vector<Correspondence> &edges,
      const std::vector<Index> &weights) {
      require_equal(node_count, weights.size(),
        "node_count and weights must have the same length");
      return parte::registration::maximum_weight_clique(
        node_count, edges, weights
      );
    },
    "node_count"_a, "edges"_a, "weights"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "compute_pch",
    [](const PointMatrix &point_matrix,
      const PointMatrix &normal_matrix,
      Normal plane_normal,
      Point plane_point,
      Scalar distance_range,
      Scalar distance_radius) {
      require_equal(point_matrix.rows(), normal_matrix.rows(),
        "points and normals must have the same length");
      auto points = points_from_matrix(point_matrix);
      auto normals = points_from_matrix(normal_matrix);
      return parte::registration::compute_pch(
        points, normals, plane_normal, plane_point,
        distance_range, distance_radius
      );
    },
    "points"_a, "normals"_a,
    "plane_normal"_a, "plane_point"_a,
    "distance_range"_a, "distance_radius"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "compute_transformation",
    [](const PointMatrix &source_point_matrix,
      const PointMatrix &target_point_matrix,
      const PlaneMatrix &source_plane_matrix,
      const PlaneMatrix &target_plane_matrix,
      const std::vector<Scalar> &plane_weights) {
      require_equal(source_point_matrix.rows(), target_point_matrix.rows(),
        "source_points and target_points must have the same length");
      require_equal(source_plane_matrix.rows(), target_plane_matrix.rows(),
        "source_planes and target_planes must have the same length");
      require_equal(source_plane_matrix.rows(), plane_weights.size(),
        "source_planes and plane_weights must have the same length");
      if(source_point_matrix.rows() == 0) {
        throw std::invalid_argument("at least one point correspondence is required");
      }
      auto source_points = points_from_matrix(source_point_matrix);
      auto target_points = points_from_matrix(target_point_matrix);
      auto source_planes = planes_from_matrix(source_plane_matrix);
      auto target_planes = planes_from_matrix(target_plane_matrix);
      return parte::registration::compute_transformation(
        source_points, target_points,
        source_planes, target_planes, plane_weights
      );
    },
    "source_points"_a, "target_points"_a,
    "source_planes"_a, "target_planes"_a, "plane_weights"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );
}
