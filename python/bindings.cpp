#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "common/preprocessing.h"
#include "registration/fpfh.h"
#include "registration/matching.h"
#include "registration/outliers.h"
#include "registration/pch.h"
#include "registration/pipeline.h"
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

template<int Columns, typename Scalar = parte::Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Columns, Eigen::RowMajor>;


template<int Columns, typename Scalar>
std::vector<Eigen::Matrix<Scalar, Columns, 1>> to_vectors(const Matrix<Columns, Scalar> &matrix)
{
  std::vector<Eigen::Matrix<Scalar, Columns, 1>> result(matrix.rows());
  for(Eigen::Index row = 0; row < matrix.rows(); ++row) {
    result[row] = matrix.row(row).transpose();
  }
  return result;
}


template<typename Vector>
auto to_matrix(const std::vector<Vector> &vectors)
{
  Matrix<Vector::RowsAtCompileTime, typename Vector::Scalar> result(vectors.size(), Vector::RowsAtCompileTime);
  for(Eigen::Index row = 0; row < result.rows(); ++row) {
    result.row(row) = vectors[row].transpose();
  }
  return result;
}


std::vector<parte::Plane> to_planes(const Matrix<6> &matrix)
{
  std::vector<parte::Plane> result(matrix.rows());
  for(Eigen::Index row = 0; row < matrix.rows(); ++row) {
    result[row] = {matrix.row(row).head<3>().transpose(), matrix.row(row).tail<3>().transpose()};
  }
  return result;
}


Matrix<6> to_matrix(const std::vector<parte::Plane> &planes)
{
  Matrix<6> result(planes.size(), 6);
  for(Eigen::Index row = 0; row < result.rows(); ++row) {
    result.row(row) << planes[row].first.transpose(), planes[row].second.transpose();
  }
  return result;
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

  nb::class_<parte::Parameters>(module, "Parameters")
    .def(nb::init<parte::Scalar>(), "voxel_size"_a)
    .def_rw("voxel_size", &parte::Parameters::voxel_size)
    .def_rw("normal_k", &parte::Parameters::normal_k)
    .def_rw("normal_radius", &parte::Parameters::normal_radius)
    .def_rw("fpfh_k", &parte::Parameters::fpfh_k)
    .def_rw("fpfh_radius", &parte::Parameters::fpfh_radius)
    .def_rw("ground_segmentation", &parte::Parameters::ground_segmentation)
    .def_rw("segmentation_thickness", &parte::Parameters::segmentation_thickness)
    .def_rw("segmentation_dispersion", &parte::Parameters::segmentation_dispersion)
    .def_rw("segmentation_min_support", &parte::Parameters::segmentation_min_support)
    .def_rw("pch_range", &parte::Parameters::pch_range)
    .def_rw("pch_2l_radius", &parte::Parameters::pch_2l_radius)
    .def_rw("pch_grouping_distance", &parte::Parameters::pch_grouping_distance)
    .def_rw("pch_grouping_angle", &parte::Parameters::pch_grouping_angle)
    .def_rw("outlier_rejection_distance", &parte::Parameters::outlier_rejection_distance)
    .def_rw("outlier_rejection_angle", &parte::Parameters::outlier_rejection_angle);

  nb::class_<parte::ProcessedCloud>(module, "ProcessedCloud")
    .def_prop_ro("points", [](const parte::ProcessedCloud &cloud) { return to_matrix(cloud.points); })
    .def_prop_ro("normals", [](const parte::ProcessedCloud &cloud) { return to_matrix(cloud.normals); })
    .def_prop_ro("planes", [](const parte::ProcessedCloud &cloud) { return to_matrix(cloud.planes); })
    .def_prop_ro("fpfh", [](const parte::ProcessedCloud &cloud) { return to_matrix(cloud.fpfh); })
    .def_ro("plane_supports", &parte::ProcessedCloud::plane_supports)
    .def_ro("non_planar", &parte::ProcessedCloud::non_planar)
    .def_ro("pch", &parte::ProcessedCloud::pch);

  nb::class_<parte::RegistrationResult>(module, "RegistrationResult")
    .def_ro("transformation", &parte::RegistrationResult::transformation)
    .def_ro("point_matches", &parte::RegistrationResult::point_matches)
    .def_ro("plane_matches", &parte::RegistrationResult::plane_matches)
    .def_ro("selected_points", &parte::RegistrationResult::selected_points)
    .def_ro("selected_planes", &parte::RegistrationResult::selected_planes);

  module.def("process_cloud", [](const Matrix<3, double> &points, const parte::Parameters &parameters) {
    return parte::process_cloud(to_vectors(points), parameters);
  }, "points"_a, "parameters"_a, nb::call_guard<nb::gil_scoped_release>());

  module.def("register_clouds", &parte::register_clouds,
    "source"_a, "target"_a, "parameters"_a, nb::call_guard<nb::gil_scoped_release>());

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
    [](const Matrix<3> &point_matrix, parte::Scalar radius, parte::Index knn) {
      auto points = to_vectors(point_matrix);
      return parte::compute_neighborhoods(points, radius, knn);
    },
    "points"_a, "radius"_a, "knn"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "compute_normals",
    [](const Matrix<3> &point_matrix,
      const std::vector<parte::Neighbors> &neighborhoods) {
      require_equal(point_matrix.rows(), neighborhoods.size(),
        "points and neighborhoods must have the same length");
      auto points = to_vectors(point_matrix);
      return to_matrix(parte::compute_normals(points, neighborhoods));
    },
    "points"_a, "neighborhoods"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "compute_plane",
    [](const Matrix<3> &point_matrix, const std::vector<parte::Index> &indices) {
      auto points = to_vectors(point_matrix);
      auto [center, normal] = parte::compute_plane(points, indices);
      parte::ScalarVector<6> plane;
      plane << center, normal;
      return plane;
    },
    "points"_a, "indices"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "group_planes",
    [](const Matrix<6> &plane_matrix,
      parte::Scalar offset_threshold,
      parte::Scalar angle_threshold) {
      auto planes = to_planes(plane_matrix);
      return parte::registration::group_planes(
        planes, offset_threshold, angle_threshold
      );
    },
    "planes"_a, "offset_threshold"_a, "angle_threshold"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "segment_planes",
    [](const Matrix<3> &point_matrix,
      const Matrix<3> &normal_matrix,
      const std::vector<parte::Neighbors> &neighborhoods,
      parte::Scalar max_thickness,
      parte::Scalar max_dispersion,
      std::size_t min_support,
      std::vector<bool> exclude) {
      require_equal(point_matrix.rows(), normal_matrix.rows(),
        "points and normals must have the same length");
      require_equal(point_matrix.rows(), neighborhoods.size(),
        "points and neighborhoods must have the same length");
      auto points = to_vectors(point_matrix);
      auto normals = to_vectors(normal_matrix);
      if(exclude.empty()) {
        exclude.resize(points.size(), false);
      }
      require_equal(points.size(), exclude.size(), "points and exclude must have the same length");
      return parte::segmentation::segment_planes(
        points, normals, neighborhoods, max_thickness, max_dispersion, min_support, std::move(exclude)
      );
    },
    "points"_a, "normals"_a, "neighborhoods"_a,
    "max_thickness"_a, "max_dispersion"_a, "min_support"_a, "exclude"_a = std::vector<bool>{},
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "segment_ground",
    [](const Matrix<3> &point_matrix, const TgsParameters &parameters) {
      auto points = to_vectors(point_matrix);
      return parte::segmentation::segment_ground(points, parameters);
    },
    "Return one boolean per input point; True marks ground.",
    "points"_a, "parameters"_a = TgsParameters{},
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "compute_fpfh",
    [](const Matrix<3> &point_matrix,
      const Matrix<3> &normal_matrix,
      const std::vector<parte::Neighbors> &neighborhoods,
      const std::vector<parte::Index> &indices) {
      require_equal(point_matrix.rows(), normal_matrix.rows(),
        "points and normals must have the same length");
      require_equal(point_matrix.rows(), neighborhoods.size(),
        "points and neighborhoods must have the same length");
      auto points = to_vectors(point_matrix);
      auto normals = to_vectors(normal_matrix);
      return to_matrix(parte::registration::compute_fpfh(
        points, normals, neighborhoods, indices
      ));
    },
    "points"_a, "normals"_a, "neighborhoods"_a, "indices"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "mutual_correspondences",
    [](const Matrix<33> &source_matrix, const Matrix<33> &target_matrix) {
      auto source = to_vectors(source_matrix);
      auto target = to_vectors(target_matrix);
      return parte::registration::mutual_correspondences<parte::registration::FPFH>(source, target);
    },
    "source"_a, "target"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "pch_matching",
    [](const std::vector<std::pair<parte::registration::PCH, parte::registration::PCH>> &source,
      const std::vector<std::pair<parte::registration::PCH, parte::registration::PCH>> &target,
      std::vector<std::vector<parte::Index>> source_groups,
      std::vector<std::vector<parte::Index>> target_groups) {
      return parte::registration::pch_matching(
        source, target, std::move(source_groups), std::move(target_groups)
      );
    },
    "source"_a, "target"_a, "source_groups"_a, "target_groups"_a,
    nb::call_guard<nb::gil_scoped_release>()
  );

  module.def(
    "consistent_correspondences",
    [](const Matrix<3> &source_point_matrix,
      const Matrix<3> &target_point_matrix,
      const Matrix<6> &source_plane_matrix,
      const Matrix<6> &target_plane_matrix,
      double max_distance,
      double max_angle) {
      require_equal(source_point_matrix.rows(), target_point_matrix.rows(),
        "source_points and target_points must have the same length");
      require_equal(source_plane_matrix.rows(), target_plane_matrix.rows(),
        "source_planes and target_planes must have the same length");
      auto source_points = to_vectors(source_point_matrix);
      auto target_points = to_vectors(target_point_matrix);
      auto source_planes = to_planes(source_plane_matrix);
      auto target_planes = to_planes(target_plane_matrix);
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
      const std::vector<parte::Correspondence> &edges,
      const std::vector<parte::Index> &weights) {
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
    [](const Matrix<3> &point_matrix,
      const Matrix<3> &normal_matrix,
      parte::Normal plane_normal,
      parte::Point plane_point,
      parte::Scalar distance_range,
      parte::Scalar distance_radius) {
      require_equal(point_matrix.rows(), normal_matrix.rows(),
        "points and normals must have the same length");
      auto points = to_vectors(point_matrix);
      auto normals = to_vectors(normal_matrix);
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
    [](const Matrix<3> &source_point_matrix,
      const Matrix<3> &target_point_matrix,
      const Matrix<6> &source_plane_matrix,
      const Matrix<6> &target_plane_matrix,
      const std::vector<parte::Scalar> &plane_weights) {
      require_equal(source_point_matrix.rows(), target_point_matrix.rows(),
        "source_points and target_points must have the same length");
      require_equal(source_plane_matrix.rows(), target_plane_matrix.rows(),
        "source_planes and target_planes must have the same length");
      require_equal(source_plane_matrix.rows(), plane_weights.size(),
        "source_planes and plane_weights must have the same length");
      auto source_points = to_vectors(source_point_matrix);
      auto target_points = to_vectors(target_point_matrix);
      auto source_planes = to_planes(source_plane_matrix);
      auto target_planes = to_planes(target_plane_matrix);
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
