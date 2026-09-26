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

namespace nb = nanobind;
using namespace nb::literals;


std::vector<parte::Plane> to_planes(const std::vector<parte::ScalarVector<6>> &rows)
{
  std::vector<parte::Plane> planes;
  for(const auto &row : rows) {
    planes.emplace_back(row.head<3>(), row.tail<3>());
  }
  return planes;
}


NB_MODULE(parte, module)
{
  module.doc() = "PARTE algorithms used by demo.py";

  module.def("compute_neighborhoods",
    [](const std::vector<parte::Point> &points, parte::Scalar radius, parte::Index knn) {
      return parte::compute_neighborhoods(points, radius, knn);
    }, "points"_a, "radius"_a, "knn"_a, nb::call_guard<nb::gil_scoped_release>());

  module.def("compute_normals",
    [](const std::vector<parte::Point> &points, const std::vector<parte::Neighbors> &neighborhoods) {
      return parte::compute_normals(points, neighborhoods);
    }, "points"_a, "neighborhoods"_a, nb::call_guard<nb::gil_scoped_release>());

  module.def("compute_plane",
    [](const std::vector<parte::Point> &points, const parte::Indices &indices) {
      auto [center, normal] = parte::compute_plane(points, indices);
      parte::ScalarVector<6> plane;
      plane << center, normal;
      return plane;
    }, "points"_a, "indices"_a, nb::call_guard<nb::gil_scoped_release>());

  module.def("group_planes",
    [](const std::vector<parte::ScalarVector<6>> &planes, parte::Scalar offset_threshold, parte::Scalar angle_threshold) {
      return parte::registration::group_planes(to_planes(planes), offset_threshold, angle_threshold);
    }, "planes"_a, "offset_threshold"_a, "angle_threshold"_a, nb::call_guard<nb::gil_scoped_release>());

  module.def("segment_planes",
    [](const std::vector<parte::Point> &points,
      const std::vector<parte::Normal> &normals,
      const std::vector<parte::Neighbors> &neighborhoods,
      parte::Scalar max_thickness, parte::Scalar max_dispersion, std::size_t min_support) {
      return parte::segmentation::segment_planes(
        points, normals, neighborhoods, max_thickness, max_dispersion, min_support,
        std::vector<bool>(points.size(), false)
      );
    }, "points"_a, "normals"_a, "neighborhoods"_a,
    "max_thickness"_a, "max_dispersion"_a, "min_support"_a, nb::call_guard<nb::gil_scoped_release>());

  module.def("compute_fpfh",
    [](const std::vector<parte::Point> &points,
      const std::vector<parte::Normal> &normals,
      const std::vector<parte::Neighbors> &neighborhoods,
      const parte::Indices &indices) {
      return parte::registration::compute_fpfh(points, normals, neighborhoods, indices);
    }, "points"_a, "normals"_a, "neighborhoods"_a, "indices"_a, nb::call_guard<nb::gil_scoped_release>());

  module.def("mutual_correspondences",
    [](const std::vector<parte::registration::FPFH> &source, const std::vector<parte::registration::FPFH> &target) {
      return parte::registration::mutual_correspondences<parte::registration::FPFH>(source, target);
    }, "source"_a, "target"_a, nb::call_guard<nb::gil_scoped_release>());

  module.def("pch_matching",
    [](const std::vector<std::pair<parte::registration::PCH, parte::registration::PCH>> &source,
      const std::vector<std::pair<parte::registration::PCH, parte::registration::PCH>> &target,
      std::vector<parte::Indices> source_groups, std::vector<parte::Indices> target_groups) {
      return parte::registration::pch_matching(source, target, std::move(source_groups), std::move(target_groups));
    }, "source"_a, "target"_a, "source_groups"_a, "target_groups"_a, nb::call_guard<nb::gil_scoped_release>());

  module.def("consistent_correspondences",
    [](const std::vector<parte::Point> &source_points,
      const std::vector<parte::Point> &target_points,
      const std::vector<parte::ScalarVector<6>> &source_planes,
      const std::vector<parte::ScalarVector<6>> &target_planes,
      double max_distance, double max_angle) {
      return parte::registration::consistent_correspondences(
        source_points, target_points, to_planes(source_planes), to_planes(target_planes), max_distance, max_angle
      );
    }, "source_points"_a, "target_points"_a, "source_planes"_a, "target_planes"_a,
    "max_distance"_a, "max_angle"_a, nb::call_guard<nb::gil_scoped_release>());

  module.def("maximum_weight_clique",
    [](std::size_t node_count, const std::vector<parte::Correspondence> &edges, const parte::Indices &weights) {
      return parte::registration::maximum_weight_clique(node_count, edges, weights);
    }, "node_count"_a, "edges"_a, "weights"_a, nb::call_guard<nb::gil_scoped_release>());

  module.def("compute_pch",
    [](const std::vector<parte::Point> &points, const std::vector<parte::Normal> &normals,
      parte::Normal plane_normal, parte::Point plane_point, parte::Scalar distance_range, parte::Scalar distance_radius) {
      return parte::registration::compute_pch(points, normals, plane_normal, plane_point, distance_range, distance_radius);
    }, "points"_a, "normals"_a, "plane_normal"_a, "plane_point"_a,
    "distance_range"_a, "distance_radius"_a, nb::call_guard<nb::gil_scoped_release>());

  module.def("compute_transformation",
    [](const std::vector<parte::Point> &source_points,
      const std::vector<parte::Point> &target_points,
      const std::vector<parte::ScalarVector<6>> &source_planes,
      const std::vector<parte::ScalarVector<6>> &target_planes,
      const std::vector<parte::Scalar> &plane_weights) {
      return parte::registration::compute_transformation(
        source_points, target_points, to_planes(source_planes), to_planes(target_planes), plane_weights
      );
    }, "source_points"_a, "target_points"_a, "source_planes"_a, "target_planes"_a,
    "plane_weights"_a, nb::call_guard<nb::gil_scoped_release>());
}
