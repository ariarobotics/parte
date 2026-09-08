#include "common/parameters.h"
#include "common/preprocessing.h"
#include "registration/fpfh.h"
#include "registration/matching.h"
#include "registration/outliers.h"
#include "registration/pch.h"
#include "registration/transformation.h"
#include "segmentation/segmentation.h"
#include "segmentation/tgs.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <open3d/Open3D.h>


namespace
{

namespace fs = std::filesystem;
namespace parameters = parte::parameters;
using Clock = std::chrono::steady_clock;
using Color = Eigen::Vector3d;
using parte::Index;
using parte::Neighbors;
using parte::Normal;
using parte::Plane;
using parte::Point;
using parte::Scalar;

struct Options
{
  fs::path source;
  fs::path target;
  Scalar voxel_size;
  bool segment_ground = false;
  bool raw = false;
};


struct PreparedCloud
{
  std::size_t input_point_count = 0;
  std::vector<Point> points;
  std::vector<Normal> normals;
  std::vector<parte::Indices> plane_supports;
  std::vector<Plane> planes;
  std::vector<std::pair<parte::registration::PCH, parte::registration::PCH>> pch;
  std::vector<std::vector<Index>> plane_groups;
  std::vector<Index> point_indices;
  std::vector<parte::registration::FPFH> fpfh;
  double preparation_seconds = 0.0;
};


struct RegistrationResult
{
  PreparedCloud source;
  PreparedCloud target;
  std::vector<parte::Correspondence> point_matches;
  std::vector<parte::Correspondence> plane_matches;
  std::vector<Scalar> plane_confidences;
  std::vector<parte::registration::wpmc_edge> graph_edges;
  std::vector<Index> clique;
  parte::ScalarMatrix<4, 4> source_to_target =
    parte::ScalarMatrix<4, 4>::Identity();
  double matching_seconds = 0.0;
  double graph_seconds = 0.0;
  double clique_seconds = 0.0;
  double estimation_seconds = 0.0;
};


void print_usage(std::ostream &stream, std::string_view executable)
{
  stream
    << "Usage: " << executable
    << " <source-cloud> <target-cloud> <voxel-size> [options]\n\n"
    << "Register source into target. Close each view to open the next.\n\n"
    << "Options:\n"
    << "  --raw             Voxel-downsample the inputs before registration\n"
    << "  --segment-ground  Use TGS before plane extraction\n"
    << "  -h, --help        Show this message\n\n"
    << "The reported 4x4 matrix transforms the source cloud into the target frame.\n";
}


double parse_positive_double(std::string_view text, std::string_view name)
{
  std::size_t parsed = 0;
  double value = std::stod(std::string(text), &parsed);
  if(parsed != text.size() || !std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return value;
}


Options parse_options(int argc, char **argv)
{
  if(argc == 2 && (std::string_view(argv[1]) == "-h" || std::string_view(argv[1]) == "--help")) {
    print_usage(std::cout, argv[0]);
    std::exit(0);
  }
  if(argc < 4) {
    print_usage(std::cerr, argv[0]);
    throw std::invalid_argument("source, target, and voxel size are required");
  }

  Options result;
  result.source = argv[1];
  result.target = argv[2];
  result.voxel_size = parse_positive_double(argv[3], "voxel size");
  for(int argument = 4; argument < argc; ++argument) {
    std::string_view option = argv[argument];
    if(option == "--segment-ground") {
      result.segment_ground = true;
    } else if(option == "--raw") {
      result.raw = true;
    } else if(option == "-h" || option == "--help") {
      print_usage(std::cout, argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  return result;
}


Scalar radians(Scalar degrees)
{
  return degrees * std::numbers::pi_v<Scalar> / 180.0f;
}


std::vector<Point> read_points(
  const fs::path &path,
  Scalar voxel_size,
  bool raw,
  std::size_t &input_point_count
)
{
  open3d::geometry::PointCloud cloud;
  if(!open3d::io::ReadPointCloud(path.string(), cloud) || cloud.IsEmpty()) {
    throw std::runtime_error("failed to read point cloud: " + path.string());
  }
  input_point_count = cloud.points_.size();
  if(raw) {
    cloud = *cloud.VoxelDownSample(voxel_size);
  }
  std::vector<Point> result;
  result.reserve(cloud.points_.size());
  for(const auto &point : cloud.points_) {
    result.push_back(point.cast<Scalar>());
  }
  return result;
}


std::vector<Neighbors> normal_neighborhoods(
  std::span<const Point> points,
  std::span<const Neighbors> fpfh_neighborhoods,
  Scalar voxel_size
)
{
  std::vector<Neighbors> result(points.size());
  Scalar radius = parameters::NormalRadius * voxel_size;
  Scalar radius_squared = radius * radius;

  #pragma omp parallel for schedule(static)
  for(Index point = 0; point < static_cast<Index>(points.size()); ++point) {
    result[point].reserve(parameters::NormalK);
    for(Index neighbor : fpfh_neighborhoods[point]) {
      if(result[point].size() == parameters::NormalK
        || (points[point] - points[neighbor]).squaredNorm() > radius_squared) {
        break;
      }
      result[point].push_back(neighbor);
    }
  }
  return result;
}


PreparedCloud prepare_cloud(
  const fs::path &path,
  Scalar voxel_size,
  bool segment_ground,
  bool raw
)
{
  const auto start = Clock::now();
  PreparedCloud result;
  result.points = read_points(
    path, voxel_size, raw, result.input_point_count
  );
  auto fpfh_neighborhoods = parte::compute_neighborhoods(
    result.points, parameters::FpfhRadius * voxel_size, parameters::FpfhK
  );
  auto neighborhoods = normal_neighborhoods(
    result.points, fpfh_neighborhoods, voxel_size
  );
  result.normals = parte::compute_normals(result.points, neighborhoods);

  std::vector<Point> non_ground_points;
  std::vector<Normal> non_ground_normals;
  Index ground_plane = -1;
  if(segment_ground) {
    auto ground = parte::segmentation::segment_ground(result.points);
    std::vector<bool> is_ground(result.points.size());
    for(Index point : ground) {
      is_ground[point] = true;
    }

    std::vector<Index> local_to_full;
    std::vector<Index> full_to_local(result.points.size(), -1);
    local_to_full.reserve(result.points.size() - ground.size());
    for(Index point = 0; point < static_cast<Index>(result.points.size()); ++point) {
      if(is_ground[point]) {
        continue;
      }
      full_to_local[point] = static_cast<Index>(local_to_full.size());
      local_to_full.push_back(point);
    }

    auto points = parte::select<Point>(result.points, local_to_full);
    auto normals = parte::select<Normal>(result.normals, local_to_full);

    std::vector<Neighbors> local_neighborhoods(points.size());
    for(Index point = 0; point < static_cast<Index>(points.size()); ++point) {
      for(Index neighbor : neighborhoods[local_to_full[point]]) {
        const Index local = full_to_local[neighbor];
        if(local != -1) {
          local_neighborhoods[point].push_back(local);
        }
      }
    }
    result.plane_supports = parte::segmentation::segment_planes(
      points, normals, local_neighborhoods,
      voxel_size, std::sin(radians(parameters::SegmentationAngle)),
      parameters::MinimumPlaneSupport
    );
    for(auto &support : result.plane_supports) {
      for(Index &point : support) {
        point = local_to_full[point];
      }
    }
    non_ground_points = std::move(points);
    non_ground_normals = std::move(normals);

    if(ground.size() >= 3) {
      const Normal ground_normal = parte::compute_plane(result.points, ground).second;
      for(Index point : ground) {
        result.normals[point] = ground_normal;
      }
      ground_plane = result.plane_supports.size();
      result.plane_supports.push_back(std::move(ground));
    }
  } else {
    result.plane_supports = parte::segmentation::segment_planes(
      result.points, result.normals, neighborhoods,
      voxel_size, std::sin(radians(parameters::SegmentationAngle)),
      parameters::MinimumPlaneSupport
    );
  }

  result.planes.reserve(result.plane_supports.size());
  std::vector<bool> is_plane_point(result.points.size());
  for(const auto &support : result.plane_supports) {
    result.planes.push_back(parte::compute_plane(result.points, support));
    for(Index point : support) {
      is_plane_point[point] = true;
    }
  }

  result.pch.resize(result.planes.size());
  #pragma omp parallel for schedule(static)
  for(Index plane = 0; plane < static_cast<Index>(result.planes.size()); ++plane) {
    const auto &[center, normal] = result.planes[plane];
    std::span<const Point> points = result.points;
    std::span<const Normal> normals = result.normals;
    // Ground supports its own PCH; other planes use only non-ground context.
    if(segment_ground && plane != ground_plane) {
      points = non_ground_points;
      normals = non_ground_normals;
    }
    result.pch[plane] = parte::registration::compute_pch(
      points, normals, normal, center,
      parameters::PchRange * voxel_size, parameters::PchRadius * voxel_size
    );
  }
  result.plane_groups = parte::registration::detail::group_planes(
    result.planes,
    parameters::GroupDistance * voxel_size,
    std::cos(radians(parameters::GroupAngle))
  );

  result.point_indices.reserve(result.points.size());
  for(Index point = 0; point < static_cast<Index>(result.points.size()); ++point) {
    if(!is_plane_point[point]) {
      result.point_indices.push_back(point);
    }
  }
  result.fpfh = parte::registration::compute_fpfh(
    result.points, result.normals, fpfh_neighborhoods, result.point_indices
  );
  result.preparation_seconds = std::chrono::duration<double>(
    Clock::now() - start
  )
                                 .count();
  return result;
}


node_t plane_weight(Scalar confidence)
{
  return 1 + static_cast<node_t>(10.0f * std::clamp(confidence, 0.0f, 1.0f));
}


RegistrationResult register_clouds(const Options &options)
{
  RegistrationResult result;
  result.source = prepare_cloud(
    options.source, options.voxel_size, options.segment_ground,
    options.raw
  );
  result.target = prepare_cloud(
    options.target, options.voxel_size, options.segment_ground,
    options.raw
  );

  auto start = Clock::now();
  result.point_matches = parte::registration::mutual_correspondences<
    parte::registration::FPFH>(result.source.fpfh, result.target.fpfh);
  if(!result.source.pch.empty() && !result.target.pch.empty()) {
    std::tie(result.plane_matches, result.plane_confidences) =
      parte::registration::pch_matching(
        result.source.pch, result.target.pch,
        result.source.plane_groups, result.target.plane_groups
      );
  }
  result.matching_seconds = std::chrono::duration<double>(
    Clock::now() - start
  )
                              .count();

  std::vector<Index> source_indices, target_indices;
  source_indices.reserve(result.point_matches.size());
  target_indices.reserve(result.point_matches.size());
  for(const auto &[source_feature, target_feature] : result.point_matches) {
    source_indices.push_back(result.source.point_indices[source_feature]);
    target_indices.push_back(result.target.point_indices[target_feature]);
  }
  auto source_points = parte::select<Point>(result.source.points, source_indices);
  auto target_points = parte::select<Point>(result.target.points, target_indices);

  source_indices.clear();
  target_indices.clear();
  source_indices.reserve(result.plane_matches.size());
  target_indices.reserve(result.plane_matches.size());
  for(const auto &[source_plane, target_plane] : result.plane_matches) {
    source_indices.push_back(source_plane);
    target_indices.push_back(target_plane);
  }
  auto source_planes = parte::select<Plane>(result.source.planes, source_indices);
  auto target_planes = parte::select<Plane>(result.target.planes, target_indices);

  start = Clock::now();
  result.graph_edges = parte::registration::consistent_correspondences(
    source_points, target_points, source_planes, target_planes,
    options.voxel_size, radians(parameters::RegistrationAngle)
  );
  result.graph_seconds = std::chrono::duration<double>(
    Clock::now() - start
  )
                           .count();

  std::vector<node_t> weights(
    result.plane_matches.size() + result.point_matches.size(), 1
  );
  for(Index plane = 0;
    plane < static_cast<Index>(result.plane_matches.size()); ++plane) {
    weights[plane] = plane_weight(result.plane_confidences[plane]);
  }

  start = Clock::now();
  result.clique = parte::registration::maximum_weight_clique(
    weights.size(), result.graph_edges, weights
  );
  result.clique_seconds = std::chrono::duration<double>(
    Clock::now() - start
  )
                            .count();
  if(result.clique.empty()) {
    throw std::runtime_error("outlier rejection returned an empty clique");
  }

  std::vector<Index> selected_points, selected_planes;
  for(Index node : result.clique) {
    if(node < static_cast<Index>(result.plane_matches.size())) {
      selected_planes.push_back(node);
    } else {
      selected_points.push_back(
        node - static_cast<Index>(result.plane_matches.size())
      );
    }
  }
  auto selected_source_points = parte::select<Point>(source_points, selected_points);
  auto selected_target_points = parte::select<Point>(target_points, selected_points);
  auto selected_source_planes = parte::select<Plane>(source_planes, selected_planes);
  auto selected_target_planes = parte::select<Plane>(target_planes, selected_planes);
  auto clique_weights = parte::select<node_t>(weights, selected_planes);
  std::vector<Scalar> selected_plane_weights(
    clique_weights.begin(), clique_weights.end()
  );
  if(selected_source_points.empty()) {
    throw std::runtime_error(
      "the selected clique contains no point correspondence; "
      "translation estimation is underconstrained"
    );
  }

  start = Clock::now();
  result.source_to_target = parte::registration::compute_transformation(
    selected_source_points, selected_target_points,
    selected_source_planes, selected_target_planes,
    selected_plane_weights
  );
  result.estimation_seconds = std::chrono::duration<double>(
    Clock::now() - start
  )
                                .count();
  return result;
}


const std::array<Color, 12> &palette()
{
  static const std::array<Color, 12> colors = {
    Color{0.80, 0.20, 0.30}, Color{0.10, 0.62, 0.50},
    Color{0.42, 0.32, 0.80}, Color{0.93, 0.60, 0.12},
    Color{0.10, 0.55, 0.78}, Color{0.78, 0.32, 0.68},
    Color{0.52, 0.68, 0.18}, Color{0.18, 0.70, 0.72},
    Color{0.65, 0.42, 0.18}, Color{0.48, 0.48, 0.82},
    Color{0.88, 0.38, 0.18}, Color{0.25, 0.66, 0.32}};
  return colors;
}


Eigen::Matrix4d translation_matrix(const Eigen::Vector3d &translation)
{
  Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
  result.block<3, 1>(0, 3) = translation;
  return result;
}


std::pair<Eigen::Matrix4d, Eigen::Matrix4d> side_by_side_transforms(
  const PreparedCloud &source,
  const PreparedCloud &target,
  Scalar voxel_size
)
{
  auto bounds = [](std::span<const Point> points) {
    Eigen::Vector3d minimum = points.front().cast<double>();
    Eigen::Vector3d maximum = minimum;
    for(const Point &point : points) {
      minimum = minimum.cwiseMin(point.cast<double>());
      maximum = maximum.cwiseMax(point.cast<double>());
    }
    return std::pair{minimum, maximum};
  };
  const auto [source_min, source_max] = bounds(source.points);
  const auto [target_min, target_max] = bounds(target.points);
  const Eigen::Vector3d source_center = 0.5 * (source_min + source_max);
  const Eigen::Vector3d target_center = 0.5 * (target_min + target_max);
  const double scale = std::max(
    (source_max - source_min).norm(), (target_max - target_min).norm()
  );
  const double gap = std::max(2.0 * static_cast<double>(voxel_size), 0.06 * scale);

  Eigen::Vector3d target_shift = -target_center;
  Eigen::Vector3d source_shift = -source_center;
  target_shift.x() += -0.5 * gap - 0.5 * (target_max.x() - target_min.x());
  source_shift.x() += 0.5 * gap + 0.5 * (source_max.x() - source_min.x());
  return {
    translation_matrix(source_shift),
    translation_matrix(target_shift)};
}


Eigen::Vector3d transformed_point(
  const Point &point,
  const Eigen::Matrix4d &transformation
)
{
  return transformation.block<3, 3>(0, 0) * point.cast<double>()
    + transformation.block<3, 1>(0, 3);
}


std::shared_ptr<open3d::geometry::PointCloud> make_cloud(
  const PreparedCloud &cloud,
  const Eigen::Matrix4d &transformation,
  const Color &base_color,
  const std::vector<Color> *colors = nullptr
)
{
  auto result = std::make_shared<open3d::geometry::PointCloud>();
  result->points_.reserve(cloud.points.size());
  result->colors_.reserve(cloud.points.size());
  for(Index point = 0; point < static_cast<Index>(cloud.points.size()); ++point) {
    result->points_.push_back(transformed_point(cloud.points[point], transformation));
    result->colors_.push_back(colors ? (*colors)[point] : base_color);
  }
  return result;
}


std::vector<Color> segmentation_colors(
  const PreparedCloud &cloud,
  const Color &base_color,
  std::size_t palette_offset = 0
)
{
  std::vector<Color> colors(cloud.points.size(), base_color);
  for(Index plane = 0;
    plane < static_cast<Index>(cloud.plane_supports.size()); ++plane) {
    const Color &color = palette()[(plane + palette_offset) % palette().size()];
    for(Index point : cloud.plane_supports[plane]) {
      colors[point] = color;
    }
  }
  return colors;
}


std::pair<std::vector<Color>, std::vector<Color>> matched_plane_colors(
  const RegistrationResult &result,
  const Color &source_base,
  const Color &target_base,
  const std::vector<bool> *selected = nullptr
)
{
  std::vector<Color> source_colors(result.source.points.size(), source_base);
  std::vector<Color> target_colors(result.target.points.size(), target_base);
  for(Index match = 0;
    match < static_cast<Index>(result.plane_matches.size()); ++match) {
    if(selected && !(*selected)[match]) {
      continue;
    }
    const auto &[source_plane, target_plane] = result.plane_matches[match];
    const Color &color = palette()[match % palette().size()];
    for(Index point : result.source.plane_supports[source_plane]) {
      source_colors[point] = color;
    }
    for(Index point : result.target.plane_supports[target_plane]) {
      target_colors[point] = color;
    }
  }
  return {std::move(source_colors), std::move(target_colors)};
}


void append_line(
  open3d::geometry::LineSet &lines,
  const Eigen::Vector3d &first,
  const Eigen::Vector3d &second,
  const Color &color
)
{
  const int start = static_cast<int>(lines.points_.size());
  lines.points_.push_back(first);
  lines.points_.push_back(second);
  lines.lines_.emplace_back(start, start + 1);
  lines.colors_.push_back(color);
}


std::shared_ptr<open3d::geometry::LineSet> point_match_lines(
  const RegistrationResult &result,
  const Eigen::Matrix4d &source_transform,
  const Eigen::Matrix4d &target_transform,
  const std::vector<bool> *selected = nullptr
)
{
  auto lines = std::make_shared<open3d::geometry::LineSet>();
  for(std::size_t match = 0; match < result.point_matches.size(); ++match) {
    const auto &[source_feature, target_feature] = result.point_matches[match];
    const std::size_t node = result.plane_matches.size() + match;
    const bool kept = !selected || (*selected)[node];
    append_line(
      *lines,
      transformed_point(
        result.source.points[result.source.point_indices[source_feature]],
        source_transform
      ),
      transformed_point(
        result.target.points[result.target.point_indices[target_feature]],
        target_transform
      ),
      selected ? (kept ? Color{0.05, 0.65, 0.25} : Color{0.85, 0.72, 0.72})
               : Color{0.18, 0.62, 0.30}
    );
  }
  return lines;
}


std::shared_ptr<open3d::geometry::LineSet> plane_match_lines(
  const RegistrationResult &result,
  const Eigen::Matrix4d &source_transform,
  const Eigen::Matrix4d &target_transform,
  const std::vector<bool> *selected = nullptr
)
{
  auto lines = std::make_shared<open3d::geometry::LineSet>();
  for(Index match = 0;
    match < static_cast<Index>(result.plane_matches.size()); ++match) {
    const bool kept = !selected || (*selected)[match];
    const auto &[source_plane, target_plane] = result.plane_matches[match];
    append_line(
      *lines,
      transformed_point(result.source.planes[source_plane].first, source_transform),
      transformed_point(result.target.planes[target_plane].first, target_transform),
      selected ? (kept ? Color{0.68, 0.16, 0.72} : Color{0.88, 0.72, 0.80})
               : Color{0.68, 0.16, 0.72}
    );
  }
  return lines;
}


void show_view(
  const std::string &name,
  const std::vector<std::shared_ptr<const open3d::geometry::Geometry>> &geometries
)
{
  std::cout << "View: " << name << " (close to continue)\n";
  open3d::visualization::Visualizer visualizer;
  if(!visualizer.CreateVisualizerWindow(name, 1440, 900)) {
    throw std::runtime_error("failed to create the Open3D visualizer window");
  }
  auto &render = visualizer.GetRenderOption();
  render.background_color_ = Color{0.97, 0.975, 0.98};
  render.point_size_ = 3.0;
  render.line_width_ = 2.0;
  render.light_on_ = false;
  for(const auto &geometry : geometries) {
    if(!geometry->IsEmpty())
      visualizer.AddGeometry(geometry);
  }
  visualizer.GetViewControl().SetZoom(0.72);
  visualizer.Run();
  visualizer.DestroyVisualizerWindow();
}


void visualize(const RegistrationResult &result, const Options &options)
{
  const Color source_color{0.92, 0.48, 0.12};
  const Color target_color{0.10, 0.47, 0.78};
  const Color muted_source{0.78, 0.72, 0.66};
  const Color muted_target{0.67, 0.73, 0.78};
  const auto [source_side, target_side] = side_by_side_transforms(
    result.source, result.target, options.voxel_size
  );

  std::vector<bool> selected(
    result.plane_matches.size() + result.point_matches.size(), false
  );
  for(Index node : result.clique) {
    selected[node] = true;
  }

  show_view("Input scans", {
    make_cloud(result.source, source_side, source_color),
    make_cloud(result.target, target_side, target_color)
  });

  auto source_segmentation = segmentation_colors(
    result.source, Color{0.76, 0.76, 0.76}
  );
  auto target_segmentation = segmentation_colors(
    result.target, Color{0.76, 0.76, 0.76}, 5
  );
  show_view("Plane extraction", {
    make_cloud(result.source, source_side, source_color, &source_segmentation),
    make_cloud(result.target, target_side, target_color, &target_segmentation)
  });

  show_view("Point matching", {
    make_cloud(result.source, source_side, muted_source),
    make_cloud(result.target, target_side, muted_target),
    point_match_lines(result, source_side, target_side)
  });

  auto [source_matched, target_matched] = matched_plane_colors(
    result, muted_source, muted_target
  );
  show_view("Plane matching", {
    make_cloud(result.source, source_side, muted_source, &source_matched),
    make_cloud(result.target, target_side, muted_target, &target_matched),
    plane_match_lines(result, source_side, target_side)
  });

  auto [source_selected, target_selected] = matched_plane_colors(
    result, muted_source, muted_target, &selected
  );
  show_view("Joint outlier rejection: selected clique", {
    make_cloud(result.source, source_side, muted_source, &source_selected),
    make_cloud(result.target, target_side, muted_target, &target_selected),
    point_match_lines(result, source_side, target_side, &selected),
    plane_match_lines(result, source_side, target_side, &selected)
  });

  Eigen::Matrix4d source_aligned = result.source_to_target.cast<double>();
  Eigen::Vector3d combined_center = Eigen::Vector3d::Zero();
  std::size_t combined_size = 0;
  for(const Point &point : result.source.points) {
    combined_center += transformed_point(point, source_aligned);
    ++combined_size;
  }
  for(const Point &point : result.target.points) {
    combined_center += point.cast<double>();
    ++combined_size;
  }
  combined_center /= static_cast<double>(combined_size);
  const Eigen::Matrix4d center = translation_matrix(-combined_center);
  source_aligned = center * source_aligned;

  auto [source_final, target_final] = matched_plane_colors(
    result, source_color, target_color, &selected
  );
  show_view("Joint alignment", {
    make_cloud(result.source, source_aligned, source_color, &source_final),
    make_cloud(result.target, center, target_color, &target_final)
  });
}


void print_result(const RegistrationResult &result, const Options &options)
{
  std::size_t selected_planes = 0;
  for(Index node : result.clique) {
    selected_planes += node < static_cast<Index>(result.plane_matches.size());
  }
  const std::size_t selected_points = result.clique.size() - selected_planes;
  const double total = result.source.preparation_seconds
    + result.target.preparation_seconds
    + result.matching_seconds
    + result.graph_seconds
    + result.clique_seconds
    + result.estimation_seconds;

  std::cout << std::fixed << std::setprecision(3)
            << "\nPARTE registration\n"
            << "  voxel size: " << options.voxel_size << " m\n"
            << "  source points: " << result.source.input_point_count << " -> "
            << result.source.points.size() << '\n'
            << "  target points: " << result.target.input_point_count << " -> "
            << result.target.points.size() << '\n'
            << "  source/target planes: " << result.source.planes.size() << " / "
            << result.target.planes.size() << '\n'
            << "  point/plane proposals: " << result.point_matches.size() << " / "
            << result.plane_matches.size() << '\n'
            << "  consistency graph: "
            << result.point_matches.size() + result.plane_matches.size()
            << " nodes, " << result.graph_edges.size() << " edges\n"
            << "  selected point/plane correspondences: " << selected_points
            << " / " << selected_planes << "\n\n"
            << "Timing\n"
            << "  source preparation: "
            << 1000.0 * result.source.preparation_seconds << " ms\n"
            << "  target preparation: "
            << 1000.0 * result.target.preparation_seconds << " ms\n"
            << "  matching: " << 1000.0 * result.matching_seconds << " ms\n"
            << "  graph construction: " << 1000.0 * result.graph_seconds << " ms\n"
            << "  maximum weighted clique: "
            << 1000.0 * result.clique_seconds << " ms\n"
            << "  transformation estimation: "
            << 1000.0 * result.estimation_seconds << " ms\n"
            << "  total excluding visualization: " << 1000.0 * total << " ms\n\n"
            << "Source-to-target transformation\n"
            << std::setprecision(9) << result.source_to_target << "\n";
}

}  // namespace


int main(int argc, char **argv)
{
  try {
    const Options options = parse_options(argc, argv);
    RegistrationResult result = register_clouds(options);
    print_result(result, options);
    visualize(result, options);
    return 0;
  } catch(const std::exception &error) {
    std::cerr << "parte: " << error.what() << '\n';
    return 1;
  }
}
