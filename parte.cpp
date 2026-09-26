#include "common/logger.h"
#include "registration/pipeline.h"

#include <cmath>
#include <format>
#include <iostream>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include <open3d/Open3D.h>


const Eigen::Vector3d orange{0.92, 0.48, 0.12};
const Eigen::Vector3d blue{0.10, 0.47, 0.78};

const Eigen::Vector3d muted_orange{0.78, 0.72, 0.66};
const Eigen::Vector3d muted_blue{0.67, 0.73, 0.78};

const Eigen::Vector3d green{0.18, 0.62, 0.30};
const Eigen::Vector3d red{0.85, 0.72, 0.72};

Eigen::Vector3d palette(std::size_t index)
{
  const double hue = index * 0.618033988749895;
  Eigen::Vector3d color;
  for(int channel = 0; channel < 3; ++channel) {
    color[channel] = 0.55 + 0.4 * std::cos(2 * std::numbers::pi * (hue + channel / 3.0));
  }
  return color;
}


void show_view(
  const std::string &name,
  const std::vector<std::shared_ptr<open3d::geometry::Geometry>> &geometries
)
{
  static std::optional<open3d::visualization::ViewParameters> camera;
  std::cout << "View: " << name << " (close to continue)\n";
  open3d::visualization::Visualizer visualizer;
  if(!visualizer.CreateVisualizerWindow(name, 1440, 900)) {
    throw std::runtime_error("failed to create the Open3D visualizer window");
  }
  for(const auto &geometry : geometries) {
    visualizer.AddGeometry(geometry);
  }

  auto &view = visualizer.GetViewControl();
  if(camera) {
    view.ConvertFromViewParameters(*camera);
  }
  visualizer.Run();
  view.ConvertToViewParameters(camera.emplace());
  visualizer.DestroyVisualizerWindow();
}

void visualize(
  const std::shared_ptr<open3d::geometry::PointCloud> &source_cloud,
  const std::shared_ptr<open3d::geometry::PointCloud> &target_cloud,
  const parte::ProcessedCloud &source,
  const parte::ProcessedCloud &target,
  const parte::RegistrationResult &registration
)
{
  auto color_points = [](auto &cloud, const auto &points, const Eigen::Vector3d &color) {
    for(auto point : points) {
      cloud.colors_[point] = color;
    }
  };

  source_cloud->PaintUniformColor(orange);
  target_cloud->PaintUniformColor(blue);

  const auto source_bounds = source_cloud->GetAxisAlignedBoundingBox();
  const auto target_bounds = target_cloud->GetAxisAlignedBoundingBox();
  const double gap = 0.05 * std::max(source_bounds.GetExtent().norm(), target_bounds.GetExtent().norm());
  Eigen::Vector3d source_offset = -source_cloud->GetCenter();
  Eigen::Vector3d target_offset = -target_cloud->GetCenter();
  source_offset.x() = gap / 2 - source_bounds.min_bound_.x();
  target_offset.x() = -gap / 2 - target_bounds.max_bound_.x();
  source_cloud->Translate(source_offset);
  target_cloud->Translate(target_offset);
  show_view("Input scans", {source_cloud, target_cloud});

  source_cloud->PaintUniformColor(muted_orange);
  target_cloud->PaintUniformColor(muted_blue);
  for(std::size_t plane = 0; plane < source.planes.size(); ++plane) {
    color_points(*source_cloud, source.plane_supports[plane], palette(plane));
  }

  for(std::size_t plane = 0; plane < target.planes.size(); ++plane) {
    color_points(*target_cloud, target.plane_supports[plane], palette(plane + 5));
  }
  show_view("Plane extraction", {source_cloud, target_cloud});

  auto point_lines = std::make_shared<open3d::geometry::LineSet>();
  for(const auto &[s, t] : registration.point_matches) {
    const int start = point_lines->points_.size();
    point_lines->points_.push_back(source_cloud->points_[source.non_planar[s]]);
    point_lines->points_.push_back(target_cloud->points_[target.non_planar[t]]);
    point_lines->lines_.emplace_back(start, start + 1);
  }
  point_lines->PaintUniformColor(red);
  show_view("Point matching", {source_cloud, target_cloud, point_lines});

  source_cloud->PaintUniformColor(muted_orange);
  target_cloud->PaintUniformColor(muted_blue);
  for(auto match : registration.selected_planes) {
    const auto &[s, t] = registration.plane_matches[match];
    const auto color = palette(match);
    color_points(*source_cloud, source.plane_supports[s], color);
    color_points(*target_cloud, target.plane_supports[t], color);
  }
  for(auto match : registration.selected_points) {
    point_lines->colors_[match] = green;
  }
  show_view("Joint outlier rejection", {source_cloud, target_cloud, point_lines});

  for(auto match : registration.selected_points) {
    const auto &[s, t] = registration.point_matches[match];
    source_cloud->colors_[source.non_planar[s]] = orange;
    target_cloud->colors_[target.non_planar[t]] = blue;
  }
  source_cloud->Translate(-source_offset);
  target_cloud->Translate(-target_offset);
  source_cloud->Transform(registration.transformation.cast<double>());
  show_view("Joint alignment", {source_cloud, target_cloud});
}

int main(int argc, char **argv)
{

  if(argc < 4) {
    std::cerr << "Usage: " 
      << argv[0] << " <source-cloud> <target-cloud> <voxel-size> [options]\n"
      << "Options:\n"
      << "  --raw             Voxel-downsample the inputs before registration\n"
      << "  --segment-ground  Use TGS to exclude ground from plane segmentation\n";
    return 1;
  }

  bool downsample = false;
  parte::Parameters parameters(std::stod(argv[3]));

  for(int argument = 4; argument < argc; ++argument) {
    std::string option = argv[argument];
    if(option == "--segment-ground") {
      parameters.ground_segmentation = true;
    } else if(option == "--raw") {
      downsample = true;
    } else {
      std::cerr << "Warning: unknown option: " << option << '\n';
      return 1;
    }
  }

  auto source_cloud = std::make_shared<open3d::geometry::PointCloud>();
  auto target_cloud = std::make_shared<open3d::geometry::PointCloud>();
  if(!open3d::io::ReadPointCloud(argv[1], *source_cloud)) {
    throw std::runtime_error("failed to read point cloud: " + std::string(argv[1]));
  }

  if(!open3d::io::ReadPointCloud(argv[2], *target_cloud)) {
    throw std::runtime_error("failed to read point cloud: " + std::string(argv[2]));
  }

  if(downsample) {
    source_cloud = source_cloud->VoxelDownSample(parameters.voxel_size);
    target_cloud = target_cloud->VoxelDownSample(parameters.voxel_size);
  }

  parte::logger.start("Total");
  auto source = parte::logger.time("Processing source cloud", [&] {
    return parte::process_cloud(source_cloud->points_, parameters);
  });

  auto target = parte::logger.time("Processing target cloud", [&] {
    return parte::process_cloud(target_cloud->points_, parameters);
  });
  
  auto registration = parte::logger.time("Registration", [&] {
    return parte::register_clouds(source, target, parameters);
  });
  parte::logger.stop("Total");

  std::cout << std::format(
    "\nPARTE registration\n"
    "  Source: {} with {} input points, {} planes and {} non-planar points\n"
    "  Target: {} with {} input points, {} planes and {} non-planar points\n"
    "  Matched: {} pts - {} planes\n"
    "  Selected: {} pts - {} planes\n\n",
    argv[1], source.points.size(), source.planes.size(), source.non_planar.size(),
    argv[2], target.points.size(), target.planes.size(), target.non_planar.size(),
    registration.point_matches.size(), registration.plane_matches.size(),
    registration.selected_points.size(), registration.selected_planes.size()
  );

  visualize(source_cloud, target_cloud, source, target, registration);
  return 0;
}
