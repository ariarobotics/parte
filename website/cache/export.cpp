#include "common/logger.h"
#include "registration/pipeline.h"

#include <open3d/Open3D.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>


template<typename T>
std::string json_array(const std::vector<T> &values)
{
  std::ostringstream s;
  s << std::setprecision(17) << '[';
  for(std::size_t i = 0; i < values.size(); ++i) {
    if(i) {
      s << ',';
    }
    s << +values[i];
  }
  return s.str() + ']';
}

struct Buffer
{
  std::vector<char> bytes;
  template<typename T>
  void add(const std::vector<T> &values)
  {
    if(values.empty()) {
      return;
    }
    const auto *begin = reinterpret_cast<const char *>(values.data());
    bytes.insert(bytes.end(), begin, begin + values.size() * sizeof(T));
  }
  void align()
  {
    while(bytes.size() % 4) {
      bytes.push_back(0);
    }
  }
  void write(const std::filesystem::path &path) const
  {
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), bytes.size());
  }
};

void export_scan(const parte::ProcessedCloud &scan, const std::filesystem::path &path)
{
  auto indices = scan.non_planar;
  std::vector<std::uint8_t> original_labels(scan.points.size());
  const auto rows = scan.planes.size() + 1;
  std::vector<float> geometry(rows * 6);
  for(std::size_t plane = 0; plane < scan.planes.size(); ++plane) {
    const auto label = plane + 1;
    for(auto point : scan.plane_supports[plane]) {
      original_labels[point] = label;
    }
    const auto &[center, normal] = scan.planes[plane];
    for(int axis = 0; axis < 3; ++axis) {
      geometry[label * 6 + axis] = center[axis];
      geometry[label * 6 + 3 + axis] = normal[axis];
    }
  }
  std::vector<std::uint8_t> labels;
  for(std::size_t point = 0; point < scan.points.size(); ++point) {
    if(original_labels[point]) {
      indices.push_back(point);
      labels.push_back(original_labels[point]);
    }
  }

  std::vector<double> bounds(6);
  for(int axis = 0; axis < 3; ++axis) {
    auto lo = scan.points[0][axis], hi = lo;
    for(const auto &point : scan.points) {
      lo = std::min(lo, point[axis]);
      hi = std::max(hi, point[axis]);
    }
    bounds[axis] = lo;
    bounds[axis + 3] = (double(hi) - lo) / 65535.0;
  }
  std::vector<std::uint16_t> points;
  points.reserve(indices.size() * 3);
  for(auto point : indices) {
    for(int axis = 0; axis < 3; ++axis) {
      const auto scale = bounds[axis + 3];
      const auto q = scale == 0 ? 0 : (scan.points[point][axis] - bounds[axis]) / scale;
      points.push_back(std::clamp(std::floor(q + 0.5), 0.0, 65535.0));
    }
  }
  Buffer buffer;
  std::vector<std::uint32_t> header(3);
  header[0] = indices.size();
  header[1] = scan.non_planar.size();
  header[2] = rows;
  buffer.add(header);
  buffer.add(bounds);
  buffer.add(points);
  buffer.add(labels);
  buffer.align();
  buffer.add(geometry);
  buffer.write(path);
}

extern "C" void export_pair(
  const char *source_path, const char *target_path,
  float voxel_size, bool segment_ground, const char *output_path
)
{
  parte::Parameters parameters(voxel_size);
  parameters.ground_segmentation = segment_ground;
  parte::logger.set_output(nullptr);

  parte::logger.start("Loading");
  open3d::geometry::PointCloud source_cloud, target_cloud;
  open3d::io::ReadPointCloud(source_path, source_cloud);
  open3d::io::ReadPointCloud(target_path, target_cloud);
  const auto loading_ms = 1000 * parte::logger.stop();

  parte::logger.start("Registration");
  const auto source = parte::process_cloud(std::move(source_cloud.points_), parameters);
  const auto target = parte::process_cloud(std::move(target_cloud.points_), parameters);
  const auto registration = parte::register_clouds(source, target, parameters);
  const auto registration_ms = 1000 * parte::logger.stop();

  const std::filesystem::path output = output_path;
  std::filesystem::create_directories(output);
  export_scan(source, output / "source.bin");
  export_scan(target, output / "target.bin");
  std::vector<std::uint32_t> points, planes;
  for(const auto &[s, t] : registration.point_matches) {
    points.push_back(s | (std::uint32_t(t) << 16));
  }
  for(const auto &[s, t] : registration.plane_matches) {
    planes.push_back((s + 1) | (std::uint32_t(t + 1) << 16));
  }
  std::vector<std::uint16_t> selected_points(registration.selected_points.begin(), registration.selected_points.end());
  std::vector<std::uint16_t> selected_planes(registration.selected_planes.begin(), registration.selected_planes.end());
  std::sort(selected_points.begin(), selected_points.end());
  std::sort(selected_planes.begin(), selected_planes.end());
  Buffer buffer;
  std::vector<std::uint32_t> header(4);
  header[0] = points.size();
  header[1] = selected_points.size();
  header[2] = planes.size();
  header[3] = selected_planes.size();
  buffer.add(header);
  buffer.add(points);
  buffer.add(selected_points);
  buffer.align();
  buffer.add(planes);
  buffer.add(selected_planes);
  buffer.write(output / "matches.bin");

  std::vector<double> matrix;
  for(int row = 0; row < 4; ++row) {
    for(int column = 0; column < 4; ++column) {
      matrix.push_back(registration.transformation(row, column));
    }
  }
  std::ofstream json(output / "trace.json");
  json << std::setprecision(10) << "{\"transform\":" << json_array(matrix)
       << ",\"registrationRuntimeMs\":" << registration_ms << ",\"loadingRuntimeMs\":" << loading_ms << "}\n";
}
