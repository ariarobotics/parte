
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
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <omp.h>

namespace
{
namespace pipeline
{
using namespace parte;
namespace parameters = parte::parameters;

struct PreparedCloud
{
  std::vector<Point> points;
  std::vector<Normal> normals;
  std::vector<Indices> supports;
  Index ground_plane = -1;
  std::vector<Plane> planes;
  std::vector<std::pair<parte::registration::PCH, parte::registration::PCH>> pch;
  std::vector<std::vector<Index>> plane_groups;
  std::vector<Index> point_indices;
  std::vector<parte::registration::FPFH> fpfh;
};


Scalar radians(Scalar degrees)
{
  return degrees * std::numbers::pi_v<Scalar> / 180.0f;
}


struct RegistrationTrace
{
  PreparedCloud source, target;
  std::vector<Correspondence> point_matches, plane_matches;
  std::vector<Scalar> plane_confidences;
  std::vector<Index> selected_points, selected_planes;
};

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
  for(Index point = 0; point < points.size(); ++point) {
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
  std::vector<Point> points,
  Scalar voxel_size,
  bool segment_ground
)
{
  PreparedCloud result;
  result.points = std::move(points);
  auto fpfh_neighborhoods = parte::compute_neighborhoods(
    result.points, parameters::FpfhRadius * voxel_size, parameters::FpfhK
  );
  auto neighborhoods = normal_neighborhoods(
    result.points, fpfh_neighborhoods, voxel_size
  );
  result.normals = parte::compute_normals(result.points, neighborhoods);

  std::vector<parte::Indices> supports;
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
    for(Index point = 0; point < result.points.size(); ++point) {
      if(is_ground[point]) {
        continue;
      }
      full_to_local[point] = local_to_full.size();
      local_to_full.push_back(point);
    }

    auto points = parte::select<Point>(result.points, local_to_full);
    auto normals = parte::select<Normal>(result.normals, local_to_full);

    std::vector<Neighbors> local_neighborhoods(points.size());
    for(Index point = 0; point < points.size(); ++point) {
      for(Index neighbor : neighborhoods[local_to_full[point]]) {
        Index local = full_to_local[neighbor];
        if(local != -1) {
          local_neighborhoods[point].push_back(local);
        }
      }
    }
    supports = parte::segmentation::segment_planes(
      points, normals, local_neighborhoods,
      voxel_size, std::sin(radians(parameters::SegmentationAngle)),
      parameters::MinimumPlaneSupport
    );
    for(auto &support : supports) {
      for(Index &point : support) {
        point = local_to_full[point];
      }
    }
    non_ground_points = std::move(points);
    non_ground_normals = std::move(normals);

    if(ground.size() >= 3) {
      Normal ground_normal = parte::compute_plane(result.points, ground).second;
      for(Index point : ground) {
        result.normals[point] = ground_normal;
      }
      ground_plane = supports.size();
      supports.push_back(std::move(ground));
    }
  } else {
    supports = parte::segmentation::segment_planes(
      result.points, result.normals, neighborhoods,
      voxel_size, std::sin(radians(parameters::SegmentationAngle)),
      parameters::MinimumPlaneSupport
    );
  }

  result.planes.reserve(supports.size());
  std::vector<bool> is_plane_point(result.points.size());
  for(const auto &support : supports) {
    result.planes.push_back(parte::compute_plane(result.points, support));
    for(Index point : support) {
      is_plane_point[point] = true;
    }
  }

  result.pch.resize(result.planes.size());
  #pragma omp parallel for schedule(static)
  for(Index plane = 0; plane < result.planes.size(); ++plane) {
    auto [center, normal] = result.planes[plane];
    std::span<const Point> points = result.points;
    std::span<const Normal> normals = result.normals;
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
  for(Index point = 0; point < result.points.size(); ++point) {
    if(!is_plane_point[point]) {
      result.point_indices.push_back(point);
    }
  }
  result.ground_plane = ground_plane;
  result.supports = std::move(supports);
  result.fpfh = parte::registration::compute_fpfh(
    result.points, result.normals, fpfh_neighborhoods, result.point_indices
  );
  return result;
}


Index plane_weight(Scalar confidence)
{
  return 1 + static_cast<Index>(10.0f * std::clamp(confidence, 0.0f, 1.0f));
}


Eigen::Matrix4d register_clouds(
  std::vector<Point> source_cloud,
  std::vector<Point> target_cloud,
  Scalar voxel_size,
  bool segment_ground,
  RegistrationTrace &result
)
{
  result = RegistrationTrace{};
  auto &source = result.source;
  auto &target = result.target;
  source = prepare_cloud(
    std::move(source_cloud), voxel_size, segment_ground
  );
  target = prepare_cloud(
    std::move(target_cloud), voxel_size, segment_ground
  );

  auto &point_matches = result.point_matches;
  point_matches = parte::registration::mutual_correspondences<
    parte::registration::FPFH>(source.fpfh, target.fpfh);
  auto &plane_matches = result.plane_matches;
  auto &plane_confidences = result.plane_confidences;
  if(!source.pch.empty() && !target.pch.empty()) {
    std::tie(plane_matches, plane_confidences) =
      parte::registration::pch_matching(
        source.pch, target.pch, source.plane_groups, target.plane_groups
      );
  }

  std::vector<Index> source_indices, target_indices;
  source_indices.reserve(point_matches.size());
  target_indices.reserve(point_matches.size());
  for(auto [source_feature, target_feature] : point_matches) {
    source_indices.push_back(source.point_indices[source_feature]);
    target_indices.push_back(target.point_indices[target_feature]);
  }
  auto source_points = parte::select<Point>(source.points, source_indices);
  auto target_points = parte::select<Point>(target.points, target_indices);

  source_indices.clear();
  target_indices.clear();
  source_indices.reserve(plane_matches.size());
  target_indices.reserve(plane_matches.size());
  for(auto [source_plane, target_plane] : plane_matches) {
    source_indices.push_back(source_plane);
    target_indices.push_back(target_plane);
  }
  auto source_planes = parte::select<Plane>(source.planes, source_indices);
  auto target_planes = parte::select<Plane>(target.planes, target_indices);

  auto edges = parte::registration::consistent_correspondences(
    source_points, target_points, source_planes, target_planes,
    voxel_size, radians(parameters::RegistrationAngle)
  );
  std::vector<Index> weights(
    plane_matches.size() + point_matches.size(), 1
  );
  for(Index plane = 0; plane < plane_matches.size(); ++plane) {
    weights[plane] = plane_weight(plane_confidences[plane]);
  }

  auto clique = parte::registration::maximum_weight_clique(
    weights.size(), edges, weights
  );
  if(clique.empty()) {
    throw std::runtime_error("no correspondences");
  }

  auto &selected_points = result.selected_points;
  auto &selected_planes = result.selected_planes;
  for(Index node : clique) {
    if(node < plane_matches.size()) {
      selected_planes.push_back(node);
    } else {
      selected_points.push_back(node - plane_matches.size());
    }
  }
  auto selected_source_points = parte::select<Point>(source_points, selected_points);
  auto selected_target_points = parte::select<Point>(target_points, selected_points);
  auto selected_source_planes = parte::select<Plane>(source_planes, selected_planes);
  auto selected_target_planes = parte::select<Plane>(target_planes, selected_planes);
  auto clique_weights = parte::select<Index>(weights, selected_planes);
  std::vector<Scalar> selected_plane_weights(
    clique_weights.begin(), clique_weights.end()
  );

  return parte::registration::compute_transformation(
    selected_source_points, selected_target_points,
    selected_source_planes, selected_target_planes,
    selected_plane_weights
  )
    .cast<double>();
}


}  // namespace pipeline
}  // namespace

namespace fs = std::filesystem;
using namespace parte;
using Clock = std::chrono::steady_clock;

std::vector<Point> read_ply(const fs::path &path)
{
  std::ifstream file(path, std::ios::binary);
  if(!file)
    throw std::runtime_error("cannot open input PLY");
  std::string line, format;
  auto next = [&] { std::getline(file, line); if(!line.empty() && line.back() == '\r') line.pop_back(); };
  next();
  if(line != "ply")
    throw std::runtime_error("expected PLY");
  std::size_t count = 0;
  std::vector<std::pair<std::string, std::string>> properties;
  bool ended = false;
  while(file && !ended) {
    next();
    std::istringstream fields(line);
    std::string key;
    fields >> key;
    if(key == "format") {
      std::string version;
      fields >> format >> version;
      if(version != "1.0")
        throw std::runtime_error("unsupported PLY version");
    } else if(key == "element") {
      std::string name;
      fields >> name >> count;
      if(name != "vertex")
        throw std::runtime_error("expected vertex-only PLY");
    } else if(key == "property") {
      std::string type, name;
      fields >> type >> name;
      properties.emplace_back(type, name);
    } else if(key == "end_header")
      ended = true;
    else if(key != "comment" && key != "obj_info" && !key.empty())
      throw std::runtime_error("unexpected PLY header");
  }
  if(!ended || !count || count > std::numeric_limits<Index>::max() || properties.size() != 3)
    throw std::runtime_error("expected nonempty XYZ PLY");
  const std::string names[] = {"x", "y", "z"};
  for(int i = 0; i < 3; ++i)
    if(properties[i].second != names[i] || (properties[i].first != "float" && properties[i].first != "double"))
      throw std::runtime_error("expected float/double XYZ properties");
  if(format != "ascii" && format != "binary_little_endian")
    throw std::runtime_error("unsupported PLY encoding");
  std::vector<Point> points(count);
  for(auto &point : points)
    for(int axis = 0; axis < 3; ++axis) {
      double value;
      if(format == "ascii")
        file >> value;
      else if(properties[axis].first == "double")
        file.read(reinterpret_cast<char *>(&value), sizeof(value));
      else {
        float v;
        file.read(reinterpret_cast<char *>(&v), sizeof(v));
        value = v;
      }
      if(!file || !std::isfinite(value))
        throw std::runtime_error("invalid/truncated PLY data");
      point[axis] = static_cast<Scalar>(value);
      if(!std::isfinite(point[axis]))
        throw std::runtime_error("PLY point exceeds float range");
    }
  return points;
}

std::string quote(const std::string &value)
{
  std::ostringstream out;
  out << '"';
  for(unsigned char c : value) {
    if(c == '"' || c == '\\')
      out << '\\' << c;
    else if(c < 32)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else
      out << c;
  }
  return out.str() + '"';
}

template <typename T>
std::string json_array(const std::vector<T> &values)
{
  std::ostringstream s;
  s << std::setprecision(17) << '[';
  for(std::size_t i = 0; i < values.size(); ++i) {
    if(i)
      s << ',';
    s << +values[i];
  }
  return s.str() + ']';
}

struct Buffer
{
  std::vector<char> bytes;
  template <typename T>
  void add(const std::vector<T> &values)
  {
    if(values.empty())
      return;
    const auto *begin = reinterpret_cast<const char *>(values.data());
    bytes.insert(bytes.end(), begin, begin + values.size() * sizeof(T));
  }
  void align()
  {
    while(bytes.size() % 4)
      bytes.push_back(0);
  }
  void write(const fs::path &path) const
  {
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), bytes.size());
    if(!out)
      throw std::runtime_error("failed to write binary cache");
  }
};

struct ScanOrder
{
  std::vector<Index> points;
  std::vector<std::uint16_t> planes;
};

ScanOrder export_scan(const pipeline::PreparedCloud &scan, const fs::path &path)
{
  if(scan.points.empty() || scan.points.size() > UINT32_MAX || scan.point_indices.size() > 65536)
    throw std::runtime_error("scan exceeds compact point index limits");
  const bool ground = scan.ground_plane >= 0;
  if(scan.planes.size() - int(ground) > 255)
    throw std::runtime_error("scan exceeds uint8 plane labels");
  ScanOrder order;
  order.points.assign(scan.points.size(), -1);
  order.planes.resize(scan.planes.size());
  std::vector<Index> indices = scan.point_indices;
  for(Index i = 0; i < indices.size(); ++i)
    order.points.at(indices[i]) = i;
  std::vector<int> original_labels(scan.points.size(), -1);
  const std::size_t rows = scan.planes.size() + !ground;
  std::vector<float> geometry(rows * 6, 0);
  std::uint16_t next = 1;
  for(Index plane = 0; plane < scan.planes.size(); ++plane) {
    auto label = plane == scan.ground_plane ? 0 : next++;
    order.planes[plane] = label;
    for(Index i : scan.supports[plane])
      original_labels.at(i) = label;
    const auto &[center, normal] = scan.planes[plane];
    for(int axis = 0; axis < 3; ++axis) {
      geometry[label * 6 + axis] = center[axis];
      geometry[label * 6 + 3 + axis] = normal[axis];
    }
  }
  std::vector<std::uint8_t> labels;
  for(Index i = 0; i < scan.points.size(); ++i) {
    if(original_labels[i] < 0) {
      if(order.points[i] < 0)
        throw std::runtime_error("missing nonplanar point index");
    } else {
      if(order.points[i] >= 0)
        throw std::runtime_error("planar point in feature subset");
      order.points[i] = indices.size();
      indices.push_back(i);
      labels.push_back(original_labels[i]);
    }
  }
  std::vector<double> bounds(6);
  for(int axis = 0; axis < 3; ++axis) {
    double lo = scan.points[0][axis], hi = lo;
    for(const auto &point : scan.points) {
      lo = std::min(lo, double(point[axis]));
      hi = std::max(hi, double(point[axis]));
    }
    bounds[axis] = lo;
    bounds[axis + 3] = (hi - lo) / 65535.0;
  }
  std::vector<std::uint16_t> points;
  points.reserve(indices.size() * 3);
  for(Index i : indices)
    for(int axis = 0; axis < 3; ++axis) {
      double scale = bounds[axis + 3];
      double q = scale == 0 ? 0 : (double(scan.points[i][axis]) - bounds[axis]) / scale;
      points.push_back(static_cast<std::uint16_t>(std::clamp(std::floor(q + 0.5), 0.0, 65535.0)));
    }
  Buffer buffer;
  buffer.add<std::uint32_t>({static_cast<std::uint32_t>(indices.size()),
    static_cast<std::uint32_t>(scan.point_indices.size()), static_cast<std::uint32_t>(rows)});
  buffer.add(bounds);
  buffer.add(points);
  buffer.add(labels);
  buffer.align();
  buffer.add(geometry);
  buffer.write(path);
  return order;
}

std::vector<std::uint16_t> selection(const std::vector<Index> &indices, std::size_t candidates)
{
  if(candidates > 65536)
    throw std::runtime_error("too many correspondence candidates");
  std::vector<std::uint16_t> result;
  for(Index i : indices) {
    if(i < 0 || i >= candidates)
      throw std::runtime_error("invalid selected correspondence");
    result.push_back(i);
  }
  std::sort(result.begin(), result.end());
  return result;
}

int main(int argc, char **argv)
{
  try {
    if(argc != 7)
      throw std::runtime_error("usage: parte-cache source.ply target.ply voxel-size segment-ground(0|1) output threads");
    if(std::endian::native != std::endian::little)
      throw std::runtime_error("little-endian host required");
    const Scalar voxel = std::stof(argv[3]);
    const int threads = std::stoi(argv[6]);
    if(!std::isfinite(voxel) || voxel <= 0 || threads < 1)
      throw std::runtime_error("invalid voxel size or thread count");
    const std::string ground_arg = argv[4];
    if(ground_arg != "0" && ground_arg != "1")
      throw std::runtime_error("invalid ground flag");
    const bool ground = ground_arg == "1";
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    const auto start = Clock::now();
    auto source = read_ply(argv[1]);
    auto target = read_ply(argv[2]);
    const auto loaded = Clock::now();
    pipeline::RegistrationTrace trace;
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    std::string error;
    try {
      transform = pipeline::register_clouds(std::move(source), std::move(target), voxel, ground, trace);
    } catch(const std::exception &e) {
      error = e.what();
    }
    const auto registered = Clock::now();
    if(trace.source.points.empty() || trace.target.points.empty()
      || trace.source.supports.size() != trace.source.planes.size()
      || trace.target.supports.size() != trace.target.planes.size())
      throw std::runtime_error("incomplete pipeline trace: " + error);
    if(!transform.allFinite()) {
      transform.setIdentity();
      error = "non-finite estimated transform";
    }
    fs::path output = argv[5];
    fs::create_directories(output);
    auto source_order = export_scan(trace.source, output / "source.bin");
    auto target_order = export_scan(trace.target, output / "target.bin");
    std::vector<std::uint32_t> points, planes;
    for(const auto &[s, t] : trace.point_matches) {
      const auto si = source_order.points.at(trace.source.point_indices.at(s));
      const auto ti = target_order.points.at(trace.target.point_indices.at(t));
      if(si < 0 || ti < 0 || si > 65535 || ti > 65535)
        throw std::runtime_error("point index overflow");
      points.push_back(std::uint32_t(si) | (std::uint32_t(ti) << 16));
    }
    for(const auto &[s, t] : trace.plane_matches)
      planes.push_back(std::uint32_t(source_order.planes.at(s)) | (std::uint32_t(target_order.planes.at(t)) << 16));
    auto selected_points = selection(trace.selected_points, points.size());
    auto selected_planes = selection(trace.selected_planes, planes.size());
    Buffer buffer;
    buffer.add<std::uint32_t>({static_cast<std::uint32_t>(points.size()), static_cast<std::uint32_t>(selected_points.size()),
      static_cast<std::uint32_t>(planes.size()), static_cast<std::uint32_t>(selected_planes.size())});
    buffer.add(points);
    buffer.add(selected_points);
    buffer.align();
    buffer.add(planes);
    buffer.add(selected_planes);
    buffer.write(output / "matches.bin");
    std::vector<double> matrix;
    for(int r = 0; r < 4; ++r)
      for(int c = 0; c < 4; ++c)
        matrix.push_back(transform(r, c));
    const auto ms = [](auto duration) { return std::chrono::duration<double, std::milli>(duration).count(); };
    std::ofstream json(output / "trace.json");
    json << std::setprecision(10) << "{\"transform\":" << json_array(matrix)
         << ",\"registrationRuntimeMs\":" << ms(registered - loaded) << ",\"loadingRuntimeMs\":" << ms(loaded - start)
         << ",\"completed\":" << (error.empty() ? "true" : "false") << ",\"error\":" << quote(error) << "}\n";
    if(!json)
      throw std::runtime_error("failed to write trace metadata");
  } catch(const std::exception &e) {
    std::cerr << "parte-cache: " << e.what() << '\n';
    return 1;
  }
}
