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
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <limits>
#include <numbers>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <omp.h>
#include <open3d/Open3D.h>


namespace
{

namespace fs = std::filesystem;
namespace parameters = parte::parameters;
using Clock = std::chrono::steady_clock;
using parte::Index;
using parte::Neighbors;
using parte::Normal;
using parte::Plane;
using parte::Point;
using parte::Scalar;

struct GroundTruth
{
  Index source = 0;
  Index target = 0;
  std::string metadata;
  Eigen::Matrix4d target_to_source = Eigen::Matrix4d::Identity();
};


struct Estimate
{
  GroundTruth ground_truth;
  Eigen::Matrix4d source_to_target = Eigen::Matrix4d::Identity();
  bool completed = false;
};


struct BenchmarkSummary
{
  std::size_t pairs = 0;
  std::size_t completed = 0;
  std::size_t successful = 0;
  double successful_translation = 0.0;
  double successful_rotation = 0.0;
  double total_runtime_ms = 0.0;
  double translation_criterion = 0.0;
  double rotation_criterion = 0.0;
};


using Summaries = std::map<std::string, BenchmarkSummary>;


struct ManifestEntry
{
  fs::path scans;
  fs::path ground_truth;
  fs::path output_subdirectory;
  bool segment_ground = false;
  double translation_criterion = 0.0;
  double rotation_criterion = 0.0;
};


struct PreparedCloud
{
  std::vector<Point> points;
  std::vector<Normal> normals;
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


double parse_positive_double(std::string_view text, std::string_view name)
{
  std::size_t parsed = 0;
  double result = std::stod(std::string(text), &parsed);
  if(parsed != text.size() || !std::isfinite(result) || result <= 0.0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return result;
}


Scalar read_voxel_size(const fs::path &scans)
{
  std::ifstream stream(scans / "voxel_size");
  Scalar result;
  if(!(stream >> result) || !std::isfinite(result) || result <= 0.0f) {
    throw std::runtime_error("invalid voxel size in " + scans.string());
  }
  return result;
}


std::vector<GroundTruth> read_ground_truth(const fs::path &path)
{
  std::ifstream stream(path);
  if(!stream) {
    throw std::runtime_error("failed to open " + path.string());
  }

  std::vector<GroundTruth> result;
  while(true) {
    GroundTruth ground_truth;
    if(!(stream >> ground_truth.source)) {
      break;
    }
    if(!(stream >> ground_truth.target >> ground_truth.metadata)) {
      throw std::runtime_error("malformed ground truth in " + path.string());
    }
    for(Eigen::Index row = 0; row < 4; ++row) {
      for(Eigen::Index column = 0; column < 4; ++column) {
        if(!(stream >> ground_truth.target_to_source(row, column))) {
          throw std::runtime_error("malformed transform in " + path.string());
        }
      }
    }
    result.push_back(std::move(ground_truth));
  }
  return result;
}


std::vector<Point> read_points(const fs::path &path)
{
  open3d::geometry::PointCloud cloud;
  if(!open3d::io::ReadPointCloud(path.string(), cloud) || cloud.IsEmpty()) {
    throw std::runtime_error("failed to read " + path.string());
  }

  std::vector<Point> result;
  result.reserve(cloud.points_.size());
  for(const auto &point : cloud.points_) {
    result.push_back(point.cast<Scalar>());
  }
  return result;
}


fs::path scan_path(const fs::path &scans, Index index)
{
  return scans / ("cloud_bin_" + std::to_string(index) + ".ply");
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
  for(Index point = 0; point < result.points.size(); ++point) {
    if(!is_plane_point[point]) {
      result.point_indices.push_back(point);
    }
  }
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
  bool segment_ground
)
{
  PreparedCloud source = prepare_cloud(
    std::move(source_cloud), voxel_size, segment_ground
  );
  PreparedCloud target = prepare_cloud(
    std::move(target_cloud), voxel_size, segment_ground
  );

  auto point_matches = parte::registration::mutual_correspondences<
    parte::registration::FPFH>(source.fpfh, target.fpfh);
  std::vector<parte::Correspondence> plane_matches;
  std::vector<Scalar> plane_confidences;
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

  std::vector<Index> selected_points, selected_planes;
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


std::pair<double, double> transform_errors(
  const Eigen::Matrix4d &estimate,
  const Eigen::Matrix4d &ground_truth
)
{
  double translation = (estimate.block<3, 1>(0, 3) - ground_truth.block<3, 1>(0, 3)).norm();
  Eigen::Matrix3d relative_rotation =
    estimate.block<3, 3>(0, 0) * ground_truth.block<3, 3>(0, 0).transpose();
  double cosine = std::clamp(
    0.5 * (relative_rotation.trace() - 1.0), -1.0, 1.0
  );
  return {translation, std::acos(cosine) * 180.0 / std::numbers::pi};
}


void write_log(const fs::path &path, const std::vector<Estimate> &estimates)
{
  fs::create_directories(path.parent_path());
  std::ofstream stream(path);
  stream << std::setprecision(17);
  for(const auto &estimate : estimates) {
    const auto &ground_truth = estimate.ground_truth;
    stream << ground_truth.source << ' ' << ground_truth.target << ' '
           << ground_truth.metadata << '\n'
           << estimate.source_to_target.inverse() << '\n';
  }
}


double average(double sum, std::size_t count)
{
  return count ? sum / count : std::numeric_limits<double>::quiet_NaN();
}


std::string duration(double seconds)
{
  auto value = static_cast<std::size_t>(seconds);
  std::ostringstream stream;
  stream << std::setfill('0') << std::setw(2) << value / 60
         << ':' << std::setw(2) << value % 60;
  return stream.str();
}


void progress_bar(std::size_t processed, std::size_t total)
{
  constexpr std::size_t width = 24;
  std::size_t filled = total ? width * processed / total : width;
  std::cout << '[' << std::string(filled, '=')
            << (filled < width ? ">" : "")
            << std::string(width - filled - (filled < width), ' ') << ']';
}


struct Progress
{
#ifdef _WIN32
  bool terminal = _isatty(_fileno(stdout));
#else
  bool terminal = isatty(fileno(stdout));
#endif
  bool visible = false;
  std::size_t name_width = 0;
  std::size_t sequences = 0, total_sequences = 0;

  void clear()
  {
    if(visible) {
      std::cout << "\r\033[1A\033[J";
      visible = false;
    }
  }

  void overall() const
  {
    std::cout << "[Sequences] ";
    progress_bar(sequences, total_sequences);
    std::cout << ' ' << sequences << '/' << total_sequences;
    std::cout << '\n';
  }

  void update(const std::string &name, const BenchmarkSummary &summary,
    std::size_t total, Clock::time_point start)
  {
    if(!terminal)
      return;
    clear();
    overall();
    double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    double rate = seconds > 0 ? summary.pairs / seconds : 0;
    std::cout << '[' << name << "] ";
    progress_bar(summary.pairs, total);
    std::cout << ' ' << summary.pairs << '/' << total
              << " [" << duration(seconds) << '<'
              << (rate > 0 ? duration((total - summary.pairs) / rate) : "--:--")
              << ", " << std::fixed << std::setprecision(2) << rate << " pairs/s]"
              << std::flush;
    visible = true;
  }
};


void add_summary(BenchmarkSummary &total, const BenchmarkSummary &current)
{
  if(total.pairs == 0) {
    total.translation_criterion = current.translation_criterion;
    total.rotation_criterion = current.rotation_criterion;
  } else {
    if(total.translation_criterion != current.translation_criterion) {
      total.translation_criterion = std::numeric_limits<double>::quiet_NaN();
    }
    if(total.rotation_criterion != current.rotation_criterion) {
      total.rotation_criterion = std::numeric_limits<double>::quiet_NaN();
    }
  }
  total.pairs += current.pairs;
  total.completed += current.completed;
  total.successful += current.successful;
  total.successful_translation += current.successful_translation;
  total.successful_rotation += current.successful_rotation;
  total.total_runtime_ms += current.total_runtime_ms;
}


BenchmarkSummary run_sequence(
  const ManifestEntry &entry,
  const fs::path &output,
  const std::string &name,
  Progress &progress
)
{
  Scalar voxel_size = read_voxel_size(entry.scans);
  auto ground_truth = read_ground_truth(entry.ground_truth);
  BenchmarkSummary summary;
  std::vector<Estimate> estimates;
  estimates.reserve(ground_truth.size());
  auto start = Clock::now();
  progress.update(name, summary, ground_truth.size(), start);

  for(std::size_t pair = 0; pair < ground_truth.size(); ++pair) {
    const auto &truth = ground_truth[pair];
    Estimate estimate;
    estimate.ground_truth = truth;
    BenchmarkSummary current;
    current.pairs = 1;
    current.translation_criterion = entry.translation_criterion;
    current.rotation_criterion = entry.rotation_criterion;
    Clock::time_point registration_start;
    bool registration_started = false;
    try {
      auto source_points = read_points(scan_path(entry.scans, truth.source));
      auto target_points = read_points(scan_path(entry.scans, truth.target));
      registration_start = Clock::now();
      registration_started = true;
      estimate.source_to_target = register_clouds(
        std::move(source_points),
        std::move(target_points),
        voxel_size,
        entry.segment_ground
      );
      current.total_runtime_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - registration_start
      )
                                   .count();
      estimate.completed = true;
      current.completed = 1;
    } catch(const std::exception &error) {
      if(registration_started) {
        current.total_runtime_ms = std::chrono::duration<double, std::milli>(
          Clock::now() - registration_start
        )
                                     .count();
      }
      progress.clear();
      std::cerr << '[' << name << ' ' << pair + 1 << '/'
                << ground_truth.size() << "] " << truth.source << " -> "
                << truth.target << " failed: " << error.what() << '\n';
    }

    auto [translation, rotation] = transform_errors(
      estimate.source_to_target, truth.target_to_source.inverse()
    );
    if(estimate.completed && translation <= entry.translation_criterion
      && rotation <= entry.rotation_criterion) {
      current.successful = 1;
      current.successful_translation = translation;
      current.successful_rotation = rotation;
    }
    add_summary(summary, current);
    estimates.push_back(std::move(estimate));
    progress.update(name, summary, ground_truth.size(), start);
  }
  progress.clear();
  write_log(output, estimates);

  std::cout << std::fixed << std::setprecision(3)
            << std::left << std::setw(progress.name_width + 2) << ('[' + name + ']')
            << std::right << " success: " << std::setw(11)
            << (std::to_string(summary.successful) + '/' + std::to_string(summary.pairs))
            << " (" << std::setw(7) << average(100.0 * summary.successful, summary.pairs)
            << "%), ATE: " << std::setw(7) << average(summary.successful_translation, summary.successful)
            << " m, ARE: " << std::setw(7) << average(summary.successful_rotation, summary.successful)
            << " deg, runtime: " << std::setw(9) << average(summary.total_runtime_ms, summary.pairs)
            << " ms\n";
  return summary;
}


void write_summary(
  const fs::path &path,
  const Summaries &datasets,
  const Summaries &sequences
)
{
  std::ofstream stream(path);
  for(const auto &[name, summary] : datasets) {
    stream << '[' << name << "]\n"
           << std::setprecision(10)
           << "successful " << summary.successful << '/' << summary.pairs << '\n';
    if(std::isnan(summary.translation_criterion)) {
      stream << "success_translation_m mixed\n";
    } else {
      stream << "success_translation_m " << summary.translation_criterion << '\n';
    }
    if(std::isnan(summary.rotation_criterion)) {
      stream << "success_rotation_deg mixed\n";
    } else {
      stream << "success_rotation_deg " << summary.rotation_criterion << '\n';
    }
    stream << "success_rate_percent " << average(100.0 * summary.successful, summary.pairs)
           << "\nATE_m " << average(summary.successful_translation, summary.successful)
           << "\nARE_deg " << average(summary.successful_rotation, summary.successful)
           << "\naverage_runtime_ms " << average(summary.total_runtime_ms, summary.pairs)
           << "\n\n";
  }

  for(const auto &[dataset, values] : datasets) {
    std::string prefix = dataset + '/';
    std::size_t width = 16;
    for(const auto &[name, summary] : sequences) {
      if(name.starts_with(prefix))
        width = std::max(width, name.size() - prefix.size());
    }
    stream << '[' << dataset << "/sequences]\n"
           << std::left << std::setw(width) << "Sequence" << std::right
           << std::setw(10) << "SR (%)" << std::setw(12) << "ATE (m)"
           << std::setw(12) << "ARE (deg)" << std::setw(14) << "Runtime (ms)" << '\n';
    auto row = [&](const std::string &name, const BenchmarkSummary &summary) {
      stream << std::left << std::setw(width) << name << std::right
             << std::fixed << std::setprecision(3)
             << std::setw(10) << average(100.0 * summary.successful, summary.pairs)
             << std::setw(12) << average(summary.successful_translation, summary.successful)
             << std::setw(12) << average(summary.successful_rotation, summary.successful)
             << std::setw(14) << average(summary.total_runtime_ms, summary.pairs) << '\n';
    };
    for(const auto &[name, summary] : sequences) {
      if(!name.starts_with(prefix))
        continue;
      row(name.substr(prefix.size()), summary);
    }
    row("Total", values);
    stream << '\n';
  }
}


std::vector<ManifestEntry> read_manifest(const fs::path &manifest)
{
  std::ifstream stream(manifest);
  if(!stream) {
    throw std::runtime_error("failed to open " + manifest.string());
  }

  fs::path base = manifest.parent_path();
  std::vector<ManifestEntry> result;
  std::string line;
  while(std::getline(stream, line)) {
    std::istringstream fields(line);
    std::string scans, ground_truth;
    if(!(fields >> scans)) {
      continue;
    }
    if(!(fields >> ground_truth)) {
      throw std::runtime_error("missing ground-truth path in manifest");
    }

    ManifestEntry entry;
    entry.scans = (base / scans).lexically_normal();
    entry.ground_truth = (base / ground_truth).lexically_normal();
    entry.output_subdirectory = entry.ground_truth.parent_path()
                                  .lexically_relative(base / "benchmarks");

    std::string option;
    while(fields >> option) {
      if(option == "--segment-ground") {
        entry.segment_ground = true;
        continue;
      }
      if(option == "--success-criteria") {
        std::string translation, rotation;
        fields >> translation >> rotation;
        entry.translation_criterion = parse_positive_double(
          translation, "translation criterion"
        );
        entry.rotation_criterion = parse_positive_double(
          rotation, "rotation criterion"
        );
        continue;
      }
      throw std::runtime_error("unsupported manifest option: " + option);
    }
    result.push_back(std::move(entry));
  }
  return result;
}

}


int main(int argc, char **argv)
{
  if(argc != 3) {
    std::cerr << "usage: parte-benchmark <manifest.txt> <output-directory>\n";
    return 2;
  }

  try {
    omp_set_dynamic(0);
    omp_set_num_threads(std::min(parameters::ThreadCount, omp_get_num_procs()));

    fs::path manifest = fs::absolute(argv[1]).lexically_normal();
    fs::path output = fs::absolute(argv[2]).lexically_normal();
    auto entries = read_manifest(manifest);

    Summaries datasets, sequences;
    Progress progress;
    for(const auto &entry : entries) {
      std::string name = entry.output_subdirectory.generic_string();
      progress.name_width = std::max(progress.name_width, name.size());
    }
    progress.total_sequences = entries.size();
    bool completed = true;
    fs::create_directories(output);
    for(const auto &entry : entries) {
      std::string sequence = entry.output_subdirectory.generic_string();
      std::string dataset = entry.output_subdirectory.begin()->string();
      fs::path directory = output / entry.output_subdirectory;
      auto current = run_sequence(entry, directory / "parte.log", sequence, progress);
      add_summary(datasets[dataset], current);
      add_summary(sequences[sequence], current);
      completed &= current.completed == current.pairs;
      ++progress.sequences;
    }
    progress.overall();
    write_summary(output / "summary.txt", datasets, sequences);
    std::cout << "wrote " << (fs::path(argv[2]) / "summary.txt").string() << '\n';
    return completed ? 0 : 1;
  } catch(const std::exception &error) {
    std::cerr << "parte-benchmark: " << error.what() << '\n';
    return 1;
  }
}
