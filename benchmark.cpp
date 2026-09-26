#include "common/logger.h"
#include "registration/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <open3d/Open3D.h>


using GroundTruth = std::tuple<int, int, int, Eigen::Matrix4d>;

struct BenchmarkSummary
{
  std::size_t pairs = 0;
  std::size_t completed = 0;
  std::size_t successful = 0;
  double successful_translation = 0.0;
  double successful_rotation = 0.0;
  double total_runtime_ms = 0.0;

  void add(double translation, double rotation, double runtime_ms, bool success)
  {
    ++pairs;
    ++completed;
    total_runtime_ms += runtime_ms;
    if(success) {
      ++successful;
      successful_translation += translation;
      successful_rotation += rotation;
    }
  }

  void add(const BenchmarkSummary &other)
  {
    pairs += other.pairs;
    completed += other.completed;
    successful += other.successful;
    successful_translation += other.successful_translation;
    successful_rotation += other.successful_rotation;
    total_runtime_ms += other.total_runtime_ms;
  }
};


struct Sequence
{
  std::string name;
  std::string dataset;
  std::filesystem::path scans;
  std::filesystem::path ground_truth;
  std::filesystem::path output_subdirectory;
  bool segment_ground = false;
  double translation_criterion = 0.0;
  double rotation_criterion = 0.0;
};

std::vector<GroundTruth> read_ground_truth(const std::filesystem::path &path)
{
  std::ifstream stream(path);
  if(!stream) {
    throw std::runtime_error("failed to open " + path.string());
  }

  std::vector<GroundTruth> result;
  int src, tgt, meta;
  while(stream >> src >> tgt >> meta) {
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    for(Eigen::Index row = 0; row < 4; ++row) {
      for(Eigen::Index column = 0; column < 4; ++column) {
        stream >> transform(row, column);
      }
    }
    result.emplace_back(src, tgt, meta, transform);
  }
  return result;
}


std::vector<Eigen::Vector3d> read_points(const std::filesystem::path &scans, parte::Index index)
{
  const auto path = scans / ("cloud_bin_" + std::to_string(index) + ".ply");
  open3d::geometry::PointCloud cloud;
  if(!open3d::io::ReadPointCloud(path.string(), cloud) || cloud.IsEmpty()) {
    throw std::runtime_error("failed to read " + path.string());
  }

  return std::move(cloud.points_);
}


std::pair<double, double> transform_errors(
  const Eigen::Matrix4d &source_to_target,
  const Eigen::Matrix4d &target_to_source
)
{
  const Eigen::Matrix4d relative = target_to_source * source_to_target;
  const double translation = relative.block<3, 1>(0, 3).norm();
  const Eigen::Matrix3d relative_rotation = relative.block<3, 3>(0, 0);
  const double cosine = std::clamp(
    0.5 * (relative_rotation.trace() - 1.0), -1.0, 1.0
  );
  return {translation, std::acos(cosine) * 180.0 / std::numbers::pi};
}


void write_log(
  const std::filesystem::path &path,
  const std::vector<GroundTruth> &ground_truth,
  const std::vector<Eigen::Matrix4d> &estimates
)
{
  std::filesystem::create_directories(path.parent_path());
  std::ofstream stream(path);
  stream << std::setprecision(17);
  for(std::size_t pair = 0; pair < estimates.size(); ++pair) {
    const auto &[src, tgt, metadata, transform] = ground_truth[pair];
    stream << src << ' ' << tgt << ' ' << metadata << '\n'  << estimates[pair].inverse() << '\n';
  }
}


double average(double sum, std::size_t count)
{
  return count ? sum / count : std::numeric_limits<double>::quiet_NaN();
}


struct Progress
{
  bool visible = false;
  std::size_t name_width = 0;
  std::size_t sequences = 0, total_sequences = 0;
  void progress_bar(std::size_t processed, std::size_t total) const
  {
    constexpr std::size_t width = 24;
    std::size_t filled = total ? width * processed / total : width;
    std::cout << std::format("[{}{}{}]", std::string(filled, '='), (filled < width ? ">" : ""), std::string(width - filled - (filled < width), ' '));
  }

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

  void update(const std::string &name, const BenchmarkSummary &summary, std::size_t total)
  {
    clear();
    overall();
    const double seconds = parte::logger.elapsed();
    const double rate = seconds > 0 ? summary.pairs / seconds : 0;
    std::cout << std::format("[{}] ", name);
    progress_bar(summary.pairs, total);
    const auto remaining = total > summary.pairs ? total - summary.pairs : 0;
    const auto elapsed = static_cast<std::size_t>(seconds);
    const auto eta = rate > 0 ? static_cast<std::size_t>((remaining / rate)) : 0;
    const auto elapsed_text = std::format("{:02}:{:02}", elapsed / 60, elapsed % 60);
    const auto eta_text = std::format("{:02}:{:02}", eta / 60, eta % 60);
    std::cout << std::format(" {} / {} [{} < {}, {:.2f} pairs/s]",
      summary.pairs, total, elapsed_text, eta_text, rate);
    std::cout << std::flush;
    visible = true;
  }
};


BenchmarkSummary run_sequence(
  const Sequence &entry,
  const std::filesystem::path &output,
  const std::string &name,
  Progress &progress
)
{
  parte::Scalar voxel_size;
  std::ifstream(entry.scans / "voxel_size") >> voxel_size;
  parte::Parameters parameters(voxel_size);
  parameters.ground_segmentation = entry.segment_ground;

  auto ground_truth = read_ground_truth(entry.ground_truth);
  BenchmarkSummary summary;
  std::vector<Eigen::Matrix4d> estimates;
  estimates.reserve(ground_truth.size());
  parte::logger.start("Sequence");
  progress.update(name, summary, ground_truth.size());
  for(std::size_t pair = 0; pair < ground_truth.size(); ++pair) {
    const auto &[src, tgt, metadata, transform] = ground_truth[pair];

    Eigen::Matrix4d estimate = Eigen::Matrix4d::Identity();
    auto source_points = read_points(entry.scans, src);
    auto target_points = read_points(entry.scans, tgt);
    parte::logger.start("Registration");
    auto source = parte::process_cloud(std::move(source_points), parameters);
    auto target = parte::process_cloud(std::move(target_points), parameters);
    estimate = parte::register_clouds(source, target, parameters).transformation.cast<double>();
    const double runtime_ms = 1000 * parte::logger.stop();
    const auto [translation, rotation] = transform_errors(estimate, transform);
    summary.add(
      translation, rotation, runtime_ms,
      translation <= entry.translation_criterion && rotation <= entry.rotation_criterion
    );
    estimates.push_back(estimate);
    progress.update(name, summary, ground_truth.size());
  }

  parte::logger.stop();
  progress.clear();
  write_log(output, ground_truth, estimates);
  const auto success_rate = average(100.0 * summary.successful, summary.pairs);
  const auto ate = average(summary.successful_translation, summary.successful);
  const auto are = average(summary.successful_rotation, summary.successful);
  const auto runtime = average(summary.total_runtime_ms, summary.pairs);
  std::cout << std::format(
    "{} success: {:>11} ({:>7.3f}%), ATE: {:>7.3f} m, ARE: {:>7.3f} deg, runtime: {:>9.3f} ms\n",
    std::string("[") + name + "]",
    std::format("{}/{}", summary.successful, summary.pairs),
    success_rate,
    ate,
    are,
    runtime
  );
  return summary;
}


void write_summary(
  const std::filesystem::path &path,
  const std::map<std::string, BenchmarkSummary> &datasets,
  const std::map<std::string, BenchmarkSummary> &sequences
)
{
  std::ofstream stream(path);
  for(const auto &[dataset, values] : datasets) {
    std::string prefix = dataset + '/';
    std::size_t width = 16;
    for(const auto &[name, summary] : sequences) {
      if(name.starts_with(prefix)) {
        width = std::max(width, name.size() - prefix.size());
      }
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
      if(!name.starts_with(prefix)) {
        continue;
      }
      row(name.substr(prefix.size()), summary);
    }
    row("Total", values);
    stream << '\n';
  }
}


std::vector<Sequence> read_sequences(const std::filesystem::path &manifest)
{
  std::ifstream stream(manifest);
  if(!stream) {
    throw std::runtime_error("failed to open " + manifest.string());
  }

  std::filesystem::path base = manifest.parent_path();
  std::vector<Sequence> result;
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

    Sequence entry;
    entry.scans = base / scans;
    entry.ground_truth = base / ground_truth;
    entry.output_subdirectory = entry.ground_truth.parent_path().lexically_relative(base / "benchmarks");
    entry.name = entry.output_subdirectory.generic_string();
    entry.dataset = entry.output_subdirectory.begin()->string();

    std::string option;
    while(fields >> option) {
      if(option == "--segment-ground") {
        entry.segment_ground = true;
        continue;
      }
      if(option == "--success-criteria") {
        std::string translation, rotation;
        fields >> translation >> rotation;
        entry.translation_criterion = std::stod(translation);
        entry.rotation_criterion = std::stod(rotation);
        continue;
      }
      throw std::runtime_error("unsupported option: " + option);
    }
    result.push_back(std::move(entry));
  }
  return result;
}

int main(int argc, char **argv)
{
  if(argc != 3) {
    std::cerr << "usage: parte-benchmark <test.txt> <output-directory>\n";
    return 1;
  }

  parte::logger.set_output(nullptr);
  const std::filesystem::path output = argv[2];
  const auto sequences = read_sequences(argv[1]);

  std::map<std::string, BenchmarkSummary> datasets, summary_by_sequence;
  Progress progress;
  for(const auto &sequence : sequences) {
    progress.name_width = std::max(progress.name_width, sequence.name.size());
  }
  
  progress.total_sequences = sequences.size();
  std::filesystem::create_directories(output);
  for(const auto &seq : sequences) {
    std::filesystem::path directory = output / seq.output_subdirectory;
    summary_by_sequence[seq.name] = run_sequence(seq, directory / "parte.log", seq.name, progress);
    datasets[seq.dataset].add(summary_by_sequence[seq.name]);
    ++progress.sequences;
  }
  progress.overall();
  write_summary(output / "summary.txt", datasets, summary_by_sequence);
  std::cout << "wrote " << (std::filesystem::path(argv[2]) / "summary.txt").string() << '\n';
  return 0;
}
