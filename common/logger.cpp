#include "common/logger.h"

#include <cassert>
#include <iomanip>
#include <sstream>

namespace parte
{

Logger &Logger::instance()
{
  static Logger instance;
  return instance;
}

Logger &logger = Logger::instance();

void Logger::log(std::string_view message) const
{
  if(output_) {
    *output_ << std::string(2 * timers_.size(), ' ') << message << '\n';
  }
}

void Logger::start(std::string_view name)
{
  timers_.push_back({std::string(name), std::chrono::steady_clock::now(), {}});
}

double Logger::stop(std::string_view name)
{
  const auto end = std::chrono::steady_clock::now();
  std::size_t depth = timers_.size();
  if(!name.empty()) {
    while(depth > 0 && timers_[depth - 1].name != name) {
      --depth;
    }
  }
  assert(depth > 0);

  double seconds = 0;
  while(timers_.size() >= depth) {
    auto timer = std::move(timers_.back());
    timers_.pop_back();
    seconds = std::chrono::duration<double>(end - timer.start).count();
    if(output_) {
      std::ostringstream message;
      message << std::string(2 * timers_.size(), ' ') << timer.name << ": "
        << std::fixed << std::setprecision(3) << 1000 * seconds << " ms\n"
        << timer.children;
      if(timers_.empty()) {
        *output_ << message.str();
      } else {
        timers_.back().children += message.str();
      }
    }
  }
  return seconds;
}

double Logger::elapsed() const
{
  assert(!timers_.empty());
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - timers_.back().start).count();
}

}
