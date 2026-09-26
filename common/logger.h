#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace parte
{

class Logger
{
public:
  static Logger &instance();
  Logger(const Logger &) = delete;
  Logger &operator=(const Logger &) = delete;

  void set_output(std::ostream *output)
  {
    output_ = output;
  }

  void log(std::string_view message) const;
  void start(std::string_view name);

  double stop(std::string_view name = {});
  double elapsed() const;

  template<typename Function>
  decltype(auto) time(std::string_view name, Function &&function)
  {
    start(name);
    struct StopTimer
    {
      Logger &logger;
      std::string_view name;

      ~StopTimer()
      {
        logger.stop(name);
      }
    } stop{*this, name};
    return std::forward<Function>(function)();
  }

private:
  Logger() = default;

  struct Timer
  {
    std::string name;
    std::chrono::steady_clock::time_point start;
    std::string children;
  };

  std::ostream *output_ = &std::cout;
  std::vector<Timer> timers_;
};

extern Logger &logger;

}
