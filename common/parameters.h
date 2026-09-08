#pragma once

#include "common/types.h"

#include <cstddef>


namespace parte::parameters
{

constexpr std::size_t NormalK = 30;
constexpr std::size_t FpfhK = 100;
constexpr Scalar NormalRadius = 2.0f;
constexpr Scalar FpfhRadius = 5.0f;
constexpr Scalar PchRange = 20.0f;
constexpr Scalar PchRadius = 20.0f;
constexpr Scalar GroupDistance = 2.0f;
constexpr Scalar SegmentationAngle = 10.0f;
constexpr Scalar GroupAngle = 5.0f;
constexpr Scalar RegistrationAngle = 5.0f;
constexpr std::size_t MinimumPlaneSupport = 100;
constexpr int ThreadCount = 16;

}
