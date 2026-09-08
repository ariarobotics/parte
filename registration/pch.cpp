#include <Eigen/Dense>
#include <span>


#include "common/types.h"
#include "registration/pch.h"


namespace parte::registration
{


std::pair<PCH, PCH> compute_pch(
  std::span<const Point> points,
  std::span<const Normal> normals,
  Normal p_normal,
  Point p_point,
  Scalar d_range,
  Scalar d_radius
)
{

  PCH pch = PCH::Zero();
  PCH r_pch = PCH::Zero();
  for(Index i = 0; i < points.size(); ++i) {
    auto dist = (points[i] - p_point).dot(p_normal);
    if(std::abs(dist) > d_range) {
      continue;
    }

    Index d_bin = static_cast<Index>((dist + d_range) * PCH_D_BINS / (2 * d_range));
    d_bin = std::clamp(d_bin, 0, PCH_D_BINS - 1);

    Scalar angle = normals[i].dot(p_normal);
    Index a_bin = static_cast<Index>((angle + 1.0f) * PCH_A_BINS / 2.0f);
    a_bin = std::clamp(a_bin, 0, PCH_A_BINS - 1);
    pch(d_bin, a_bin) += 1.0f;


    if(d_radius > 0.0) {
      if((points[i] - p_point - dist * p_normal).squaredNorm() <= d_radius * d_radius) {
        r_pch(d_bin, a_bin) += 1.0f;
      }
    }
  }
  pch = pch / std::max(pch.sum(), 1.0f);
  r_pch = r_pch / std::max(r_pch.sum(), 1.0f);
  return {pch, r_pch};
}

}  // namespace parte::registration
