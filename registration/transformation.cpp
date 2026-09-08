#include "registration/transformation.h"

namespace parte::registration
{

ScalarMatrix<4, 4> compute_transformation(
  std::span<const Point> source_points, std::span<const Point> target_points,
  std::span<const Plane> source_planes, std::span<const Plane> target_planes,
  std::span<const Scalar> plane_weights
)
{
  constexpr Index max_iter = 50;
  constexpr Scalar threshold = 1e-9;
  const Index n_points = source_points.size(), n_planes = source_planes.size();

  Point src_center = Point::Zero(), tgt_center = Point::Zero();
  for(Index i = 0; i < n_points; ++i) {
    src_center += source_points[i] / n_points;
    tgt_center += target_points[i] / n_points;
  }

  ScalarMatrix<3, 3> H = ScalarMatrix<3, 3>::Zero();
  for(Index i = 0; i < n_points; ++i) {
    H += target_points[i] * source_points[i].transpose();
  }

  Point b = Point::Zero();
  ScalarMatrix<3, 3> A = ScalarMatrix<3, 3>::Identity() * n_points;
  for(Index i = 0; i < n_planes; ++i) {
    const auto w = plane_weights[i];

    auto [pc, pn] = source_planes[i];
    auto [qc, qn] = target_planes[i];

    H += w * qn * pn.transpose();
    A += w * qn * qn.transpose();
    b += w * (qn.dot(qc) - pn.dot(pc)) * qn;
  }

  ScalarMatrix<3, 3> R = ScalarMatrix<3, 3>::Identity();
  Point t = tgt_center;

  Eigen::ColPivHouseholderQR<ScalarMatrix<3, 3>> A_solver(A);
  for(Index iter = 0; iter < max_iter; ++iter) {
    const auto R_prev = R;
    const auto t_prev = t;

    ScalarMatrix<3, 3> H_adjusted = H - n_points * (t * src_center.transpose());

    // update R
    auto svd = Eigen::JacobiSVD<ScalarMatrix<3, 3>>(H_adjusted, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto &U = svd.matrixU();
    const auto &V = svd.matrixV();
    Point diag(1.0, 1.0, (U * V.transpose()).determinant());
    R = U * diag.asDiagonal() * V.transpose();

    // Update t
    Point b_adjusted = b + n_points * (tgt_center - R * src_center);
    t = A_solver.solve(b_adjusted);
    if((R - R_prev).norm() < threshold && (t - t_prev).norm() < threshold) {
      break;
    }
  }

  ScalarMatrix<4, 4> T_out = ScalarMatrix<4, 4>::Identity();
  T_out.block<3, 3>(0, 0) = R;
  T_out.block<3, 1>(0, 3) = t;
  return T_out;
}

}  // namespace registration
