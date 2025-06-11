/*
Copyright 2020 University of Minnesota and Clemson University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/*
This file contains the implementations of the dynamics models for each agent type: Velocity, Acceleration, Differential Drive, Smooth Differential Drive, Simple Car, Smooth Car
In addition, any dynamics model specific features are handled here (such as special constraint projections or collision center computations).
*/
#include <sgd/ttc_sgd_problem_models.h>

#include <iostream>

// --------------------------------------------
// ******************* V **********************
// --------------------------------------------
void VTTCSGDProblem::ContDynamics(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, Eigen::VectorXf* x_dot,
                                  Eigen::MatrixXf* dxdot_du, Eigen::MatrixXf* dxdot_dx) {
  if (x_dot != nullptr) {
    (*x_dot) = Eigen::VectorXf::Zero(2);
    (*x_dot)[0] = u[0];
    (*x_dot)[1] = u[1];
  }
  if (dxdot_du != nullptr) {
    (*dxdot_du) = Eigen::MatrixXf::Identity(2,2);
  }
  if (dxdot_dx != nullptr) {
    (*dxdot_dx) = Eigen::MatrixXf::Zero(2,2);
  }
}

void VTTCSGDProblem::GetDynamicsDeriv(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                                      Eigen::MatrixXf* dxtpdt_du, Eigen::MatrixXf* dxtpdt_dx) {
  if (dxtpdt_dx != nullptr) {
    (*dxtpdt_dx) = Eigen::Matrix2f::Identity();
  }
  if (dxtpdt_du != nullptr) {
    (*dxtpdt_du) = dt * Eigen::Matrix2f::Identity();
  }
}

TTCObstacle* VTTCSGDProblem::CreateObstacle() {
  TTCObstacle* o = new VTTCObstacle();
  FillObstacle(o);
  return o;
}

// --------------------------------------------
// ******************* A **********************
// --------------------------------------------
void ATTCSGDProblem::ContDynamics(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, Eigen::VectorXf* x_dot,
                                  Eigen::MatrixXf* dxdot_du, Eigen::MatrixXf* dxdot_dx) {
  float scale_a = 1.0f;
  // Leaky velocity constraint
  if (x.tail<2>().norm() > params.vel_limit && x.tail<2>().dot(u) > 0.0f) {
    scale_a = 1e-2f;
  }
  if (x_dot != nullptr) {
    (*x_dot) = Eigen::VectorXf::Zero(4);
    x_dot->head<2>() = x.tail<2>();
    x_dot->tail<2>() = scale_a * u;
  }
  if (dxdot_du != nullptr) {
    (*dxdot_du) = Eigen::MatrixXf::Zero(4,2);
    (*dxdot_du)(2,0) = scale_a;
    (*dxdot_du)(3,1) = scale_a;
  }
  if (dxdot_dx != nullptr) {
    (*dxdot_dx) = Eigen::MatrixXf::Zero(4,4);
    (*dxdot_dx)(0,2) = 1.0f;
    (*dxdot_dx)(1,3) = 1.0f;
  }
}

void ATTCSGDProblem::GetDynamicsDeriv(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                                      Eigen::MatrixXf* dxtpdt_du, Eigen::MatrixXf* dxtpdt_dx) {
  if (dxtpdt_dx != nullptr) {
    (*dxtpdt_dx) = Eigen::MatrixXf::Identity(4, 4);
    (*dxtpdt_dx)(0, 2) = dt;
    (*dxtpdt_dx)(1, 3) = dt;
  }
  if (dxtpdt_du != nullptr) {
    // leaky constraints on velocity
    float scale_a = 1.0f;
    if (x.tail<2>().norm() > params.vel_limit && x.tail<2>().dot(u) > 0.0f) {
      scale_a = 1e-2f;
    }
    (*dxtpdt_du) = Eigen::MatrixXf::Zero(4, 2);
    (*dxtpdt_du)(0, 0) = 0.5f * dt * dt * scale_a;
    (*dxtpdt_du)(1, 1) = 0.5f * dt * dt * scale_a;
    (*dxtpdt_du)(2, 0) = dt * scale_a;
    (*dxtpdt_du)(3, 1) = dt * scale_a;
  }
}

void ATTCSGDProblem::AdditionalCosts(const Eigen::VectorXf &u, float* cost, Eigen::VectorXf* dc_du) {
  // Velocity Constraint Cost:
  Eigen::VectorXf x_t;
  Eigen::MatrixXf dxt_du;
  if (dc_du != nullptr) {
    DynamicsLong(u, params.x_0, 0.0f, 1.0f, &x_t, &dxt_du, nullptr);
  } else {
    DynamicsLong(u, params.x_0, 0.0f, 1.0f, &x_t);
  }
  if (x_t.tail<2>().norm() > params.vel_limit) {
    if (cost != nullptr) {
      (*cost) += params.k_v_constr * (x_t.tail<2>().squaredNorm() - params.vel_limit * params.vel_limit);
    }
    if (dc_du != nullptr) {
      (*dc_du) += 2.0 * params.k_v_constr * x_t.tail<2>().transpose() * dxt_du.bottomRows<2>();
    }
  }
}

void ATTCSGDProblem::ProjConstr(Eigen::VectorXf& u) {
  // Trim u to control constraints:
  TTCSGDProblem::ProjConstr(u);

  // Trim u to state constraints:
  Eigen::Vector2f v_new = params.x_0.tail<2>() + params.dt_step * u;
  if (v_new.norm() > params.vel_limit) {
    v_new = params.vel_limit / v_new.norm() * v_new;
    u = (v_new - params.x_0.tail<2>()) / params.dt_step;
  }
}

TTCObstacle* ATTCSGDProblem::CreateObstacle() {
  TTCObstacle* o = new ATTCObstacle();
  FillObstacle(o);
  return o;
}

// --------------------------------------------
// ******************* DD *********************
// --------------------------------------------
void DDTTCSGDProblem::ContDynamics(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, Eigen::VectorXf* x_dot,
                                   Eigen::MatrixXf* dxdot_du, Eigen::MatrixXf* dxdot_dx) {
  if (x_dot != nullptr) {
    (*x_dot) = Eigen::VectorXf::Zero(3);
    (*x_dot)[0] = u[0] * std::cos(x[2]);
    (*x_dot)[1] = u[0] * std::sin(x[2]);
    (*x_dot)[2] = u[1];
  }
  if (dxdot_du != nullptr) {
    (*dxdot_du) = Eigen::MatrixXf::Zero(3,2);
    (*dxdot_du)(0,0) = std::cos(x[2]);
    (*dxdot_du)(1,0) = std::sin(x[2]);
    (*dxdot_du)(2,1) = 1.0f;
  }
  if (dxdot_dx != nullptr) {
    (*dxdot_dx) = Eigen::MatrixXf::Zero(3,3);
    (*dxdot_dx)(0,2) = -u[0] * std::sin(x[2]);
    (*dxdot_dx)(1,2) = u[0] * std::cos(x[2]);
  }
}

void DDTTCSGDProblem::GetDynamicsDeriv(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                                      Eigen::MatrixXf* dxtpdt_du, Eigen::MatrixXf* dxtpdt_dx) {
  float th_new = x[2] + dt * u[1];
  if (dxtpdt_dx != nullptr) {
    (*dxtpdt_dx) = Eigen::Matrix3f::Identity();
    (*dxtpdt_dx)(0, 2) = -0.5f * dt * u[0] * (std::sin(x[2]) + std::sin(th_new));
    (*dxtpdt_dx)(1, 2) = 0.5f * dt * u[0] * (std::cos(x[2]) + std::cos(th_new));
  }
  if (dxtpdt_du != nullptr) {
    (*dxtpdt_du) = Eigen::MatrixXf::Zero(3, 2);
    (*dxtpdt_du)(0, 0) = 0.5f * dt * (std::cos(x[2]) + std::cos(th_new));
    (*dxtpdt_du)(1, 0) = 0.5f * dt * (std::sin(x[2]) + std::sin(th_new));
    (*dxtpdt_du)(0, 1) = -0.5f * dt * dt * u[0] * std::sin(th_new);
    (*dxtpdt_du)(1, 1) = 0.5f * dt * dt * u[0] * std::cos(th_new);
    (*dxtpdt_du)(2, 1) = dt;
  }
}

TTCObstacle* DDTTCSGDProblem::CreateObstacle() {
  TTCObstacle* o = new DDTTCObstacle();
  FillObstacle(o);
  return o;
}

// --------------------------------------------
// ****************** ADD *********************
// --------------------------------------------
void ADDTTCSGDProblem::ContDynamics(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, Eigen::VectorXf* x_dot,
                                    Eigen::MatrixXf* dxdot_du, Eigen::MatrixXf* dxdot_dx) {
  // leaky constraints on velocities
  float scale_v =
      ((x[3] > params.vel_limit && u[0] > 0.0f) || (x[3] < -params.vel_limit && u[0] < 0.0f)) ? 1e-2f
                                                                      : 1.0f;
  float scale_w =
      ((x[4] > params.rot_vel_limit && u[1] > 0.0f) || (x[4] < -params.rot_vel_limit && u[1] < 0.0f)) ? 1e-2f
                                                                      : 1.0f;
  if (x_dot != nullptr) {
    (*x_dot) = Eigen::VectorXf::Zero(5);
    (*x_dot)[0] = x[3] * std::cos(x[2]);
    (*x_dot)[1] = x[3] * std::sin(x[2]);
    (*x_dot)[2] = x[4];
    (*x_dot)[3] = scale_v * u[0];
    (*x_dot)[4] = scale_w * u[1];
  }
  if (dxdot_du != nullptr) {
    (*dxdot_du) = Eigen::MatrixXf::Zero(5,2);
    (*dxdot_du)(3,0) = scale_v;
    (*dxdot_du)(4,1) = scale_w;
  }
  if (dxdot_dx != nullptr) {
    (*dxdot_dx) = Eigen::MatrixXf::Zero(5,5);
    (*dxdot_dx)(0,2) = -x[3] * std::sin(x[2]);
    (*dxdot_dx)(0,3) = std::cos(x[2]);
    (*dxdot_dx)(1,2) = x[3] * std::cos(x[2]);
    (*dxdot_dx)(1,3) = std::sin(x[2]);
    (*dxdot_dx)(2,4) = 1.0f;
  }
}

void ADDTTCSGDProblem::GetDynamicsDeriv(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                                      Eigen::MatrixXf* dxtpdt_du, Eigen::MatrixXf* dxtpdt_dx) {
  // leaky constraints on velocities
  float scale_v =
      ((x[3] > params.vel_limit && u[0] > 0.0f) || (x[3] < -params.vel_limit && u[0] < 0.0f)) ? 1e-2f
                                                                      : 1.0f;
  float scale_w =
      ((x[4] > params.rot_vel_limit && u[1] > 0.0f) || (x[4] < -params.rot_vel_limit && u[1] < 0.0f)) ? 1e-2f
                                                                      : 1.0f;

  float v_new = x[3] + dt * scale_v * u[0];
  float w_new = x[4] + dt * scale_w * u[1];
  float th_new = x[2] + 0.5f * dt * (x[4] + w_new);

  if (dxtpdt_dx != nullptr) {
    (*dxtpdt_dx) = Eigen::MatrixXf::Identity(5, 5);
    (*dxtpdt_dx)(0, 2) = -0.5f * dt * (x[3] * std::sin(x[2]) + v_new * std::sin(th_new));
    (*dxtpdt_dx)(1, 2) = 0.5f * dt * (x[3] * std::cos(x[2]) + v_new * std::cos(th_new));
    (*dxtpdt_dx)(0, 3) = 0.5f * dt * (std::cos(x[2]) + std::cos(th_new));
    (*dxtpdt_dx)(1, 3) = 0.5f * dt * (std::sin(x[2]) + std::sin(th_new));
    (*dxtpdt_dx)(2, 4) = dt;

    (*dxtpdt_dx)(0, 4) = -0.5f * dt * v_new * std::sin(th_new) * (*dxtpdt_dx)(2, 4);
    (*dxtpdt_dx)(1, 4) = 0.5f * dt * v_new * std::cos(th_new) * (*dxtpdt_dx)(2, 4);
  }
  if (dxtpdt_du != nullptr) {
    (*dxtpdt_du) = Eigen::MatrixXf::Zero(5, 2);
    (*dxtpdt_du)(3, 0) = dt * scale_v;
    (*dxtpdt_du)(4, 1) = dt * scale_w;

    (*dxtpdt_du)(2, 1) = 0.5f * dt * (*dxtpdt_du)(4, 1);

    (*dxtpdt_du)(0, 0) = 0.5f * dt * std::cos(th_new) * (*dxtpdt_du)(3, 0);
    (*dxtpdt_du)(1, 0) = 0.5f * dt * std::sin(th_new) * (*dxtpdt_du)(3, 0);

    (*dxtpdt_du)(0, 1) = -0.5f * dt * v_new * std::sin(th_new) * (*dxtpdt_du)(2, 1);
    (*dxtpdt_du)(1, 1) = 0.5f * dt * v_new * std::cos(th_new) * (*dxtpdt_du)(2, 1);
  }
}

void ADDTTCSGDProblem::AdditionalCosts(const Eigen::VectorXf &u, float* cost, Eigen::VectorXf* dc_du) {
  // Velocity Constraint Cost:
  Eigen::VectorXf x_t;
  Eigen::MatrixXf dxt_du;
  if (dc_du != nullptr) {
    DynamicsLong(u, params.x_0, 0.0f, 1.0f, &x_t, &dxt_du, nullptr);
  } else {
    DynamicsLong(u, params.x_0, 0.0f, 1.0f, &x_t);
  }
  if (std::abs(x_t[3]) > params.vel_limit) {
    float err;
    if (x_t[3] > params.vel_limit) {
      err = x_t[3] - params.vel_limit;
    } else {
      err = x_t[3] + params.vel_limit; // x_t[3] is negative
    }
    if (cost != nullptr) {
      (*cost) += params.k_v_constr * err * err;
    }
    if (dc_du != nullptr) {
      (*dc_du) += 2.0 * params.k_v_constr * err * dxt_du.row(3);
    }
  }
}

void ADDTTCSGDProblem::ProjConstr(Eigen::VectorXf& u) {
  // Trim u to control constraints
  TTCSGDProblem::ProjConstr(u);

  // Trim u to state constraints
  float v_new = params.x_0[3] + params.dt_step * u[0];
  if (v_new > params.vel_limit) {
    u[0] = (params.vel_limit - params.x_0[3]) / params.dt_step;
  }
  float w_new = params.x_0[4] + params.dt_step * u[1];
  if (w_new > params.rot_vel_limit) {
    u[1] = (params.rot_vel_limit - params.x_0[4]) / params.dt_step;
  }
}

TTCObstacle* ADDTTCSGDProblem::CreateObstacle() {
  TTCObstacle* o = new ADDTTCObstacle();
  FillObstacle(o);
  return o;
}

// --------------------------------------------
// ****************** CAR *********************
// --------------------------------------------
void CARTTCSGDProblem::ContDynamics(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, Eigen::VectorXf* x_dot,
                                   Eigen::MatrixXf* dxdot_du, Eigen::MatrixXf* dxdot_dx) {
  float len_scale = 2.0f / std::sqrt(5.0f);
  float L = 2.0f * len_scale * params.radius;
  if (x_dot != nullptr) {
    (*x_dot) = Eigen::VectorXf::Zero(3);
    (*x_dot)[0] = u[0] * std::cos(x[2]);
    (*x_dot)[1] = u[0] * std::sin(x[2]);
    (*x_dot)[2] = u[0] * std::tan(u[1]) / L;
  }
  if (dxdot_du != nullptr) {
    (*dxdot_du) = Eigen::MatrixXf::Zero(3,2);
    (*dxdot_du)(0,0) = std::cos(x[2]);
    (*dxdot_du)(1,0) = std::sin(x[2]);
    (*dxdot_du)(2,0) = std::tan(u[1]) / L;
    (*dxdot_du)(2,1) = u[0] / (std::cos(u[1]) * std::cos(u[1]) * L);
  }
  if (dxdot_dx != nullptr) {
    (*dxdot_dx) = Eigen::MatrixXf::Zero(3,3);
    (*dxdot_dx)(0,2) = -u[0] * std::sin(x[2]);
    (*dxdot_dx)(1,2) = u[0] * std::cos(x[2]);
  }
}

void CARTTCSGDProblem::GetDynamicsDeriv(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                                        Eigen::MatrixXf* dxtpdt_du, Eigen::MatrixXf* dxtpdt_dx) {
  float len_scale = 2.0f / std::sqrt(5.0f);
  float L = 2.0f * len_scale * params.radius;
  float th_new = x[2] + dt * u[0] / L * std::tan(u[1]);

  if (dxtpdt_dx != nullptr) {
    (*dxtpdt_dx) = Eigen::MatrixXf::Identity(3, 3);
    (*dxtpdt_dx)(0, 2) = -0.5f * dt * u[0] * (std::sin(x[2]) + std::sin(th_new));
    (*dxtpdt_dx)(1, 2) = 0.5f * dt * u[0] * (std::cos(x[2]) + std::cos(th_new));
  }

  if (dxtpdt_du != nullptr) {
    (*dxtpdt_du) = Eigen::MatrixXf::Zero(3, 2);
    (*dxtpdt_du)(2, 0) = dt * std::tan(u[1]) / L;
    (*dxtpdt_du)(2, 1) = u[0] * dt / (L * std::cos(u[1]) * std::cos(u[1]));

    (*dxtpdt_du)(0, 0) =
        0.5f * dt *
        (std::cos(x[2]) + std::cos(th_new) - u[0] * std::sin(th_new) * (*dxtpdt_du)(2, 0));
    (*dxtpdt_du)(1, 0) =
        0.5f * dt *
        (std::sin(x[2]) + std::sin(th_new) + u[0] * std::cos(th_new) * (*dxtpdt_du)(2, 0));

    (*dxtpdt_du)(0, 1) = -0.5f * dt * u[0] * std::sin(th_new) * (*dxtpdt_du)(2, 1);
    (*dxtpdt_du)(1, 1) = 0.5f * dt * u[0] * std::cos(th_new) * (*dxtpdt_du)(2, 1);
  }
}

Eigen::Vector2f CARTTCSGDProblem::GetCollisionCenter(const Eigen::VectorXf &x, Eigen::MatrixXf* dc_dx) {
  float len_scale = 2.0f / std::sqrt(5.0f);
  if (dc_dx != nullptr) {
    (*dc_dx) = Eigen::MatrixXf::Identity(2,x.size());
    (*dc_dx)(0,2) = -len_scale * params.radius * std::sin(x[2]);
    (*dc_dx)(1,2) = len_scale * params.radius * std::cos(x[2]);
  }
  return x.head<2>() + len_scale * params.radius * Eigen::Vector2f(std::cos(x[2]), std::sin(x[2]));
}

TTCObstacle* CARTTCSGDProblem::CreateObstacle() {
  TTCObstacle* o = new CARTTCObstacle();
  FillObstacle(o);
  return o;
}

// --------------------------------------------
// ****************** A Car *******************
// --------------------------------------------
void ACARTTCSGDProblem::ContDynamics(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, Eigen::VectorXf* x_dot,
                                    Eigen::MatrixXf* dxdot_du, Eigen::MatrixXf* dxdot_dx) {
  float len_scale = 2.0f / std::sqrt(5.0f);
  float L = 2.0f * len_scale * params.radius;
  // leaky constraints on velocities
  float scale_v =
      ((x[3] > params.vel_limit && u[0] > 0.0f) || (x[3] < -params.vel_limit && u[0] < 0.0f)) ? 1e-2f
                                                                      : 1.0f;
  float scale_phi =
      ((x[4] > params.steer_limit && u[1] > 0.0f) || (x[4] < -params.steer_limit && u[1] < 0.0f)) ? 1e-2f
                                                                      : 1.0f;
  if (x_dot != nullptr) {
    (*x_dot) = Eigen::VectorXf::Zero(5);
    (*x_dot)[0] = x[3] * std::cos(x[2]);
    (*x_dot)[1] = x[3] * std::sin(x[2]);
    (*x_dot)[2] = x[3] * std::tan(x[4]) / L;
    (*x_dot)[3] = scale_v * u[0];
    (*x_dot)[4] = scale_phi * u[1];
  }
  if (dxdot_du != nullptr) {
    (*dxdot_du) = Eigen::MatrixXf::Zero(5,2);
    (*dxdot_du)(3,0) = scale_v;
    (*dxdot_du)(4,1) = scale_phi;
  }
  if (dxdot_dx != nullptr) {
    (*dxdot_dx) = Eigen::MatrixXf::Zero(5,5);
    (*dxdot_dx)(0,2) = -x[3] * std::sin(x[2]);
    (*dxdot_dx)(0,3) = std::cos(x[2]);
    (*dxdot_dx)(1,2) = x[3] * std::cos(x[2]);
    (*dxdot_dx)(1,3) = std::sin(x[2]);
    (*dxdot_dx)(2,3) = std::tan(x[4]) / L;
    (*dxdot_dx)(2,4) = x[3] / (std::cos(x[4]) * std::cos(x[4]) * L);
  }
}

void ACARTTCSGDProblem::GetDynamicsDeriv(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                                      Eigen::MatrixXf* dxtpdt_du, Eigen::MatrixXf* dxtpdt_dx) {
  float len_scale = 2.0f / std::sqrt(5.0f);
  float L = 2.0f * len_scale * params.radius;
  // leaky constraints on velocities
  float scale_v =
      ((x[3] > params.vel_limit && u[0] > 0.0f) || (x[3] < -params.vel_limit && u[0] < 0.0f)) ? 1e-2f
                                                                      : 1.0f;
  float scale_phi =
      ((x[4] > params.steer_limit && u[1] > 0.0f) || (x[4] < -params.steer_limit && u[1] < 0.0f)) ? 1e-2f
                                                                      : 1.0f;

  float v_new = x[3] + dt * scale_v * u[0];

  float tan_phi = std::tan(x[4]);
  float cos_phi = std::cos(x[4]);
  float phi_new = x[4] + dt * scale_phi * u[1];
  float tan_phi_new = std::tan(phi_new);
  float cos_phi_new = std::cos(phi_new);
  
  float cos_th = std::cos(x[2]);
  float sin_th = std::sin(x[2]);
  float th_new = x[2] + 0.5f * dt / L * (x[3]*tan_phi + v_new*tan_phi_new);
  float cos_th_new = std::cos(th_new);
  float sin_th_new = std::sin(th_new);

  if (dxtpdt_dx != nullptr) {
    (*dxtpdt_dx) = Eigen::MatrixXf::Identity(5, 5);

    (*dxtpdt_dx)(2, 3) = 0.5f * dt / L * (tan_phi + tan_phi_new);
    (*dxtpdt_dx)(2, 4) = 0.5f * dt / L * (x[3] / (cos_phi * cos_phi) + v_new / (cos_phi_new * cos_phi_new));

    (*dxtpdt_dx)(0, 2) = -0.5f * dt * (x[3] * sin_th + v_new * sin_th_new);
    (*dxtpdt_dx)(0, 3) = 0.5f * dt * (cos_th + cos_th_new - v_new * sin_th_new * (*dxtpdt_dx)(2, 3));
    (*dxtpdt_dx)(0, 4) = -0.5f * dt * v_new * sin_th_new * (*dxtpdt_dx)(2, 4);

    (*dxtpdt_dx)(1, 2) = 0.5f * dt * (x[3] * cos_th + v_new * cos_th_new);
    (*dxtpdt_dx)(1, 3) = 0.5f * dt * (sin_th + sin_th_new + v_new * cos_th_new * (*dxtpdt_dx)(2, 3));
    (*dxtpdt_dx)(1, 4) = 0.5f * dt * v_new * cos_th_new * (*dxtpdt_dx)(2, 4);
  }
  if (dxtpdt_du != nullptr) {
    (*dxtpdt_du) = Eigen::MatrixXf::Zero(5, 2);
    (*dxtpdt_du)(3, 0) = dt * scale_v;
    (*dxtpdt_du)(4, 1) = dt * scale_phi;

    (*dxtpdt_du)(2, 0) = 0.5f * dt / L * tan_phi_new * (*dxtpdt_du)(3, 0);
    (*dxtpdt_du)(2, 1) = 0.5f * dt * v_new * (*dxtpdt_du)(4, 1) / (L * cos_phi_new * cos_phi_new);

    (*dxtpdt_du)(0, 0) = 0.5f * dt * (cos_th_new * (*dxtpdt_du)(3, 0) - v_new * sin_th_new * (*dxtpdt_du)(2, 0));
    (*dxtpdt_du)(0, 1) = -0.5f * dt * v_new * sin_th_new * (*dxtpdt_du)(2, 1);

    (*dxtpdt_du)(1, 0) = 0.5f * dt * (sin_th_new * (*dxtpdt_du)(3, 0) + v_new * cos_th_new * (*dxtpdt_du)(2, 0));
    (*dxtpdt_du)(1, 1) = 0.5f * dt * v_new * cos_th_new * (*dxtpdt_du)(2, 1);
  }
}

Eigen::Vector2f ACARTTCSGDProblem::GetCollisionCenter(const Eigen::VectorXf &x, Eigen::MatrixXf* dc_dx) {
  float len_scale = 2.0f / std::sqrt(5.0f);
  float s_th = std::sin(x[2]);
  float c_th = std::cos(x[2]);
  if (dc_dx != nullptr) {
    (*dc_dx) = Eigen::MatrixXf::Identity(2,x.size());
    (*dc_dx)(0,2) = -len_scale * params.radius * s_th;
    (*dc_dx)(1,2) = len_scale * params.radius * c_th;
  }
  return x.head<2>() + len_scale * params.radius * Eigen::Vector2f(c_th, s_th);
}

void ACARTTCSGDProblem::AdditionalCosts(const Eigen::VectorXf &u, float* cost, Eigen::VectorXf* dc_du) {
  // Velocity Constraint Cost:
  Eigen::VectorXf x_t;
  Eigen::MatrixXf dxt_du;
  if (dc_du != nullptr) {
    DynamicsLong(u, params.x_0, 0.0f, 1.0f, &x_t, &dxt_du, nullptr);
  } else {
    DynamicsLong(u, params.x_0, 0.0f, 1.0f, &x_t);
  }
  if (std::abs(x_t[3]) > params.vel_limit) {
    float err;
    if (x_t[3] > params.vel_limit) {
      err = x_t[3] - params.vel_limit;
    } else {
      err = x_t[3] + params.vel_limit; // x_t[3] is negative
    }
    if (cost != nullptr) {
      (*cost) += params.k_v_constr * err * err;
    }
    if (dc_du != nullptr) {
      (*dc_du) += 2.0 * params.k_v_constr * err * dxt_du.row(3);
    }
  }
  if (std::abs(x_t[4]) > params.steer_limit) {
    float err;
    if (x_t[4] > params.steer_limit) {
      err = x_t[4] - params.steer_limit;
    } else {
      err = x_t[4] + params.steer_limit; // x_t[3] is negative
    }
    if (cost != nullptr) {
      (*cost) += params.k_v_constr * err * err;
    }
    if (dc_du != nullptr) {
      (*dc_du) += 2.0 * params.k_v_constr * err * dxt_du.row(4);
    }
  }
}

void ACARTTCSGDProblem::ProjConstr(Eigen::VectorXf& u) {
  // Trim u to control constraints
  TTCSGDProblem::ProjConstr(u);

  // Trim u to state constraints
  float v_new = params.x_0[3] + params.dt_step * u[0];
  if (v_new > params.vel_limit) {
    u[0] = (params.vel_limit - params.x_0[3]) / params.dt_step;
  }
  float phi_new = params.x_0[4] + params.dt_step * u[1];
  if (phi_new > params.steer_limit) {
    u[1] = (params.steer_limit - params.x_0[4]) / params.dt_step;
  }
}

TTCObstacle* ACARTTCSGDProblem::CreateObstacle() {
  TTCObstacle* o = new ACARTTCObstacle();
  FillObstacle(o);
  return o;
}