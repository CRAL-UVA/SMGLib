/*
Copyright 2020 University of Minnesota and Clemson University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/*
This file contains the implementations of the dynamics models for each obstacle type: Velocity, Acceleration, Differential Drive, Smooth Differential Drive, Simple Car, Smooth Car
*/

#include <sgd/ttc_obstacles.h>

void TTCObstacle::Dynamics(const Eigen::VectorXf &u_t,
                           const Eigen::VectorXf &x_t,
                           const float t,
                           float dt,
                           Eigen::VectorXf* x_tpdt) const {
  if (x_tpdt != nullptr) {
    // RK4
    Eigen::VectorXf k1, k2, k3, k4;
    Eigen::VectorXf x2, x3, x4;
    ContDynamics(u_t, x_t, t, &k1);
    x2 = x_t + 0.5f * dt * k1;
    ContDynamics(u_t, x2, t + 0.5f * dt, &k2);
    x3 = x_t + 0.5f * dt * k2;
    ContDynamics(u_t, x3, t + 0.5f * dt, &k3);
    x4 = x_t + dt * k3;
    ContDynamics(u_t, x4, t + dt, &k4);

    (*x_tpdt) = x_t + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
  }
}

Eigen::Vector2f TTCObstacle::GetCollisionCenter(const Eigen::VectorXf &x_t, Eigen::MatrixXf* dc_dx) {
  if (dc_dx != nullptr) {
    (*dc_dx) = Eigen::MatrixXf::Identity(2,x_t.size());
  }
  return x_t.head<2>();
}

void VTTCObstacle::ContDynamics(const Eigen::VectorXf &u_t,
                                const Eigen::VectorXf &x_t,
                                const float t,
                                Eigen::VectorXf* x_dot) const {
  if (x_dot != nullptr) {
    (*x_dot) = u_t;
  }
}

void ATTCObstacle::ContDynamics(const Eigen::VectorXf &u_t,
                                const Eigen::VectorXf &x_t,
                                const float t,
                                Eigen::VectorXf* x_dot) const {
  if (x_dot != nullptr) {
    (*x_dot) = Eigen::VectorXf::Zero(4);
    x_dot->head<2>() = x_t.tail<2>();
    x_dot->tail<2>() = u_t;
  }
}

void DDTTCObstacle::ContDynamics(const Eigen::VectorXf &u_t,
                                const Eigen::VectorXf &x_t,
                                const float t,
                                Eigen::VectorXf* x_dot) const {
  if (x_dot != nullptr) {
    (*x_dot) = Eigen::VectorXf::Zero(3);
    (*x_dot)[0] = u_t[0] * std::cos(x_t[2]);
    (*x_dot)[1] = u_t[0] * std::sin(x_t[2]);
    (*x_dot)[2] = u_t[1];
  }
}

void ADDTTCObstacle::ContDynamics(const Eigen::VectorXf &u_t,
                                const Eigen::VectorXf &x_t,
                                const float t,
                                Eigen::VectorXf* x_dot) const {
  if (x_dot != nullptr) {
    (*x_dot) = Eigen::VectorXf::Zero(5);
    (*x_dot)[0] = x_t[3] * std::cos(x_t[2]);
    (*x_dot)[1] = x_t[3] * std::sin(x_t[2]);
    (*x_dot)[2] = x_t[4];
    (*x_dot)[3] = u_t[0];
    (*x_dot)[4] = u_t[1];
  }
}

void CARTTCObstacle::ContDynamics(const Eigen::VectorXf &u_t,
                                const Eigen::VectorXf &x_t,
                                const float t,
                                Eigen::VectorXf* x_dot) const {
  float len_scale = 2.0f * std::sqrt(5.0f) / 5.0f;
  float L = 2.0f * len_scale * radius;
  if (x_dot != nullptr) {
    (*x_dot) = Eigen::VectorXf::Zero(3);
    (*x_dot)[0] = u_t[0] * std::cos(x_t[2]);
    (*x_dot)[1] = u_t[0] * std::sin(x_t[2]);
    (*x_dot)[2] = u_t[0] * std::tan(u_t[1]) / L;
  }
}
Eigen::Vector2f CARTTCObstacle::GetCollisionCenter(const Eigen::VectorXf &x_t, Eigen::MatrixXf* dc_dx) {
  float len_scale = 2.0f * std::sqrt(5.0f) / 5.0f;
  if (dc_dx != nullptr) {
    (*dc_dx) = Eigen::MatrixXf::Identity(2,x_t.size());
    (*dc_dx)(0,2) = -len_scale * radius * std::sin(x_t[2]);
    (*dc_dx)(1,2) = len_scale * radius * std::cos(x_t[2]);
  }
  return x_t.head<2>() + len_scale * radius * Eigen::Vector2f(std::cos(x_t[2]), std::sin(x_t[2]));
}

void ACARTTCObstacle::ContDynamics(const Eigen::VectorXf &u_t,
                                const Eigen::VectorXf &x_t,
                                const float t,
                                Eigen::VectorXf* x_dot) const {
  float len_scale = 2.0f * std::sqrt(5.0f) / 5.0f;
  float L = 2.0f * len_scale * radius;
  if (x_dot != nullptr) {
    (*x_dot) = Eigen::VectorXf::Zero(5);
    (*x_dot)[0] = x_t[3] * std::cos(x_t[2]);
    (*x_dot)[1] = x_t[3] * std::sin(x_t[2]);
    (*x_dot)[2] = x_t[3] * std::tan(x_t[4]) / L;
    (*x_dot)[3] = u_t[0];
    (*x_dot)[4] = u_t[1];
  }
}
Eigen::Vector2f ACARTTCObstacle::GetCollisionCenter(const Eigen::VectorXf &x_t, Eigen::MatrixXf* dc_dx) {
  float len_scale = 2.0f * std::sqrt(5.0f) / 5.0f;
  if (dc_dx != nullptr) {
    (*dc_dx) = Eigen::MatrixXf::Identity(2,x_t.size());
    (*dc_dx)(0,2) = -len_scale * radius * std::sin(x_t[2]);
    (*dc_dx)(1,2) = len_scale * radius * std::cos(x_t[2]);
  }
  return x_t.head<2>() + len_scale * radius * Eigen::Vector2f(std::cos(x_t[2]), std::sin(x_t[2]));
}