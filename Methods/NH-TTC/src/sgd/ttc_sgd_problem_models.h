/*
Copyright 2020 University of Minnesota and Clemson University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#pragma once
#include <sgd/ttc_sgd_problem.h>

class VTTCSGDProblem : public TTCSGDProblem {
public:
  VTTCSGDProblem(TTCParams params_in) {
    params = params_in;
  }
  void ContDynamics(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, Eigen::VectorXf* x_dot,
                    Eigen::MatrixXf* dxdot_du, Eigen::MatrixXf* dxdot_dx);
  void GetDynamicsDeriv(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                                Eigen::MatrixXf* dxtpdt_du, Eigen::MatrixXf* dxtpdt_dx);
  TTCObstacle* CreateObstacle();
};

class ATTCSGDProblem : public TTCSGDProblem {
public:
  ATTCSGDProblem(TTCParams params_in) {
    params = params_in;
  }
  void ContDynamics(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, Eigen::VectorXf* x_dot,
                    Eigen::MatrixXf* dxdot_du, Eigen::MatrixXf* dxdot_dx);
  void GetDynamicsDeriv(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                                Eigen::MatrixXf* dxtpdt_du, Eigen::MatrixXf* dxtpdt_dx);
  void AdditionalCosts(const Eigen::VectorXf &u, float* cost, Eigen::VectorXf* dc_du);
  void ProjConstr(Eigen::VectorXf& u);
  TTCObstacle* CreateObstacle();
};

class DDTTCSGDProblem : public TTCSGDProblem {
public:
  DDTTCSGDProblem(TTCParams params_in) {
    params = params_in;
  }
  void ContDynamics(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, Eigen::VectorXf* x_dot,
                    Eigen::MatrixXf* dxdot_du, Eigen::MatrixXf* dxdot_dx);
  void GetDynamicsDeriv(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                                Eigen::MatrixXf* dxtpdt_du, Eigen::MatrixXf* dxtpdt_dx);
  TTCObstacle* CreateObstacle();
};

class ADDTTCSGDProblem : public TTCSGDProblem {
public:
  ADDTTCSGDProblem(TTCParams params_in) {
    params = params_in;
  }
  void ContDynamics(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, Eigen::VectorXf* x_dot,
                    Eigen::MatrixXf* dxdot_du, Eigen::MatrixXf* dxdot_dx);
  void GetDynamicsDeriv(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                                Eigen::MatrixXf* dxtpdt_du, Eigen::MatrixXf* dxtpdt_dx);
  void AdditionalCosts(const Eigen::VectorXf &u, float* cost, Eigen::VectorXf* dc_du);
  void ProjConstr(Eigen::VectorXf& u);
  TTCObstacle* CreateObstacle();
};

class CARTTCSGDProblem : public TTCSGDProblem {
public:
  CARTTCSGDProblem(TTCParams params_in) {
    params = params_in;
  }
  void ContDynamics(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, Eigen::VectorXf* x_dot,
                    Eigen::MatrixXf* dxdot_du, Eigen::MatrixXf* dxdot_dx);
  void GetDynamicsDeriv(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                                Eigen::MatrixXf* dxtpdt_du, Eigen::MatrixXf* dxtpdt_dx);
  Eigen::Vector2f GetCollisionCenter(const Eigen::VectorXf &x, Eigen::MatrixXf* dc_dx);
  TTCObstacle* CreateObstacle();
};

class ACARTTCSGDProblem : public TTCSGDProblem {
public:
  ACARTTCSGDProblem(TTCParams params_in) {
    params = params_in;
  }
  void ContDynamics(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, Eigen::VectorXf* x_dot,
                    Eigen::MatrixXf* dxdot_du, Eigen::MatrixXf* dxdot_dx);
  void GetDynamicsDeriv(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                                Eigen::MatrixXf* dxtpdt_du, Eigen::MatrixXf* dxtpdt_dx);
  Eigen::Vector2f GetCollisionCenter(const Eigen::VectorXf &x, Eigen::MatrixXf* dc_dx);
  void AdditionalCosts(const Eigen::VectorXf &u, float* cost, Eigen::VectorXf* dc_du);
  void ProjConstr(Eigen::VectorXf& u);
  TTCObstacle* CreateObstacle();
};