/*
Copyright 2020 University of Minnesota and Clemson University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#pragma once
#include <sgd/sgd.h>
#include <sgd/ttc_obstacles.h>

#include <vector>

#ifndef M_PI
#define M_PI 3.141592
#endif

enum class TTCPow { ONE, TWO };

struct TTCParams {
  float k_goal = 1.0f;
  std::vector<float> ts_goal_check = {1.0f};
  std::vector<Eigen::Vector2f> goals;

  bool use_ttc_cost = true;
  float k_ttc = 1.0f;
  TTCPow ttc_power = TTCPow::ONE;
  bool find_all_ttcs = true;

  float k_dist = 0.1f;
  float dist_smooth = -10.0f;

  Eigen::VectorXf x_0;
  Eigen::VectorXf u_curr;
  std::vector<TTCObstacle*> obsts;

  float dt_step = 0.1f;
  float max_ttc = 5.0f;

  // Collision Parameters
  float radius = 0.1f;
  float safety_radius = 0.05f;

  // Constraint Parameters
  float vel_limit = 0.3f;
  float rot_vel_limit = 1.0f;
  float steer_limit = 0.25f * static_cast<float>(M_PI);

  float k_constr = 10.0f;
  float k_v_constr = 10.0f;
  bool box_constraint = true;
  Eigen::VectorXf u_lb, u_ub;
  Eigen::VectorXf x_lb, x_ub;
  float circle_u_limit = 0.3f;
};

class TTCSGDProblem : public SGDProblem {
public:
  TTCParams params;
  std::vector<std::vector<Eigen::VectorXf>> obst_poses;
  std::vector<std::vector<Eigen::Vector2f>> obst_centers;
  std::vector<float> obst_comb_rad;
  
  void InitProblem();

  float GetTTCLin(const float x1_x, const float x1_y, const float x2_x, const float x2_y,
                  const float o1_x, const float o1_y, const float o2_x, const float o2_y,
                  const float comb_rad);

  std::vector<float> GetTTCDisc(const Eigen::VectorXf &u,
                                std::vector<Eigen::VectorXf> *P = nullptr, std::vector<Eigen::VectorXf> *O = nullptr,
                                std::vector<Eigen::MatrixXf> *dPdu = nullptr, std::vector<Eigen::VectorXf> *dPdt = nullptr,
                                std::vector<Eigen::VectorXf> *dOdt = nullptr);

  void TTCCost(const Eigen::VectorXf &u, float* cost, Eigen::VectorXf* dc_du);
  void DistCost(const Eigen::VectorXf& u, float* cost, Eigen::VectorXf* dc_du);

  void GoalCost(const Eigen::VectorXf &u, float* cost, Eigen::VectorXf* dc_du);

  virtual void ProjConstr(Eigen::VectorXf& u);

  virtual void AdditionalCosts(const Eigen::VectorXf &u, float* cost, Eigen::VectorXf* dc_du) {};

  void Cost(const Eigen::VectorXf& u, float* cost, Eigen::VectorXf* dc_du);

  virtual void ContDynamics(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, Eigen::VectorXf* x_dot,
                            Eigen::MatrixXf* dxdot_du = nullptr, Eigen::MatrixXf* dxdot_dx = nullptr) = 0;

  void Dynamics(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                Eigen::VectorXf* x_tpdt, Eigen::MatrixXf* dxtpdt_du = nullptr, Eigen::MatrixXf* dxtpdt_dx = nullptr);
  virtual void GetDynamicsDeriv(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                                Eigen::MatrixXf* dxtpdt_du, Eigen::MatrixXf* dxtpdt_dx) = 0;

  void DynamicsLong(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                    Eigen::VectorXf* x_tpdt, Eigen::MatrixXf* dxtpdt_du = nullptr, Eigen::MatrixXf* dxtpdt_dx = nullptr);

  void SetParams(TTCParams params_in) {
    params = params_in;
  };

  virtual Eigen::Vector2f GetCollisionCenter(const Eigen::VectorXf &x, Eigen::MatrixXf* dc_dx = nullptr);

  virtual TTCObstacle* CreateObstacle() = 0;
  void FillObstacle(TTCObstacle* o);

  virtual ~TTCSGDProblem(){};
};