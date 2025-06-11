/*
Copyright 2020 University of Minnesota and Clemson University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/*
This file contains the base collision avoidance optmization problem.  Both the collision cost and the goal cost are implemented here.
*/

#include <sgd/ttc_sgd_problem.h>

#include <iostream>
#include <iomanip>

void TTCSGDProblem::InitProblem() {
  // Propagate the obstacles forward to compute trajectories for use in collision checking
  obst_poses.clear();
  obst_centers.clear();
  obst_comb_rad.clear();
  if (params.obsts.size() == 0) {
    return;
  }
  Eigen::Vector2f x_c = GetCollisionCenter(params.x_0);
  // Place initial poses:
  obst_comb_rad = std::vector<float>(params.obsts.size());
  obst_poses.push_back(std::vector<Eigen::VectorXf>(params.obsts.size()));
  obst_centers.push_back(std::vector<Eigen::Vector2f>(params.obsts.size()));
  for (size_t o_idx = 0; o_idx < params.obsts.size(); ++o_idx) {
    obst_poses[0][o_idx] = params.obsts[o_idx]->p;

    obst_comb_rad[o_idx] = params.radius + params.obsts[o_idx]->radius + params.safety_radius;
    Eigen::Vector2f o_c = params.obsts[o_idx]->GetCollisionCenter(obst_poses[0][o_idx]);
    obst_centers[0][o_idx] = o_c;
    float init_dist = (o_c - x_c).norm();
    // If we start in collision, modify radius so we're slightly out of it
    if (init_dist <= obst_comb_rad[o_idx]) {
      obst_comb_rad[o_idx] = init_dist - 0.01f;
    }
  }

  int max_steps = static_cast<int>(std::ceil(params.max_ttc / params.dt_step));
  // Propagate forward:
  for (int t = 0; t < max_steps; ++t) {
    obst_poses.push_back(std::vector<Eigen::VectorXf>(params.obsts.size()));
    obst_centers.push_back(std::vector<Eigen::Vector2f>(params.obsts.size()));
    for (size_t o_idx = 0; o_idx < params.obsts.size(); ++o_idx) {
      params.obsts[o_idx]->Dynamics(params.obsts[o_idx]->u, obst_poses[t][o_idx], t * params.dt_step, params.dt_step, &(obst_poses[t+1][o_idx]));
      obst_centers.back()[o_idx] = params.obsts[o_idx]->GetCollisionCenter(obst_poses[t+1][o_idx]);
    }
  }
}

float TTCSGDProblem::GetTTCLin(const float x1_x, const float x1_y, const float x2_x, const float x2_y,
                               const float o1_x, const float o1_y, const float o2_x, const float o2_y,
                               const float comb_rad) {
  float pox = o1_x - x1_x;
  float poy = o1_y - x1_y;
  float xvx = (x2_x - x1_x) / params.dt_step;
  float xvy = (x2_y - x1_y) / params.dt_step;
  float ovx = (o2_x - o1_x) / params.dt_step;
  float ovy = (o2_y - o1_y) / params.dt_step;
  float dvx = xvx - ovx;
  float dvy = xvy - ovy;

  float a = dvx*dvx + dvy*dvy;
  float b = -(dvx*pox + dvy*poy);
  float c = pox*pox + poy*poy - comb_rad*comb_rad;

  float ttc = std::numeric_limits<float>::max();
  if (c < 0.0f) {
    ttc = 0.0f;
    return ttc;
  }

  if (b < 0.0f && std::abs(a) > 0.0f) { // to have a collision that agents must
                                        // approach each other (i.e. b<0)
    float disc = b * b - a * c;
    if (disc > 0.0f) {
      float t =
          (-b - std::sqrt(disc)) /
          a; // the smallest possible positive root is always given by this
      if (t > 0.0f) {
        ttc = t;
      }
    }
  }
  return ttc;
}

std::vector<float> TTCSGDProblem::GetTTCDisc(const Eigen::VectorXf &u,
                                std::vector<Eigen::VectorXf> *P, std::vector<Eigen::VectorXf> *O,
                                std::vector<Eigen::MatrixXf> *dPdu, std::vector<Eigen::VectorXf> *dPdt,
                                std::vector<Eigen::VectorXf> *dOdt) {
  if (P != nullptr) { P->clear(); }
  if (O != nullptr) { O->clear(); }
  if (dPdu != nullptr) { dPdu->clear(); }
  if (dPdt != nullptr) { dPdt->clear(); }
  if (dOdt != nullptr) { dOdt->clear(); }
  std::vector<float> ttcs;
  Eigen::VectorXf x_t = params.x_0;
  Eigen::Vector2f xt_c = GetCollisionCenter(x_t);
  Eigen::MatrixXf B = Eigen::MatrixXf::Zero(x_t.size(), u.size());
  std::vector<bool> found_coll(params.obsts.size(), false);
  size_t n_coll_found = 0;

  bool compute_derivs = (P != nullptr);

  int max_steps = static_cast<int>(std::ceil(params.max_ttc / params.dt_step));
  for (int i = 0; i < max_steps; ++i) {
    float t = i * params.dt_step;
    const std::vector<Eigen::VectorXf>& os_t = obst_poses[i];
    const std::vector<Eigen::Vector2f>& os_c_t = obst_centers[i];
    const std::vector<Eigen::VectorXf>& os_tp1 = obst_poses[i+1];
    const std::vector<Eigen::Vector2f>& os_c_tp1 = obst_centers[i+1];
    // Get new positions:
    Eigen::VectorXf x_tp1;
    Eigen::MatrixXf A_t, B_t;
    if (compute_derivs) {
      Dynamics(u, x_t, t, params.dt_step, &x_tp1, &B_t, &A_t);
    } else {
      Dynamics(u, x_t, t, params.dt_step, &x_tp1);
    }
    
    // Check if there are any collisions:
    std::vector<std::pair<float,int>> colls;
    Eigen::Vector2f xtp1_c = GetCollisionCenter(x_tp1);
    for (size_t o_idx = 0; o_idx < os_tp1.size(); ++o_idx) {
      // Skip any obstacle we've already found a collision with
      if (found_coll[o_idx]) { continue; }
      float comb_rad = obst_comb_rad[o_idx];      
      const Eigen::Vector2f& ot_c = os_c_t[o_idx];
      const Eigen::Vector2f& otp1_c = os_c_tp1[o_idx];
      float ttc = GetTTCLin(xt_c[0],xt_c[1], xtp1_c[0],xtp1_c[1], ot_c[0],ot_c[1], otp1_c[0],otp1_c[1], comb_rad);
      if (ttc >= 0.0 && ttc <= params.dt_step) {
        colls.emplace_back(ttc, o_idx);
      }
    }
    // Sort the collisions by increasing ttc
    std::sort(colls.begin(), colls.end());

    // If we've found a collision, compute necessary derivatives:
    for (const std::pair<float,int>& coll : colls) {
      if (compute_derivs) {
        Eigen::VectorXf P_c, O_c, dPdt_c, dOdt_c;
        Eigen::MatrixXf A_c, B_c;
        Dynamics(u, x_t, t, coll.first, &P_c, &B_c, &A_c);
        params.obsts[coll.second]->Dynamics(params.obsts[coll.second]->u, os_t[coll.second], t, coll.first, &O_c);

        ContDynamics(u, P_c, t+coll.first, &dPdt_c);
        params.obsts[coll.second]->ContDynamics(params.obsts[coll.second]->u, O_c, t+coll.first, &dOdt_c);

        Eigen::MatrixXf dxc_dP, doc_dO;
        Eigen::Vector2f xt_col = GetCollisionCenter(P_c, &dxc_dP);
        Eigen::Vector2f ot_col = params.obsts[coll.second]->GetCollisionCenter(O_c, &doc_dO);

        if (P != nullptr) { P->push_back(xt_col); }
        if (O != nullptr) { O->push_back(ot_col); }
        if (dPdu != nullptr) { dPdu->push_back(dxc_dP * (A_c * B + B_c)); }
        if (dPdt != nullptr) { dPdt->push_back(dxc_dP * dPdt_c); }
        if (dOdt != nullptr) { dOdt->push_back(doc_dO * dOdt_c); }
      }

      ttcs.push_back(static_cast<float>(i) * params.dt_step  + coll.first);
      found_coll[coll.second] = true;
      n_coll_found++;
      // Break once we've found all collisions, or the first collision if find_all_ttcs is false
      if (n_coll_found == os_t.size() || !params.find_all_ttcs) {
        return ttcs;
      }
    }

    // Othewise update B, x, and o
    if (compute_derivs) {
      B = A_t * B + B_t;
    }
    x_t = x_tp1;
    xt_c = xtp1_c;
  }

  // Return any found collisions
  return ttcs;
}

void TTCSGDProblem::TTCCost(const Eigen::VectorXf &u, float* cost, Eigen::VectorXf* dc_du) {
  if (params.obsts.size() == 0) {
    return;
  }
  std::vector<float> ttc;
  std::vector<Eigen::VectorXf> P, O, dPdt, dOdt;
  std::vector<Eigen::MatrixXf> dPdu;
  if (dc_du != nullptr) {
    ttc = GetTTCDisc(u, &P, &O, &dPdu, &dPdt, &dOdt);
  } else {
    ttc = GetTTCDisc(u);
  }

  for (size_t ttc_idx = 0; ttc_idx < ttc.size(); ++ttc_idx) {
    if (ttc[ttc_idx] > 0.0f && ttc[ttc_idx] != std::numeric_limits<float>::max()) {
      if (cost != nullptr) {
        if (params.ttc_power == TTCPow::ONE) {
          (*cost) += params.k_ttc / ttc[ttc_idx];
        } else if (params.ttc_power == TTCPow::TWO) {
          (*cost) += params.k_ttc / (ttc[ttc_idx] * ttc[ttc_idx]);
        } else {
          std::cerr << "Unimplemented TTC Power!" << std::endl;
          exit(-1);
        }
      }
      if (dc_du != nullptr) {
        float dC_dttc;
        if (params.ttc_power == TTCPow::ONE) {
          dC_dttc = -1.0f * params.k_ttc / (ttc[ttc_idx] * ttc[ttc_idx]);
        } else if (params.ttc_power == TTCPow::TWO) {
          dC_dttc = -2.0f * params.k_ttc / (ttc[ttc_idx] * ttc[ttc_idx] * ttc[ttc_idx]);
        } else {
          std::cerr << "Unimplemented TTC Power Deriv!" << std::endl;
          exit(-1);
        }

        float dx = P[ttc_idx](0) - O[ttc_idx](0);
        float dy = P[ttc_idx](1) - O[ttc_idx](1);

        // Note: this is missing a factor of 2, but the factor drops out in the
        // division to get dttc_du
        float df_dttc = dx * (dPdt[ttc_idx](0) - dOdt[ttc_idx](0)) + dy * (dPdt[ttc_idx](1) - dOdt[ttc_idx](1));
        Eigen::VectorXf df_du = dx * dPdu[ttc_idx].row(0) + dy * dPdu[ttc_idx].row(1);
        Eigen::VectorXf dttc_du = -df_du / df_dttc;

        (*dc_du) += dC_dttc * dttc_du;
      }
    }
  }
}

void TTCSGDProblem::DistCost(const Eigen::VectorXf& u, float* cost, Eigen::VectorXf* dc_du) {
  if (params.obsts.size() == 0) {
	return;
  }
  Eigen::VectorXf x_t = params.x_0;
  Eigen::MatrixXf dxt_du = Eigen::MatrixXf::Zero(x_t.size(), u.size());
  
  bool compute_derivs = (dc_du != nullptr);
  
  int max_steps = static_cast<int>(std::ceil(params.max_ttc / params.dt_step));
  for (int i = 0; i < max_steps; ++i) {
    float t = i * params.dt_step;
    const std::vector<Eigen::Vector2f>& os_c_t = obst_centers[i];

    // Get new position:
    Eigen::VectorXf x_tp1;
    Eigen::Vector2f c_t;
    Eigen::MatrixXf dct_dxt;
    if (compute_derivs) {
      Eigen::MatrixXf A_t, B_t;
      Dynamics(u, x_t, t, params.dt_step, &x_tp1, &B_t, &A_t);
      dxt_du = A_t * dxt_du + B_t;
      c_t = GetCollisionCenter(x_tp1, &dct_dxt);
    } else {
      Dynamics(u, x_t, t, params.dt_step, &x_tp1);
      c_t = GetCollisionCenter(x_tp1);
    }
    x_t = x_tp1;

    for (size_t j = 0; j < os_c_t.size(); ++j) {
      Eigen::Vector2f dist_vec = os_c_t[j] - c_t;
      float d = dist_vec.norm();
      d = d - obst_comb_rad[j];
      float penalty = 0.0f;
      if (d < 0) {
        penalty = std::exp(params.dist_smooth * d);
      }

      if (cost != nullptr) {
        (*cost) += params.k_dist * penalty;
      }
      if (compute_derivs) {
        (*dc_du) += params.k_dist * penalty * params.dist_smooth * dist_vec.normalized().transpose() * dct_dxt * dxt_du;
      }
    }
  }
}

void TTCSGDProblem::GoalCost(const Eigen::VectorXf &u, float* cost, Eigen::VectorXf* dc_du) {
  Eigen::VectorXf cen_pos_0 = GetCollisionCenter(params.x_0);
  for (size_t i = 0; i < params.goals.size(); ++i) {
    float curr_dist = (cen_pos_0 - params.goals[i]).norm();
    float t_goal_check = params.ts_goal_check[i];
    Eigen::VectorXf x_t;
    Eigen::MatrixXf dxt_du;
    if (dc_du != nullptr) {
      DynamicsLong(u, params.x_0, 0.0f, t_goal_check, &x_t, &dxt_du, nullptr);
    } else {
      DynamicsLong(u, params.x_0, 0.0f, t_goal_check, &x_t);
    }

    Eigen::Vector2f cen_pos;
    Eigen::MatrixXf dcen_dx;
    if (dc_du != nullptr) {
      cen_pos = GetCollisionCenter(x_t, &dcen_dx);
    } else {
      cen_pos = GetCollisionCenter(x_t);
    }

    Eigen::Vector2f goal_vec = cen_pos - params.goals[i];
    if (cost != nullptr) {
      (*cost) += params.k_goal * (goal_vec.norm() - curr_dist);
    }
    if (dc_du != nullptr) {
      Eigen::Vector2f dd_dcen = params.k_goal * goal_vec.normalized();
      (*dc_du) += dd_dcen.transpose() * dcen_dx * dxt_du;
    }
  }
}

void TTCSGDProblem::ProjConstr(Eigen::VectorXf& u) {
  if (params.box_constraint) {
    for (int i = 0; i < u.size(); ++i) {
      u[i] = std::min(params.u_ub[i], std::max(params.u_lb[i], u[i]));
    }
  } else {
    if (u.norm() > params.circle_u_limit) {
      u *= params.circle_u_limit / u.norm();
    }
  }
}

void TTCSGDProblem::Cost(const Eigen::VectorXf& u, float* cost, Eigen::VectorXf* dc_du) {
  if (cost != nullptr) {
    (*cost) = 0;
  }
  if (dc_du != nullptr) {
    (*dc_du) = Eigen::VectorXf::Zero(u.size());
  }

  GoalCost(u, cost, dc_du);

  if (params.use_ttc_cost) {
    TTCCost(u, cost, dc_du);
  } else {
    DistCost(u, cost, dc_du);
  }

  // Add any dynamics model specific costs
  AdditionalCosts(u, cost, dc_du);
}

void TTCSGDProblem::Dynamics(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                             Eigen::VectorXf* x_tpdt, Eigen::MatrixXf* dxtpdt_du, Eigen::MatrixXf* dxtpdt_dx) {
  if (dxtpdt_dx != nullptr || dxtpdt_du != nullptr) {
    // In general it's faster to manually compute per model, as opposed to using the analytical RK2 derivative.
    GetDynamicsDeriv(u, x, t, dt, dxtpdt_du, dxtpdt_dx);
    // // RK2 analytical derivative
    // Eigen::VectorXf k1;
    // Eigen::MatrixXf dk1_dx, dk1_du;
    // Eigen::MatrixXf dk2_dx_plus, dk2_du;
    // ContDynamics(u, x, &k1, &dk1_du, &dk1_dx);
    // // x_plus = x + 0.5f * dt * k1
    // ContDynamics(u, x + 0.5f * dt * k1, nullptr, &dk2_du, &dk2_dx_plus);

    // int x_dim = x.size();
    // if (dxtpdt_dx != nullptr) {
    //   (*dxtpdt_dx) = Eigen::MatrixXf::Identity(x_dim,x_dim) + dt * dk2_dx_plus * (Eigen::MatrixXf::Identity(x_dim, x_dim) + 0.5f * dt * dk1_dx);
    // }
    // if (dxtpdt_du != nullptr) {
    //   (*dxtpdt_du) = dt * dk2_du + 0.5f * dt * dt * dk2_dx_plus * dk1_du;
    // }
  }
  if (x_tpdt != nullptr) {
    // RK4
    Eigen::VectorXf k1, k2, k3, k4;
    Eigen::VectorXf x2, x3, x4;
    ContDynamics(u, x, t, &k1);
    x2 = x + 0.5f * dt * k1;
    ContDynamics(u, x2, t + 0.5f * dt, &k2);
    x3 = x + 0.5f * dt * k2;
    ContDynamics(u, x3, t + 0.5f * dt, &k3);
    x4 = x + dt * k3;
    ContDynamics(u, x4, t + dt, &k4);

    (*x_tpdt) = x + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
  }
}

void TTCSGDProblem::DynamicsLong(const Eigen::VectorXf &u, const Eigen::VectorXf &x, const float t, const float dt,
                                 Eigen::VectorXf* x_tpdt, Eigen::MatrixXf* dxtpdt_du, Eigen::MatrixXf* dxtpdt_dx) {
  int n_steps = static_cast<int>(std::floor(dt / params.dt_step));
  float rem_dt = dt - (n_steps * params.dt_step);

  bool comp_derivs = (dxtpdt_dx != nullptr || dxtpdt_du != nullptr);

  int x_dim = static_cast<int>(x.size()), u_dim = static_cast<int>(u.size());

  Eigen::VectorXf x_t = x;
  Eigen::MatrixXf A, B;
  Eigen::MatrixXf A_t, B_t;
  if (comp_derivs) {
    A = Eigen::MatrixXf::Identity(x_dim,x_dim);
    B = Eigen::MatrixXf::Zero(x_dim, u_dim);
  }
  for (int i = 0; i < n_steps; ++i) {
    if (comp_derivs) {
      Dynamics(u, x_t, t + i * params.dt_step, params.dt_step, &x_t, &B_t, &A_t);
      A.applyOnTheLeft(A_t);
      B.applyOnTheLeft(A_t);
      B += B_t;
    } else {
      Dynamics(u, x_t, t + i * params.dt_step, params.dt_step, &x_t);
    }
  }
  if (comp_derivs) {
      Dynamics(u, x_t, dt - rem_dt, rem_dt, &x_t, &B_t, &A_t);
      A.applyOnTheLeft(A_t);
      B.applyOnTheLeft(A_t);
      B += B_t;
  } else {
    Dynamics(u, x_t, dt - rem_dt, rem_dt, &x_t);
  }

  if (x_tpdt != nullptr) {
    (*x_tpdt) = x_t;
  }
  if (dxtpdt_du != nullptr) {
    (*dxtpdt_du) = B;
  }
  if (dxtpdt_dx != nullptr) {
    (*dxtpdt_dx) = A;
  }
}

Eigen::Vector2f TTCSGDProblem::GetCollisionCenter(const Eigen::VectorXf &x, Eigen::MatrixXf* dc_dx) {
  if (dc_dx != nullptr) {
    (*dc_dx) = Eigen::MatrixXf::Identity(2,x.size());
  }
  return x.head<2>();
}

void TTCSGDProblem::FillObstacle(TTCObstacle* o) {
  o->u = params.u_curr;
  o->p = params.x_0;
  o->radius = params.radius;
}