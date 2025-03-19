/*
Copyright 2020 University of Minnesota and Clemson University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/*
This file contains the main implementation of the subgradient descent optimizer, in addition to a sampling-based optimizer.
In our experiments, the subgradient descent-based optimizer obtained better quality paths.
*/

#include <sgd/sgd.h>

#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <fstream>

inline bool CheckExitAndUpdateTime(std::chrono::time_point<std::chrono::high_resolution_clock> t_start,
                        std::chrono::time_point<std::chrono::high_resolution_clock> t_iter_start,
                        float& max_iter_time,
                        const float max_time) {
  auto t_now = std::chrono::high_resolution_clock::now();
  
  auto iter_dur = std::chrono::duration_cast<std::chrono::nanoseconds>(t_now - t_iter_start).count();
  float iter_dur_ms = static_cast<float>(iter_dur) / 1e6f;
  max_iter_time = std::max(max_iter_time, iter_dur_ms);

  auto total_dur = std::chrono::duration_cast<std::chrono::nanoseconds>(t_now - t_start).count();
  float total_dur_ms = static_cast<float>(total_dur) / 1e6f;
  if (total_dur_ms + max_iter_time > max_time) {
    return true;
  } else {
    return false;
  }
}

void SGD::GetCostGrid(SGDProblem *p,
                      const SGDOptParams &params,
                      int n_samples,
                      std::string file_name) {
  p->InitProblem();
  int x_dim = static_cast<int>(params.x_0.size());

  if (params.x_lb.size() != x_dim || params.x_ub.size() != x_dim) {
    std::cerr << "Dimension Mismatch between x_0 and x_lb and x_ub!" << std::endl;
    exit(-1);
  }
  if (x_dim != 2) {
    std::cerr << "GetCostGrid undefined for this number of dimensions" << std::endl;
    exit(-1);
  }

  std::ofstream f_out(file_name);

  for (int i = 0; i < n_samples; ++i) {
    float x_0 = (i + 0.5f) / n_samples * (params.x_ub[0] - params.x_lb[0]) + params.x_lb[0];
    for (int j = 0; j < n_samples; ++j) {
      float x_1 = (j + 0.5f) / n_samples * (params.x_ub[1] - params.x_lb[1]) + params.x_lb[1];
      Eigen::VectorXf x_k = Eigen::Vector2f(x_0,x_1);
      float cost_k;
      p->Cost(x_k, &cost_k, nullptr);
      f_out << x_0 << "," << x_1 << "," << cost_k << std::endl;
    }
  }
  f_out.close();
}

Eigen::VectorXf SGD::SolveSGD(SGDProblem *p,
                              const SGDOptParams &params,
                              float *best_cost) {
  auto t_start = std::chrono::high_resolution_clock::now();
  p->InitProblem();
  
  int x_dim = static_cast<int>(params.x_0.size());
  float cost_best, cost_k;
  Eigen::VectorXf x_k = params.x_0, x_best = params.x_0, gk = Eigen::VectorXf::Zero(x_dim);
  p->Cost(x_k, &cost_k, &gk);
  cost_best = cost_k;

  float longest_iter = 0.0f;
  Eigen::VectorXf skm1 = Eigen::VectorXf::Zero(x_dim);
  size_t k = 0;
  for (;; ++k) {
    auto t_iter_start = std::chrono::high_resolution_clock::now();
    float beta_k;
    Eigen::VectorXf sk;
    if (params.sk_mode == SGDSkMode::CFM) {
      // ---- CFM ----
      if (skm1.squaredNorm() > 0) {
        float gamma = 1.5f;
        beta_k = std::max(0.0f, -gamma * (skm1.dot(gk)) / (skm1.squaredNorm()));
      } else {
        beta_k = 0.0;
      }
      sk = gk + beta_k * skm1;
    } else if (params.sk_mode == SGDSkMode::Filtered) {
      // ---- Filtered ----
      sk = 0.9f * gk + 0.1f * skm1;
    } else {
      std::cerr << "Unimplemented SGDSkMode!" << std::endl;
      exit(-1);
    }

    float alpha;
    if (params.alpha_mode == SGDAlphaMode::PolyakKnown) {
      // ---- Polyak step with known size ----
      alpha = (cost_k - params.polyak_best) / (sk.squaredNorm());
    } else if (params.alpha_mode == SGDAlphaMode::PolyakSemiKnown) {
      // ---- Polyak step with semi-known size ----
      float gamma = 10.0f / (10.0f + k);
      float cost_guess = (1.0f - gamma) * cost_best + gamma * params.polyak_best;
      alpha = (cost_k - cost_guess) / (sk.squaredNorm());
    } else if (params.alpha_mode == SGDAlphaMode::PolyakUnknown) {
      // ---- Polyak step with unknown size ----
      float gamma = 10.0f / (10.0f + k);
      alpha = (cost_k - cost_best + gamma) / (sk.squaredNorm());
    } else if (params.alpha_mode == SGDAlphaMode::SmallAlpha) {
      // ---- Small Alpha Step ----
      alpha = 0.001f;
    } else {
      std::cerr << "Unimplemented SGDAlphaMode!" << std::endl;
      exit(-1);
    }
    x_k = x_k - alpha * sk;
    skm1 = sk;

    if (params.projectedSGD) {
      p->ProjConstr(x_k);
    }

    p->Cost(x_k, &cost_k, &gk);
    if (cost_k < cost_best) {
      cost_best = cost_k;
      x_best = x_k;
    }

    bool should_exit = CheckExitAndUpdateTime(t_start, t_iter_start, longest_iter, params.max_time);
    if (should_exit) {
      break;
    }
  }

  if (params.projectedSGD || params.projectFinal) {
    p->ProjConstr(x_best);
  }
  if (best_cost != nullptr) {
    (*best_cost) = cost_best;
  }
  return x_best;
}

Eigen::VectorXf SGD::SolveSampled(SGDProblem *p,
                                  const SGDOptParams &params,
                                  float *best_cost) {
  auto t_start = std::chrono::high_resolution_clock::now();
  p->InitProblem();
  int x_dim = static_cast<int>(params.x_0.size());
  float cost_best, cost_k;
  Eigen::VectorXf x_k = params.x_0, x_best = params.x_0;
  p->Cost(x_k, &cost_k, nullptr);
  cost_best = cost_k;

  if (params.x_lb.size() != x_dim || params.x_ub.size() != x_dim) {
    std::cerr << "Dimension Mismatch between x_0 and x_lb and x_ub!" << std::endl;
    exit(-1);
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);

  float longest_iter = 0.0f;
  for (size_t k = 0;; ++k) {
    auto t_iter_start = std::chrono::high_resolution_clock::now();
    // Sample random control
    for (int idx = 0; idx < x_dim; ++idx) {
      x_k[idx] = params.x_lb[idx] + dis(gen) * (params.x_ub[idx] - params.x_lb[idx]);
    }
    p->ProjConstr(x_k);

    // Compute cost and update best control
    p->Cost(x_k, &cost_k, nullptr);
    if (cost_k < cost_best) {
      x_best = x_k;
      cost_best = cost_k;
    }

    if (CheckExitAndUpdateTime(t_start, t_iter_start, longest_iter, params.max_time)) {
      break;
    }
  }

  p->ProjConstr(x_best);
  if (best_cost != nullptr) {
    (*best_cost) = cost_best;
  }
  return x_best;
}

Eigen::VectorXf SGD::Solve(SGDProblem* p, const SGDOptParams& params, float* cost_best) {
  if (params.opt_mode == OptMode::SGD) {
    return SolveSGD(p, params, cost_best);
  } else if (params.opt_mode == OptMode::Sampled) {
    return SolveSampled(p, params, cost_best);
  } else {
    std::cerr << "OptMode Not Implemented!" << std::endl;
    exit(-1);
  }
}