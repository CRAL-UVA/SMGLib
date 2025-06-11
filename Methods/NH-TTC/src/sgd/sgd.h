/*
Copyright 2020 University of Minnesota and Clemson University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#pragma once
#include <Eigen/Core>

class SGDProblem {
public:
  virtual void InitProblem() = 0;
  virtual void ProjConstr(Eigen::VectorXf& x) = 0;
  virtual void Cost(const Eigen::VectorXf& x, float* cost, Eigen::VectorXf* dc_du) = 0;

  virtual ~SGDProblem(){};
};


enum class OptMode { SGD, Sampled, Unset };

enum class SGDAlphaMode { PolyakUnknown, PolyakKnown, PolyakSemiKnown, SmallAlpha, Unset };

enum class SGDSkMode { CFM, Filtered, Unset };

struct SGDOptParams {
  // General Parameters
  float max_time = 10.0f;
  OptMode opt_mode = OptMode::Unset;
  Eigen::VectorXf x_0;

  // Parameters for SGD
  SGDAlphaMode alpha_mode = SGDAlphaMode::Unset;
  SGDSkMode sk_mode = SGDSkMode::Unset;

  float polyak_best = 0.0f;

  bool projectedSGD = true;
  bool projectFinal = true;

  // Parameters for Sampled
  Eigen::VectorXf x_lb, x_ub;
};

class SGD {
  static Eigen::VectorXf SolveSGD(SGDProblem* p, const SGDOptParams& params, float* cost_best);
  static Eigen::VectorXf SolveSampled(SGDProblem* p, const SGDOptParams& params, float* cost_best);

public:
  static Eigen::VectorXf Solve(SGDProblem* p, const SGDOptParams& params, float* cost_best);

  static void GetCostGrid(SGDProblem *p, const SGDOptParams &params, int n_samples, std::string file_name);
};