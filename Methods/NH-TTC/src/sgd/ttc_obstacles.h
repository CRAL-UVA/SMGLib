/*
Copyright 2020 University of Minnesota and Clemson University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#pragma once
#include <Eigen/Core>

class TTCObstacle {
public:
  Eigen::VectorXf u, p;
  float radius = 0.1f;

  virtual void ContDynamics(const Eigen::VectorXf &u,
                                       const Eigen::VectorXf &x,
                                       const float t,
                                       Eigen::VectorXf* x_dot) const = 0;
  void Dynamics(const Eigen::VectorXf &u,
                const Eigen::VectorXf &x,
                const float t,
                float dt, Eigen::VectorXf* x_tpdt) const;
  virtual Eigen::Vector2f GetCollisionCenter(const Eigen::VectorXf &x, Eigen::MatrixXf* dc_dx = nullptr);
  virtual ~TTCObstacle(){};
};

class VTTCObstacle : public TTCObstacle {
public:
  explicit VTTCObstacle(){};
  void ContDynamics(const Eigen::VectorXf &u,
                    const Eigen::VectorXf &x,
                    const float t,
                    Eigen::VectorXf* x_dot) const;
  ~VTTCObstacle(){};
};

class ATTCObstacle : public TTCObstacle {
public:
  explicit ATTCObstacle(){};
  void ContDynamics(const Eigen::VectorXf &u,
                               const Eigen::VectorXf &x,
                               const float t,
                               Eigen::VectorXf* x_dot) const;
  ~ATTCObstacle(){};
};

class DDTTCObstacle : public TTCObstacle {
public:
  explicit DDTTCObstacle(){};
  void ContDynamics(const Eigen::VectorXf &u,
                               const Eigen::VectorXf &x,
                               const float t,
                               Eigen::VectorXf* x_dot) const;
  ~DDTTCObstacle(){};
};

class ADDTTCObstacle : public TTCObstacle {
public:
  explicit ADDTTCObstacle(){};
  void ContDynamics(const Eigen::VectorXf &u,
                               const Eigen::VectorXf &x,
                               const float t,
                               Eigen::VectorXf* x_dot) const;
  ~ADDTTCObstacle(){};
};

class CARTTCObstacle : public TTCObstacle {
public:
  explicit CARTTCObstacle(){};
  void ContDynamics(const Eigen::VectorXf &u,
                               const Eigen::VectorXf &x,
                               const float t,
                               Eigen::VectorXf* x_dot) const;
  Eigen::Vector2f GetCollisionCenter(const Eigen::VectorXf &x, Eigen::MatrixXf* dc_dx);
  ~CARTTCObstacle(){};
};

class ACARTTCObstacle : public TTCObstacle {
public:
  explicit ACARTTCObstacle(){};
  void ContDynamics(const Eigen::VectorXf &u,
                               const Eigen::VectorXf &x,
                               const float t,
                               Eigen::VectorXf* x_dot) const;
  Eigen::Vector2f GetCollisionCenter(const Eigen::VectorXf &x, Eigen::MatrixXf* dc_dx);
  ~ACARTTCObstacle(){};
};
