/*
Copyright 2020 University of Minnesota and Clemson University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#pragma once
#include <nhttc_opengl/opengl_utils.h>

#include <nhttc_opengl/test_utils.h>

#include <thread>

class NHTTCSim {
public:
  NHTTCSim();
  void Step(float dt);
  void Draw(GLFWwindow* window);
  ~NHTTCSim();

  // SDL event handling:
  void handleMouseScroll(double x, double y);
  void handleMouseButtonDown(GLFWwindow* wind, int button);
  void handleMouseButtonUp(GLFWwindow* wind, int button);
  void handleMouseMotion(double x, double y);
  void handleKeyup(GLFWwindow* wind, int key);

private:
  void setActiveProgram(GLint program);
  std::vector<std::string> GetVirtAgentParts();
  void AddVirtualAgent();
  void ConvertMouseXY(float x, float y, float& m_x, float& m_y);
  
  void DrawAgents();
  void DrawBackground();
  static constexpr int N_PT_PER_AGENT = 100;
  static constexpr int PT_SIZE = 5;
  static constexpr float bg_size = 0.2f;

  void PlanAllAgents();

  int w, h;

  static constexpr int MAX_AGENTS = 2000;
  static constexpr int A_SIZE = 9;
  GLuint vbo[1];
  float* agent_buffer;
  GLuint tex0, tex1, tex2, tex3, tex4, tex5, tex6;
  GLint shaderProgram;
  glm::mat4 proj,view;

  float view_scale = 1.0;
  float cen_x = 0.0f, cen_y = 0.0f;

  // Planning Options:
  static constexpr float PLANNING_TIME = 0.1f;
  float time_since_last_plan = PLANNING_TIME + 1.0f;
  std::vector<Agent> agents, planning_agents, new_planning_agents;
  std::vector<Eigen::Vector2f> agent_start_poses;
  SGDOptParams opt_params;
  std::thread planning_thread;
  bool reset_planning_agents = false;


  // Key Flags:
  bool left_mouse_down = false;
  float held_pos_x = 0.0f, held_pos_y = 0.0f;
  float x_offset = 0.0f, y_offset = 0.0f;

  bool adding_mode = false;
  bool add_pos_mode = false, add_orient_mode = false, add_goal_mode = false;
  float virt_agent_x = 0.0f, virt_agent_y = 0.0f;
  float virt_agent_th = 0.0f;
  int virt_agent_type = 0;
  bool virt_agent_react = true;
  float virt_agent_gx = 0.0f, virt_agent_gy = 0.0f;

  bool paused = true;
};