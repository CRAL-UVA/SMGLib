/*
Copyright 2020 University of Minnesota and Clemson University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/*
This file contains both the main control optimization loop, as well as the graphical drawing functions.
*/

#include <nhttc_opengl/nhttc_sim.h>

#include <nativefd/nfd.h>
#include <atomic>

std::atomic<bool> is_done(false);
std::atomic<bool> should_plan(false);

NHTTCSim::NHTTCSim() {
  opt_params.max_time = 10.0f;
  opt_params.opt_mode = OptMode::SGD;
  opt_params.alpha_mode = SGDAlphaMode::PolyakSemiKnown;
  opt_params.sk_mode = SGDSkMode::Filtered;

  agent_buffer = new float[MAX_AGENTS * A_SIZE];
  agent_buffer[0] = 0.0f; // x
  agent_buffer[1] = 0.0f; // y
  agent_buffer[2] = 0.0f; // id
  agent_buffer[3] = 0.0f; // th
  agent_buffer[4] = 50000.0f; // r
  agent_buffer[5] = -1.0f; // type
  agent_buffer[6] = 0.0f; // gx
  agent_buffer[7] = 0.0f; // gy
  agent_buffer[8] = 1.0f; // reactive

  glGenBuffers(1, vbo);  //Create 1 buffer called vbo
  glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); //Set the vbo as the active array buffer (Only one buffer can be active at a time)
  glBufferData(GL_ARRAY_BUFFER, MAX_AGENTS*A_SIZE*sizeof(float), agent_buffer, GL_STREAM_DRAW); //upload vertices to vbo

  const char* fShader2_n = "shaders/fragShaderPt.glsl";
  const char* vShader2_n = "shaders/vertShaderPt.glsl";
  shaderProgram = InitShader(vShader2_n, fShader2_n);

  // Load in all agent textures
  std::string check_texture_n = "textures/checker.png";
  std::string v_texture_n = "textures/v.png";
  std::string a_texture_n = "textures/a.png";
  std::string dd_texture_n = "textures/dd.png";
  std::string add_texture_n = "textures/add.png";
  std::string car_texture_n = "textures/car.png";
  std::string acar_texture_n = "textures/acar.png";
  int tex_w, tex_h;
  glActiveTexture(GL_TEXTURE0);
  tex0 = loadTexture(check_texture_n.c_str(), tex_w, tex_h);
  glActiveTexture(GL_TEXTURE1);
  tex1 = loadTexture(v_texture_n.c_str(), tex_w, tex_h);
  glActiveTexture(GL_TEXTURE2);
  tex2 = loadTexture(a_texture_n.c_str(), tex_w, tex_h);
  glActiveTexture(GL_TEXTURE3);
  tex3 = loadTexture(dd_texture_n.c_str(), tex_w, tex_h);
  glActiveTexture(GL_TEXTURE4);
  tex4 = loadTexture(add_texture_n.c_str(), tex_w, tex_h);
  glActiveTexture(GL_TEXTURE5);
  tex5 = loadTexture(car_texture_n.c_str(), tex_w, tex_h);
  glActiveTexture(GL_TEXTURE6);
  tex6 = loadTexture(acar_texture_n.c_str(), tex_w, tex_h);

  planning_thread = std::thread(&NHTTCSim::PlanAllAgents, this);
}

void NHTTCSim::PlanAllAgents() {
	while (true) {
        // Wait until done or a new planning cycle starts
        while (!is_done.load() && !should_plan.load()) {}
        if (is_done.load()) { should_plan.store(false); return; }
		if (planning_agents.empty()) {
			should_plan.store(false);
			continue;
		}

		// Forward propagate all agents by PLANNING_TIME:
   float n_controlled = 0.0f;
   for (size_t a_idx = 0; a_idx < planning_agents.size(); ++a_idx) {
        planning_agents[a_idx].prob->DynamicsLong(planning_agents[a_idx].prob->params.u_curr, planning_agents[a_idx].prob->params.x_0, 0.0f, PLANNING_TIME, &(planning_agents[a_idx].prob->params.x_0));

        // Check if the agent is one of the last two
        if (a_idx >= planning_agents.size() - 2) {
        // Print the position (x_0) of the agent
            std::cout << "Agent " << a_idx << " Position: " << planning_agents[a_idx].prob->params.x_0 << std::endl;

        // Print the current velocity (u_curr) of the agent
            std::cout << "Agent " << a_idx << " Current Velocity: " << planning_agents[a_idx].prob->params.u_curr << std::endl;
    }

        if (planning_agents[a_idx].controlled) {
            n_controlled = n_controlled + 1.0f;
        }
   }
    


    

		// float n_controlled = 0.0f;
		// for (size_t a_idx = 0; a_idx < planning_agents.size(); ++a_idx) {
		// 	planning_agents[a_idx].prob->DynamicsLong(planning_agents[a_idx].prob->params.u_curr, planning_agents[a_idx].prob->params.x_0, 0.0f, PLANNING_TIME, &(planning_agents[a_idx].prob->params.x_0));
		// 	if (planning_agents[a_idx].controlled) {
		// 		n_controlled = n_controlled + 1.0f;
		// 	}
		// }

		if (n_controlled == 0.0f) {
			should_plan.store(false);
			continue;
		}

		// Create Obstacle Lists
		std::vector<TTCObstacle*> all_obsts = BuildObstacleList(planning_agents);

		// Plan for each agent for PLANNING_TIME / planning_agents.size()
        // *1000 to convert to ms, *0.95 to leave a little spare time
		float agent_plan_time_ms = 0.95f * PLANNING_TIME / n_controlled * 1000.0f;
		for (size_t a_idx = 0; a_idx < planning_agents.size(); ++a_idx) {
			if (!planning_agents[a_idx].controlled) {
				continue; // Skip any not controlled agent
			}
			planning_agents[a_idx].opt_params.max_time = agent_plan_time_ms;
			SetAgentObstacleList(planning_agents[a_idx], a_idx, all_obsts);

			// Ensure goals set properly
			planning_agents[a_idx].prob->params.goals.clear();
			for (size_t i = 0; i < planning_agents[a_idx].prob->params.ts_goal_check.size(); ++i) {
				planning_agents[a_idx].prob->params.goals.push_back(planning_agents[a_idx].goal);
			}

			// Optimize controls
			planning_agents[a_idx].PrepareSGDParams();
			float sgd_opt_cost;
			Eigen::VectorXf u_new = SGD::Solve(planning_agents[a_idx].prob, planning_agents[a_idx].opt_params, &sgd_opt_cost);
			planning_agents[a_idx].prob->params.u_curr = 0.5f * (u_new + planning_agents[a_idx].prob->params.u_curr); // Reciprocity
		}
		should_plan.store(false);
	}
}

void NHTTCSim::Step(float dt) {
  time_since_last_plan += dt;
  bool join_plan = time_since_last_plan > PLANNING_TIME;

  if (join_plan) {
    // synchronize with planning thread
	while (should_plan.load()) {}

    // Update current controls
    if (!reset_planning_agents) {
      for (size_t i = 0; i < planning_agents.size(); ++i) {
        agents[i].prob->params.u_curr = planning_agents[i].prob->params.u_curr;
      }
    }
  }

  if (!paused) {
    // Step each agent
    for (size_t a_idx = 0; a_idx < agents.size(); ++a_idx) {
      agents[a_idx].prob->DynamicsLong(agents[a_idx].prob->params.u_curr, agents[a_idx].prob->params.x_0, 0.0f, dt, &(agents[a_idx].prob->params.x_0));
    }
  }

  if (join_plan) {
    // Kick off new planning cycle
    if (reset_planning_agents) {
      planning_agents.clear();
      reset_planning_agents = false;
    }
    // Include any new agents in the planning
    for (size_t i = 0; i < new_planning_agents.size(); ++i) {
      planning_agents.push_back(new_planning_agents[i]);
    }
    new_planning_agents.clear();
    // Update initial planning points
    for (size_t i = 0; i < agents.size(); ++i) {
      planning_agents[i].prob->params.x_0 = agents[i].prob->params.x_0;
      planning_agents[i].prob->params.u_curr = agents[i].prob->params.u_curr;
      planning_agents[i].goal = agents[i].goal;
    }

	should_plan.store(true);
    time_since_last_plan = 0.0f;
  }
}

int TypeToInt(AType t) {
  switch (t) {
    case AType::V:
      return 0;
    case AType::A:
      return 1;
    case AType::DD:
      return 2;
    case AType::ADD:
      return 3;
    case AType::CAR:
      return 4;
    case AType::ACAR:
      return 5;
    default:
      std::cerr << "Invalid Agent Type!" << std::endl;
      return 0;
  }
}

void NHTTCSim::DrawBackground() {
  glDepthMask(GL_FALSE);
  float max_y = view_scale, min_y = -max_y;
  float max_x = max_y * static_cast<float>(w) / static_cast<float>(h), min_x = -max_x;
  
  glUniform1i(glGetUniformLocation(shaderProgram, "agent_type"), -1);
  glUniform1i(glGetUniformLocation(shaderProgram, "reactive"), true);

  // Bottom Left:
  agent_buffer[0 * PT_SIZE + 0] = min_x + cen_x + x_offset;
  agent_buffer[0 * PT_SIZE + 1] = min_y + cen_y + y_offset;
  agent_buffer[0 * PT_SIZE + 2] = 0.0f;
  agent_buffer[0 * PT_SIZE + 3] = agent_buffer[0 * PT_SIZE + 0] / (2.0f * bg_size);
  agent_buffer[0 * PT_SIZE + 4] = agent_buffer[0 * PT_SIZE + 1] / (2.0f * bg_size);

  // Bottom Right:
  agent_buffer[1 * PT_SIZE + 0] = max_x + cen_x + x_offset;
  agent_buffer[1 * PT_SIZE + 1] = min_y + cen_y + y_offset;
  agent_buffer[1 * PT_SIZE + 2] = 0.0f;
  agent_buffer[1 * PT_SIZE + 3] = agent_buffer[1 * PT_SIZE + 0] / (2.0f * bg_size);
  agent_buffer[1 * PT_SIZE + 4] = agent_buffer[1 * PT_SIZE + 1] / (2.0f * bg_size);

  // Top Left:
  agent_buffer[2 * PT_SIZE + 0] = min_x + cen_x + x_offset;
  agent_buffer[2 * PT_SIZE + 1] = max_y + cen_y + y_offset;
  agent_buffer[2 * PT_SIZE + 2] = 0.0f;
  agent_buffer[2 * PT_SIZE + 3] = agent_buffer[2 * PT_SIZE + 0] / (2.0f * bg_size);
  agent_buffer[2 * PT_SIZE + 4] = agent_buffer[2 * PT_SIZE + 1] / (2.0f * bg_size);

  // Top Right:
  agent_buffer[3 * PT_SIZE + 0] = max_x + cen_x + x_offset;
  agent_buffer[3 * PT_SIZE + 1] = max_y + cen_y + y_offset;
  agent_buffer[3 * PT_SIZE + 2] = 0.0f;
  agent_buffer[3 * PT_SIZE + 3] = agent_buffer[3 * PT_SIZE + 0] / (2.0f * bg_size);
  agent_buffer[3 * PT_SIZE + 4] = agent_buffer[3 * PT_SIZE + 1] / (2.0f * bg_size);

  glBufferSubData(GL_ARRAY_BUFFER, 0, 4 * PT_SIZE * sizeof(float), agent_buffer); //upload vertices to vbo
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  glDepthMask(GL_TRUE);
}

void NHTTCSim::DrawAgents() {
  // Add Virtual Agent:
  if (adding_mode) {
    std::vector<std::string> parts = GetVirtAgentParts();
    agents.emplace_back(parts, opt_params);
  }


  // Draw the Goals:
  for (size_t i = 0; i < agents.size(); ++i) {
    Agent& a = agents[i];
    Eigen::Vector2f xy = a.goal;

    // Set uniform for agent type
    glUniform1i(glGetUniformLocation(shaderProgram, "agent_type"), TypeToInt(a.a_type));
    glUniform1i(glGetUniformLocation(shaderProgram, "reactive"), a.reactive);

    // Add Central point:
    int buffer_pt_idx = 0;
    agent_buffer[buffer_pt_idx * PT_SIZE + 0] = xy[0];
    agent_buffer[buffer_pt_idx * PT_SIZE + 1] = xy[1];
    agent_buffer[buffer_pt_idx * PT_SIZE + 2] = 0.0f;
    agent_buffer[buffer_pt_idx * PT_SIZE + 3] = 0.5f;
    agent_buffer[buffer_pt_idx * PT_SIZE + 4] = 0.5f;
    buffer_pt_idx++;

    for (int j = 0; j < 11; ++j) {
      float th_draw = 2.0f * static_cast<float>(M_PI) * j / (10.0f);
      float r_scale = (j%2 == 0 ? 0.75f : 0.4f);
      agent_buffer[buffer_pt_idx * PT_SIZE + 0] = std::cos(th_draw) * r_scale * a.prob->params.radius + xy[0];
      agent_buffer[buffer_pt_idx * PT_SIZE + 1] = std::sin(th_draw) * r_scale * a.prob->params.radius + xy[1];
      agent_buffer[buffer_pt_idx * PT_SIZE + 2] = 0.0f;
      agent_buffer[buffer_pt_idx * PT_SIZE + 3] = 0.5f;
      agent_buffer[buffer_pt_idx * PT_SIZE + 4] = 0.5f;
      buffer_pt_idx++;
    }

    glBufferSubData(GL_ARRAY_BUFFER, 0, buffer_pt_idx * PT_SIZE * sizeof(float), agent_buffer); //upload vertices to vbo
    glDrawArrays(GL_TRIANGLE_FAN, 0, buffer_pt_idx);
  }

  // Draw the agents themselves:
  for (size_t i = 0; i < agents.size(); ++i) {
    Agent& a = agents[i];
    Eigen::Vector2f xy = a.prob->GetCollisionCenter(a.prob->params.x_0, nullptr);

    // Set uniform for agent type
    glUniform1i(glGetUniformLocation(shaderProgram, "agent_type"), TypeToInt(a.a_type));
    glUniform1i(glGetUniformLocation(shaderProgram, "reactive"), a.reactive);

    // Add Central point:
    int buffer_pt_idx = 0;
    agent_buffer[buffer_pt_idx * PT_SIZE + 0] = xy[0];
    agent_buffer[buffer_pt_idx * PT_SIZE + 1] = xy[1];
    agent_buffer[buffer_pt_idx * PT_SIZE + 2] = static_cast<float>(i) + 0.1f;
    agent_buffer[buffer_pt_idx * PT_SIZE + 3] = 0.5f;
    agent_buffer[buffer_pt_idx * PT_SIZE + 4] = 0.5f;
    buffer_pt_idx++;

    for (int j = 0; j < N_PT_PER_AGENT; ++j) {
      float th_draw = 2.0f * static_cast<float>(M_PI) * j / (N_PT_PER_AGENT - 1.0f);
      float th_ag = 0.0f;
      if (a.a_type != AType::V && a.a_type != AType::A) {
        th_ag = a.prob->params.x_0[2];
      }
      agent_buffer[buffer_pt_idx * PT_SIZE + 0] = std::cos(th_draw + th_ag) * a.prob->params.radius + xy[0];
      agent_buffer[buffer_pt_idx * PT_SIZE + 1] = std::sin(th_draw + th_ag) * a.prob->params.radius + xy[1];
      agent_buffer[buffer_pt_idx * PT_SIZE + 2] = static_cast<float>(i) + 0.1f;
      agent_buffer[buffer_pt_idx * PT_SIZE + 3] = 0.5f + 0.5f * std::cos(th_draw);
      agent_buffer[buffer_pt_idx * PT_SIZE + 4] = 0.5f + 0.5f * std::sin(th_draw);
      buffer_pt_idx++;
    }

    glBufferSubData(GL_ARRAY_BUFFER, 0, buffer_pt_idx * PT_SIZE * sizeof(float), agent_buffer); //upload vertices to vbo
    glDrawArrays(GL_TRIANGLE_FAN, 0, buffer_pt_idx);
  }

  if (adding_mode) {
    // Remove the agent we're currently placing from the agents list
    delete agents.back().prob;
    agents.pop_back();
  }
}

void NHTTCSim::Draw(GLFWwindow* window) {
  glfwGetWindowSize(window, &w, &h);
  int fb_w, fb_h;
  glfwGetFramebufferSize(window, &fb_w, &fb_h);
  glViewport(0,0,fb_w,fb_h);
  float max_y = view_scale, min_y = -max_y;
  float max_x = max_y * static_cast<float>(w) / static_cast<float>(h), min_x = -max_x;
  proj = glm::ortho(min_x + cen_x + x_offset, max_x + cen_x + x_offset, min_y + cen_y + y_offset, max_y + cen_y + y_offset, 1.0f, 10.0f + 1.0f * static_cast<float>(MAX_AGENTS+1)); //FOV, aspect, near, far
	
  view = glm::lookAt(
    glm::vec3(0.0f,0.0f,1.0f * static_cast<float>(MAX_AGENTS+1)),  //Cam Position
		glm::vec3(0.0f, 0.0f, 0.0f), //Look at point
		glm::vec3(0.0f, 1.0f, 0.0f)); //Up
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  setActiveProgram(shaderProgram);
  glUniform1i(glGetUniformLocation(shaderProgram, "texs[0]"), 0);
  glUniform1i(glGetUniformLocation(shaderProgram, "texs[1]"), 1);
  glUniform1i(glGetUniformLocation(shaderProgram, "texs[2]"), 2);
  glUniform1i(glGetUniformLocation(shaderProgram, "texs[3]"), 3);
  glUniform1i(glGetUniformLocation(shaderProgram, "texs[4]"), 4);
  glUniform1i(glGetUniformLocation(shaderProgram, "texs[5]"), 5);
  glUniform1i(glGetUniformLocation(shaderProgram, "texs[6]"), 6);

  // Set Agent Buffer:
  DrawBackground();
  DrawAgents();

  glDisable(GL_BLEND);
}

NHTTCSim::~NHTTCSim() {
	is_done.store(true);
  if (planning_thread.joinable()) {
    planning_thread.join();
  }
  delete[] agent_buffer;
  glDeleteBuffers(1, vbo);
}

void NHTTCSim::setActiveProgram(GLint program) {
	glUseProgram(program);
	
	GLint uniView = glGetUniformLocation(program, "view");
	glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));
	
	GLint uniProj = glGetUniformLocation(program, "proj");
	glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));
	
	//Tell OpenGL how to set shader input 
	GLint posAttrib = glGetAttribLocation(program, "pos");
	glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, PT_SIZE*sizeof(float), 0);
	glEnableVertexAttribArray(posAttrib);
	GLint texAttrib = glGetAttribLocation(program, "tex_coord");
	glVertexAttribPointer(texAttrib, 2, GL_FLOAT, GL_FALSE, PT_SIZE*sizeof(float), (void*)(3*sizeof(float)));
	glEnableVertexAttribArray(texAttrib);
}

std::vector<std::string> NHTTCSim::GetVirtAgentParts() {
  std::string type;
  int p_dim, u_dim;
  switch (virt_agent_type) {
    case 0:
      type = "v";
      p_dim = 2; u_dim = 2;
      break;
    case 1:
      type = "a";
      p_dim = 4; u_dim = 2;
      break;
    case 2:
      type = "dd";
      p_dim = 3; u_dim = 2;
      break;
    case 3:
      type = "add";
      p_dim = 5; u_dim = 2;
      break;
    case 4:
      type = "car";
      p_dim = 3; u_dim = 2;
      break;
    case 5:
      type = "acar";
      p_dim = 5; u_dim = 2;
      break;
    default:
      std::cerr << "Invalid virtual agent type: " << virt_agent_type << std::endl;
      exit(-1);
  }
  Eigen::VectorXf p = Eigen::VectorXf::Zero(p_dim);
  Eigen::VectorXf u = Eigen::VectorXf::Zero(u_dim);
  p[0] = virt_agent_x;
  p[1] = virt_agent_y;
  if (virt_agent_type > 1) {
    p[2] = virt_agent_th;
  }
  std::vector<std::string> parts(3 + p_dim + u_dim + 2);
  parts[0] = type;
  parts[1] = "y";
  parts[2] = (virt_agent_react ? "y" : "n");
  for (int i = 0; i < p.size(); ++i) {
    parts[3+i] = std::to_string(p[i]);
  }
  for (int i = 0; i < u.size(); ++i) {
    parts[3 + p_dim + i] = std::to_string(u[i]);
  }
  parts[3 + p_dim + u_dim] = std::to_string(virt_agent_gx);
  parts[3 + p_dim + u_dim + 1] = std::to_string(virt_agent_gy);

  return parts;
}

void NHTTCSim::AddVirtualAgent() {
  std::vector<std::string> parts = GetVirtAgentParts();
  Eigen::Vector2f xy(virt_agent_x, virt_agent_y);

  agents.emplace_back(parts, opt_params);
  agent_start_poses.push_back(xy);
  new_planning_agents.emplace_back(parts, opt_params);
}

// SDL event handling:
void NHTTCSim::handleMouseScroll(double x, double y) {
  if (!adding_mode) {
    view_scale = std::min(10.0f, std::max(0.1f, view_scale - view_scale / 20 * static_cast<float>(y)));
  } else if (add_pos_mode) {
    if (y > 0) {
      virt_agent_type += 1;
    } else if (y < 0) {
      virt_agent_type -= 1;
    }
    // Wrap Agent Type
    if (virt_agent_type < 0) {
      virt_agent_type = 5;
      virt_agent_react = !virt_agent_react;
    } else if (virt_agent_type > 5) {
      virt_agent_type = 0;
      virt_agent_react = !virt_agent_react;
    }
  }
}

void NHTTCSim::handleMouseButtonDown(GLFWwindow* wind, int button) {
  if (button == GLFW_MOUSE_BUTTON_1) {
    left_mouse_down = true;
    double x, y;
    glfwGetCursorPos(wind, &x, &y);
    held_pos_x = static_cast<float>(x);
    held_pos_y = static_cast<float>(y);
  }
}

void NHTTCSim::handleMouseButtonUp(GLFWwindow * wind, int button) {
  if (button == GLFW_MOUSE_BUTTON_1) { // left click
    left_mouse_down = false;
    if (adding_mode) {
      // Step new Agent setup Phase
      if (add_pos_mode) {
        if (virt_agent_type > 1) { // Not V or A agent
          add_orient_mode = true;
          add_pos_mode = false;
        } else {
          add_goal_mode = true;
          add_pos_mode = false;
          double x,y;
          glfwGetCursorPos(wind, &x, &y);
          ConvertMouseXY(static_cast<float>(x), static_cast<float>(y), virt_agent_gx, virt_agent_gy);
        }
      } else if (add_orient_mode) {
        add_goal_mode = true;
        add_orient_mode = false;
        double x, y;
        glfwGetCursorPos(wind, &x, &y);
        ConvertMouseXY(static_cast<float>(x), static_cast<float>(y), virt_agent_gx, virt_agent_gy);
      } else if (add_goal_mode) {
        AddVirtualAgent();
        adding_mode = false;
        add_goal_mode = false;
      }
    } else {
      cen_x += x_offset;
      cen_y += y_offset;
    }
    x_offset = 0.0f;
    y_offset = 0.0f;
  }
}

void NHTTCSim::ConvertMouseXY(float x, float y, float& m_x, float& m_y) {
  m_x = 2.0f * (x / static_cast<float>(w) - 0.5f);
  m_x *= view_scale * static_cast<float>(w) / static_cast<float>(h);
  m_x += cen_x;
  m_y = -2.0f * (y / static_cast<float>(h) - 0.5f);
  m_y *= view_scale;
  m_y += cen_y;
}

void NHTTCSim::handleMouseMotion(double x, double y) {
  if (adding_mode) {
    if (add_pos_mode) {
      ConvertMouseXY(static_cast<float>(x), static_cast<float>(y), virt_agent_x, virt_agent_y);
    } else if (add_orient_mode) {
      float m_x, m_y;
      ConvertMouseXY(static_cast<float>(x), static_cast<float>(y), m_x, m_y);
      virt_agent_th = std::atan2(m_y - virt_agent_y, m_x - virt_agent_x);
    } else if (add_goal_mode) {
      float m_x, m_y;
      ConvertMouseXY(static_cast<float>(x), static_cast<float>(y), m_x, m_y);
      virt_agent_gx = m_x;
      virt_agent_gy = m_y;
    }
  } else if (left_mouse_down) {
    x_offset = -2.0f * (static_cast<float>(x) - held_pos_x) / static_cast<float>(w) * view_scale * static_cast<float>(w) / static_cast<float>(h);
    y_offset = 2.0f * (static_cast<float>(y) - held_pos_y) / static_cast<float>(h) * view_scale;
  }
}

void NHTTCSim::handleKeyup(GLFWwindow* wind, int key) {
  if (key == GLFW_KEY_A && !adding_mode) {
    double x, y;
    glfwGetCursorPos(wind, &x, &y);
    ConvertMouseXY(static_cast<float>(x), static_cast<float>(y), virt_agent_x, virt_agent_y);
    adding_mode = true;
    add_pos_mode = true;
    paused = true;
  }
  if (key == GLFW_KEY_SPACE && !adding_mode) {
    paused = !paused;
  }
  if (key == GLFW_KEY_R && !adding_mode) {
    for (size_t i = 0; i < agents.size(); ++i) {
      Eigen::VectorXf tmp = agents[i].goal;
      agents[i].goal = agent_start_poses[i];
      agent_start_poses[i] = tmp;
    }
  }
  if (key == GLFW_KEY_C && !adding_mode) {
    agents.clear();
    agent_start_poses.clear();
    reset_planning_agents = true;
    new_planning_agents.clear();
  }
  if (key == GLFW_KEY_L && !adding_mode) {
    paused = true;
    // Clear the current scene
    agents.clear();
    agent_start_poses.clear();
    reset_planning_agents = true;
    new_planning_agents.clear();
    // Get File to Load:
    nfdchar_t *outPath = NULL;
    nfdresult_t result = NFD_OpenDialog( "scn", NULL, &outPath );
    if ( result == NFD_OKAY ) {
      // Parse file and add agents:
      std::string file_name(outPath);

      std::vector<std::vector<std::string>> file_parsed = LoadFileByToken(file_name, 0, ',');

      for (size_t i = 0; i < file_parsed.size(); ++i) {
        if (!file_parsed[i].empty()) {
          agents.emplace_back(file_parsed[i], opt_params);
          agent_start_poses.push_back(agents.back().prob->params.x_0.head<2>());
          new_planning_agents.emplace_back(file_parsed[i], opt_params);
        }
      }

      free(outPath);
    } else if ( result == NFD_CANCEL ) {
      std::cout << "user cancelled load." << std::endl;
    } else {
      std::cerr << "Error in file load: " << NFD_GetError() << std::endl;
    }
  }
  if (key == GLFW_KEY_S && !adding_mode) {
    paused = true;

    nfdchar_t *savePath = NULL;
    nfdresult_t result = NFD_SaveDialog( "scn", NULL, &savePath );
    if ( result == NFD_OKAY ) {
      // Ensure the saved file ends with ".scn"
      std::string scn_end = ".scn";
      std::string scn_file(savePath);
      if (scn_file.length() < scn_end.length() || scn_file.compare(scn_file.length() - scn_end.length(), scn_end.length(), scn_end) != 0) {
        scn_file = scn_file + scn_end;
      }
      
      std::ofstream f_out(scn_file);

      for (Agent& a : agents) {
        f_out << a.type_name << ",";
        f_out << (a.controlled ? "y" : "n") << ",";
        f_out << (a.reactive ? "y" : "n");
        Eigen::VectorXf& pos = a.prob->params.x_0;
        for (int i = 0; i < pos.size(); ++i) {
          f_out << "," << pos[i];
        }
        Eigen::VectorXf& u = a.prob->params.u_curr;
        for (int i = 0; i < u.size(); ++i) {
          f_out << "," << u[i];
        }
        for (int i = 0; i < a.goal.size(); ++i) {
          f_out << "," << a.goal[i];
        }
        f_out << std::endl;
      }

      free(savePath);
    } else if ( result == NFD_CANCEL ) {
      std::cout << "user cancelled save." << std::endl;
    } else {
      std::cerr << "Error in file save: " << NFD_GetError() << std::endl;
    }
  }
  if (key == GLFW_KEY_UP && add_pos_mode) {
    virt_agent_type += 1;
    if (virt_agent_type > 5) {
      virt_agent_type = 0;
      virt_agent_react = !virt_agent_react;
    }
  }
  if (key == GLFW_KEY_DOWN && add_pos_mode) {
    virt_agent_type -= 1;
    if (virt_agent_type < 0) {
      virt_agent_type = 5;
      virt_agent_react = !virt_agent_react;
    }
  }
  if ((key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT) && add_pos_mode) {
    virt_agent_react = !virt_agent_react;
  }
}
