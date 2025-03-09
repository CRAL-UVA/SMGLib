/*
Copyright 2020 University of Minnesota and Clemson University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/*
This file contains the main loop, as well as the input callbacks.
*/

#include <nhttc_opengl/opengl_utils.h>
#include <nhttc_opengl/nhttc_sim.h>

bool quit = false;
bool fullscreen = false;

NHTTCSim* sim;

void handleEvents(GLFWwindow* window);
void mouse_button_cb(GLFWwindow*, int, int, int);
void mouse_motion_cb(GLFWwindow*, double, double);
void mouse_scroll_cb(GLFWwindow*, double, double);
void key_cb(GLFWwindow*, int, int, int, int);

void print_instructions() {
	std::cout << "Click and drag to move the camera. Scroll the mouse to zoom in or out." << std::endl;
	std::cout << "'Space': Pauses or Resumes execution." << std::endl;
	std::cout << "'C' : Clears all agents from the simulation." << std::endl;
	std::cout << "'L' : Brings up a file dialog to load a scene file. Clears the current scene first." << std::endl;
	std::cout << "'S' : Brings up a file dialog to save the current agent positions and goal positions to a scene file." << std::endl;
	std::cout << "'R' : Swaps the initial position of an agent and its goal." << std::endl;
	std::cout << "'A' : Enters adding mode" << std::endl;
	std::cout << " - First, use the scroll wheel or arrow keys to select an agent type." << std::endl;
	std::cout << " - Next, for any agent with orientation, move the mouse to orient the agent." << std::endl;
	std::cout << " - Finally, move the mouse to the desired goal location." << std::endl;
	std::cout << " - Note that entering adding mode automatically pauses the simulation." << std::endl;
}

int main(){
	print_instructions();
    float timePast = 0;
    float lastTime = 0;

    GLFWwindow* window = GL_GLFW_init("NH-TTC");
    srand (static_cast <unsigned> (time(0)));

	// Initialize simulation -- Note this has to be after GL_GLFW_init
	sim = new NHTTCSim();

	// Initialize all callbacks:
	glfwSetMouseButtonCallback(window, mouse_button_cb);
	glfwSetCursorPosCallback(window, mouse_motion_cb);
	glfwSetScrollCallback(window, mouse_scroll_cb);
	glfwSetKeyCallback(window, key_cb);
	
	glEnable(GL_DEPTH_TEST);
	// Loop forever until exit
	while (!quit){
      handleEvents(window);
      
      // Clear the screen to default color
      glClearColor(0.5f, 0.0f, 0.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      
	  timePast = static_cast<float>(glfwGetTime());
      float dt = timePast-lastTime;
      lastTime = timePast;
      
	  // Update sim and draw
	  sim->Step(dt);
	  sim->Draw(window);

	  glfwSwapBuffers(window); //Double buffering
	}
	GL_GLFW_del();
	delete sim;
	
	return 0;
}


void mouse_button_cb(GLFWwindow* window, int button, int action, int mods) {
	if (action == GLFW_PRESS) {
		sim->handleMouseButtonDown(window, button);
	} else if (action == GLFW_RELEASE) {
		sim->handleMouseButtonUp(window, button);
	}
}

void mouse_motion_cb(GLFWwindow* window, double x, double y) {
	sim->handleMouseMotion(x, y);
}

void mouse_scroll_cb(GLFWwindow* window, double x, double y) {
	sim->handleMouseScroll(x, y);
}

void key_cb(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_RELEASE) {
		if (key == GLFW_KEY_ESCAPE) {
			quit = true;
			return;
		}
		sim->handleKeyup(window, key);
	}
}

void handleEvents(GLFWwindow *window) {
	if (glfwWindowShouldClose(window)) {
		quit = true;
		return;
	}
	glfwPollEvents();
}