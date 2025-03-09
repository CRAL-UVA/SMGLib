/*
Copyright 2020 University of Minnesota and Clemson University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define PI 3.141592f

#define GLM_FORCE_RADIANS
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/projection.hpp>
#include <glm/gtx/string_cast.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <string>
#include <vector>
#include <fstream>

GLFWwindow* GL_GLFW_init(const char* window_name);
void GL_GLFW_del();

char* file_read(const char* filename);
void print_log(GLuint object);
GLuint create_shader(const char* filename, GLenum type);

GLuint InitShader(const char* vShaderFileName, const char* fShaderFileName, const char* gShaderFileName);
GLuint InitShader(const char* vShaderFileName, const char* fShaderFileName);

int loadModel(std::string fName, float *&model1);
GLuint loadTexture(const char* file, int &width, int &height);