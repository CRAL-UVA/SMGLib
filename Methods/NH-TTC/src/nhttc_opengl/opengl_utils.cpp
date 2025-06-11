/*
Copyright 2020 University of Minnesota and Clemson University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/*
This file contains helper utilities to interface with GLFW and STB
*/

#include <nhttc_opengl/opengl_utils.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

std::vector<int> shaderprogs;
GLuint vao;

using namespace std;

static void error_callback(int error, const char* description) {
	std::cerr << "GLFW Error Code: " << error << " Error: " << description << std::endl;
}

GLFWwindow* GL_GLFW_init(const char* window_name) {
	glfwSetErrorCallback(error_callback);
	if (!glfwInit()) {
		exit(-1);
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_SAMPLES, 4);
	GLFWwindow* window = glfwCreateWindow(800, 600, window_name, NULL, NULL);
	if (!window) {
		glfwTerminate();
		exit(-1);
	}

	glfwMakeContextCurrent(window);
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to gladLoadGLLoader" << std::endl;
		glfwTerminate();
		exit(-1);
	}

	glfwSwapInterval(1);
	
	glEnable(GL_MULTISAMPLE);
	
	//Build a Vertex Array Object. This stores the VBO and attribute mappings in one object
	glGenVertexArrays(1, &vao); //Create a VAO
	glBindVertexArray(vao); //Bind the above created VAO to the current context

	// Hack to get drawing
	{
		int x, y;
		glfwGetWindowPos(window, &x, &y);
		glfwSetWindowPos(window, 100, 100);
		glfwSetWindowPos(window, x, y);
	}
	
	return window;
}

void GL_GLFW_del() {
	//Clean Up
	for (unsigned int i = 0; i < shaderprogs.size(); i++) {
		glDeleteProgram(shaderprogs[i]);
	}
    glDeleteVertexArrays(1, &vao);

	glfwTerminate();
}

// Create a GLSL program object from vertex and fragment shader files
GLuint InitShader(const char* vShaderFileName, const char* fShaderFileName, const char* gShaderFileName)
{
	GLint link_ok;
	GLuint vs,gs,fs;
	if ((vs = create_shader(vShaderFileName, GL_VERTEX_SHADER))   == 0) exit(-1);
	if ((gs = create_shader(gShaderFileName, GL_GEOMETRY_SHADER))   == 0) exit(-1);
	if ((fs = create_shader(fShaderFileName, GL_FRAGMENT_SHADER)) == 0) exit(-1);

	GLuint program = glCreateProgram();
	glAttachShader(program, vs);
	glAttachShader(program, gs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &link_ok);
	if (!link_ok) {
	fprintf(stderr, "glLinkProgram:");
	print_log(program);
	return 0;
	}
	glUseProgram(program);
	
	shaderprogs.push_back(program);
	
	return program;
}

// Create a GLSL program object from vertex and fragment shader files
GLuint InitShader(const char* vShaderFileName, const char* fShaderFileName)
{
	GLint link_ok;
	GLuint vs,fs;
	if ((vs = create_shader(vShaderFileName, GL_VERTEX_SHADER))   == 0) exit(-1);
	if ((fs = create_shader(fShaderFileName, GL_FRAGMENT_SHADER)) == 0) exit(-1);

	GLuint program = glCreateProgram();
	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glGetProgramiv(program, GL_LINK_STATUS, &link_ok);
	if (!link_ok) {
	fprintf(stderr, "glLinkProgram:");
	return 0;
	}
	glUseProgram(program);
	
	shaderprogs.push_back(program);
	
	return program;
}

char* file_read(const char* filename)
{
  FILE* input = fopen(filename, "rb");
  if(input == NULL) return NULL;
 
  if(fseek(input, 0, SEEK_END) == -1) return NULL;
  long size = ftell(input);
  if(size == -1) return NULL;
  if(fseek(input, 0, SEEK_SET) == -1) return NULL;
 
  /*if using c-compiler: dont cast malloc's return value*/
  char *content = (char*) malloc( (size_t) size +1  ); 
  if(content == NULL) return NULL;
 
  size_t rLen = fread(content, 1, (size_t)size, input);
  if(ferror(input) || rLen == 0) {
    free(content);
    return NULL;
  }
 
  fclose(input);
  content[size] = '\0';
  return content;
}

void print_log(GLuint object)
{
  GLint log_length = 0;
  if (glIsShader(object))
    glGetShaderiv(object, GL_INFO_LOG_LENGTH, &log_length);
  else if (glIsProgram(object))
    glGetProgramiv(object, GL_INFO_LOG_LENGTH, &log_length);
  else {
    fprintf(stderr, "printlog: Not a shader or a program\n");
    return;
  }
 
  char* log = (char*)malloc(log_length);
 
  if (glIsShader(object))
    glGetShaderInfoLog(object, log_length, NULL, log);
  else if (glIsProgram(object))
    glGetProgramInfoLog(object, log_length, NULL, log);
 
  fprintf(stderr, "%s", log);
  free(log);
}

GLuint create_shader(const char* filename, GLenum type)
{
	const GLchar* source = file_read(filename);
	if (source == NULL) {
		fprintf(stderr, "Error opening %s: ", filename); perror("");
		return 0;
	}
	GLuint res = glCreateShader(type);
	const GLchar* sources[2] = {"",	source };
	glShaderSource(res, 2, sources, NULL);
	free((void*)source);

	glCompileShader(res);
	GLint compile_ok = GL_FALSE;
	glGetShaderiv(res, GL_COMPILE_STATUS, &compile_ok);
	if (compile_ok == GL_FALSE) {
		fprintf(stderr, "%s:", filename);
		print_log(res);
		glDeleteShader(res);
		return 0;
	}
 
  return res;
}

int loadModel(string fName, float* &model1) {
	ifstream modelFile;
	modelFile.open(fName.c_str());
	int numLines = 0;
	modelFile >> numLines;
	model1 = new float[numLines];
	for (int i = 0; i < numLines; i++){
		modelFile >> model1[i];
	}
	modelFile.close();
	return numLines/8;
}

GLuint loadTexture(const char* file, int &width, int &height) {
	int channels;
	stbi_set_flip_vertically_on_load(true);
	unsigned char *image = stbi_load(file,
									&width,
									&height,
									&channels,
									STBI_rgb_alpha);

   //Now generate the OpenGL texture object
   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
   glTexImage2D(GL_TEXTURE_2D,0, GL_RGBA, width, height, 0,
       GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*) image);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
   glGenerateMipmap(GL_TEXTURE_2D);
 
   //clean up memory and close stuff
   stbi_image_free(image);
 
   return texture;
 }