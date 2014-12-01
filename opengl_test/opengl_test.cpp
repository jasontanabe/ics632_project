#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <array>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#define bwidth 1280
#define bheight 720

std::string loadSource(std::string fname);
GLuint compileShader(std::string fname, GLenum shaderType);
GLuint createShaderProgram(std::string vertShaderCode, std::string fragShaderCode);

int main(int argc, char *argv[])
{
	GLuint vao, vbo, eab;
	GLuint tex;

	// GLFW initialization
	if (glfwInit() != GL_TRUE) {
		return 1;
	}

	// get primary display/desktop properties
	GLFWmonitor *primary = glfwGetPrimaryMonitor();
	const GLFWvidmode *vidmode = glfwGetVideoMode(primary);
	GLint pWidth = vidmode->width;
	GLint pHeight = vidmode->height;
	// make sure defined window isnt too large to display on your screen
	if (bwidth > pWidth - 150 || bheight > pHeight - 150) {
		std::cerr << "Window should be <= " << pWidth - 150 << " x " << pHeight - 150 << std::endl;
		glfwTerminate();
		return 1;
	}

	// specify standards and create GL window/context
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	GLFWwindow *window = glfwCreateWindow(bwidth, bheight, "GL test", NULL, NULL);
	glfwMakeContextCurrent(window);
	//get actual obtained framebuffer size in pixels
	glfwGetFramebufferSize(window, &pWidth, &pHeight);

	// GLEW initialization
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (err != GLEW_OK) {
		std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
		glfwTerminate();
		return 1;
	}

	// get some GPU and OpenGL info...
	std::string renderer = std::string(reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
	std::string version = std::string(reinterpret_cast<const char*>(glGetString(GL_VERSION)));
	std::cout << "Renderer: " << renderer << std::endl;
	std::cout << "Version: " << version << std::endl;

	// create the pixel/texture data; normalized RGBA float pixels
	GLint maxNorm = pWidth + pHeight - 2;
	std::vector<GLfloat> texData;
	for (int i = 0; i < pHeight; i++) {
		for (int j = 0; j < pWidth; j++) {
			GLfloat val = static_cast<GLfloat>(i+j) / maxNorm;
			texData.push_back(val);		// R
			texData.push_back(0.0f);	// G
			texData.push_back(1.0f - val);	// B
			texData.push_back(1.0f);	// A
		}
	}

	// define the triangle vertices for quad (whole window)
	// (x,y) pair from LL, LR, UR, UL (counter-clockwise around quad)
	std::array<GLfloat, 8> quadVertices = { {
		-1.0f, -1.0f,
		1.0f, -1.0f,
		1.0f, 1.0f,
		-1.0f, 1.0f
	} };
	
	// map texture vertices to quad vertices
	std::array<GLfloat, 8> texVertices = { {
		0.0f, 1.0f,
		1.0f, 1.0f,
		1.0f, 0.0f,
		0.0f, 0.0f
	} };

	// element coordinate of triangles that make up quad
	std::array<GLuint, 6> quadIndices = { {
		0, 1, 2,
		2, 3, 0
	} };

	// generate, bind, and populate buffers
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, 16*sizeof(GLfloat), NULL, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, 8*sizeof(GLfloat), &quadVertices.at(0));
	glBufferSubData(GL_ARRAY_BUFFER, 8*sizeof(GLfloat), 8*sizeof(GLfloat), &texVertices.at(0));
	
	glGenBuffers(1, &eab);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eab);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6*sizeof(GLuint), &quadIndices.at(0), GL_STATIC_DRAW);
	
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, bwidth, bheight, 0, GL_RGBA, GL_FLOAT, &texData.at(0));
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// load and compile shaders
	GLuint shaderProgram = createShaderProgram("vertex_shader.glsl", "fragment_shader.glsl");

	// shader specs
	GLint positionAttrib = glGetAttribLocation(shaderProgram, "position");
	glVertexAttribPointer(positionAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(positionAttrib);
	
	GLint texAttrib = glGetAttribLocation(shaderProgram, "texCoord");
	glVertexAttribPointer(texAttrib, 2, GL_FLOAT, GL_FALSE, 0, (GLvoid*)(8*sizeof(GLfloat)));
	glEnableVertexAttribArray(texAttrib);

	//glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	//glViewport(0, 0, bwidth, bheight);

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT);
		glBindBuffer(GL_ARRAY_BUFFER, vao);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	
	return 0;
}

std::string loadSource(std::string fname)
{
	std::ifstream shaderFile(fname);
	std::stringstream shaderBuffer;
	shaderBuffer << shaderFile.rdbuf();
	
	return shaderBuffer.str();
}

GLuint compileShader(std::string fname, GLenum shaderType)
{
	std::string shaderSource = loadSource(fname);
	
	GLuint shader = glCreateShader(shaderType);
	glShaderSource(shader, 1, reinterpret_cast<const GLchar**>(&shaderSource), NULL);
	glCompileShader(shader);

	return shader;
}

GLuint createShaderProgram(std::string vertShaderCode, std::string fragShaderCode)
{
	GLuint vertexShader = compileShader(vertShaderCode, GL_VERTEX_SHADER);
	GLuint fragmentShader = compileShader(fragShaderCode, GL_FRAGMENT_SHADER);
	
	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	glLinkProgram(shaderProgram);
	glUseProgram(shaderProgram);
	
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return shaderProgram;
}
