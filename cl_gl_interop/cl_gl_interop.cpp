#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef linux
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#elif defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#elif defined(TARGET_OS_MAC)
#define GLFW_EXPOSE_NATIVE_COCOA
#define GLFW_EXPOSE_NATIVE_NSGL
#endif

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <array>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <CL/cl.h>
#include <CL/cl_gl.h>

#define NSIZE 600
#define iters 100

std::string loadSource(std::string fname);
GLuint compileShader(std::string fname, GLenum shaderType);
GLuint createShaderProgram(std::string vertShaderCode, std::string fragShaderCode);
void display(GLFWwindow *window, GLuint vao);

int main(int argc, char* argv[])
{
	int N = NSIZE; // texture/buffer dimensions

	GLuint vao, vbo, eab;
	GLuint tex;

	cl_platform_id platformID;
	cl_device_id deviceID;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem texBuffA, texBuffB, bufferN;

	std::string platformVendor, platformName, platformVersion;
	std::string kernelCode;

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
	if (NSIZE > pWidth - 150 || NSIZE > pHeight - 150) {
		std::cerr << "Window should be <= " << pWidth - 150 << " x " << pHeight - 150 << std::endl;
		glfwTerminate();
		return 1;
	}

	// specify standards and create GL window/context
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	GLFWwindow *window = glfwCreateWindow(NSIZE, NSIZE, "GL test", NULL, NULL);
	glfwMakeContextCurrent(window);

	// GLEW initialization
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (err != GLEW_OK) {
		std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
		glfwTerminate();
		return 1;
	}

	// create the pixel/texture data; normalized RGBA float pixels
	GLint maxNorm = 2*(N*N*N);
	std::vector<GLfloat> texData;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			GLfloat val = static_cast<GLfloat>(i*i*i + j*j*j) / maxNorm;
			texData.push_back(val);			// R
			texData.push_back(0.0f);		// G
			texData.push_back(1.0f - val);	// B
			texData.push_back(1.0f);		// A
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
	glBufferData(GL_ARRAY_BUFFER, 16 * sizeof(GLfloat), NULL, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, 8 * sizeof(GLfloat), &quadVertices.at(0));
	glBufferSubData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), 8 * sizeof(GLfloat), &texVertices.at(0));

	glGenBuffers(1, &eab);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eab);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(GLuint), &quadIndices.at(0), GL_STATIC_DRAW);

	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, N, N, 0, GL_RGBA, GL_FLOAT, &texData.at(0));
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// load and compile shaders
	GLuint shaderProgram = createShaderProgram("vertex_shader.glsl", "fragment_shader.glsl");

	// shader specs
	GLint positionAttrib = glGetAttribLocation(shaderProgram, "position");
	glVertexAttribPointer(positionAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(positionAttrib);

	GLint texAttrib = glGetAttribLocation(shaderProgram, "texCoord");
	glVertexAttribPointer(texAttrib, 2, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<GLvoid*>(8 * sizeof(GLfloat)));
	glEnableVertexAttribArray(texAttrib);

	cl_int clerr;
	clerr = clGetPlatformIDs(1, &platformID, NULL);
	clerr = clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, NULL);

	// although GLFW window creation/management is cross-platform, CL context creation from GL currently is not...
	// get native context handle depending on platform...
	#ifdef linux
	cl_context_properties cprops[] = {
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties)(platformID),
		CL_GL_CONTEXT_KHR, reinterpret_cast<cl_context_properties)(glXGetCurrentContext()),
		CL_GLX_DISPLAY_KHR, reinterpret_cast<cl_context_properties>(glXGetCurrentDisplay()), 
		0
	};
	#elif defined(_WIN32)
	cl_context_properties cprops[] = {
		CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platformID),
		CL_GL_CONTEXT_KHR, reinterpret_cast<cl_context_properties>(wglGetCurrentContext()),
		CL_WGL_HDC_KHR, reinterpret_cast<cl_context_properties>(wglGetCurrentDC()),
		0
	};
	#elif defined(TARGET_OS_MAC)
	CGLContextObj glContext = CGLGetCurrentContext();
	CGLShareGroupObj shareGroup = CGLGetShareGroup(glContext);
	cl_context_properties cprops[] = {
		CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
		reinterpret_cast<cl_context_properties>(shareGroup),
		0
	};
	#endif

	context = clCreateContext(cprops, 1, &deviceID, NULL, NULL, &clerr);
	queue = clCreateCommandQueue(context, deviceID, NULL, &clerr);

	kernelCode = loadSource("stencil_kernel.cl");
	const char *c_kernelCode = kernelCode.c_str();
	program = clCreateProgramWithSource(context, 1, &c_kernelCode, NULL, &clerr);
	clerr = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "stencil", &clerr);

	cl_image_format clf;
	clf.image_channel_order = CL_RGBA;
	clf.image_channel_data_type = CL_FLOAT;
	texBuffA = clCreateFromGLTexture2D(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, tex, &clerr);
	texBuffB = clCreateImage2D(context, CL_MEM_READ_WRITE, &clf, N, N, 0, NULL, &clerr);

	clerr = clSetKernelArg(kernel, 0, sizeof(cl_mem), &texBuffA);
	clerr = clSetKernelArg(kernel, 1, sizeof(cl_mem), &texBuffB);

	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { N, N, 1 };
	size_t global[2] = { N, N };
	double curTime = glfwGetTime();
	double refreshRate = 10.0; // in frames per second
	for (int num_iter = 0; num_iter < iters; num_iter++) {
		//texBuffA = clCreateFromGLTexture2D(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, tex, &clerr);
		clerr = clEnqueueAcquireGLObjects(queue, tex, &texBuffA, 0, NULL, NULL);
		clerr = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
		clerr = clEnqueueCopyImage(queue, texBuffB, texBuffA, origin, origin, region, 0, NULL, NULL);
		clerr = clEnqueueReleaseGLObjects(queue, tex, &texBuffA, 0, NULL, NULL);
		clFinish(queue);
		while ((glfwGetTime() - curTime) < 1.0 / refreshRate);
		display(window, vao);
		curTime = glfwGetTime();
	}

	glfwTerminate();

	return 0;
}

void display(GLFWwindow *window, GLuint vao)
{
	glClear(GL_COLOR_BUFFER_BIT);
	glBindBuffer(GL_ARRAY_BUFFER, vao);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
	glfwSwapBuffers(window);
	glfwPollEvents();
}

std::string loadSource(std::string fname)
{
	std::ifstream sourceFile(fname);
	std::stringstream sourceBuffer;
	sourceBuffer << sourceFile.rdbuf();

	return sourceBuffer.str();
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