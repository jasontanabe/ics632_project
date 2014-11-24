#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <CL/cl.hpp>

int main(int argc, char* argv[])
{
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;
	cl::Context context;
	cl::CommandQueue queue;
	cl::Program::Sources source;
	cl::Program program;
	cl::Kernel kernel;
	cl::Buffer bufferA, bufferB, bufferN;

	std::string platformVendor;
	std::string sourceCode;

	if (argc < 3) {
		std::cerr << "Usage: ./vector_add <N_size> <num_iters>" << std::endl;
		return 1;
	}

	int N = std::stoi(argv[1]);
	int iters = std::stoi(argv[2]);
	int bufferSize = N * N * sizeof(float);
	std::vector<float> A(N*N), B(N*N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A.at(i*N+j) = static_cast<float>(i+j);
			std::cout << A.at(i*N+j) << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::ifstream sourceFile("stencil_kernel.cl");
	std::stringstream sourceBuffer;
	sourceBuffer << sourceFile.rdbuf();
	sourceCode = sourceBuffer.str();

	try {
		cl::Platform::get(&platforms);
		platforms[0].getInfo(static_cast<cl_platform_info>(CL_PLATFORM_VENDOR), &platformVendor);
		std::cerr << "Platform number is: " << platforms.size() << std::endl;
		std::cerr << "Platform is by: " << platformVendor << std::endl;

		cl_context_properties cprops[3] = {
			CL_CONTEXT_PLATFORM,
			reinterpret_cast<cl_context_properties>(platforms[0]()),
			0
		};
		context = cl::Context(CL_DEVICE_TYPE_GPU, cprops);
		devices = context.getInfo<CL_CONTEXT_DEVICES>();
	
		source = cl::Program::Sources(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
		program = cl::Program(context, source);
		program.build(devices);
		kernel = cl::Kernel(program, "stencil");

		bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, bufferSize);
		bufferB = cl::Buffer(context, CL_MEM_READ_WRITE, bufferSize);
		bufferN = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));

		queue = cl::CommandQueue(context, devices[0]);
		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, bufferSize, &A.at(0));
		queue.enqueueWriteBuffer(bufferN, CL_TRUE, 0, sizeof(int), &N);

		cl::NDRange global(N, N);
		cl::NDRange local(1, 1);
		kernel.setArg(0, bufferA);
		kernel.setArg(1, bufferB);
		kernel.setArg(2, bufferN);
		for (int num_iter = 0; num_iter < iters; num_iter++) {
			queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
			queue.enqueueCopyBuffer(bufferB, bufferA, 0, 0, bufferSize);
		}

		queue.enqueueReadBuffer(bufferB, CL_TRUE, 0, bufferSize, &B.at(0));
	} catch(cl::Error error) {
		std::cerr << error.what() << " (" << error.err() << ")" << std::endl;
	}

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			std::cout << std::fixed << B.at(i*N+j) << " ";
		}
		std::cout << std::endl;
	}

	return 0;
}

