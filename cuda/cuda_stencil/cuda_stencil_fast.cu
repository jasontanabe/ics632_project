#include <iostream>
#include <stdio.h>

#define N  100
#define BLOCK_SIZE 5
#define ITERS 5

texture<float> texture_a;
texture<float> texture_b;

__global__ void stencil(float* b) {
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  int offset = x + y*N;

  int left = offset - 1;
  int right = offset + 1;
  int top = offset - N;
  int bot = offset + N;

  float update = 0.0;
  if (x > 0) {
    update += tex1Dfetch(texture_a, left);
  }
  if (x < N-1) {
    update += tex1Dfetch(texture_a, right);
  }
  if (y > 0) {
    update += tex1Dfetch(texture_a, top);
  } 
  if (y < N-1) {
    update += tex1Dfetch(texture_a, bot);
  }

  b[offset] = update / 4.0;
}

__global__ void copy(float* a) {
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  int offset = x + y*N;
  a[offset] = tex1Dfetch(texture_b, offset);
}

int main() {
  float a[N*N], b[N*N];
  float *dev_a, *dev_b;

  dim3 blocks(N/BLOCK_SIZE, N/BLOCK_SIZE);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

  cudaMalloc((void**)&dev_a, N*N*sizeof(float));
  cudaMalloc((void**)&dev_b, N*N*sizeof(float));

  cudaBindTexture(NULL, texture_a, dev_a, N*N*sizeof(float));
  cudaBindTexture(NULL, texture_b, dev_b, N*N*sizeof(float));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i*N+j] = static_cast<float>(i+j);
    }
  }

  cudaMemcpy(dev_a, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N*N*sizeof(float), cudaMemcpyHostToDevice);

  for (int num_it = 0; num_it < ITERS; num_it++) {
    stencil<<<blocks, threads>>>(dev_b);
    copy<<<blocks, threads>>>(dev_a);
  }

  cudaMemcpy(b, dev_b, N*N*sizeof(float), cudaMemcpyDeviceToHost);

  // print out the new array b
  std::cout << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
    }
  }

  // find sum
  float sum = 0.0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << b[i*N+j] << " ";
      sum += b[i*N+j];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::cout << "sum is " << sum << std::endl;

  cudaUnbindTexture(texture_a);
  cudaUnbindTexture(texture_b);
  cudaFree(dev_a);
  cudaFree(dev_b);
}


