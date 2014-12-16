#include <iostream>
#include <stdio.h>

#define N  100
#define ITERS 5

__global__ void stencil(float* a, float* b) {
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = x + y * N;

  float update = 0.0;
  if (y > 0) {
    update += a[(y-1)*N+x];
  }
  if (y < N-1) {
    update += a[(y+1)*N+x];
  }
  if (x > 0) {
    update += a[y*N+(x-1)];
  }
  if (x < N-1) {
    update += a[y*N+(x+1)];
  }

  b[offset] = update / 4.0;
}

__global__ void copy(float* to, float* from) {
  int offset = blockIdx.x + blockIdx.y * N;
  to[offset] = from[offset];
}

int main() {
  float a[N*N], b[N*N];
  float *dev_a, *dev_b;

  dim3 blocks(N, N);

  cudaMalloc((void**)&dev_a, N*N*sizeof(float));
  cudaMalloc((void**)&dev_b, N*N*sizeof(float));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i*N+j] = static_cast<float>(i+j);
    }
  }

  cudaMemcpy(dev_a, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N*N*sizeof(float), cudaMemcpyHostToDevice);

  for (int num_it = 0; num_it < ITERS; num_it++) {
    stencil<<<blocks, 1>>>(dev_a, dev_b);
    copy<<<blocks, 1>>>(dev_a, dev_b);
  }

  cudaMemcpy(b, dev_b, N*N*sizeof(float), cudaMemcpyDeviceToHost);

  // print out the new array b
  std::cout << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
    }
  }
  std::cout << std::endl;

  // find sum
  float sum = 0.0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sum += b[i*N+j];
      std::cout << b[i*N+j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "sum is " << sum << std::endl;
  cudaFree(dev_a);
  cudaFree(dev_b);
}


