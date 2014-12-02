#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N  300
#define NUM_THREADS 16

__global__ void matmul(int* a, int* b, int* c) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= N || col >= N) {
    return;
  }
  int c_val = 0.0;
  for (int i = 0; i < N; i++) {
    c_val += a[row*N+i] * b[i*N+col];
  }
  c[row*N+col] = c_val;
}

int main() {
  int a[N*N], b[N*N], c[N*N]; int *dev_a, *dev_b, *dev_c;

  cudaMalloc((void**)&dev_a, N*N*sizeof(int));
  cudaMalloc((void**)&dev_b, N*N*sizeof(int));
  cudaMalloc((void**)&dev_c, N*N*sizeof(int));

  srand(5);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i*N+j] = rand();
      b[i*N+j] = rand();
      c[i*N+j] = 0.0;
    }
  }

  cudaMemcpy(dev_a, a, N*N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N*N*sizeof(int), cudaMemcpyHostToDevice);

  dim3 threads(NUM_THREADS, NUM_THREADS);
  dim3 blocks((N+NUM_THREADS-1)/NUM_THREADS, (N+NUM_THREADS-1)/NUM_THREADS);

  matmul<<<blocks, threads>>>(dev_a, dev_b, dev_c);

  cudaThreadSynchronize();

  cudaMemcpy(c, dev_c, N*N*sizeof(int), cudaMemcpyDeviceToHost);

  // find sum
  int sum = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sum += c[i*N+j];
    }
  }
  std::cout << "sum is " << sum << std::endl;
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}


