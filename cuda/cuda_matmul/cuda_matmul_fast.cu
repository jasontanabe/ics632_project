#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N  300
#define BLOCK_SIZE 15

struct timeval start, end;

// get global offset of a given block and given index in block
__device__ int global_offset(int block_row, int block_col, int row, int col) {
  return block_row*BLOCK_SIZE*N + N*row + block_col*BLOCK_SIZE + col;
}

__global__ void matmul(int* a, int* b, int* c) {
  // which block we're in
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // row and col within block
  int row = threadIdx.y;
  int col = threadIdx.x;

  int c_val = 0;

  // for all blocks
  for (int i = 0; i < gridDim.x; i++) {
    // shared memory buffers
    __shared__ int as[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int bs[BLOCK_SIZE][BLOCK_SIZE];

    // copy data to shared memory buffer
    as[row][col] = a[global_offset(block_row, i, row, col)];
    bs[row][col] = b[global_offset(i, block_col, row, col)];
    __syncthreads();

    // matrix multiplication for block
    for (int j = 0; j < BLOCK_SIZE; j++) {
      c_val += as[row][j] * bs[j][col];
    }

    __syncthreads();
  }
  c[global_offset(block_row, block_col, row, col)] = c_val;
}

int main() {
  int a[N*N], b[N*N], c[N*N]; 
  int *dev_a, *dev_b, *dev_c;

  // allocate memory on the device
  cudaMalloc((void**)&dev_a, N*N*sizeof(int));
  cudaMalloc((void**)&dev_b, N*N*sizeof(int));
  cudaMalloc((void**)&dev_c, N*N*sizeof(int));

  // fill arbitrary data into arrays
  srand(5);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      a[i*N+j] = rand();
      b[i*N+j] = rand();
      c[i*N+j] = 0.0;
    }
  }

  // copy data from host to device
  cudaMemcpy(dev_a, a, N*N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N*N*sizeof(int), cudaMemcpyHostToDevice);

  // thread and block sizes
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blocks((N+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE);

  gettimeofday(&start, NULL);

  // matrix multiplication kernel
  matmul<<<blocks, threads>>>(dev_a, dev_b, dev_c);

  cudaThreadSynchronize();

  gettimeofday(&end, NULL);

  // copy data from device to host
  cudaMemcpy(c, dev_c, N*N*sizeof(int), cudaMemcpyDeviceToHost);

  // find sum
  int sum = 0;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      sum += c[i*N+j];
    }
  }
  std::cout << "sum is " << sum << std::endl;
  printf("Seconds elapsed: %f\n",
      (end.tv_sec*1000000.0 + end.tv_usec - start.tv_sec*1000000.0 - 
       start.tv_usec) / 1000000.0);


  // free the memory on device
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}


