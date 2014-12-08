#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#define N 300

int a[N][N];
int b[N][N];
int c[N][N];

struct timeval start, end;

int main() {
  int i = 0; 
  int j = 0;
  int k = 0;
  int sum = 0;

  srand(5);
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      a[i][j] = rand();
      b[i][j] = rand();
      c[i][j] = 0;
    }
  }

  omp_set_num_threads(2);

  gettimeofday(&start, NULL);

#pragma omp parallel shared(a, b, c) private(i, j, k) 
{
  #pragma omp for 
  for (i = 0; i < N; ++i) {
    for (k = 0; k < N; ++k)  {
      for (j = 0; j < N; ++j)  {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

  gettimeofday(&end, NULL);
  sum = 0;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      sum += c[i][j];
    }
  }
  printf("Sum is %d\n", sum);

  printf("Seconds elapsed: %f\n",
      (end.tv_sec*1000000.0 + end.tv_usec - start.tv_sec*1000000.0 - 
       start.tv_usec) / 1000000.0);
}
