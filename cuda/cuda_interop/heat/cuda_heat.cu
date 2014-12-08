#include <iostream>
#include <stdio.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#define N  600
#define BLOCK_SIZE 15
#define ITERS 100

texture<float> texture_a;
texture<float> texture_b;
GLuint buffer_obj;
cudaGraphicsResource* resource;
float *dev_a, *dev_b;

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

__global__ void data_to_color(uchar4* display, float* b) {
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  int offset = x + y*N;

  display[offset].x = b[offset];
  display[offset].y = 0;  
  display[offset].z = 255 - b[offset];  
  display[offset].w = 255;  
}

static void key_func(unsigned char key, int x, int y) {
  switch(key) {
    // esc key
    case 27:
      cudaGraphicsUnregisterResource(resource);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
      glDeleteBuffers(1, &buffer_obj);

      cudaUnbindTexture(texture_a);
      cudaUnbindTexture(texture_b);
      cudaFree(dev_a);
      cudaFree(dev_b);
      exit(0);
  }
}

static void draw_func() {
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  glDrawPixels(N, N, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glutSwapBuffers();
}


static void update_data() {
  dim3 blocks(N/BLOCK_SIZE, N/BLOCK_SIZE);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  uchar4* dev_ptr;
  size_t size;
  // get the pointer mappped to the cuda resource
  cudaGraphicsMapResources(1, &resource, NULL);
  cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, resource);

  stencil<<<blocks, threads>>>(dev_b);
  copy<<<blocks, threads>>>(dev_a);
  data_to_color<<<blocks, threads>>>(dev_ptr, dev_b);

  cudaGraphicsUnmapResources(1, &resource, NULL);

  glutPostRedisplay();

}


static void idle_func() {
  update_data();
}

int main(int argc, char** argv) {
  float a[N*N], b[N*N];
  cudaDeviceProp prop;
  int dev;

  // find the right cuda device to use
  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 1;
  prop.minor = 0;
  cudaChooseDevice(&dev, &prop);
  cudaGLSetGLDevice(dev);

  // initialize the window
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(N, N);
  glutCreateWindow("cuda_interop/heat");

  // create a pixel buffer object to be used in OpenGL
  glewInit();
  glGenBuffers(1, &buffer_obj);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer_obj);
  // allocate the NxN 32-bit values on the GPU
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, N*N*sizeof(float), NULL,
               GL_DYNAMIC_DRAW_ARB);

  // notify the runtime that we're gonna share the bufffer with cuda
  cudaGraphicsGLRegisterBuffer(&resource, buffer_obj, cudaGraphicsMapFlagsNone);

  cudaMalloc((void**)&dev_a, N*N*sizeof(float));
  cudaMalloc((void**)&dev_b, N*N*sizeof(float));

  cudaBindTexture(NULL, texture_a, dev_a, N*N*sizeof(float));
  cudaBindTexture(NULL, texture_b, dev_b, N*N*sizeof(float));

  int mid = N/2;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i > mid - 50 && i < mid + 50 && 
          j > mid - 50 && j < mid + 50) {
        a[j+i*N] = 255; 
      } else {
        a[j+i*N] = 0;
      }
    }
  }

  cudaMemcpy(dev_a, a, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N*N*sizeof(float), cudaMemcpyHostToDevice);


  glutKeyboardFunc(key_func);
  glutDisplayFunc(draw_func);
  glutIdleFunc(idle_func);

  glutMainLoop();

  cudaFree(dev_a);
  cudaFree(dev_b);
}
