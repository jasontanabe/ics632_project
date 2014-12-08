#include <iostream>
#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#define DIM 512

GLuint buffer_obj;
cudaGraphicsResource* resource;

__global__ void kernel(uchar4* ptr) {
  int x = threadIdx.x + blockIdx.x*blockDim.x;  
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  int offset = x + y*blockDim.x*gridDim.x;

  float fx = x/(float)DIM - 0.5f;
  float fy = y/(float)DIM - 0.5f;
  unsigned char green = 128 + 127*sin(abs(fx*100) - abs(fy*100));
  ptr[offset].x = 0;
  ptr[offset].y = green;
  ptr[offset].z = 0;
  ptr[offset].w = 255;
}

static void draw_func(void) {
  glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glutSwapBuffers();
}

static void key_func(unsigned char key, int x, int y) {
  switch(key) {
    case 27:
      cudaGraphicsUnregisterResource(resource);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
      glDeleteBuffers(1, &buffer_obj);
      exit(0);
  }
}

int main(int argc, char** argv) {
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
  glutInitWindowSize(DIM, DIM);
  glutCreateWindow("cuda_interop/waves");

  // createa a pixel buffer object to be used in OpenGL
  glewInit();
  glGenBuffers(1, &buffer_obj);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer_obj);
  // allocate DIMxDIM 32-bit values and fill it with no data (NULL)
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM*DIM*4, NULL, 
               GL_DYNAMIC_DRAW_ARB);
  
  // notify the runtime that we're gonna share the buffer with cuda
  cudaGraphicsGLRegisterBuffer(&resource, buffer_obj, cudaGraphicsMapFlagsNone);

  // device pointer mapped to the cuda resource
  uchar4* dev_ptr;
  size_t size;
  cudaGraphicsMapResources(1, &resource, NULL);
  cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, resource);

  dim3 grids(DIM/16, DIM/16);
  dim3 threads(16, 16);
  kernel<<<grids, threads>>>(dev_ptr);

  // important to make this call prior to rendering 
  // (provides synchronization between CUDA and graphics portion)
  cudaGraphicsUnmapResources(1, &resource, NULL);

  // callbacks for the glut window
  glutDisplayFunc(draw_func);
  glutKeyboardFunc(key_func);

  glutMainLoop();

  return 1;
}
