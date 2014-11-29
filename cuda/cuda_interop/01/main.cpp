#include <GL/glut.h>

#define DIM 512

GLuint bufferObj;
cudaGraphicsResource *resource;

int main(int argc, char** argv) {
  cudaDeviceProp prop;
  int dev;

  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 1;
  prop.minor = 0;
  cudaChooseDevice(&dev, &prop);
  cudaGLSetGLDevice(dev);
}
