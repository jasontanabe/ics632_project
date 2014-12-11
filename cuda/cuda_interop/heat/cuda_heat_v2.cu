#include <iostream>
#include <stdio.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>

#define N  600
// should be a multiple of N
#define BLOCK_SIZE 15
// number of iterations per screen refresh (should be odd)
#define ITERATIONS_BEFORE_REFRESH 10
// MAX TEMP VALUE
#define MAX_TEMP_VALUE 5000

texture<float> texture_a;
texture<float> texture_b;
texture<float> texture_const;
GLuint buffer_obj;
cudaGraphicsResource* resource;
float* dev_a;
float* dev_b; 
float* dev_const;
float* const_data;
bool run_anim = false;
bool left_button_down = false;

__global__ void stencil_a(float* out) {
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

  out[offset] = update / 4.0;
}

__global__ void stencil_b(float* out) {
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  int offset = x + y*N;

  int left = offset - 1;
  int right = offset + 1;
  int top = offset - N;
  int bot = offset + N;

  float update = 0.0;
  if (x > 0) {
    update += tex1Dfetch(texture_b, left);
  }
  if (x < N-1) {
    update += tex1Dfetch(texture_b, right);
  }
  if (y > 0) {
    update += tex1Dfetch(texture_b, top);
  } 
  if (y < N-1) {
    update += tex1Dfetch(texture_b, bot);
  }

  out[offset] = update / 4.0;
}


__global__ void copy_const(float* output) {
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  int offset = x + y*N;
  int temp;
  temp = tex1Dfetch(texture_const, offset);
  if (temp != 0) {
    output[offset] = temp;
  }
}

__global__ void data_to_color(uchar4* display, float* b) {
  int x = threadIdx.x + blockIdx.x*blockDim.x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;
  int offset = x + y*N;

  int out = b[offset];
  if (out > 255) {
    out = 255;
  }
  display[offset].x = out;
  display[offset].y = 0;  
  display[offset].z = 255 - out;
  display[offset].w = 255;  
}


bool in_range(int x, int y) {
  return x > 0 && x < N && y > 0 && y < N;
}

uchar4* get_mapped_ptr() {
    uchar4* dev_ptr;
    size_t size;
    // get the pointer mappped to the cuda resource
    cudaGraphicsMapResources(1, &resource, NULL);
    cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, resource);
    return dev_ptr;
}

void clear_data() {
  dim3 blocks(N/BLOCK_SIZE, N/BLOCK_SIZE);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      const_data[j+i*N] = 0;
    }
  }

  cudaMemcpy(dev_a, const_data, N*N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_const, const_data, N*N*sizeof(float), cudaMemcpyHostToDevice);

  uchar4* dev_ptr = get_mapped_ptr();
  data_to_color<<<blocks, threads>>>(dev_ptr, dev_const);
  cudaGraphicsUnmapResources(1, &resource, NULL);

}

void update_data() {
  dim3 blocks(N/BLOCK_SIZE, N/BLOCK_SIZE);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  uchar4* dev_ptr = get_mapped_ptr();
  bool a_input = true;

  for (int i = 0; i < ITERATIONS_BEFORE_REFRESH; i++) {
    if (a_input) {
      copy_const<<<blocks, threads>>>(dev_a);
      stencil_a<<<blocks, threads>>>(dev_b);
    } else {
      copy_const<<<blocks, threads>>>(dev_b);
      stencil_b<<<blocks, threads>>>(dev_a);
    }
    a_input = !a_input;
  }
  data_to_color<<<blocks, threads>>>(dev_ptr, dev_b);
  cudaGraphicsUnmapResources(1, &resource, NULL);

  glutPostRedisplay();
}

void draw_func() {
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  glDrawPixels(N, N, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glutSwapBuffers();
}

void key_func(unsigned char key, int x, int y) {
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
    case 's':
      run_anim = true;
      break;
    case 'r':
      run_anim = false;
      clear_data();
      break;
  }
}

void mouse_func(int button, int state, int x, int y) {
  if (button == GLUT_LEFT_BUTTON) {
    if (state == GLUT_DOWN) {
      left_button_down = true;
    } else {
      left_button_down = false;
    }
  }
}

void mouse_motion_func(int x, int y) {
  if (left_button_down && in_range(x, y) && !run_anim) {
    dim3 blocks(N/BLOCK_SIZE, N/BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    uchar4* dev_ptr = get_mapped_ptr();

    const_data[(N-y)*N+x] = MAX_TEMP_VALUE;

    cudaMemcpy(dev_const, const_data, N*N*sizeof(float), 
               cudaMemcpyHostToDevice);
    data_to_color<<<blocks, threads>>>(dev_ptr, dev_const);
  }
  cudaGraphicsUnmapResources(1, &resource, NULL);

  glutPostRedisplay();
}

void idle_func() {
  if (run_anim) {
    update_data();
  }
}


int main(int argc, char** argv) {
  dim3 blocks(N/BLOCK_SIZE, N/BLOCK_SIZE);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  cudaDeviceProp prop;
  int dev;

  // find the right cuda device to use
  memset(&prop, 0, sizeof(cudaDeviceProp));
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

  const_data = (float*)malloc(N*N*sizeof(float));

  cudaMalloc((void**)&dev_a, N*N*sizeof(float));
  cudaMalloc((void**)&dev_b, N*N*sizeof(float));
  cudaMalloc((void**)&dev_const, N*N*sizeof(float));

  cudaBindTexture(NULL, texture_a, dev_a, N*N*sizeof(float));
  cudaBindTexture(NULL, texture_b, dev_b, N*N*sizeof(float));
  cudaBindTexture(NULL, texture_const, dev_const, N*N*sizeof(float));

  clear_data();

  glutKeyboardFunc(key_func);
  glutDisplayFunc(draw_func);
  glutIdleFunc(idle_func);
  glutMouseFunc(mouse_func);
  glutMotionFunc(mouse_motion_func);

  std::cout << "draw to initialize" << std::endl;
  std::cout << "press s to start" << std::endl;
  std::cout << "press esc key to exit" << std::endl;

  glutMainLoop();

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_const);
}
