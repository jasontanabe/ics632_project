# PyQt4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
# PyOpenCL imports
import pyopencl as cl
from pyopencl.tools import get_gl_sharing_context_properties

from ctypes import sizeof, c_float, c_void_p, c_uint
from array import array
import time


# OpenCL kernel that generates a sine function.
clkernel = """
__kernel void clkernel(__global float4* clcol, __global float4*glcol, __global float2* clpos, __global float2* glpos)
{
    unsigned int x = get_global_id(0)%302;
    unsigned int y = get_global_id(0)/302;
    if((x > 0 && x < (302-1)) && (y > 0 && y < (302-1))) {
        float current = clcol[x+y*302].x*3+clcol[x+y*302].y*2+clcol[x+y*302].z;
        float val_up = clcol[x+(y-1)*302].x*3+clcol[x+(y-1)*302].y*2+clcol[x+(y-1)*302].z;
        float val_left = clcol[x-1+y*302].x*3+clcol[x-1+y*302].y*2+clcol[x-1+y*302].z;
        float val_right = clcol[x+1+y*302].x*3+clcol[x+1+y*302].y*2+clcol[x+1+y*302].z;
        float val_down = clcol[x+(y+1)*302].x*3+clcol[x+(y+1)*302].y*2+clcol[x+(y+1)*302].z;
        float val = (val_up+val_left+val_right+val_down)*0.251;
        if (val > current) {
            if (val > 2) {
                glcol[x+y*302].x = val/3.0;
                glcol[x+y*302].y = 0;
                glcol[x+y*302].z = 0;
            }
            else if (val > 1) {
                glcol[x+y*302].x = 0;
                glcol[x+y*302].y = val/2;
                glcol[x+y*302].z = 0;
            }
            else {
                glcol[x+y*302].x = 0;
                glcol[x+y*302].y = 0;
                glcol[x+y*302].z = val;
            }
            //glcol[x+y*302].y = glcol[x+y*302].z = 0
        }
        else {
            glcol[x+y*302].x = clcol[x+y*302].x;
            glcol[x+y*302].y = clcol[x+y*302].y;
            glcol[x+y*302].z = clcol[x+y*302].z;
        }
        glcol[x+y*302].w = 1.0;
        clcol[x+y*302].x = glcol[x+y*302].x;
        clcol[x+y*302].y = glcol[x+y*302].y;
        clcol[x+y*302].z = glcol[x+y*302].z;
        clcol[x+y*302].w = glcol[x+y*302].w;
        glpos[x+y*302].x = clpos[x+y*302].x;
        glpos[x+y*302].y = clpos[x+y*302].y;
    }
}
"""

def clinit():
    """Initialize OpenCL with GL-CL interop.
    """
    plats = cl.get_platforms()
    # handling OSX
    if sys.platform == "darwin":
        ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                             devices=[])
    else:
        ctx = cl.Context(properties=[
                            (cl.context_properties.PLATFORM, plats[0])]
                            + get_gl_sharing_context_properties())
    queue = cl.CommandQueue(ctx)
    return ctx, queue

class GLPlotWidget(QGLWidget):
    # default window size
    width, height = 600, 600

    def set_data(self, data):
        """Load 2D data as a Nx2 Numpy array.
        """
        self.data = data
        self.data_count = data.shape[0]

    def set_color(self, color):
        """Load 2D data as a Nx2 Numpy array.
        """
        self.color = color
        self.color_count = color.shape[0]

    def initialize_buffers(self):
        """Initialize OpenGL and OpenCL buffers and interop objects,
        and compile the OpenCL kernel.
        """
        # empty OpenGL VBO and color buffer
        self.glbuf = glvbo.VBO(data=np.zeros(self.data.shape),
                               usage=gl.GL_DYNAMIC_DRAW,
                               target=gl.GL_ARRAY_BUFFER)
        self.glbuf.bind()
        self.glcolbuf = glvbo.VBO(data=np.zeros(self.color.shape),
                               usage=gl.GL_DYNAMIC_DRAW,
                               target=gl.GL_ARRAY_BUFFER)
        self.glcolbuf.bind()
        # initialize the CL context
        self.ctx, self.queue = clinit()
        # create read/write OpenCL buffers
        self.clbuf = cl.Buffer(self.ctx,
                            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                            hostbuf=self.data)
        self.clcolbuf = cl.Buffer(self.ctx,
                            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                            hostbuf=self.color)
        # create an interop object to access to GL VBO from OpenCL
        self.glclbuf = cl.GLBuffer(self.ctx, cl.mem_flags.READ_WRITE,
                            int(self.glbuf.buffers[0]))
        self.glclcolbuf = cl.GLBuffer(self.ctx, cl.mem_flags.READ_WRITE,
                            int(self.glcolbuf.buffers[0]))
        # build the OpenCL program
        self.program = cl.Program(self.ctx, clkernel).build()
        # release the PyOpenCL queue
        self.queue.finish()

    def execute(self):
        """Execute the OpenCL kernel.
        """
        # get secure access to GL-CL interop objects
        cl.enqueue_acquire_gl_objects(self.queue, [self.glclbuf, self.glclcolbuf])
        # arguments to the OpenCL kernel
        kernelargs = (self.clcolbuf,
                      self.glclcolbuf,
                      self.clbuf,
                      self.glclbuf)
        # execute the kernel
        self.program.clkernel(self.queue, (self.color_count,), None, *kernelargs)
        # release access to the GL-CL interop objects
        cl.enqueue_release_gl_objects(self.queue, [self.glclbuf, self.glclcolbuf])
        self.queue.finish()

    def update_buffer(self):
        """Update the GL buffer from the CL buffer
        """
        # execute the kernel before rendering
        self.execute()
        gl.glFlush()

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc.
        """
        # initialize OpenCL first
        self.initialize_buffers()
        # set background color
        gl.glClearColor(0,0,0,0)
        # update the GL buffer from the CL buffer
        self.update_buffer()

    def paintGL(self):
        """Paint the scene.
        """
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glPointSize(10)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        #bind the color buffer
        self.glcolbuf.bind()
        gl.glColorPointer(4, gl.GL_FLOAT, 0, None)
        # bind the VBO
        self.glbuf.bind()
        # these vertices contain 2 simple precision coordinates
        gl.glVertexPointer(2, gl.GL_FLOAT, 0, None)

        # tell OpenGL that the VBO contains an array of vertices
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)

        # draw "count" points from the VBO
        gl.glDrawArrays(gl.GL_POINTS, 0, self.color_count)

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glDisable(gl.GL_BLEND)

        #Iterations per draw
        for i in range(1, 1000):
            self.update_buffer()

    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport.
        """
        # update the window size
        self.width, self.height = width, height
        # paint within the whole window
        gl.glViewport(0, 0, width, height)
        # set orthographic projection (2D only)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        # the window corner OpenGL coordinates are (-+1, -+1)
        gl.glOrtho(-1, 1, 1, -1, -1, 1)

if __name__ == '__main__':
    import sys
    import numpy as np

    # define a Qt window with an OpenGL widget inside it
    class TestWindow(QtGui.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()
            # Generate initial data
            self.data = np.zeros((302*302,2))
            self.color = np.zeros((302*302,4))
            for i in range (1, 302):
                for j in range (1, 302):
                    self.data[i*302+j] = ((i-151.0)/151.0,(j-151.0)/151.0)
            self.color[150+150*302] = (1, 0, 0, 1)
            self.color[10+20*302] = (1, 0, 0, 1)
            self.color[10+40*302] = (1, 0, 0, 1)
            self.color[20+35*302] = (1, 0, 0, 1)
            self.color[155+200*302] = (1, 0, 0, 1)
            self.color[135+45*302] = (1, 0, 0, 1)
            self.color[270+270*302] = (1, 0, 0, 1)
            self.data = np.array(self.data, dtype=np.float32)
            self.color = np.array(self.color, dtype=np.float32)

            # initialize the GL widget
            self.widget = GLPlotWidget()
            self.widget.set_data(self.data)
            self.widget.set_color(self.color)
            # put the window at the screen position (100, 100)
            self.setGeometry(100, 100, self.widget.width, self.widget.height)
            self.setCentralWidget(self.widget)
            self.show()

    # create the Qt App and window
    app = QtGui.QApplication(sys.argv)
    window = TestWindow()
    window.show()
    app.exec_()

