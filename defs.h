#ifndef DEFS_H_
#define DEFS_H_

#include "GL/glew.h"
#include "GL/freeglut.h"


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_timer.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda_gl.h>
#include <helper_functions.h>

#include "curand_kernel.h"

//debug defines:
#define DEBUG_UGRIDTETRAHEDRIZER 0
#define DEBUG_VBOCREATOR 0
#define DEBUG_CUDARENDERER 1
#define DEBUG_CU_FILE 0

#define REFRESH_DELAY	  10 //ms

#define TUMOR 0
#define DRAGON 0
#define HEART 0
#define ISABEL 1

#define SDATA 1
#define UDATA 0


#define SI3D 0
static const size_t SUBPIXELLEVEL = 4;
static const unsigned int ZDEPTH = 1;
static const unsigned int WINDOWWIDTH = 1200;
static const unsigned int WINDOWHEIGHT = 800;
static const unsigned int MAX_RNG_INIT_BLOCKS = 200;

static const unsigned int BLOCKSIZE_PROJECTION = 128;
//vector, vertex and tetras for VBO:
typedef float vector4[4];
typedef float vector3[3];
typedef unsigned int RGBA;
typedef float depthValue;

template<typename T> struct vertex{
  vector3 pos;
  T scalar;
  T var;
};


template<typename T> struct tetra{
  vertex<T> v[4];
};

template<typename T> struct voxel{
    vertex<T> v[8];
};

typedef vertex<float> vertexFloat;
typedef tetra<float> tetraFloat;
typedef voxel<float> voxelFloat;


struct transFuncOpacityPoint{
  float scalar;
  float opacity;
};

struct transFuncColorPoint{
  float scalar;
  RGBA color;
};

struct transFuncUncertainty{
  float var;
  RGBA color;
};

template<unsigned int nO, unsigned int nC> struct transFunc{
  //unsigned int nU;
  transFuncOpacityPoint opacityPoints[nO];
  transFuncColorPoint   colorPoints[nC];
  //transFuncUncertainty uncertaintyPoint[nU];
  
  unsigned int numOpPoints;
  unsigned int numColPoints;
  unsigned int numUncPoints;
};

#endif
