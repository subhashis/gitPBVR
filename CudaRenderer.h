#ifndef CUDARENDERER_H_
#define CUDARENDERER_H_

#include "defs.h"
#include "Singleton.h"


#include <iostream>

class CudaRenderer : public Singleton<CudaRenderer>{

  friend class Singleton<CudaRenderer>;

public:
  ~CudaRenderer(){};
  void launchKernel(unsigned int windowWidth, unsigned int windowHeight);
  void mapOpenGLMatrixToDevice();
  void mapNoiseDataToDevice();
  void initCuda();
  void initTransferFunction();

  void updateCamRotation(float rotX, float rotY);
  
protected:
  CudaRenderer();
  CudaRenderer(CudaRenderer& dontcopy){};

  void initRNG();
  void mapPixelBuffer();
  void unmapPixelBuffer();
  void computeFPS();

  bool m_cudaIsInit;

  struct cudaGraphicsResource* m_vboStatisticsCuda;
  struct cudaGraphicsResource* m_pboSubPixelsCuda;
  struct cudaGraphicsResource* m_pboSubPixelsDepthCuda;
  struct cudaGraphicsResource* m_pboPixelsCuda;
  
  curandState* m_rngStates;
  char* m_className;

  float m_rotX;
  float m_rotY;
  float m_z;

  int m_numParticles;
  int m_fpsCount;        // FPS count for averaging
  int m_fpsLimit;        // FPS limit for sampling
  int m_frameCount;
  //unsigned int m_timer;
  StopWatchInterface *m_timer;

  //tetra buffers:
  unsigned int m_numTetras;
  tetraFloat* m_tetras;
  tetraFloat* m_tetrasD;
  void initTetraBufferD();

  //voxel buffers:
  unsigned int m_numVoxels;
  voxelFloat* m_voxels;
  voxelFloat* m_voxelsD;
  void initVoxelBufferD();

  //subpixels:
  RGBA* m_subPixelsD;
  void initSubPixelsD();

  //subpixel depth buffer:
  float* m_subPixelsDepthD;
  void initSubPixelsDepthD();

  //tetrahedron volume ratios:
  float* m_volumeRatiosD;
  void initVolumeRatioBufferD();
  



};

#endif
