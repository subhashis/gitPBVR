
#include "CudaRenderer.h"
#include "VBOCreator.h"
#include "PerlinNoise.h"



#include <cstdlib> 
#include <ctime> 

#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/lu.hpp"
#include <boost/numeric/ublas/io.hpp>

//#include "windows.h"

extern "C" void mapMVPMatrixToDevice(float matrix[4][4]);
extern "C" void mapTFToDevice(transFunc<6,4> transferFunction);
extern "C" void mapPNToDevice(int rand_gradient[256]);
extern "C" void launchProjection(curandState* rngStatesD, unsigned int numParticles, float* volumeRatios, unsigned int numTetras, unsigned int windowWidth, unsigned int windowHeight, tetraFloat* tetrasD, RGBA* subPixelsD, depthValue* subPixelsDepthD);
extern "C" void launchProjectionSD(curandState* rngStatesD, unsigned int numParticles, float* volumeRatios, unsigned int numVoxels, unsigned int windowWidth, unsigned int windowHeight, voxelFloat* voxelsD, RGBA* subPixelsD, depthValue* subPixelsDepthD);
extern "C" void launchSuperimposingSpatial(unsigned int windowWidth, unsigned int windowHeight, unsigned int subPixelLevel, RGBA* pixels, RGBA* subPixels, depthValue* subPixelsDepthD);
extern "C" curandState* initRNGCuda(unsigned int numTetras, unsigned int seed);



CudaRenderer::CudaRenderer(){

  m_className = "CudaRenderer";
  m_cudaIsInit = false;

  m_fpsCount = 0;
  m_fpsLimit = 1;
  m_frameCount = 0;
  m_timer = 0;

  m_rotX = 0.0f;
  m_rotY = 0.0f;

  unsigned int million = 1000000;
#if TUMOR
  m_numParticles = 12 * million;
#endif

#if DRAGON
  m_numParticles = 130 * million;
#endif

#if HEART
  m_numParticles = 60 * million;
#endif

#if ISABEL
  //it was 100million for isabel DS
  m_numParticles = 50 * million;
#endif

  m_rngStates = NULL;

  sdkCreateTimer(&m_timer);
}

void CudaRenderer::initCuda(){
#if UDATA
  initTetraBufferD();
#endif
#if SDATA
  initVoxelBufferD();
#endif
  initSubPixelsD();
  initSubPixelsDepthD();
  initVolumeRatioBufferD();


  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&m_pboPixelsCuda, VBOCreator::getInstance()->getPBOPixels(), cudaGraphicsMapFlagsNone));
  

  initRNG();
  
  m_cudaIsInit = true;
}


void CudaRenderer::mapPixelBuffer(){
  checkCudaErrors(cudaGraphicsMapResources(1, &m_pboPixelsCuda));
}

void CudaRenderer::unmapPixelBuffer(){
  checkCudaErrors(cudaGraphicsUnmapResources(1, &m_pboPixelsCuda));
}

void CudaRenderer::initRNG(){
  unsigned int seed = 0;
  srand((unsigned)time(0)); 
  seed = rand(); 
#if SDATA
  unsigned int numVoxels = VBOCreator::getInstance()->getNumVoxels();
  //std::cout << "Init rng: num tetras = " << numTetras << std::endl;
  m_rngStates = initRNGCuda(numVoxels,seed);
#endif
#if UDATA
  unsigned int numTetras = VBOCreator::getInstance()->getNumTetras();
  //std::cout << "Init rng: num tetras = " << numTetras << std::endl;
  m_rngStates = initRNGCuda(numTetras,seed);
#endif
}



void CudaRenderer::computeFPS()
{
  m_frameCount++;
  m_fpsCount++;

  if (m_fpsCount == m_fpsLimit) {
    char fps[256];
    float ifps = 1.f / (sdkGetAverageTimerValue(&m_timer) / 1000.f);
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", ifps);  

    std::cout << fps << std::endl;
    m_fpsCount = 0; 
    m_fpsLimit = (int)MAX(ifps, 1.f);

#if FPS_QA
    if(ifps < MIN_FPS){
      float ratio = static_cast<float>(ifps) / static_cast<float>(MIN_FPS);
      m_numParticles *= ratio;
    }
    if(ifps > MAX_FPS){
      float ratio = static_cast<float>(ifps) / static_cast<float>(MAX_FPS);
      m_numParticles *= ratio;
    }
      

    std::cout << "Num particles = " << m_numParticles << std::endl;
#endif
    sdkResetTimer(&m_timer); 
  }
}



void CudaRenderer::launchKernel(unsigned int windowWidth, unsigned int windowHeight){

  //used deprecated functions:
  //tetraFloat* tetrasD = VBOCreator::getInstance()->mapTetrasToDevice();
  //subPixelConcrete* subPixelsD = VBOCreator::getInstance()->mapSubPixelsToDevice();
  if(!m_cudaIsInit)
    initCuda();

  sdkStartTimer(&m_timer);

  mapOpenGLMatrixToDevice();
  //mapNoiseDataToDevice();
  initTransferFunction();

  //Make some noise
  //PerlinNoise pn;


  mapPixelBuffer();

  size_t numBytes;
  RGBA* pixelsD;

  unsigned int numSubPixels = WINDOWWIDTH*WINDOWHEIGHT*SUBPIXELLEVEL*SUBPIXELLEVEL*ZDEPTH;
  checkCudaErrors(cudaMemset(m_subPixelsD, 0xff, numSubPixels * sizeof(RGBA)));
  checkCudaErrors(cudaMemset(m_subPixelsDepthD, 0x00, numSubPixels * sizeof(RGBA)));

  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&pixelsD, &numBytes, m_pboPixelsCuda));
  checkCudaErrors(cudaMemset(pixelsD, 0xff, windowWidth*windowHeight*4));


  //std::cout << "Error before kernels: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
#if UDATA
  launchProjection(m_rngStates,m_numParticles,m_volumeRatiosD,m_numTetras, windowWidth, windowHeight,m_tetrasD,m_subPixelsD, m_subPixelsDepthD);
#endif
#if SDATA
  launchProjectionSD(m_rngStates,m_numParticles,m_volumeRatiosD,m_numVoxels, windowWidth, windowHeight,m_voxelsD,m_subPixelsD, m_subPixelsDepthD);
#endif
  //std::cout << "Projection error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
  cudaDeviceSynchronize();
  launchSuperimposingSpatial(windowWidth,windowHeight,SUBPIXELLEVEL,pixelsD,m_subPixelsD, m_subPixelsDepthD);
  cudaDeviceSynchronize();

  unmapPixelBuffer();

  sdkStopTimer(&m_timer);
  computeFPS();
  
  
}

void CudaRenderer::updateCamRotation(float rotX, float rotY){
  m_rotX = rotX;
  m_rotY = rotY;
}

void CudaRenderer::mapOpenGLMatrixToDevice(){

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

#if TUMOR
  glTranslatef(0,0,60);
  //gluLookAt(0,0,0,0,0,-1,0,1,0);
#endif

#if DRAGON
  glTranslatef(0,0,170);
  //gluLookAt(0,0,0,0,0,-1,0,1,0);
#endif

#if HEART
  glTranslatef(0,0,300);
#endif

#if ISABEL
  //for isabel DS
  //glTranslatef(200,-250,800);
  //gluLookAt(0,0,0,0,0,1,0,1,0);
  //darkmatter DS

  //glTranslatef(-50,-25,100);
  glTranslatef(-25,30,150);
  //gluLookAt(0,0,0,0,0,1,0,1,0);
#endif
  glRotatef(m_rotX,1.0,0.0,0.0);
  glRotatef(m_rotY,0.0,1.0,0.0);


  float zNear = 1.0f;
  float zFar = 1000.0f;


  float mv[16];
  glGetFloatv(GL_MODELVIEW_MATRIX,mv);
  
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective (45, (float)WINDOWWIDTH/(float)WINDOWHEIGHT, zNear, zFar);
  float pr[16];
  glGetFloatv(GL_PROJECTION_MATRIX,pr);


  /*

  boost::numeric::ublas::matrix<float> mvBoost(4,4);
  boost::numeric::ublas::matrix<float> prBoost(4,4);
  boost::numeric::ublas::matrix<float> mvpBoost(4,4);
  
  for(int i = 0 ; i < 4 ; i++){
    for(int j = 0 ; j < 4 ;j++){
      mvBoost(i,j) = mv[4*i+j];
      prBoost(i,j) = pr[4*i+j];
    }
  }
  mvBoost = trans(mvBoost);
  prBoost = trans(prBoost);
  mvpBoost = prod(prBoost,mvBoost);

  float mvpArray[4][4];
  for(int i = 0 ; i < 4 ; i++){
    for(int j = 0 ; j < 4 ;j++){
      mvpArray[i][j] = mvpBoost(i,j);
    }
  }*/

     // Get The Current PROJECTION Matrix From OpenGL
   float proj[16];
   float modl[16];
   float clip[16];
   glGetFloatv( GL_PROJECTION_MATRIX, proj );
   // Get The Current MODELVIEW Matrix From OpenGL
   glGetFloatv( GL_MODELVIEW_MATRIX, modl );
   // Combine The Two Matrices (Multiply Projection By Modelview)
   clip[ 0] = modl[ 0] * proj[ 0] + modl[ 1] * proj[ 4] + modl[ 2] * proj[ 8] +    modl[ 3] * proj[12];
   clip[ 1] = modl[ 0] * proj[ 1] + modl[ 1] * proj[ 5] + modl[ 2] * proj[ 9] +    modl[ 3] * proj[13];
   clip[ 2] = modl[ 0] * proj[ 2] + modl[ 1] * proj[ 6] + modl[ 2] * proj[10] +    modl[ 3] * proj[14];
   clip[ 3] = modl[ 0] * proj[ 3] + modl[ 1] * proj[ 7] + modl[ 2] * proj[11] +    modl[ 3] * proj[15];
   clip[ 4] = modl[ 4] * proj[ 0] + modl[ 5] * proj[ 4] + modl[ 6] * proj[ 8]    + modl[ 7] * proj[12];
   clip[ 5] = modl[ 4] * proj[ 1] + modl[ 5] * proj[ 5] + modl[ 6] * proj[ 9] +    modl[ 7] * proj[13];
   clip[ 6] = modl[ 4] * proj[ 2] + modl[ 5] * proj[ 6] + modl[ 6] * proj[10] +    modl[ 7] * proj[14];
   clip[ 7] = modl[ 4] * proj[ 3] + modl[ 5] * proj[ 7] + modl[ 6] * proj[11] +    modl[ 7] * proj[15];
   clip[ 8] = modl[ 8] * proj[ 0] + modl[ 9] * proj[ 4] + modl[10] * proj[ 8]    + modl[11] * proj[12];
   clip[ 9] = modl[ 8] * proj[ 1] + modl[ 9] * proj[ 5] + modl[10] * proj[ 9] +    modl[11] * proj[13];
   clip[10] = modl[ 8] * proj[ 2] + modl[ 9] * proj[ 6] + modl[10] * proj[10] +    modl[11] * proj[14];
   clip[11] = modl[ 8] * proj[ 3] + modl[ 9] * proj[ 7] + modl[10] * proj[11] +    modl[11] * proj[15];
   clip[12] = modl[12] * proj[ 0] + modl[13] * proj[ 4] + modl[14] * proj[ 8]    + modl[15] * proj[12];
   clip[13] = modl[12] * proj[ 1] + modl[13] * proj[ 5] + modl[14] * proj[ 9] +    modl[15] * proj[13];
   clip[14] = modl[12] * proj[ 2] + modl[13] * proj[ 6] + modl[14] * proj[10] +    modl[15] * proj[14];
   clip[15] = modl[12] * proj[ 3] + modl[13] * proj[ 7] + modl[14] * proj[11] +    modl[15] * proj[15];

   float mvp[4][4];
   for(int i = 0 ; i < 4 ; i++){
     for(int j = 0 ; j < 4 ; j++){
       mvp[i][j] = clip[4*j+i];
     }
   }
   mapMVPMatrixToDevice(mvp);

}

void CudaRenderer::mapNoiseDataToDevice(){

    int noise_arr[256] = {
        151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
        8,99,37,240,21,10,23,190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
        35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,
        134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
        55,46,245,40,244,102,143,54, 65,25,63,161,1,216,80,73,209,76,132,187,208, 89,
        18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,
        250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
        189,28,42,223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167,
        43,172,9,129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,
        97,228,251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,
        107,49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
        138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180 };

    mapPNToDevice(noise_arr);

}

void CudaRenderer::initTransferFunction(){

#if TUMOR
  transFunc<6,4> tf;
  tf.numOpPoints = 6;
  tf.numColPoints = 4;
  transFuncOpacityPoint op1,op2,op3,op4,op5,op6;
  op1.scalar = -1.0f;
  op1.opacity = 0.0f;

  op2.scalar = 0.7f;
  op2.opacity = 0.0f;

  op3.scalar = 0.75f;
  op3.opacity = 0.5f;

  op4.scalar = 1.0f;
  op4.opacity = 0.7f;
  
  op5.scalar = 1.15f;
  op5.opacity = 0.7f;

  op6.scalar = 2.0f;
  op6.opacity = 0.7f;


  tf.opacityPoints[0] = op1;
  tf.opacityPoints[1] = op2;
  tf.opacityPoints[2] = op3;
  tf.opacityPoints[3] = op4;
  tf.opacityPoints[4] = op5;
  tf.opacityPoints[5] = op6;

  transFuncColorPoint col1,col2,col3,col4;
  col1.scalar = 0.69f;
  col1.color = 0x00ff0000;

  //col2.scalar = 0.7f;
  //col2.color = 0xffc04c3b;
  col2.scalar = 0.8f;
  col2.color = 0x00ff0000;

  //col3.scalar = 1.0f;
  //col3.color = 0xff2604b4;
  col3.scalar = 1.0f;
  col3.color = 0x000000ff;

  col4.scalar = 1.5f;
  col4.color = 0x000000ff;;

  tf.colorPoints[0] = col1;
  tf.colorPoints[1] = col2;
  tf.colorPoints[2] = col3;
  tf.colorPoints[3] = col4;

  mapTFToDevice(tf);

#endif

#if DRAGON
  transFunc<6,4> tf;
  tf.numOpPoints = 6;
  tf.numColPoints = 4;
  transFuncOpacityPoint op1,op2,op3,op4,op5,op6;
  /*op1.scalar = -1.0f;
  op1.opacity = 0.0f;

  op2.scalar = 0.0f;
  op2.opacity = 0.0f;

  op3.scalar = 0.0015f;
  op3.opacity = 0.0f;

  op4.scalar = 0.005f;
  op4.opacity = 0.7f;

  op5.scalar = 0.02116f;
  op5.opacity = 1.0f;

  op6.scalar = 1.0f;
  op6.opacity = 1.0f;*/

  op1.scalar = -1.0f;
  op1.opacity = 0.0f;

  op2.scalar = 0.0f;
  op2.opacity = 0.0f;

  op3.scalar = 0.0015f;
  op3.opacity = 0.0f;

  op4.scalar = 0.005f;
  op4.opacity = 1.0f;

  op5.scalar = 0.02116f;
  op5.opacity = 1.0f;

  op6.scalar = 1.0f;
  op6.opacity = 1.0f;

  tf.opacityPoints[0] = op1;
  tf.opacityPoints[1] = op2;
  tf.opacityPoints[2] = op3;
  tf.opacityPoints[3] = op4;
  tf.opacityPoints[4] = op5;
  tf.opacityPoints[5] = op6;

  transFuncColorPoint col1,col2,col3,col4;
  col1.scalar = -2.0f;
  col1.color = 0x00ff0000;
  //col1.color = 0x3b4cc000;

  col2.scalar = -0.00063f;
  col2.color = 0x00ff0000;
  //col2.color = 0x3b4cc000;

  col3.scalar = 0.0211f;
  col3.color = 0x0000ff00;
  //col3.color = 0xb4042600;

  col4.scalar = 2.0f;
  col4.color = 0x0000ff00;
  //col4.color = 0xb4042600;

  tf.colorPoints[0] = col1;
  tf.colorPoints[1] = col2;
  tf.colorPoints[2] = col3;
  tf.colorPoints[3] = col4;

  mapTFToDevice(tf);

#endif

#if HEART
    transFunc<6,4> tf;
  tf.numOpPoints = 6;
  tf.numColPoints = 4;
  transFuncOpacityPoint op1,op2,op3,op4,op5,op6;
  op1.scalar = 0.0f;
  op1.opacity = 0.0f;

  op2.scalar = 180.0f;
  op2.opacity = 0.3f;

  op3.scalar = 320.0f;
  op3.opacity = 0.0f;

  op4.scalar = 481.0f;
  op4.opacity = 1.0f;

  op5.scalar = 482.0f;
  op5.opacity = 1.0f;

  op6.scalar = 483.0f;
  op6.opacity = 1.0f;


  tf.opacityPoints[0] = op1;
  tf.opacityPoints[1] = op2;
  tf.opacityPoints[2] = op3;
  tf.opacityPoints[3] = op4;
  tf.opacityPoints[4] = op5;
  tf.opacityPoints[5] = op6;

  transFuncColorPoint col1,col2,col3,col4;
  col1.scalar = 180.0f;
  col1.color = 0x00ff0000;
  //col1.color = 0x3b4cc000;

  col2.scalar = 330.0f;
  col2.color = 0x0000ff00;
  //col2.color = 0x3b4cc000;

  col3.scalar = 480.0f;
  col3.color = 0x000000ff;
  //col3.color = 0xb4042600;

  col4.scalar = 481.0f;
  col4.color = 0x000000ff;
  //col4.color = 0xb4042600;

  tf.colorPoints[0] = col1;
  tf.colorPoints[1] = col2;
  tf.colorPoints[2] = col3;
  tf.colorPoints[3] = col4;

  mapTFToDevice(tf);
#endif

#if ISABEL
   /* transFunc<6,4> tf;
  tf.numOpPoints = 6;
  tf.numColPoints = 4;
  transFuncOpacityPoint op1,op2,op3,op4,op5,op6;
  op1.scalar = 0.0f;
  op1.opacity = 0.3f;

  op2.scalar = 2000.0f;
  op2.opacity = 0.0f;

  op3.scalar = -2000.0f;
  op3.opacity = 1.0f;

  op4.scalar = 2315.0f;
  op4.opacity = 0.0f;

  op5.scalar = -3777.0f;
  op5.opacity = 1.0f;

  op6.scalar = 1500.0f;
  op6.opacity = 0.5f;


  tf.opacityPoints[0] = op1;
  tf.opacityPoints[1] = op2;
  tf.opacityPoints[2] = op3;
  tf.opacityPoints[3] = op4;
  tf.opacityPoints[4] = op5;
  tf.opacityPoints[5] = op6;

  transFuncColorPoint col1,col2,col3,col4;
  col1.scalar = -3777.0f;
  col1.color = 0x00ff0000;
  //col1.color = 0x000000;
  //col1.color = 0xff3300ff;

  col2.scalar = -2000.0f;
  col2.color = 0x0000ff00;
  //col2.color = 0x191919;
  //col2.color = 0xff704d0ff;

  col3.scalar = 0.0f;
  col3.color = 0x000000ff;
  //col3.color = 0x4d4d4d;
  //col3.color = 0xffad99ff;

  col4.scalar = 2315.570f;
  col4.color = 0xffffffff;
  //col4.color = 0x999999;
  //col4.color = 0x9999ff00;

  tf.colorPoints[0] = col1;
  tf.colorPoints[1] = col2;
  tf.colorPoints[2] = col3;
  tf.colorPoints[3] = col4;

  mapTFToDevice(tf);*/
  transFunc<6,4> tf;
    tf.numOpPoints = 6;
    tf.numColPoints = 4;
    transFuncOpacityPoint op1,op2,op3,op4,op5,op6;
    op1.scalar = 0.0f;
    op1.opacity = 0.2f;

    op2.scalar = 100.0f;
    op2.opacity = 0.5f;

    op3.scalar = 4000.0f;
    op3.opacity = 0.7f;

    op4.scalar = 8000.0f;
    op4.opacity = 1.0f;

    op5.scalar = 12000.0f;
    op5.opacity = 1.0f;

    op6.scalar = 14000.0f;
    op6.opacity = 1.0f;


    tf.opacityPoints[0] = op1;
    tf.opacityPoints[1] = op2;
    tf.opacityPoints[2] = op3;
    tf.opacityPoints[3] = op4;
    tf.opacityPoints[4] = op5;
    tf.opacityPoints[5] = op6;

transFuncColorPoint col1,col2,col3,col4;
col1.scalar = 0.0f;
col1.color = 0x081a3600;
//col1.color = 0x000000;
//col1.color = 0xff3300ff;

col2.scalar = 2000.0f;
col2.color = 0x0000ff00;
//col2.color = 0x191919;
//col2.color = 0xff704d0ff;

col3.scalar =12000.0f;
col3.color = 0x0000ffff;
//col3.color = 0x4d4d4d;
//col3.color = 0xffad99ff;

col4.scalar = 14000.0f;
col4.color = 0x0000ffff;
//col4.color = 0x999999;
//col4.color = 0x9999ff00;

tf.colorPoints[0] = col1;
tf.colorPoints[1] = col2;
tf.colorPoints[2] = col3;
tf.colorPoints[3] = col4;

mapTFToDevice(tf);
#endif

}


void CudaRenderer::initTetraBufferD(){
  m_tetras = VBOCreator::getInstance()->getTetras();
  m_numTetras = VBOCreator::getInstance()->getNumTetras();

  checkCudaErrors(cudaMalloc((void **)&m_tetrasD, m_numTetras * sizeof(tetraFloat)));  
  checkCudaErrors(cudaMemcpy(m_tetrasD,m_tetras,m_numTetras*sizeof(tetraFloat),cudaMemcpyHostToDevice));
  std::cout << "Allocating tetras: " << m_numTetras*sizeof(tetraFloat) /(double)(1024*1024) << " mb" << std::endl;
}

void CudaRenderer::initVoxelBufferD(){
  m_voxels = VBOCreator::getInstance()->getVoxels();
  m_numVoxels = VBOCreator::getInstance()->getNumVoxels();

  std::cout << "inside initVoxelBufferD, m_numVoxels=" << m_numVoxels << endl;

  checkCudaErrors(cudaMalloc((void **)&m_voxelsD, m_numVoxels * sizeof(voxelFloat)));
  checkCudaErrors(cudaMemcpy(m_voxelsD,m_voxels,m_numVoxels*sizeof(voxelFloat),cudaMemcpyHostToDevice));
  std::cout << "Allocating voxels: " << m_numVoxels*sizeof(voxelFloat) /(double)(1024*1024) << " mb" << std::endl;
}

void CudaRenderer::initVolumeRatioBufferD(){
#if UDATA
  float* volumes = VBOCreator::getInstance()->getTetraVolumes();
  float totalVolume = UGridTetrahedrizer::getInstance()->getGridVolume();
  float* volumeRatiosH = new float[m_numTetras];

  for(int i = 0 ; i < m_numTetras ; i++){
    volumeRatiosH[i] = volumes[i]/totalVolume;
  }

  checkCudaErrors(cudaMalloc((void**)&m_volumeRatiosD, m_numTetras*sizeof(float)));
  std::cout << "Allocating ratios: " << m_numTetras*sizeof(float) /(double)(1024*1024) << " mb" << std::endl;
  checkCudaErrors(cudaMemcpy(m_volumeRatiosD,volumeRatiosH,m_numTetras*sizeof(float),cudaMemcpyHostToDevice));
#endif

#if SDATA
  float* volumes = VBOCreator::getInstance()->getVoxelVolumes();
  float totalVolume = SGrid::getInstance()->getGridVolume();
  float* volumeRatiosH = new float[m_numVoxels];

  for(int i = 0 ; i < m_numVoxels ; i++){
    volumeRatiosH[i] = volumes[i]/totalVolume;
  }

  checkCudaErrors(cudaMalloc((void**)&m_volumeRatiosD, m_numVoxels*sizeof(float)));
  std::cout << "Allocating ratios: " << m_numVoxels*sizeof(float) /(double)(1024*1024) << " mb" << std::endl;
  checkCudaErrors(cudaMemcpy(m_volumeRatiosD,volumeRatiosH,m_numVoxels*sizeof(float),cudaMemcpyHostToDevice));
#endif


}

void CudaRenderer::initSubPixelsD(){
  unsigned int numSubPixels = WINDOWWIDTH*WINDOWHEIGHT*SUBPIXELLEVEL*SUBPIXELLEVEL*ZDEPTH;
  std::cout << "Allocating subpixels: " << numSubPixels * sizeof(RGBA) /(double)(1024*1024) << " mb" << std::endl;
  checkCudaErrors(cudaMalloc((void **)&m_subPixelsD, numSubPixels * sizeof(RGBA)));  
  checkCudaErrors(cudaMemset(m_subPixelsD, 0xff, numSubPixels * sizeof(RGBA)));
}

void CudaRenderer::initSubPixelsDepthD(){
  unsigned int numSubPixels = WINDOWWIDTH*WINDOWHEIGHT*SUBPIXELLEVEL*SUBPIXELLEVEL*ZDEPTH;
  checkCudaErrors(cudaMalloc((void **)&m_subPixelsDepthD, numSubPixels * sizeof(float)));  
  std::cout << "Allocating depth: " << numSubPixels * sizeof(float) /(double)(1024*1024) << " mb" << std::endl;
  checkCudaErrors(cudaMemset(m_subPixelsDepthD, 0x00, numSubPixels * sizeof(float)));
}
