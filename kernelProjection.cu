#include <stdio.h>
#include "defs.h"
#include "PerlinNoise.h"

__constant__ float mvpMatrixDevice[4][4];
__constant__ transFunc<6,4> tfDevice;
__constant__ int p[256];




extern "C" void mapMVPMatrixToDevice(float matrix[4][4]){
  cudaMemcpyToSymbol(mvpMatrixDevice, matrix, sizeof(float)*16);
}

extern "C" void mapTFToDevice(transFunc<6,4> transferFunction){
  cudaMemcpyToSymbol(tfDevice,&transferFunction,sizeof(transFunc<6,4>));
}

extern "C" void mapPNToDevice(int rand_gradient[256]){
    cudaMemcpyToSymbol(p, rand_gradient, sizeof(int)*256);
}

__device__ float opacityFromTF(float scalar){
  for(int i = 0 ; i < tfDevice.numOpPoints - 1 ; i++){
  float lowerScalar = tfDevice.opacityPoints[i].scalar;
  float upperScalar = tfDevice.opacityPoints[i+1].scalar;
    if(scalar >  lowerScalar && scalar < upperScalar){
        float delta = (scalar - lowerScalar) / (upperScalar - lowerScalar);
        return (tfDevice.opacityPoints[i].opacity + (tfDevice.opacityPoints[i+1].opacity - tfDevice.opacityPoints[i].opacity) * delta);
    }
  }
  return 0.0f;
  /***Hardcoded TF for dragon only*
  if(scalar >= -1.0f && scalar <= 0.0015f)
  	return 0.0f;
  else if(scalar >= 0.005f && scalar <= 1.0f)
  	return 1.0f;
  else if(scalar > 0.0015f && scalar < 0.005f)
  	return((scalar - 0.0015f)/0.0035f);
  else 
  	return 0.0f;
  	*/
}

__device__ RGBA colorFromTF(float scalar){
  RGBA ret = 0x00000000;
  unsigned char color[4] = {0,0,0,0};
  //unsigned char uColor[4] = {255, 255, 0, 0};
  //first, opacity:
  color[3] = static_cast<unsigned char>(opacityFromTF(scalar) * 255.0f);
  //next, color:
  for(int i = 0 ; i < tfDevice.numColPoints - 1 ; i++){
    float lowerScalar = tfDevice.colorPoints[i].scalar;
    float upperScalar = tfDevice.colorPoints[i+1].scalar;
    if(scalar > lowerScalar && scalar < upperScalar){
      
      float delta = (scalar - lowerScalar) / (upperScalar - lowerScalar);
      unsigned char tfColCharLow[4];
      //rgbaUintToByteArray(&tfDevice.colorPoints[i].color,tfColCharLow);
      memcpy(tfColCharLow,&tfDevice.colorPoints[i].color,4);
      unsigned char tfColCharHigh[4];
      //rgbaUintToByteArray(&tfDevice.colorPoints[i+1].color,tfColCharHigh);
      memcpy(tfColCharHigh,&tfDevice.colorPoints[i+1].color,4);
      
      color[0] = tfColCharLow[0] + (char)((float)(tfColCharHigh[0] - tfColCharLow[0])*delta);
      color[1] = tfColCharLow[1] + (char)((float)(tfColCharHigh[1] - tfColCharLow[1])*delta);
      color[2] = tfColCharLow[2] + (char)((float)(tfColCharHigh[2] - tfColCharLow[2])*delta);
      
    }
  }
  
  //printf("red=%i,green=%i,blue=%ialpha=%i\n",color[0],color[1],color[2],color[3]);

  //charArrayToRGBA(color,&ret);
  memcpy(&ret,color,4);

  return ret;
}



__device__ double fade(double t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

__device__ double lerp(double t, double a, double b) {
    return a + t * (b - a);
}

__device__ double grad(int hash, double x, double y, double z) {
    int h = hash & 15;
    // Convert lower 4 bits of hash inot 12 gradient directions
    double u = h < 8 ? x : y,
           v = h < 4 ? y : h == 12 || h == 14 ? x : z;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}


__device__ float noise(float x, float y, float z){
    // Find the unit cube that contains the point
    int X = (int) floor(x) & 255;
    int Y = (int) floor(y) & 255;
    int Z = (int) floor(z) & 255;

    // Find relative x, y,z of point in cube
    x -= floor(x);
    y -= floor(y);
    z -= floor(z);

    // Compute fade curves for each of x, y, z
    double u = fade(x);
    double v = fade(y);
    double w = fade(z);

    // Hash coordinates of the 8 cube corners
    int A = p[X] + Y;
    int AA = p[A] + Z;
    int AB = p[A + 1] + Z;
    int B = p[X + 1] + Y;
    int BA = p[B] + Z;
    int BB = p[B + 1] + Z;

    // Add blended results from 8 corners of cube
    double res = lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x-1, y, z)), lerp(u, grad(p[AB], x, y-1, z), grad(p[BB], x-1, y-1, z))),	lerp(v, lerp(u, grad(p[AA+1], x, y, z-1), grad(p[BA+1], x-1, y, z-1)), lerp(u, grad(p[AB+1], x, y-1, z-1),	grad(p[BB+1], x-1, y-1, z-1))));
    return (res + 1.0)/2.0;
}

__device__ RGBA noiseOutput(float pos0, float pos1, float pos2, float scalar){

    RGBA ret = 0x00000000;
    unsigned char color[4] = {0,0,0,0};
    double n = noise(10*pos0,10*pos1, 10*pos2);

    n = 60*noise(pos0, pos1, pos2);
    n = n - floor(n);

    int red = floor(255 * n);
    int green = floor(255 * n);
    int blue = floor(255 * n);

    color[0] = red;
    color[1] = green;
    color[2] = blue;
    color[3] = static_cast<unsigned char>(opacityFromTF(scalar) * 255.0f);//proper get the alpha value

    //printf("red=%i,green=%i,blue=%ialpha=%i\n",color[0],color[1],color[2],color[3]);



    memcpy(&ret,color,4);

    return ret;

}

  __device__ void projectParticle(float pos0, float pos1, float pos2, float scalar, RGBA* subPixelsD, depthValue* subPixelsDepthD,unsigned int windowWidth, unsigned int windowHeight, int flag){
  float clipped[4];

  // vertex * MVPmatrix:
  clipped[0] = pos0 * mvpMatrixDevice[0][0] + pos1 * mvpMatrixDevice[0][1] + pos2 * mvpMatrixDevice[0][2] + 1.0f * mvpMatrixDevice[0][3];
  clipped[1] = pos0 * mvpMatrixDevice[1][0] + pos1 * mvpMatrixDevice[1][1] + pos2 * mvpMatrixDevice[1][2] + 1.0f * mvpMatrixDevice[1][3];
  clipped[2] = pos0 * mvpMatrixDevice[2][0] + pos1 * mvpMatrixDevice[2][1] + pos2 * mvpMatrixDevice[2][2] + 1.0f * mvpMatrixDevice[2][3];
  clipped[3] = pos0 * mvpMatrixDevice[3][0] + pos1 * mvpMatrixDevice[3][1] + pos2 * mvpMatrixDevice[3][2] + 1.0f * mvpMatrixDevice[3][3];

  
  // normalized device coordinates:
  if(clipped[3] < 0.0001f && clipped[3] > -0.0001f)
    return;
  float reciW = 1.0f/clipped[3];
  clipped[0] *= reciW;
  clipped[1] *= reciW;
  clipped[2] *= reciW;
  clipped[3] = 1.0f;

  /*float xPos = (clipped[0] * 0.5 + 0.5) * static_cast<float>(windowWidth * SUBPIXELLEVEL);
  float yPos = (clipped[1] * 0.5 + 0.5) * static_cast<float>(windowHeight * SUBPIXELLEVEL);*/
  float zPos = (1.0 + clipped[2]) * 0.5;

  //get screen coordinates for buffer access:
  int xPixel = static_cast<int>((clipped[0] * 0.5 + 0.5) * static_cast<float>(windowWidth * SUBPIXELLEVEL));
  int yPixel = static_cast<int>((clipped[1] * 0.5 + 0.5) * static_cast<float>(windowHeight * SUBPIXELLEVEL));

  //compare x and y here to return if out of bounds:
  if(xPixel < 0 || xPixel >= windowWidth*SUBPIXELLEVEL)
    return;
  if(yPixel <0 ||yPixel >= windowHeight*SUBPIXELLEVEL)
    return;
  
  unsigned int arrayPos = yPixel*windowWidth*SUBPIXELLEVEL + xPixel;
  //depth test:
  if(zPos > subPixelsDepthD[arrayPos] && subPixelsDepthD[arrayPos] > 0.001f)
    return;




  /*if(flag == 0)
      subPixelsD[arrayPos] = colorFromTF(scalar);
  if(flag == 1)
      subPixelsD[arrayPos] = noiseOutput(pos0, pos1, pos2, scalar);*/
  subPixelsD[arrayPos] = colorFromTF(scalar);
  subPixelsDepthD[arrayPos] = zPos;

}


__global__ void kernelProjection(curandState* rngStatesD, unsigned int numParticles, float* volumeRatios, unsigned int numTetras, unsigned int subpixelLevel, unsigned int windowWidth, unsigned int windowHeight, tetraFloat* tetrasD, RGBA* subPixelsD, float* subPixelsDepthD){
  
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  if(x > numTetras)
    return;

  curandState localRNGState = rngStatesD[x];


  //create some noise
  //PerlinNoise pn;

  //get the model coordinates of the 4 points of the current tetra:
  __shared__ vertexFloat v0[BLOCKSIZE_PROJECTION];
  __shared__ vertexFloat v1[BLOCKSIZE_PROJECTION];
  __shared__ vertexFloat v2[BLOCKSIZE_PROJECTION];
  __shared__ vertexFloat v3[BLOCKSIZE_PROJECTION];
  v0[threadIdx.x] = tetrasD[x].v[0];
  v1[threadIdx.x] = tetrasD[x].v[1];
  v2[threadIdx.x] = tetrasD[x].v[2];
  v3[threadIdx.x] = tetrasD[x].v[3];

  //calculating scalar gradient
  /*float maxScalar=1000.0f, minScalar=1000.0f;
  float maxPos[3],minPos[3];

  if(v0[threadIdx.x].scalar >= maxScalar)
  {
      maxScalar = v0[threadIdx.x].scalar;
      maxPos[0] = v0[threadIdx.x].pos[0];
      maxPos[1] = v0[threadIdx.x].pos[1];
      maxPos[2] = v0[threadIdx.x].pos[2];
  }
  if(v1[threadIdx.x].scalar >= maxScalar)
  {
      maxScalar = v1[threadIdx.x].scalar;
      maxPos[0] = v1[threadIdx.x].pos[0];
      maxPos[1] = v1[threadIdx.x].pos[1];
      maxPos[2] = v1[threadIdx.x].pos[2];
  }
  if(v2[threadIdx.x].scalar >= maxScalar)
  {
      maxScalar = v2[threadIdx.x].scalar;
      maxPos[0] = v2[threadIdx.x].pos[0];
      maxPos[1] = v2[threadIdx.x].pos[1];
      maxPos[2] = v2[threadIdx.x].pos[2];
  }
  if(v3[threadIdx.x].scalar >= maxScalar)
  {
      maxScalar = v3[threadIdx.x].scalar;
      maxPos[0] = v3[threadIdx.x].pos[0];
      maxPos[1] = v3[threadIdx.x].pos[1];
      maxPos[2] = v3[threadIdx.x].pos[2];
  }

  if(v0[threadIdx.x].scalar <= minScalar)
  {
      minScalar = v0[threadIdx.x].scalar;
      minPos[0] = v0[threadIdx.x].pos[0];
      minPos[1] = v0[threadIdx.x].pos[1];
      minPos[2] = v0[threadIdx.x].pos[2];
  }
  if(v1[threadIdx.x].scalar <= minScalar)
  {
      minScalar = v1[threadIdx.x].scalar;
      minPos[0] = v1[threadIdx.x].pos[0];
      minPos[1] = v1[threadIdx.x].pos[1];
      minPos[2] = v1[threadIdx.x].pos[2];
  }
  if(v2[threadIdx.x].scalar <= minScalar)
  {
      minScalar = v2[threadIdx.x].scalar;
      minPos[0] = v2[threadIdx.x].pos[0];
      minPos[1] = v2[threadIdx.x].pos[1];
      minPos[2] = v2[threadIdx.x].pos[2];
  }
  if(v3[threadIdx.x].scalar <= minScalar)
  {
      minScalar = v3[threadIdx.x].scalar;
      minPos[0] = v3[threadIdx.x].pos[0];
      minPos[1] = v3[threadIdx.x].pos[1];
      minPos[2] = v3[threadIdx.x].pos[2];
  }

  */
  
  //check out the uncertainty value for each grid point.
 // printf("CHECK: variance value for the grid points= %f, %f, %f, %f \n", v0[threadIdx.x].var, v1[threadIdx.x].var, v2[threadIdx.x].var, v3[threadIdx.x].var);

  if((opacityFromTF(v0[threadIdx.x].scalar) + opacityFromTF(v1[threadIdx.x].scalar) + opacityFromTF(v2[threadIdx.x].scalar) + opacityFromTF(v3[threadIdx.x].scalar))/4.0f < 0.05f)
    return;

  int cellParticles = static_cast<int>(volumeRatios[x] * static_cast<float>(numParticles));
  
  //generate the given amount of particles:
  //for(int i = 0 ; i <= cellParticles ; i++){
  for(int i = 0 ; i <= cellParticles ; i++){

    //generate 3 parameters randomly in the range 0..1
    float par0 = curand_uniform(&localRNGState);
    float par1 = curand_uniform(&localRNGState);
    float par2 = curand_uniform(&localRNGState);
    //float par3 = curand_uniform(&localRNGState);

    int flag = 0;
    if(i%2 == 0) flag =0; // real value
    else flag = 1; //noise value

    //if the sum of the 3 parameters is > 1, calculate 1 - parameter for each
    if(par0 + par1 + par2 > 1.0f){
      par0 = 1.0f - par0;
      par1 = 1.0f - par1;
      par2 = 1.0f - par2;
    }

    //fourth parameter is calculated here:
    float par3 = 1.0f - par0 - par1 - par2;
    

    //particle opacity equals emission probability:
    float particleScalar = v0[threadIdx.x].scalar * par0 + v1[threadIdx.x].scalar * par1 + v2[threadIdx.x].scalar * par2 + v3[threadIdx.x].scalar * par3;
    
   
    //if opacity of particle >= random variable, project it. otherwise, reject it.
    if(opacityFromTF(particleScalar) >= curand_uniform(&localRNGState)){

      //finalize particle and project it:
      //position is calculated on the fly, minimize mem access
      projectParticle(v0[threadIdx.x].pos[0] * par0 + v1[threadIdx.x].pos[0] * par1 + v2[threadIdx.x].pos[0] * par2 + v3[threadIdx.x].pos[0] * par3,
                      v0[threadIdx.x].pos[1] * par0 + v1[threadIdx.x].pos[1] * par1 + v2[threadIdx.x].pos[1] * par2 + v3[threadIdx.x].pos[1] * par3,
                      v0[threadIdx.x].pos[2] * par0 + v1[threadIdx.x].pos[2] * par1 + v2[threadIdx.x].pos[2] * par2 + v3[threadIdx.x].pos[2] * par3,
                      particleScalar,subPixelsD,subPixelsDepthD,windowWidth,windowHeight, flag);

      /*float particleX = v0[threadIdx.x].pos[0] * par0 + v1[threadIdx.x].pos[0] * par1 + v2[threadIdx.x].pos[0] * par2 + v3[threadIdx.x].pos[0] * par3;
      float particleY = v0[threadIdx.x].pos[1] * par0 + v1[threadIdx.x].pos[1] * par1 + v2[threadIdx.x].pos[1] * par2 + v3[threadIdx.x].pos[1] * par3;
      float particleZ = v0[threadIdx.x].pos[2] * par0 + v1[threadIdx.x].pos[2] * par1 + v2[threadIdx.x].pos[2] * par2 + v3[threadIdx.x].pos[2] * par3;

      if(particleScalar >= maxScalar)
      {
          maxScalar = particleScalar;
          maxPos[0] = particleX;
          maxPos[1] = particleY;
          maxPos[2] = particleZ;
      }
      if(particleScalar <= minScalar)
      {
          minScalar = particleScalar;
          minPos[0] = particleX;
          minPos[1] = particleY;
          minPos[2] = particleZ;
      }*/

    }
    
    
    
  }

  //printf("maxSacalar = %d\n", maxScalar);
  //printf("minScalar = %d\n", minScalar);

  /***** Metropolis particle generation*******
  float particleOpacityOld;


  //generate the given amount of particles:
  //for(int i = 0 ; i <= cellParticles ; i++){
  for(int i = 0 ; i <= cellParticles ; i++){

    //generate 3 parameters randomly in the range 0..1
    float par0 = curand_uniform(&localRNGState);
    float par1 = curand_uniform(&localRNGState);
    float par2 = curand_uniform(&localRNGState);

    //if the sum of the 3 parameters is > 1, calculate 1 - parameter for each
    if(par0 + par1 + par2 > 1.0f){
      par0 = 1.0f - par0;
      par1 = 1.0f - par1;
      par2 = 1.0f - par2;
    }

    //fourth parameter is calculated here:
    float par3 = 1.0f - par0 - par1 - par2;
    

    //particle opacity equals emission probability:
    float particleScalar = v0[threadIdx.x].scalar * par0 + v1[threadIdx.x].scalar * par1 + v2[threadIdx.x].scalar * par2 + v3[threadIdx.x].scalar * par3;
    
    //opacity of the new generated particle 
    float particleOpacity = opacityFromTF(particleScalar);
    float ratio;
    int testVar = 0;
    if( i == 0 )
    {
    	particleOpacityOld = particleOpacity;
    	ratio = 1.0f;
    	testVar = 1;
    }
    else
    {
    	ratio = particleOpacity/particleOpacityOld;
    	if( ratio >= 1.0f )
    	{
    		particleOpacityOld = particleOpacity;
    		testVar = 1;
    	}
    }
    
    if(testVar == 0){



    //if opacity of particle >= random variable, project it. otherwise, reject it.
    if(particleOpacity >= curand_uniform(&localRNGState)){

      //finalize particle and project it:
      //position is calculated on the fly, minimize mem access
      particleOpacityOld = particleOpacity;
      projectParticle(v0[threadIdx.x].pos[0] * par0 + v1[threadIdx.x].pos[0] * par1 + v2[threadIdx.x].pos[0] * par2 + v3[threadIdx.x].pos[0] * par3,
                      v0[threadIdx.x].pos[1] * par0 + v1[threadIdx.x].pos[1] * par1 + v2[threadIdx.x].pos[1] * par2 + v3[threadIdx.x].pos[1] * par3,
                      v0[threadIdx.x].pos[2] * par0 + v1[threadIdx.x].pos[2] * par1 + v2[threadIdx.x].pos[2] * par2 + v3[threadIdx.x].pos[2] * par3,
                      particleScalar,subPixelsD,subPixelsDepthD,windowWidth,windowHeight);
    }
    
    }
    else if(testVar == 1)
    {
    	
      projectParticle(v0[threadIdx.x].pos[0] * par0 + v1[threadIdx.x].pos[0] * par1 + v2[threadIdx.x].pos[0] * par2 + v3[threadIdx.x].pos[0] * par3,
                      v0[threadIdx.x].pos[1] * par0 + v1[threadIdx.x].pos[1] * par1 + v2[threadIdx.x].pos[1] * par2 + v3[threadIdx.x].pos[1] * par3,
                      v0[threadIdx.x].pos[2] * par0 + v1[threadIdx.x].pos[2] * par1 + v2[threadIdx.x].pos[2] * par2 + v3[threadIdx.x].pos[2] * par3,
                      particleScalar,subPixelsD,subPixelsDepthD,windowWidth,windowHeight);
    }
	
  }
*/

/****** trying to generate only the 4 vertex points

  for(int i = 0 ; i < 4 ; i++){
	float particleScalar = 0.0f;
	if(i==0) 
		{
			particleScalar = v0[threadIdx.x].scalar;
			projectParticle(v0[threadIdx.x].pos[0] , v0[threadIdx.x].pos[1] , v0[threadIdx.x].pos[2] , particleScalar,subPixelsD,subPixelsDepthD,windowWidth,windowHeight);
		}
	else if (i==1) {
			particleScalar = v1[threadIdx.x].scalar;
			projectParticle(v1[threadIdx.x].pos[0] , v1[threadIdx.x].pos[1] , v1[threadIdx.x].pos[2] , particleScalar,subPixelsD,subPixelsDepthD,windowWidth,windowHeight);
		}
	else if (i==2) {
			particleScalar = v2[threadIdx.x].scalar;
			projectParticle(v2[threadIdx.x].pos[0] , v2[threadIdx.x].pos[1] , v2[threadIdx.x].pos[2] , particleScalar,subPixelsD,subPixelsDepthD,windowWidth,windowHeight);
		}
	else if (i==3) {
			particleScalar = v3[threadIdx.x].scalar;
			projectParticle(v3[threadIdx.x].pos[0] , v3[threadIdx.x].pos[1] , v3[threadIdx.x].pos[2] , particleScalar,subPixelsD,subPixelsDepthD,windowWidth,windowHeight);
		}

	
}
*/
  //write back updated state:
  rngStatesD[x] = localRNGState;

}

__global__ void kernelProjectionSD(curandState* rngStatesD, unsigned int numParticles, float* volumeRatios, unsigned int numVoxels, unsigned int subpixelLevel, unsigned int windowWidth, unsigned int windowHeight, voxelFloat* voxelsD, RGBA* subPixelsD, float* subPixelsDepthD){

  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  if(x > numVoxels)
    return;

  curandState localRNGState = rngStatesD[x];

  //create some noise
  //PerlinNoise pn;

  //get the model coordinates of the 8 points of the current voxel:
  __shared__ vertexFloat v0[BLOCKSIZE_PROJECTION];
  __shared__ vertexFloat v1[BLOCKSIZE_PROJECTION];
  __shared__ vertexFloat v2[BLOCKSIZE_PROJECTION];
  __shared__ vertexFloat v3[BLOCKSIZE_PROJECTION];
  __shared__ vertexFloat v4[BLOCKSIZE_PROJECTION];
  __shared__ vertexFloat v5[BLOCKSIZE_PROJECTION];
  __shared__ vertexFloat v6[BLOCKSIZE_PROJECTION];
  __shared__ vertexFloat v7[BLOCKSIZE_PROJECTION];
  v0[threadIdx.x] = voxelsD[x].v[0];
  v1[threadIdx.x] = voxelsD[x].v[1];
  v2[threadIdx.x] = voxelsD[x].v[2];
  v3[threadIdx.x] = voxelsD[x].v[3];
  v4[threadIdx.x] = voxelsD[x].v[4];
  v5[threadIdx.x] = voxelsD[x].v[5];
  v6[threadIdx.x] = voxelsD[x].v[6];
  v7[threadIdx.x] = voxelsD[x].v[7];


  //check out the uncertainty value for each grid point.
 // printf("CHECK: variance value for the grid points= %f, %f, %f, %f \n", v0[threadIdx.x].var, v1[threadIdx.x].var, v2[threadIdx.x].var, v3[threadIdx.x].var);

  //if((opacityFromTF(v0[threadIdx.x].scalar) + opacityFromTF(v1[threadIdx.x].scalar) + opacityFromTF(v2[threadIdx.x].scalar) + opacityFromTF(v3[threadIdx.x].scalar) + opacityFromTF(v4[threadIdx.x].scalar) + opacityFromTF(v5[threadIdx.x].scalar) + opacityFromTF(v6[threadIdx.x].scalar) + opacityFromTF(v7[threadIdx.x].scalar))/8.0f < 0.05f)
  //  return;

  //for cosmo dataset
  float averageDensity = (v0[threadIdx.x].scalar + v1[threadIdx.x].scalar + v2[threadIdx.x].scalar + v3[threadIdx.x].scalar +v4[threadIdx.x].scalar + v5[threadIdx.x].scalar + v6[threadIdx.x].scalar + v7[threadIdx.x].scalar)/8.0f;
  int nop = floor(averageDensity);

  //int cellParticles = static_cast<int>(volumeRatios[x] * static_cast<float>(numParticles));
  if(nop > 1000.0f)
      nop = 500.0f;

  //generate the given amount of particles:
  //for(int i = 0 ; i <= cellParticles ; i++){
  for(int i = 0 ; i <= nop ; i++){

    //generate 3 parameters randomly in the range 0..1
    float par0 = curand_uniform(&localRNGState);
    float par1 = curand_uniform(&localRNGState);
    float par2 = curand_uniform(&localRNGState);
    //float par3 = curand_uniform(&localRNGState);

    int flag = 0;
    //if(i%2 == 0) flag =0; // real value
    //else flag = 1; //noise value

    //if the sum of the 3 parameters is > 1, calculate 1 - parameter for each
    if(par0 + par1 + par2 > 1.0f){
      par0 = 1.0f - par0;
      par1 = 1.0f - par1;
      par2 = 1.0f - par2;
    }

    //fourth parameter is calculated here:
    float par3 = 1.0f - par0 - par1 - par2;


    //particle opacity equals emission probability:
    float particleScalar = v0[threadIdx.x].scalar * par0 + v1[threadIdx.x].scalar * par1 + v2[threadIdx.x].scalar * par2 + v3[threadIdx.x].scalar * par3;


    //if opacity of particle >= random variable, project it. otherwise, reject it.
    //if(opacityFromTF(averageDensity) >= curand_uniform(&localRNGState)){
    if(true){
      //finalize particle and project it:
      //position is calculated on the fly, minimize mem access
      float particleX,particleY,particleZ;
      //if(i%2 == 0)
      //{
          particleX = v0[threadIdx.x].pos[0] * par0 + v1[threadIdx.x].pos[0] * par1 + v2[threadIdx.x].pos[0] * par2 + v3[threadIdx.x].pos[0] * par3;
          particleY = v0[threadIdx.x].pos[1] * par0 + v1[threadIdx.x].pos[1] * par1 + v2[threadIdx.x].pos[1] * par2 + v3[threadIdx.x].pos[1] * par3;
          particleZ = v0[threadIdx.x].pos[2] * par0 + v1[threadIdx.x].pos[2] * par1 + v2[threadIdx.x].pos[2] * par2 + v3[threadIdx.x].pos[2] * par3;
      /*}
      else{
          particleX = v4[threadIdx.x].pos[0] * par0 + v5[threadIdx.x].pos[0] * par1 + v6[threadIdx.x].pos[0] * par2 + v7[threadIdx.x].pos[0] * par3;
          particleY = v4[threadIdx.x].pos[1] * par0 + v5[threadIdx.x].pos[1] * par1 + v6[threadIdx.x].pos[1] * par2 + v7[threadIdx.x].pos[1] * par3;
          particleZ = v4[threadIdx.x].pos[2] * par0 + v5[threadIdx.x].pos[2] * par1 + v6[threadIdx.x].pos[2] * par2 + v7[threadIdx.x].pos[2] * par3;
      }*/
      projectParticle(particleX,particleY, particleZ, 13000.0f ,subPixelsD,subPixelsDepthD,windowWidth,windowHeight, flag);
    }



  }

  //write back updated state:
  rngStatesD[x] = localRNGState;

}


extern "C" void launchProjection(curandState* rngStatesD, unsigned int numParticles, float* volumeRatios, unsigned int numTetras, unsigned int windowWidth, unsigned int windowHeight, tetraFloat* tetrasD, RGBA* subPixelsD, float* subPixelsDepthD){
  dim3 block(BLOCKSIZE_PROJECTION,1,1);
  double gridSize = static_cast<double>(numTetras)/static_cast<double>(BLOCKSIZE_PROJECTION);
  dim3 grid(static_cast<int>(gridSize)+1,1,1);

  
    printf("LaunchKernel projection");
    printf(" \n num tetras = %d\n Block Size = %d x 1 x 1 \n Grid Size = %d",BLOCKSIZE_PROJECTION,numTetras,static_cast<int>(gridSize)+1);
    printf(" \n window width = %d, window height = %d\n",windowWidth,windowHeight);
  

  kernelProjection<<< grid, block>>>(rngStatesD, numParticles, volumeRatios, numTetras, SUBPIXELLEVEL, windowWidth, windowHeight, tetrasD, subPixelsD, subPixelsDepthD);


}

extern "C" void launchProjectionSD(curandState* rngStatesD, unsigned int numParticles, float* volumeRatios, unsigned int numVoxels, unsigned int windowWidth, unsigned int windowHeight, voxelFloat* voxelsD, RGBA* subPixelsD, float* subPixelsDepthD){
  dim3 block(BLOCKSIZE_PROJECTION,1,1);
  double gridSize = static_cast<double>(numVoxels)/static_cast<double>(BLOCKSIZE_PROJECTION);
  dim3 grid(static_cast<int>(gridSize)+1,1,1);


    printf("LaunchKernel projection");
    printf(" \n num voxels = %d\n Block Size = %d x 1 x 1 \n Grid Size = %d",BLOCKSIZE_PROJECTION,numVoxels,static_cast<int>(gridSize)+1);
    printf(" \n window width = %d, window height = %d\n",windowWidth,windowHeight);


  kernelProjectionSD<<< grid, block>>>(rngStatesD, numParticles, volumeRatios, numVoxels, SUBPIXELLEVEL, windowWidth, windowHeight, voxelsD, subPixelsD, subPixelsDepthD);


}



