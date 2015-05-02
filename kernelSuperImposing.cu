#include <stdio.h>
#include <iostream>
#include "defs.h"




__device__ void rgbaUintToByteArray2(RGBA* input, unsigned char* output){
  memcpy(output,input,4);
}

__device__ void charArrayToRGBA2(unsigned char* input, RGBA* output){
  memcpy(output,input,4);
}

__global__ void kernelSuperimposing(unsigned int windowWidth, unsigned int windowHeight, unsigned int subPixelLevel, RGBA* pixelsD, RGBA* subPixelsD, depthValue* subPixelsDepthD){
  
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

  if(x >= windowWidth || y >= windowHeight)
    return;

  float minDepth = 1.0f;
  float maxDepth = 0.0f;
  for(unsigned int xStart = x * subPixelLevel ; xStart < (x+1) * subPixelLevel ; xStart ++){
    for(unsigned int yStart = y * subPixelLevel ; yStart < (y+1) * subPixelLevel ; yStart ++){
      float currentDepth = subPixelsDepthD[yStart*windowWidth*subPixelLevel + xStart];
      if(currentDepth < minDepth)
        minDepth = currentDepth;
      if(currentDepth > maxDepth)
        maxDepth = currentDepth;
    }
  }


  float distance = maxDepth - minDepth;


  //need ints here to prevent overflow:
  unsigned int pixelComponentsSum[4] = {0,0,0,0};
  for(unsigned int xStart = x * subPixelLevel ; xStart < (x+1) * subPixelLevel ; xStart ++){
    for(unsigned int yStart = y * subPixelLevel ; yStart < (y+1) * subPixelLevel ; yStart ++){

      float currentDepth = subPixelsDepthD[yStart*windowWidth*subPixelLevel + xStart];
      float depthRatio = (currentDepth - minDepth) / distance;
      //float depthRatio = 1.0f - (maxDepth - currentDepth) / distance;

      unsigned char currentSubPixel[4];
      rgbaUintToByteArray2(&(subPixelsD[yStart * windowWidth * subPixelLevel + xStart]),currentSubPixel);
      
      pixelComponentsSum[0] += static_cast<unsigned char>(static_cast<float>(currentSubPixel[0]));
      pixelComponentsSum[1] += static_cast<unsigned char>(static_cast<float>(currentSubPixel[1]));
      pixelComponentsSum[2] += static_cast<unsigned char>(static_cast<float>(currentSubPixel[2]));
      pixelComponentsSum[3] += static_cast<unsigned char>(static_cast<float>(currentSubPixel[3]) * (depthRatio));

    }
  }
  unsigned char pixelComponents[4];
  pixelComponents[0] = (unsigned char)((float)pixelComponentsSum[0] / (float)(subPixelLevel*subPixelLevel));
  pixelComponents[1] = (unsigned char)((float)pixelComponentsSum[1] / (float)(subPixelLevel*subPixelLevel));
  pixelComponents[2] = (unsigned char)((float)pixelComponentsSum[2] / (float)(subPixelLevel*subPixelLevel));
  pixelComponents[3] = (unsigned char)((float)pixelComponentsSum[3] / (float)(subPixelLevel*subPixelLevel));

  RGBA pixel = 0;
  charArrayToRGBA2(pixelComponents,&pixel);  

  pixelsD[y*windowWidth + x] = pixel;
}




extern "C" void launchSuperimposingSpatial(unsigned int windowWidth, unsigned int windowHeight, unsigned int subPixelLevel, RGBA* pixels, RGBA* subPixels, depthValue* subPixelsDepthD){
  unsigned int x = 16;
  unsigned int y = 16;
  dim3 block(x,y,1);
  dim3 grid((int)(windowWidth/x)+1,(int)(windowHeight/y)+1,1);
  
    printf("LaunchKernel superimposing");
    /*std::cout <<  "Block Size = " << block[0] << " x " << block[1] << " x " << block[2] << std::endl;
    std::cout <<  "Grid Size  = " << grid[0] << " x " << grid[1] << " x " << grid[2] << std::endl;*/
    printf(" \n window width = %d, window height = %d\n",windowWidth,windowHeight);
 
  kernelSuperimposing<<< grid, block >>>(windowWidth,windowHeight,subPixelLevel,pixels,subPixels, subPixelsDepthD);

}

