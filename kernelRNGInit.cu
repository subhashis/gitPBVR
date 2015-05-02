#include <stdio.h>
#include "defs.h"



__global__ void kernelRNGSetup(unsigned int numTetras, curandState *state, unsigned int seed, unsigned int roundNumber, unsigned int numInitsPerRound){
  unsigned int id = numInitsPerRound * roundNumber + blockIdx.x*blockDim.x + threadIdx.x;
  if(id > numTetras)
    return;
  curand_init(seed, id, 0, &state[id]);
}



extern "C" curandState* initRNGCuda(unsigned int numTetras, unsigned int seed){
  curandState* devStates;
  unsigned int blockSize = 256;
  checkCudaErrors(cudaMalloc((void **)&devStates, numTetras * sizeof(curandState)));
  
  //subdivide inits into managable portions - avoid timeout from watchdog
  unsigned int numInitsPerRound = MAX_RNG_INIT_BLOCKS * blockSize;
  unsigned int numRounds = static_cast<unsigned int>(static_cast<float>(numTetras)/static_cast<float>(numInitsPerRound)) + 1;

  dim3 block(blockSize,1,1);
  dim3 grid(MAX_RNG_INIT_BLOCKS,1,1);

  printf("------------ Initializing curand ------------\n");
  printf("Number of inits per round = %d\n",numInitsPerRound);
  printf("Number of rounds = %d\n",numRounds);
  printf("---------------------------------------------\n");

  for(unsigned int i = 0 ; i < numRounds ; i++){
    kernelRNGSetup<<<grid, block>>>(numTetras, devStates, seed, i, numInitsPerRound);
    cudaDeviceSynchronize();
  }
  
  return devStates;
}

