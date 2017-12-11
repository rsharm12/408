// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__shared__ float partialSum[2*BLOCK_SIZE];

/* add scanned block sums to all scanned block i+1 */
__global__ void final_scan(float *input, float *output, int len) {
  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockIdx.x*blockDim.x + t;
  if(blockIdx.x > 0) {
    if(start < len)
       output[start] += input[blockIdx.x-1];
    if(start + blockDim.x < len)
       output[start+blockDim.x] += input[blockIdx.x-1];
  }
}

/* 2nd kernel call used to scan block sums calculated from 1st scan */
__global__ void aux_scan(float *input, float *output, int len) {
  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockIdx.x*blockDim.x + t;
  
  if(start < len)
    partialSum[t] = input[start];
  else
    partialSum[t] = 0.0;
  if(start+blockDim.x < len)
    partialSum[blockDim.x + t] = input[start+blockDim.x];
  else
    partialSum[blockDim.x + t] = 0.0;
  
  __syncthreads();
  
  int stride = 1;
  while(stride <= 2*BLOCK_SIZE)  // calculate first half
  {
       int index = (t+1)*stride*2 - 1;
       if(index < 2*BLOCK_SIZE && (index-stride) >= 0)
          partialSum[index] += partialSum[index-stride];
       stride *= 2;
       __syncthreads();
  }
  
  stride = BLOCK_SIZE/2;    // calculate second half
  while(stride > 0)
  {
       int index = (t+1)*stride*2 - 1;
       if((index+stride) < 2*BLOCK_SIZE)
       {
	        partialSum[index+stride] += partialSum[index];
       }				
       stride /= 2;	
       __syncthreads();
  }
  
  __syncthreads();
  
  if(start < len)
    output[start] = partialSum[t];
  if(start+blockDim.x < len)
    output[start+blockDim.x] = partialSum[blockDim.x+t];
}

/* compute block sums into an auxillary array */
__global__ void scan(float *input, float *output, float *aux, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockIdx.x*blockDim.x + t;
  
  if(start < len)
    partialSum[t] = input[start];
  else
    partialSum[t] = 0.0;
  if(start+blockDim.x < len)
    partialSum[blockDim.x + t] = input[start+blockDim.x];
  else
    partialSum[blockDim.x + t] = 0.0;
  
  __syncthreads();
  
  int stride = 1;
  while(stride <= 2*BLOCK_SIZE)  // calculate first half
  {
       int index = (t+1)*stride*2 - 1;
       if(index < 2*BLOCK_SIZE && (index-stride) >= 0)
          partialSum[index] += partialSum[index-stride];
       stride *= 2;
       __syncthreads();
  }
  
  stride = BLOCK_SIZE/2;    // calculate second half
  while(stride > 0)
  {
       int index = (t+1)*stride*2 - 1;
       if((index+stride) < 2*BLOCK_SIZE)
       {
	        partialSum[index+stride] += partialSum[index];
       }				
       stride /= 2;	
       __syncthreads();
  }
  
  __syncthreads();
  
  if(start < len)
    output[start] = partialSum[t];
  if(start+blockDim.x < len)
    output[start+blockDim.x] = partialSum[blockDim.x+t];
  if(t == blockDim.x-1)
    aux[blockIdx.x] = partialSum[2*blockDim.x-1];
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceAuxInput;
  float *deviceAuxOutput;
  int numElements; // number of elements in the list
  int numAuxElements;
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);
  numAuxElements = 2*BLOCK_SIZE;
  
  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceAuxInput, numAuxElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceAuxOutput, numAuxElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numElements/(1.0*2*BLOCK_SIZE)), 1, 1);
  dim3 dimGridAux(1,1,1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceAuxInput, numElements);
  cudaDeviceSynchronize();
  aux_scan<<<dimGridAux, dimBlock>>>(deviceAuxInput, deviceAuxOutput, 2*BLOCK_SIZE);
  cudaDeviceSynchronize();
  final_scan<<<dimGrid,dimBlock>>>(deviceAuxOutput, deviceOutput, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAuxInput);
  cudaFree(deviceAuxOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
