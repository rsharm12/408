#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_WIDTH 8
#define O_TILE_WIDTH 6

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_WIDTH*MASK_WIDTH*MASK_WIDTH];


__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  
  int row_o = blockIdx.y*O_TILE_WIDTH + ty;
  int col_o = blockIdx.x*O_TILE_WIDTH + tx;
  int dep_o = blockIdx.z*O_TILE_WIDTH + tz;
  
  int row_i = row_o - (MASK_WIDTH/2);
  int col_i = col_o - (MASK_WIDTH/2);
  int dep_i = dep_o - (MASK_WIDTH/2);
    
  __shared__ float N_ds[O_TILE_WIDTH+MASK_WIDTH-1][O_TILE_WIDTH+MASK_WIDTH-1][O_TILE_WIDTH+MASK_WIDTH-1];
  
  if( (row_i >= 0) && (row_i < y_size) && (col_i >= 0) && (col_i < x_size) && (dep_i >= 0) && (dep_i < z_size) ) {
    N_ds[tz][ty][tx] = input[col_i + (row_i * x_size) + (dep_i * y_size * x_size)];
  } else {
    N_ds[tz][ty][tx] = 0.0f;
  }
  __syncthreads();
  
  float Pvalue = 0.0f;
  if(ty < O_TILE_WIDTH && tx < O_TILE_WIDTH && tz < O_TILE_WIDTH) {
    for(int z=0; z<MASK_WIDTH; z++) {
      for(int y=0; y<MASK_WIDTH; y++) {
        for(int x=0; x<MASK_WIDTH; x++) {
          Pvalue += N_ds[z+tz][y+ty][x+tx] * deviceKernel[x + (y*MASK_WIDTH) + (z*MASK_WIDTH*MASK_WIDTH)];
        }
      }
    }
    if(row_o < y_size && col_o < x_size && dep_o < z_size )
      output[col_o + (row_o * x_size) + (dep_o * y_size * x_size)] = Pvalue;
  }
  
  __syncthreads(); 
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);
  
  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
  
  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceInput, (inputLength-3)*sizeof(float));
  cudaMalloc((void **) &deviceOutput, (inputLength-3)*sizeof(float));
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  cudaMemcpy(deviceInput, &hostInput[3], (inputLength-3)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength*sizeof(float));
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(x_size/(1.0*O_TILE_WIDTH)), ceil(y_size/(1.0*O_TILE_WIDTH)), ceil(z_size/(1.0*O_TILE_WIDTH)));
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
  
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  cudaMemcpy(&hostOutput[3], deviceOutput, (inputLength-3)*sizeof(float), cudaMemcpyDeviceToHost);
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
