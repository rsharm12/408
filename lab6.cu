// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 16

__global__ void cast_fp_to_uchar(unsigned char *out, float *in, int width, int height, int channels)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if (ii < width * height * channels) {
    out[ii] = (unsigned char)(255 * in[ii]);
  }
}

__global__ void conv_RGB_to_GrayScale(unsigned char *out, unsigned char *in, int width, int height, int channels)
{
  unsigned char r, g, b;

  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  int ii = Row * width + Col;

  if (ii < width * height * channels)
  {
    r = in[3 * ii];
    g = in[3 * ii + 1];
    b = in[3 * ii + 2];
    out[ii] = (unsigned char)(0.21 * r + 0.71 * g + 0.07 * b);
  }
}

__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo)
{
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x < HISTOGRAM_LENGTH) histo_private[threadIdx.x] = 0;
  __syncthreads();

  /* int stride = blockDim.x * gridDim.x;
   while(i < size)
   {
       atomicAdd(&(histo_private[buffer[i]]), 1);
       i += stride;
   }
   __syncthreads();
  */
  if (i < size)
    atomicAdd(&(histo_private[buffer[i]]), 1);
  __syncthreads();

  if (threadIdx.x < HISTOGRAM_LENGTH)
  {
    atomicAdd( &(histo[threadIdx.x]), histo_private[threadIdx.x]);
  }

}

__global__ void histo_cdf_kernel(float *output, unsigned int *input, int width, int height) {

  __shared__ float cdf[HISTOGRAM_LENGTH];

  unsigned int t = threadIdx.x;
  unsigned int start = blockDim.x;
  float size = width * height;

  if (t < HISTOGRAM_LENGTH)
    cdf[t] = (float)input[t] / size;
  else
    cdf[t] = 0.0f;
  if (start + t < HISTOGRAM_LENGTH)
    cdf[start + t] = (float)input[start + t] / size;
  else
    cdf[start + t] = 0.0f;

  __syncthreads();

  int stride = 1;
  while (stride <= blockDim.x) // calculate first half
  {
    int index = (t + 1) * stride * 2 - 1;
    if (index < HISTOGRAM_LENGTH ) //&& (index-stride) >= 0)
      cdf[index] += cdf[index - stride];
    stride *= 2;
    __syncthreads();
  }

  stride = blockDim.x / 2;  // calculate second half
  while (stride > 0)
  {
    int index = (t + 1) * stride * 2 - 1;
    if ((index + stride) < HISTOGRAM_LENGTH)
    {
      cdf[index + stride] += cdf[index];
    }
    stride /= 2;
    __syncthreads();
  }

  __syncthreads();

  if (t < HISTOGRAM_LENGTH) {
    output[t] = cdf[t];
  }
  if (start + t < HISTOGRAM_LENGTH) {
    output[start + t] = cdf[start + t];
  }
}

__device__ unsigned char clamp(float x, float start, float end)
{
  return min(max(x, start), end);
}

__device__ unsigned char correct_color(float *cdf, unsigned char val, float cdfmin)
{
  return clamp(255 * (cdf[val] - cdfmin) / (1.0 - cdfmin), 0, 255.0);
}

__global__ void histoEqualize(unsigned char *uchar_image, float *histo_cdf, int width, int height, int channels)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if (ii < width * height * channels)
    uchar_image[ii] = correct_color(histo_cdf, uchar_image[ii], histo_cdf[0]);

}

__global__ void cast_uchar_to_fp(float *out, unsigned char *in, int width, int height, int channels)
{
  int ii = blockIdx.x * blockDim.x + threadIdx.x;

  if (ii < width * height * channels)
    out[ii] = (float)(in[ii] / 255.0);
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  int imageSize;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *d_histo_cdf;
  unsigned int *d_histogram;
  unsigned char *d_grayImage;
  unsigned char *d_ucharImage;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  imageSize = imageWidth * imageHeight * imageChannels;

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceInputImageData, imageSize * sizeof(float));
  cudaMalloc((void **) &deviceOutputImageData, imageSize * sizeof(float));
  cudaMalloc((void **) &d_histo_cdf, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **) &d_histogram, imageWidth * imageHeight * sizeof(unsigned int));
  cudaMalloc((void **) &d_grayImage, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **) &d_ucharImage, imageSize * sizeof(unsigned char));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  wbTime_start(Compute, "Performing CUDA computation");
  /* cast image data to unsigned char */
  dim3 dimGrid(ceil(imageSize / (1.0 * HISTOGRAM_LENGTH)), 1, 1);
  dim3 dimBlock(HISTOGRAM_LENGTH, 1, 1);
  cast_fp_to_uchar <<< dimGrid, dimBlock>>>(d_ucharImage, deviceInputImageData, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  /* convert unsigned char image data to grayscale */
  dim3 dimGrid_G(ceil(imageWidth / (1.0 * BLOCK_SIZE)), ceil(imageHeight / (1.0 * BLOCK_SIZE)), 1);
  dim3 dimBlock_G(BLOCK_SIZE, BLOCK_SIZE, 1);
  conv_RGB_to_GrayScale <<< dimGrid_G, dimBlock_G>>>(d_grayImage, d_ucharImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  /* compute histogram of grayscale image */
  dim3 dimGrid_H(ceil((imageWidth * imageHeight) / (1.0 * HISTOGRAM_LENGTH)), 1, 1);
  dim3 dimBlock_H(HISTOGRAM_LENGTH, 1, 1);
  histo_kernel <<< dimGrid_H, dimBlock_H>>>(d_grayImage, imageWidth * imageHeight, d_histogram);
  cudaDeviceSynchronize();

  /* compute cumulative distribution function of histogram */
  dim3 dimGrid_cdf(1, 1, 1);
  dim3 dimBlock_cdf(HISTOGRAM_LENGTH / 2, 1, 1);
  histo_cdf_kernel <<< dimGrid_cdf, dimBlock_cdf>>>(d_histo_cdf, d_histogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  /* correct input image via histogram equalization */
  dim3 dimGrid_eq(ceil(imageSize / (1.0 * HISTOGRAM_LENGTH)), 1, 1);
  dim3 dimBlock_eq(HISTOGRAM_LENGTH, 1, 1);
  histoEqualize <<< dimGrid_eq, dimBlock_eq>>>(d_ucharImage, d_histo_cdf, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  /* cast image back to floating */
  cast_uchar_to_fp <<< dimGrid, dimBlock>>>(deviceOutputImageData, d_ucharImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(d_histo_cdf);
  cudaFree(d_histogram);
  cudaFree(d_grayImage);
  cudaFree(d_ucharImage);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, outputImage);

  return 0;
}
