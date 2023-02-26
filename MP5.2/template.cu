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

__global__ void scan(float *input, float *output, int len, float *addition) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float XY[BLOCK_SIZE * 2];
  
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  
  if (i < len) XY[threadIdx.x] = input[i];
  else XY[threadIdx.x] = 0;

  if ((i + blockDim.x) < len) XY[blockDim.x + threadIdx.x] = input[i + blockDim.x];
  else XY[blockDim.x + threadIdx.x] = 0;

  int stride = 1;
  while (stride < 2 * blockDim.x) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < 2 * blockDim.x && (index - stride) >= 0)
      XY[index] += XY[index - stride];
    stride = stride * 2;
  }

  stride = blockDim.x / 2;
  while(stride > 0) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if ((index + stride) < 2 * blockDim.x) XY[index + stride] += XY[index];
    stride = stride / 2;
  }

  __syncthreads();
  if (i < len) output[i] = XY[threadIdx.x];
  if ((i + blockDim.x) < len) output[i + blockDim.x] = XY[blockDim.x + threadIdx.x];
  if (threadIdx.x == (blockDim.x - 1)) addition[blockIdx.x] = XY[blockDim.x * 2 - 1];
}

__global__ void add(float *output, int len, float *auxiliary){
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  if (i < len && blockIdx.x != false) output[i] += auxiliary[blockIdx.x - 1];
  if ((i + blockDim.x) < len && blockIdx.x != false) output[i + blockDim.x] += auxiliary[blockIdx.x - 1]; 
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *result;
  float *sum;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&result, ceil(numElements/float(BLOCK_SIZE * 2.0)) * sizeof(float)));
  wbCheck(cudaMalloc((void **)&sum, ceil(numElements/float(BLOCK_SIZE * 2.0)) * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numElements/float(BLOCK_SIZE * 2.0)));
  dim3 dim_one(1);
  dim3 dimBlock(BLOCK_SIZE);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements, sum);
  cudaDeviceSynchronize();
  scan<<<dim_one, dimBlock>>>(sum, result, ceil(numElements/float(BLOCK_SIZE * 2.0)), deviceInput);
  cudaDeviceSynchronize();
  add<<<dimGrid, dimBlock>>>(deviceOutput, numElements, result);
  cudaDeviceSynchronize();
  
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
