#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16

__constant__ float Mk[3200];

// Op1 and 4
__global__ void conv_forward_kernel(float *output, const float *input, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Xtile_width = TILE_WIDTH + K - 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a
    extern __shared__ float sh_mem[];
    float* X_shd = &sh_mem[0];
    #define X_shd2d(i1, i0) X_shd[(i1) * (Xtile_width) + i0]
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) Mk[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here


    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    const int tiles = ceil(Width_out * 1.0 / TILE_WIDTH);
    int h = (blockIdx.z / tiles) * TILE_WIDTH + tx;
    int w = (blockIdx.z % tiles) * TILE_WIDTH + ty;
    float acc = 0.0f;

    int hb = (blockIdx.z / tiles) * TILE_WIDTH;
    int wb = (blockIdx.z % tiles) * TILE_WIDTH;

    for(int c = 0; c < Channel; ++c){
        for(int ii = h; ii < hb + Xtile_width; ii += TILE_WIDTH){
            for(int jj = w; jj < wb + Xtile_width; jj += TILE_WIDTH){
                if(ii < Height && jj < Width){
                    X_shd2d(ii - hb, jj - wb) = in_4d(bx, c, ii, jj);
                }
            }
        }
        __syncthreads();
        if (h < Height_out && w < Width_out) {
             for (int p = 0; p < K; p++) {
	       	for (int q = 0; q < K; q++) {
	            acc += X_shd2d(tx + p, ty + q) * mask_4d(by, c, p, q);
	        }
	     }
        }
        __syncthreads();
    }

    if (h < Height_out && w < Width_out)
	out_4d(bx, by, h, w) = acc;

    #undef out_4d
    #undef in_4d
    #undef X_shd2d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

     cudaMalloc((void **) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
     cudaMalloc((void **) device_output_ptr, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float));
     cudaMalloc((void **) device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
     cudaMemset(*device_output_ptr, 0, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float)); 
  
     cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
     cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // Set the kernel dimensions and call the kernel
    
    int h = ceil(1.0 * Height_out / TILE_WIDTH);
    int w = ceil(1.0 * Width_out / TILE_WIDTH);
    dim3 gridDim(Batch, Map_out, h*w);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1.0);
    size_t sh_size = sizeof(float) * (TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1);
    cudaMemcpyToSymbol(Mk, device_mask, Channel * K * K * Map_out * sizeof(float));
    conv_forward_kernel<<<gridDim, blockDim, sh_size>>>(device_output, device_input, Batch, Map_out, Channel, Height, Width, K);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
