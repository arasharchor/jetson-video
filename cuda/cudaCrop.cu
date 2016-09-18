/*
 * http://github.com/dusty-nv/jetson-video
 */

#include "cudaCrop.h"
#include "cudaMath.h"


__global__ void gpuCrop_uchar( unsigned char* input, int inputWidth, unsigned char* output, int outputWidth, int outputHeight )
{
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	
	if( x >= outputWidth || y >= outputHeight )
		return;

	output[y * outputWidth + x] = input[y * inputWidth + x];
} 


cudaError_t cudaCrop( uint8_t* input, const dim3& inputSize, uint8_t* output, const dim3& outputSize )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputSize.x == 0 || inputSize.y == 0 || inputSize.z == 0 || outputSize.x == 0 || outputSize.y == 0 || outputSize.z == 0 ) 
		return cudaErrorInvalidValue;

	const int inputAlignedWidth  = inputSize.z  / sizeof(uint8_t);
	const int outputAlignedWidth = outputSize.z / sizeof(uint8_t);
	

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputSize.x,blockDim.x), iDivUp(outputSize.y,blockDim.y));

	gpuCrop_uchar<<<gridDim, blockDim>>>(input, inputAlignedWidth, output, outputAlignedWidth, outputSize.y);

	return CUDA(cudaGetLastError());
}




