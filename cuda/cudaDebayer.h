/*
 * http://github.com/dusty-nv/jetson-video
 */

#ifndef __CUDA_DEBAYER_H__
#define __CUDA_DEBAYER_H__


#include "cudaUtility.h"

 
/**
 * cudaBayerType
 */
enum cudaBayerType
{
	CUDA_BAYER_GR = 0,
	CUDA_BAYER_RG,
	CUDA_BAYER_GB,
	CUDA_BAYER_BG
};


/**
 * cudaBayerToRGBA
 */
cudaError_t cudaBayerToRGBA( uint8_t* input, uchar4* output, size_t width, size_t height, cudaBayerType colorspace );
		
/**
 * cudaBayerToRGBA
 */
cudaError_t cudaBayerToRGBA( uint8_t* input, float4* output, size_t width, size_t height, cudaBayerType colorspace );
								
						
/**
 * cudaBayerToRGBA
 */
cudaError_t cudaBayerToRGBA( uint8_t* input, size_t inputPitch, 
							 uchar4* output, size_t outputPitch, 
							 size_t width,   size_t height, 
                             cudaBayerType colorspace );

/**
 * cudaBayerToRGBA
 */
cudaError_t cudaBayerToRGBA( uint8_t* input, size_t inputPitch, 
							 float4* output, size_t outputPitch, 
							 size_t width,   size_t height, 
                             cudaBayerType colorspace );
							 

/**
 * cudaBayerToRGBA (with scaling)
 */
cudaError_t cudaBayerToRGBA( uint8_t* input, size_t inputPitch, size_t inputWidth, size_t inputHeight, 
					         uchar4* output, size_t outputPitch, size_t outputWidth, size_t outputHeight, 
                             cudaBayerType colorspace );

				
/**
 * cudaBayerToRGBA (with scaling)
 */
cudaError_t cudaBayerToRGBA( uint8_t* input, size_t inputPitch, size_t inputWidth, size_t inputHeight, 
					         float4* output, size_t outputPitch, size_t outputWidth, size_t outputHeight, 
                             cudaBayerType colorspace );

							 
#endif
