/*
 * http://github.com/dusty-nv/jetson-video
 */

#ifndef __CUDA_CROP_H_
#define __CUDA_CROP_H_


#include "cudaUtility.h"


/**
 * Crop a single-channel unsigned char 8-bit image.
 * The output should be smaller than the input in some dimension.
 */
cudaError_t cudaCrop( uint8_t* input, const dim3& inputSize, uint8_t* output, const dim3& outputSize );




#endif

