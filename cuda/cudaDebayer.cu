/*
 * http://github.com/dusty-nv/jetson-video
 */

#include "cudaDebayer.h"
#include "cudaMath.h"


texture<uchar1, 2, cudaReadModeElementType> bayerTex;
//texture<ushort1, 2, cudaReadModeNormalizedFloat> bayerTex;



template<int xBayerOffset, int yBayerOffset>
__device__ float4 bayerToRgba( const int x, const int y )
{
	#define fetch(xOffset, yOffset) 	tex2D(bayerTex, x + xOffset, y + yOffset).x

	float C = fetch(0,0);
	const float4 kC = make_float4(4.0f, 6.0f, 5.0f, 5.0f) / 8.0f;


    // Determine which of four types of pixels we are on.
	// note:  if first red pixel offset is not (0,0), adjust that here
	const int2 alternate = make_int2((x + xBayerOffset) % 2, (y + yBayerOffset) % 2);

	

	float4 Dvec = make_float4( fetch(-1,-1),
							   fetch(-1, 1),
							   fetch( 1,-1),
							   fetch( 1, 1) );


	float4 PATTERN = make_float4( kC.x * C,
								  kC.y * C,
								  kC.z * C,
								  kC.z * C );

	// sum Dvec
	Dvec.x += Dvec.y + Dvec.z + Dvec.w;


	// sample other pixels in 5x5 neighborhood
	float4 value = make_float4( fetch( 0,-2),
							    fetch( 0,-1),
							    fetch(-1, 0),
							    fetch(-2, 0) );

	float4 temp  = make_float4( fetch(0,2),
								fetch(0,1),
								fetch(2,0),
								fetch(1,0) );

    // Even the simplest compilers should be able to constant-fold these to avoid the division.
	#define kWeights(x,y,z,w)		make_float4(x/8.0f, y/8.0f, z/8.0f, w/8.0f)

	const float4 kA = kWeights(-1.0f,-1.5f, 0.5f,-1.0f);
	const float4 kB = kWeights( 2.0f, 0.0f, 0.0f, 4.0f);
	const float4 kD = kWeights( 0.0f, 2.0f,-1.0f,-1.0f);
	const float4 kE = kWeights(-1.0f,-1.5f,-1.0f, 0.5f);	// (kA.xywz)
	const float4 kF = kWeights( 2.0f, 0.0f, 4.0f, 0.0f);	// (kB.xywz)

    
	value += temp;

    
    // There are five filter patterns (identity, cross, checker,
    // theta, phi).  Precompute the terms from all of them and then
    // use swizzles to assign to color channels. 
    //
    // Channel   Matches
    //   x       cross   (e.g., EE G)
    //   y       checker (e.g., EE B)
    //   z       theta   (e.g., EO R)
    //   w       phi     (e.g., EO R)
    
    #define A (value.x)
    #define B (value.y)
    #define D (Dvec.x)
    #define E (value.z)
    #define F (value.w)
    
	PATTERN.y += kD.y * D;
	PATTERN.z += kD.z * D;
	PATTERN.w += kD.z * D;

	PATTERN.x += kA.x * A + kE.x * E;
	PATTERN.y += kA.y * A + kE.y * E;
	PATTERN.z += kA.z * A + kE.x * E;
	PATTERN.w += kA.x * A + kE.w * E;

	PATTERN.x += kB.x * B;
	PATTERN.w += kB.w * B;

	PATTERN.x += kF.x * F;
	PATTERN.z += kF.z * F;


	// generate output
	const float4 px = (alternate.y == 0) ?
        	((alternate.x == 0) ? make_float4(C, PATTERN.x, PATTERN.y, 1.0f)  :	// even row, even column
								  make_float4(PATTERN.z, C, PATTERN.w, 1.0f)) :	// even row, odd column
        	((alternate.x == 0) ? make_float4(PATTERN.w, C, PATTERN.z, 1.0f)  :	// odd row, even column
								  make_float4(PATTERN.y, PATTERN.x, C, 1.0f));	// odd row, odd column
	
	/*const float4 px = (alternate.y == 0) ?
        	((alternate.x == 0) ? make_float4(C, C, C, 1.0f)  :	// even row, even column
								  make_float4(C, C,C, 1.0f)) :	// even row, odd column
        	((alternate.x == 0) ? make_float4(C, C, C, 1.0f)  :	// odd row, even column
								  make_float4(C, C, C, 1.0f));*/	

	return px;
}


// gpuBayerToRGBA kernel
template<int xBayerOffset, int yBayerOffset>
__global__ void gpuBayerToRGBA( uchar4* output, int alignedWidth, float2 scale )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const float4 px = bayerToRgba<xBayerOffset, yBayerOffset>(x * scale.x, y * scale.y);

	output[y*alignedWidth+x] = make_uchar4(min(255.0f,px.x), min(255.0f,px.y), min(255.0f,px.z), 255);
}


// gpuBayerToRGBA_float kernel
template<int xBayerOffset, int yBayerOffset>
__global__ void gpuBayerToRGBA_float( float4* output, int alignedWidth, float2 scale )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	const float4 rgba = bayerToRgba<xBayerOffset, yBayerOffset>(x * scale.x, y * scale.y);

	output[y*alignedWidth+x] = make_float4(rgba.x / 255.0f, rgba.y / 255.0f, rgba.z / 255.0f, 1.0f);
}


__device__ float4 bayerToRgba_no_tex( uint8_t* input, int inputAlignedWidth, const int x, const int y )
{
	#define fetchg(xOffset, yOffset) 	input[(y + yOffset) * inputAlignedWidth + x + xOffset]

	float C = fetchg(0,0);
	const float4 kC = make_float4(4.0f, 6.0f, 5.0f, 5.0f) / 8.0f;


    // Determine which of four types of pixels we are on.
	// note:  if first red pixel offset is not (0,0), adjust that here
	const int2 alternate = make_int2((x + 0) % 2, (y + 0) % 2);

	

	float4 Dvec = make_float4( fetchg(-1,-1),
							   fetchg(-1, 1),
							   fetchg( 1,-1),
							   fetchg( 1, 1) );


	float4 PATTERN = make_float4( kC.x * C,
								  kC.y * C,
								  kC.z * C,
								  kC.z * C );

	// sum Dvec
	Dvec.x += Dvec.y + Dvec.z + Dvec.w;


	// sample other pixels in 5x5 neighborhood
	float4 value = make_float4( fetchg( 0,-2),
							    fetchg( 0,-1),
							    fetchg(-1, 0),
							    fetchg(-2, 0) );

	float4 temp  = make_float4( fetchg(0,2),
								fetchg(0,1),
								fetchg(2,0),
								fetchg(1,0) );

    // Even the simplest compilers should be able to constant-fold these to avoid the division.
	#define kWeights(x,y,z,w)		make_float4(x/8.0f, y/8.0f, z/8.0f, w/8.0f)

	const float4 kA = kWeights(-1.0f,-1.5f, 0.5f,-1.0f);
	const float4 kB = kWeights( 2.0f, 0.0f, 0.0f, 4.0f);
	const float4 kD = kWeights( 0.0f, 2.0f,-1.0f,-1.0f);
	const float4 kE = kWeights(-1.0f,-1.5f,-1.0f, 0.5f);	// (kA.xywz)
	const float4 kF = kWeights( 2.0f, 0.0f, 4.0f, 0.0f);	// (kB.xywz)

    
	value += temp;

    
    // There are five filter patterns (identity, cross, checker,
    // theta, phi).  Precompute the terms from all of them and then
    // use swizzles to assign to color channels. 
    //
    // Channel   Matches
    //   x       cross   (e.g., EE G)
    //   y       checker (e.g., EE B)
    //   z       theta   (e.g., EO R)
    //   w       phi     (e.g., EO R)
    
    #define A (value.x)
    #define B (value.y)
    #define D (Dvec.x)
    #define E (value.z)
    #define F (value.w)
    
	PATTERN.y += kD.y * D;
	PATTERN.z += kD.z * D;
	PATTERN.w += kD.z * D;

	PATTERN.x += kA.x * A + kE.x * E;
	PATTERN.y += kA.y * A + kE.y * E;
	PATTERN.z += kA.z * A + kE.x * E;
	PATTERN.w += kA.x * A + kE.w * E;

	PATTERN.x += kB.x * B;
	PATTERN.w += kB.w * B;

	PATTERN.x += kF.x * F;
	PATTERN.z += kF.z * F;


	// generate output
	const float4 px = (alternate.y == 0) ?
        	((alternate.x == 0) ? make_float4(C, PATTERN.x, PATTERN.y, 1.0f)  :	// even row, even column
								  make_float4(PATTERN.z, C, PATTERN.w, 1.0f)) :	// even row, odd column
        	((alternate.x == 0) ? make_float4(PATTERN.w, C, PATTERN.z, 1.0f)  :	// odd row, even column
								  make_float4(PATTERN.y, PATTERN.x, C, 1.0f));	// odd row, odd column
	
	/*const float4 px = (alternate.y == 0) ?
        	((alternate.x == 0) ? make_float4(C, C, C, 1.0f)  :	// even row, even column
								  make_float4(C, C,C, 1.0f)) :	// even row, odd column
        	((alternate.x == 0) ? make_float4(C, C, C, 1.0f)  :	// odd row, even column
								  make_float4(C, C, C, 1.0f));*/	

	return px;
}


// gpuBayerToRGBA kernel
__global__ void gpuBayerToRGBA_no_tex( uint8_t* input, int inputAlignedWidth, uchar4* output, int alignedWidth, float2 scale, int width, int height )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	float4 px = make_float4(0,0,0,255);

	if( x > 5 && y > 5 && x < (width-5) && y < (height-5) )
		px = bayerToRgba_no_tex(input, inputAlignedWidth, x * scale.x, y * scale.y);

	output[y*alignedWidth+x] = make_uchar4(min(255.0f,px.x), min(255.0f,px.y), min(255.0f,px.z), 255);
}


// cudaBayerToRGBA
cudaError_t cudaBayerToRGBA( uint8_t* input, size_t inputPitch, size_t inputWidth, size_t inputHeight, 
					    uchar4* output, size_t outputPitch, size_t outputWidth, size_t outputHeight,
					    cudaBayerType colorspace )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputPitch == 0 || outputPitch == 0 || inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( (float)inputWidth / (float)outputWidth,
							    (float)inputHeight / (float)outputHeight );

	const int outputAlignedWidth = outputPitch / sizeof(uint32_t);

	const dim3 blockDim(8, 8);
	const dim3 gridDim(outputWidth/blockDim.x, outputHeight/blockDim.y);
	
	// Set texture parameters (default)
	bayerTex.addressMode[0] = cudaAddressModeClamp;
	bayerTex.addressMode[1] = cudaAddressModeClamp;
	bayerTex.filterMode	    = cudaFilterModePoint;		// no bilinear filtering
	bayerTex.normalized	    = false;					// do not normalize coordinates

	// bind textures
	if( CUDA_FAILED(cudaBindTexture2D(NULL, bayerTex, input, cudaCreateChannelDesc<uint8_t>(), inputWidth, inputHeight, inputPitch)))
	{
		gpuBayerToRGBA_no_tex<<<gridDim, blockDim>>>(input, inputPitch, output, outputAlignedWidth, scale, inputWidth, inputHeight);
	}
	else
	{
		if( colorspace == CUDA_BAYER_RG )		gpuBayerToRGBA<0,0><<<gridDim, blockDim>>>(output, outputAlignedWidth, scale);
		else if( colorspace == CUDA_BAYER_GR )	gpuBayerToRGBA<1,0><<<gridDim, blockDim>>>(output, outputAlignedWidth, scale);
		else if( colorspace == CUDA_BAYER_GB )	gpuBayerToRGBA<0,1><<<gridDim, blockDim>>>(output, outputAlignedWidth, scale);
		else if( colorspace == CUDA_BAYER_BG )	gpuBayerToRGBA<1,1><<<gridDim, blockDim>>>(output, outputAlignedWidth, scale);

		CUDA(cudaUnbindTexture(bayerTex));
	}

	return CUDA(cudaGetLastError());
}


// cudaBayerToRGBA
cudaError_t cudaBayerToRGBA( uint8_t* input, size_t inputPitch, uchar4* output, size_t outputPitch, size_t width, size_t height, cudaBayerType colorspace )
{
	return cudaBayerToRGBA(input, inputPitch, width, height, output, outputPitch, width, height, colorspace);
}


// cudaBayerToRGBA
cudaError_t cudaBayerToRGBA( uint8_t* input, uchar4* output, size_t width, size_t height, cudaBayerType colorspace )
{
	return cudaBayerToRGBA(input, width * sizeof(uint8_t), width, height, output, width * sizeof(uchar4), width, height, colorspace);
}




// cudaBayerToRGBA
cudaError_t cudaBayerToRGBA( uint8_t* input, size_t inputPitch, size_t inputWidth, size_t inputHeight, 
					    float4* output, size_t outputPitch, size_t outputWidth, size_t outputHeight,
					    cudaBayerType colorspace )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputPitch == 0 || outputPitch == 0 || inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( (float)inputWidth / (float)outputWidth,
							    (float)inputHeight / (float)outputHeight );

	const int outputAlignedWidth = outputPitch / sizeof(uint32_t);

	const dim3 blockDim(8, 8);
	const dim3 gridDim(outputWidth/blockDim.x, outputHeight/blockDim.y);
	
	// Set texture parameters (default)
	bayerTex.addressMode[0] = cudaAddressModeClamp;
	bayerTex.addressMode[1] = cudaAddressModeClamp;
	bayerTex.filterMode	    = cudaFilterModePoint;		// no bilinear filtering
	bayerTex.normalized	    = false;					// do not normalize coordinates

	// bind textures
	if( CUDA_FAILED(cudaBindTexture2D(NULL, bayerTex, input, cudaCreateChannelDesc<uint8_t>(), inputWidth, inputHeight, inputPitch)))
	{
		return CUDA(cudaGetLastError());
	}
	else
	{
		if( colorspace == CUDA_BAYER_RG )	   gpuBayerToRGBA_float<0,0><<<gridDim, blockDim>>>(output, outputAlignedWidth, scale);
		else if( colorspace == CUDA_BAYER_GR ) gpuBayerToRGBA_float<1,0><<<gridDim, blockDim>>>(output, outputAlignedWidth, scale);
		else if( colorspace == CUDA_BAYER_GB ) gpuBayerToRGBA_float<0,1><<<gridDim, blockDim>>>(output, outputAlignedWidth, scale);
		else if( colorspace == CUDA_BAYER_BG ) gpuBayerToRGBA_float<1,1><<<gridDim, blockDim>>>(output, outputAlignedWidth, scale);

		CUDA(cudaUnbindTexture(bayerTex));
	}

	return CUDA(cudaGetLastError());
}


// cudaBayerToRGBA
cudaError_t cudaBayerToRGBA( uint8_t* input, size_t inputPitch, float4* output, size_t outputPitch, size_t width, size_t height, cudaBayerType colorspace )
{
	return cudaBayerToRGBA(input, inputPitch, width, height, output, outputPitch, width, height, colorspace);
}


// cudaBayerToRGBA
cudaError_t cudaBayerToRGBA( uint8_t* input, float4* output, size_t width, size_t height, cudaBayerType colorspace )
{
	return cudaBayerToRGBA(input, width * sizeof(uint8_t), width, height, output, width * sizeof(float4), width, height, colorspace);
}


