/*
 * http://github.com/dusty-nv/jetson-video
 */
 
#ifndef __JETSON_CAMERA_H__
#define __JETSON_CAMERA_H__


#include <stdint.h>


/**
 * Abstract camera interface.
 */
class Camera
{
public:
	/**
	 * Constructor
	 */
	Camera();
	
	/**
	 * Destructor
	 */
	virtual ~Camera();
		
	/**
	 * Open camera for streaming
	 */
	virtual bool Open() = 0;
	
	/**
	 * Close camera stream.
	 */
	virtual bool Close() = 0;
	
	/**
	 * Get width, in pixels, of camera image.
	 */
	inline uint32_t GetWidth() const					{ return mWidth; }
	
	/**
	 * Retrieve height, in pixels, of camera image.
	 */
	inline uint32_t GetHeight() const					{ return mHeight; }

	/**
 	 * Return the size in bytes of one line of the image.
	 */
	inline uint32_t GetPitch() const					{ return mPitch; }

	/**
	 * Return the bit depth per pixel.
	 */
	inline uint32_t GetDepth() const					{ return mDepth; }
	
	/**
	 * Return the size (in bytes) of one image frame.
	 */
	inline uint32_t GetSize() const						{ return mSize; }

protected:

	uint32_t mWidth;
	uint32_t mHeight;
	uint32_t mPitch;
	uint32_t mDepth;
	uint32_t mSize;
};


#endif