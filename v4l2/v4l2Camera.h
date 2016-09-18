/*
 * inference-101
 */

#ifndef __V4L2_CAPTURE_H
#define __V4L2_CAPTURE_H

#include "Camera.h"
#include <linux/videodev2.h>

#include <stdint.h>
#include <string>
#include <vector>



struct v4l2_mmap
{
	struct v4l2_buffer buf;
	void*  ptr;
};


/**
 * Video4Linux2 camera capture streaming.
 */
class v4l2Camera : public Camera
{
public:	
	/**
	 * Create V4L2 interface
	 * @param path Filename of the video device (e.g. /dev/video0)
	 */
	static v4l2Camera* Create( const char* device_path );

	/**
	 * Destructor
	 */	
	virtual ~v4l2Camera();

	/**
 	 * Start streaming
	 */
	virtual bool Open();

	/**
	 * Stop streaming
	 */
	virtual bool Close();

	/**
	 * Return the next image.
	 */
	void* Capture( size_t timeout=0 );

private:

	v4l2Camera( const char* device_path );

	bool init();
	bool initCaps();
	bool initFormats();
	bool initStream();

	bool initUserPtr();
	bool initMMap();

	int 	mFD;
	int	    mRequestFormat;
	uint32_t mRequestWidth;
	uint32_t mRequestHeight;

	v4l2_mmap* mBuffersMMap;
	size_t mBufferCountMMap;

	std::vector<v4l2_fmtdesc> mFormats;
	std::string mDevicePath;
};


#endif


