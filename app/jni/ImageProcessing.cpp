/*
*  ImageProcessing.cpp
* package com.cabatuan.fastcornersnative;
*/
#include <jni.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat * temp = NULL;

extern "C"
jboolean
Java_com_cabatuan_fastcornersnative_CameraPreview_ImageProcessing(
		JNIEnv* env, jobject thiz,
		jint width, jint height,
		jbyteArray NV21FrameData, jintArray outPixels)
{
	jbyte * pNV21FrameData = env->GetByteArrayElements(NV21FrameData, 0);
	jint * poutPixels = env->GetIntArrayElements(outPixels, 0);

	if ( temp == NULL )
    	{
    		temp = new Mat(height, width, CV_8UC1);
    	}

	Mat mGray(height, width, CV_8UC1, (unsigned char *)pNV21FrameData);
	Mat mResult(height, width, CV_8UC4, (unsigned char *)poutPixels);

	Mat FastImg = *temp;
    GoodFeaturesToTrackDetector harris_detector( 5000, 0.001, 5, 3, true );
    vector<KeyPoint> keypoints;

    FASTX( mGray, keypoints, 30, true, FastFeatureDetector::TYPE_9_16 );

    drawKeypoints( mGray, keypoints, FastImg, Scalar( 0, 255, 0));

	cvtColor(FastImg, mResult, CV_BGR2BGRA);

	env->ReleaseByteArrayElements(NV21FrameData, pNV21FrameData, 0);
	env->ReleaseIntArrayElements(outPixels, poutPixels, 0);
	return true;
}

//void FASTX(InputArray image, vector<KeyPoint>& keypoints, int threshold, bool nonmaxSuppression, int type)
//Parameters:
//image – grayscale image where keypoints (corners) are detected.
//keypoints – keypoints detected on the image.
//threshold – threshold on difference between intensity of the central pixel and pixels of a circle around this pixel.
//nonmaxSuppression – if true, non-maximum suppression is applied to detected corners (keypoints).
//type – one of the three neighborhoods as defined in the paper: FastFeatureDetector::TYPE_9_16,
// FastFeatureDetector::TYPE_7_12, FastFeatureDetector::TYPE_5_8