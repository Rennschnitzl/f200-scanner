#ifndef CAMERAWRAPPER_H
#define CAMERAWRAPPER_H

#include <opencv2/opencv.hpp>
#include "realsense-core/cameradriver.h"

#include <memory>

class CameraWrapper
{
public:
    CameraWrapper(int filtermode);
    ~CameraWrapper();
    cv::Mat getCoeffs();
    cv::Mat getCameraMatrix();
    void displayCameraProperties();
    void recordStack(int frames, std::vector<cv::Mat> &irlist, std::vector<cv::Mat> &depthlist);
	
private:
    void loadCameraMatrix();
    void clearBuffer();
	
    cv::Mat convertIRtoCV(std::vector<std::vector<u_int8_t> > ir);
    cv::Mat convertDepthtoCV(std::vector<std::vector<u_int16_t> > depth);
    
private:	
    int height;
    int width;
    double factor;
    double offset;
	
    CameraDriver* depthcam;
	
	timespec slptm;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
};

#endif // CAMERAWRAPPER_H
