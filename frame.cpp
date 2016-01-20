#include "frame.h"

Frame::Frame(int width, int height, cv::Mat CameraMatrix, cv::Mat coeffs)
{
    this->width = width;
    this->height = height;
    this->cameraMatrix  = CameraMatrix;
    this->coefficients = coeffs;

    this->cloudptr = this->cloud.makeShared();
}

