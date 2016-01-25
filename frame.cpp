#include "frame.h"


Frame::Frame(cv::Mat CameraMatrix_depth, cv::Mat coeffs_depth, cv::Mat CameraMatrix_color, cv::Mat coeffs_color)
{
    this->cameraMatrix_color = CameraMatrix_color;
    this->cameraMatrix_depth = CameraMatrix_depth;
    this->coefficients_color = coeffs_color;
    this->coefficients_depth = coeffs_depth;
}
