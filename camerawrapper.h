#ifndef CAMERAWRAPPER_H
#define CAMERAWRAPPER_H

#include <opencv2/opencv.hpp>
#include <librealsense/rs.h>
#include <librealsense/rsutil.h>
#include <stdio.h>
#include "frame.h"

//#include <memory>

class CameraWrapper
{
public:
    CameraWrapper(int frames);
    ~CameraWrapper();
    Frame record();
    void setStackSize(int frames);
private:
    rs_intrinsics depth_intrin;
    rs_intrinsics color_intrin;
    rs_intrinsics ir_intrin;
    rs_extrinsics depth_to_color;
    rs_error * e;
    rs_context * ctx;
    rs_device * dev;
    cv::Mat getCameraMatrix_depth() const;
    cv::Mat getCameraMatrix_color() const;
    cv::Mat getCoefficients_depth() const;
    cv::Mat getCoefficients_color() const;
    int framesToRecord;
    void check_error();
    float depthScale;
    void convertIntrinsicToOpenCV(const rs_intrinsics & in_intrinsics, cv::Mat & out_cammat, cv::Mat & out_coeffs);
    rs_intrinsics getIntrinsicsFromOpenCV(const cv::Mat & in_cammat, const cv::Mat & in_coeffs);
    // TODO converterfunctions for different points
    //cv::Mat convertIRtoCV(std::vector<std::vector<u_int8_t> > ir);
    //cv::Mat convertDepthtoCV(std::vector<std::vector<u_int16_t> > depth);
};

#endif // CAMERAWRAPPER_H
