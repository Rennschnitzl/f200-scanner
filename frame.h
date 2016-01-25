#ifndef FRAME_H
#define FRAME_H

#include <iostream>
#include <opencv2/opencv.hpp>
//#include <pcl/common/common_headers.h>
//#include <pcl/point_types.h>
//#include <pcl/common/transforms.h>

class Frame
{
public:
    Frame();

    // camera related
    cv::Mat cameraMatrix_depth;
    cv::Mat coefficients_depth;
    cv::Mat cameraMatrix_color;
    cv::Mat coefficients_color;
    cv::Mat cameraMatrix_ir;
    cv::Mat coefficients_ir;

    // images
    cv::Mat belief;
    cv::Mat processedImageDepth;
    cv::Mat processedImageIR;
    cv::Mat processedImageRGB;


    // tracking
//    Eigen::Vector4f origin;
//    Eigen::Quaternionf orientation;
//    Eigen::Affine3f matrix;

    // point cloud
//    pcl::PointCloud<pcl::PointXYZI> cloud;
//    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudptr;

    // getter and setter for calibration
//    cv::Mat getCameraMatrix_depth() const;
//    void setCameraMatrix_depth(const cv::Mat &value);

//    cv::Mat getCoefficients_depth() const;
//    void setCoefficients_depth(const cv::Mat &value);

//    cv::Mat getCameraMatrix_color() const;
//    void setCameraMatrix_color(const cv::Mat &value);

//    cv::Mat getCoefficients_color() const;
//    void setCoefficients_color(const cv::Mat &value);

//    cv::Mat getCameraMatrix_ir() const;
//    void setCameraMatrix_ir(const cv::Mat &value);

//    cv::Mat getCoefficients_ir() const;
//    void setCoefficients_ir(const cv::Mat &value);

private:

};

#endif // FRAME_H
