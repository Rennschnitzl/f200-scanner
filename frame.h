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
    Frame(cv::Mat CameraMatrix_depth, cv::Mat coeffs_depth, cv::Mat CameraMatrix_color, cv::Mat coeffs_color);

    // camera related
    cv::Mat cameraMatrix_depth;
    cv::Mat coefficients_depth;
    cv::Mat cameraMatrix_color;
    cv::Mat coefficients_color;

    // images
    cv::Mat belief;
    cv::Mat processedImageDepth;
    cv::Mat processedImageIR;


    // tracking
//    Eigen::Vector4f origin;
//    Eigen::Quaternionf orientation;
//    Eigen::Affine3f matrix;

    // point cloud
//    pcl::PointCloud<pcl::PointXYZI> cloud;
//    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudptr;
private:

};

#endif // FRAME_H
