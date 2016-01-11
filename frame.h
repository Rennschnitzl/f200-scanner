#ifndef FRAME_H
#define FRAME_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/common/common_headers.h>
//#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

class Frame
{
public:
    Frame(int width, int height, cv::Mat CameraMatrix, cv::Mat coeffs);

    // camera related
    cv::Mat cameraMatrix;
    cv::Mat coefficients;
    int width;
    int height;

    // images
    cv::Mat belief;
    cv::Mat processedImageDepth;
    cv::Mat processedImageIR;
    std::vector<cv::Mat> rawStackIR;
    std::vector<cv::Mat> rawStackDepth;

    // tracking
    Eigen::Vector4f origin;
    Eigen::Quaternionf orientation;
    Eigen::Affine3f matrix;

    // point cloud
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudptr;
private:

};

#endif // FRAME_H
