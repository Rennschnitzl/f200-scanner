#ifndef CONVERTER_H
#define CONVERTER_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <pcl/common/common_headers.h>


//#include <highgui.h>
//#include <time.h>
//#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/io/io.h>
//#include <pcl/io/pcd_io.h>
//#include <pcl/point_types.h>
//#include <cmath>
//#include <boost/thread/thread.hpp>
//#include <iostream>
//#include <string.h>
//#include <linux/videodev2.h>
//#include <linux/uvcvideo.h>
//#include <linux/usb/video.h>
//#include <fcntl.h>
//#include <stdio.h>
//#include <sys/ioctl.h>
//#include <sys/mman.h>
//#include <errno.h>
//#include <unistd.h>

class Converter
{
public:
    Converter();
    static void analyseStack(std::vector<cv::Mat> &stack, cv::Mat &believe, cv::Mat &result);
    static void averageIR(const std::vector<cv::Mat> & IRstack, cv::Mat & ir_avg);
    static double calculateVariance(std::vector<int> var_values);
    static std::string type2str(int type);
    static void rescaleDepth(cv::InputArray in_in, int depth, cv::OutputArray out_out);
    static void depthTo3d(const cv::Mat& in_depth, const cv::Mat& K, pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, const cv::Mat& belief);
    static void undistortDepth(cv::Mat &in_depth, const cv::Mat &camMatrix, const cv::Mat &coeffs, cv::Mat &belief);
};

#endif // CONVERTER_H
