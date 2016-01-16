#include <iostream>
#include <opencv2/opencv.hpp>
#include "camerawrapper.h"
#include "frame.h"
#include "converter.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>


#define RECORD_STACK_SIZE 5
#define IMAGE_HEIGHT 480
#define IMAGE_WIDTH 640

using namespace std;

void display(Frame frame1)
{
    cv::Mat depth_cv_8;
    cv::Mat depth_cv_rgb(IMAGE_HEIGHT,IMAGE_WIDTH,CV_8UC3);

    frame1.processedImageDepth.convertTo(depth_cv_8,CV_8U,1.0/256.0);

    cv::cvtColor(depth_cv_8,depth_cv_rgb,CV_GRAY2RGB);

    cv::Mat ir_cv_rgb(IMAGE_HEIGHT,IMAGE_WIDTH,CV_8UC3);

    cv::cvtColor(frame1.processedImageIR,ir_cv_rgb,CV_GRAY2RGB);

    cv::imshow("depth", depth_cv_rgb);
    cv::imshow("ir", ir_cv_rgb);
}

int main()
{
    CameraWrapper cw(0);
    Frame frame1(IMAGE_WIDTH, IMAGE_HEIGHT, cw.getCameraMatrix(),cw.getCoeffs());

    // RECORD & PROCESS
    cw.recordStack(RECORD_STACK_SIZE,frame1.rawStackIR, frame1.rawStackDepth);
    Converter::analyseStack(frame1.rawStackDepth, frame1.belief, frame1.processedImageDepth);
    Converter::averageIR(frame1.rawStackIR, frame1.processedImageIR);
    cv::Mat rview;
    Converter::undistortDepth(frame1.processedImageDepth, rview, frame1.cameraMatrix, frame1.coefficients);

    // SAVE IMAGES TO DISK
    imwrite( "processed.png", frame1.processedImageDepth );
    imwrite( "undistorted.png", rview );

    // CONVERT TO CLOUD
    pcl::PointCloud<pcl::PointXYZ> cloudpro;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudptrpro;
    cloudptrpro = cloudpro.makeShared();
    Converter::depthTo3d(frame1.processedImageDepth,cw.getCameraMatrix(),cloudptrpro);

    pcl::PointCloud<pcl::PointXYZ> cloudund;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudptrund;
    cloudptrund = cloudund.makeShared();
    Converter::depthTo3d(rview,cw.getCameraMatrix(),cloudptrund);

    // DISPLAY
    display(frame1);
    cv::imshow("undistorted", rview);


    // PCL VIEWER
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2 (new pcl::PointCloud<pcl::PointXYZ>);
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->initCameraParameters ();
    viewer->addPointCloud<pcl::PointXYZ> (cloudptrpro, "processed");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloudptrund, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cloudptrund, single_color, "undistorted");

    viewer->addCoordinateSystem(1.0, "marker");

    cv::waitKey(0);

    viewer->spin();

    return 0;
}
