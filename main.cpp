#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "camerawrapper.h"
#include "frame.h"
#include "converter.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>


#define RECORD_STACK_SIZE 40
#define IMAGE_HEIGHT 480
#define IMAGE_WIDTH 640

float lastx = 0.0, lasty = 0.0, lastz = 0.0;

using namespace std;

void pp_callback(const pcl::visualization::PointPickingEvent& event, void* viewer_void)
{
    //std::cout << "Picking event active" << std::endl;
    if(event.getPointIndex()!=-1)
    {
        float x,y,z;
        event.getPoint(x,y,z);
        // http://www.calculatorsoup.com/calculators/geometry-solids/distance-two-points.php ;)
        std::cout << x<< "," << y<<"," << z << std::endl;

        long double distance=sqrt(pow(x-lastx,2.0)+pow(y-lasty,2.0)+pow(z-lastz,2.0));
        lastx = x;
        lasty = y;
        lastz = z;
        std::cout << "euclidean Distance to last point: " << distance << std::endl;
    }
}

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
    cv::imshow("belief", frame1.belief);
}

int main()
{
    CameraWrapper cw(0);
    Frame frame1(IMAGE_WIDTH, IMAGE_HEIGHT, cw.getCameraMatrix(),cw.getCoeffs());

    // RECORD & PROCESS
    cw.recordStack(RECORD_STACK_SIZE,frame1.rawStackIR, frame1.rawStackDepth);
    Converter::analyseStack(frame1.rawStackDepth, frame1.belief, frame1.processedImageDepth);
    Converter::averageIR(frame1.rawStackIR, frame1.processedImageIR);
    Converter::undistortDepth(frame1.processedImageDepth, frame1.cameraMatrix, frame1.coefficients, frame1.belief);

    // CONVERT TO CLOUD
    pcl::PointCloud<pcl::PointXYZI> cloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloudptr;
    cloudptr = cloud.makeShared();
    Converter::depthTo3d(frame1.processedImageDepth,cw.getCameraMatrix(),cloudptr, frame1.belief);

    // DISPLAY
    display(frame1);

    // PCL VIEWER
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->registerPointPickingCallback(pp_callback, (void*)&viewer);
    viewer->setBackgroundColor (0, 0, 0);
    viewer->initCameraParameters ();
    viewer->addCoordinateSystem(1.0, "Origin");

    // filter cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZI>);
//    pcl::RadiusOutlierRemoval<pcl::PointXYZI> outrem;
//    // build the filter
//    outrem.setInputCloud(cloudptr);
//    outrem.setRadiusSearch(0.15);
//    outrem.setMinNeighborsInRadius (5);
//    // apply filter
//    outrem.filter (*cloud_filtered);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud (cloudptr);
    sor.setMeanK (50);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud_filtered);
    std::cout << "filter removed " << (cloudptr->size() - cloud_filtered->size()) << " points" << std::endl;

    // add point cloud
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distribution(cloud_filtered, "intensity");
    viewer->addPointCloud<pcl::PointXYZI>(cloud_filtered,intensity_distribution,"frame");

    cv::waitKey(0);

    viewer->spin();

    return 0;
}
