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

#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>


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

    /// **************
    /// TRIANGULATION
    /// **************
    // Normal estimation*
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud (cloud_filtered);
    n.setInputCloud (cloud_filtered);
    n.setSearchMethod (tree);
    n.setKSearch (20);
    n.compute (*normals);
    //* normals should not contain the point normals + surface curvatures

    // Concatenate the XYZ and normal fields*
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields (*cloud_filtered, *normals, *cloud_with_normals);
    //* cloud_with_normals = cloud + normals

    // Create search tree*
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud (cloud_with_normals);

    // Initialize objects
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    pcl::PolygonMesh triangles;

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius (0.025);

    // Set typical values for the parameters
    gp3.setMu (2.5);
    gp3.setMaximumNearestNeighbors (100);
    gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
    gp3.setMinimumAngle(M_PI/18); // 10 degrees
    gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
    gp3.setNormalConsistency(false);

    // Get result
    gp3.setInputCloud (cloud_with_normals);
    gp3.setSearchMethod (tree2);
    gp3.reconstruct (triangles);

    pcl::io::saveVTKFile ("mesh.vtk", triangles);

    // Additional vertex information
    std::vector<int> parts = gp3.getPartIDs();
    std::vector<int> states = gp3.getPointStates();

    /// **************
    /// /TRIANGULATION
    /// **************

    // add point cloud
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distribution(cloud_filtered, "intensity");
    viewer->addPointCloud<pcl::PointXYZI>(cloud_filtered,intensity_distribution,"frame");

    cv::waitKey(0);

    viewer->spin();

    return 0;
}
