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
#include <pcl/surface/mls.h>
#include <pcl/registration/icp.h>

#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>
#include <pcl/point_types_conversion.h>
#include <pcl/surface/impl/mls.hpp> // MSL for PointXYZINormals


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

void triangulation(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered)
{

    // convert XYZI to normal cloud
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_convert (new pcl::PointCloud <pcl::PointXYZ>);
    //pcl::PointXYZINormal

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
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::concatenateFields (*cloud_filtered, *normals, *cloud_with_normals);
    //* cloud_with_normals = cloud + normals

    // Create search tree*
    pcl::search::KdTree<pcl::PointXYZINormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZINormal>);
    tree2->setInputCloud (cloud_with_normals);

    // Initialize objects
    pcl::GreedyProjectionTriangulation<pcl::PointXYZINormal> gp3;
    pcl::PolygonMesh triangles;

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius (0.15);

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
}

void interpolate(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered, pcl::PointCloud<pcl::PointXYZINormal> mls_points)
{
    // Create a KD-Tree
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZI>);


    // Init object (second point type is for the normals, even if unused)
    pcl::MovingLeastSquares<pcl::PointXYZI, pcl::PointXYZINormal> mls;

    mls.setComputeNormals (true);

    // Set parameters
    mls.setInputCloud (cloud_filtered);
    mls.setPolynomialFit (true);
    mls.setSearchMethod (tree);
    mls.setSearchRadius (0.02);

    // Reconstruct
    mls.process (mls_points);
}

void statistical(pcl::PointCloud<pcl::PointXYZI>::Ptr cloudptr, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered)
{
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
}

Frame getMeAFuckingFrame(CameraWrapper &cw)
{

    std::cout << "hier kommt das voegelchen..." << std::endl;

    Frame frame1(IMAGE_WIDTH, IMAGE_HEIGHT, cw.getCameraMatrix(),cw.getCoeffs());

    // RECORD & PROCESS
    cw.recordStack(RECORD_STACK_SIZE,frame1.rawStackIR, frame1.rawStackDepth);
    Converter::analyseStack(frame1.rawStackDepth, frame1.belief, frame1.processedImageDepth);
    Converter::averageIR(frame1.rawStackIR, frame1.processedImageIR);
    Converter::undistortDepth(frame1.processedImageDepth, frame1.cameraMatrix, frame1.coefficients, frame1.belief);

    // CONVERT TO CLOUD
    Converter::depthTo3d(frame1.processedImageDepth,cw.getCameraMatrix(),frame1.cloudptr, frame1.belief);

    return frame1;
}

int main()
{
    /// **************
    /// Init viewer
    /// **************
    //Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->registerPointPickingCallback(pp_callback, (void*)&viewer);
    viewer->setBackgroundColor (0, 0, 0);
    viewer->initCameraParameters ();
    viewer->addCoordinateSystem(1.0, "Origin");
    pcl::PointCloud<pcl::PointXYZI>::Ptr dummy (new pcl::PointCloud<pcl::PointXYZI>);
    viewer->addPointCloud<pcl::PointXYZI> (dummy, "frame1");
    viewer->addPointCloud<pcl::PointXYZI> (dummy, "frame2");


    /// **************
    /// Init other stuff
    /// **************
    CameraWrapper cw(0);
    std::vector<Frame> framevector;
    Frame savedFrame(IMAGE_WIDTH, IMAGE_HEIGHT, cw.getCameraMatrix(),cw.getCoeffs());

    /// **************
    /// Main Loop
    /// **************

    Frame temporary_frame = getMeAFuckingFrame(cw);

    // DISPLAY
    display(temporary_frame);

    cv::waitKey(0);


    /// **************
    /// Statistical outlier removal
    /// **************
//    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZI>);
//    statistical(cloudptr, cloud_filtered);

    /// **************
    /// Triangulation
    /// writes vtk mesh to disk
    /// **************
    //triangulation(cloud_filtered);


    /// **************
    /// Interpolation
    /// **************
//    pcl::PointCloud<pcl::PointXYZINormal> mls_points;
//    pcl::PointCloud<pcl::PointXYZINormal>::Ptr mls_ptr = mls_points.makeShared();

//    interpolate(cloud_filtered, mls_points);
//    viewer->addPointCloud<pcl::PointXYZINormal>(mls_ptr, "frame");


    /// **************
    /// Display in Viewer
    /// **************
    // add point cloud
//    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> intensity_distribution(cloudptr, "intensity");
//    viewer->addPointCloud<pcl::PointXYZI>(cloudptr,intensity_distribution,"framei");
//    viewer->addPointCloud<pcl::PointXYZI>(cloudptr2,"target");
//    viewer->addPointCloud<pcl::PointXYZI>(Finalptr,"frame");

    while(true)
    {
        // a => stop; rest => next frame
        int key = cv::waitKey(0);
        /// strange opencv workaround
        /// see: http://stackoverflow.com/questions/9172170/python-opencv-cv-waitkey-spits-back-weird-output-on-ubuntu-modulo-256-maps-corre
        key -= 0x100000;
        std::cout << key << std::endl;
        // "a"
        if(key == 97)
            break;
//        if(key == 119)
//        {
//            viewer->updateCoordinateSystemPose("marker", mpose.inverse());
//            viewer->updatePointCloud(fr.getCloudPointer(), "preview");
//            viewer->updatePointCloudPose("preview", mpose.inverse());
//            viewer->spin();
//        }
        // "s"
        if(key == 115)
        {
            savedFrame = temporary_frame;
            temporary_frame = getMeAFuckingFrame(cw);
            display(temporary_frame);
        }
//        // "space"
        if(key == 32)
        {
            /// **************
            /// ICP
            /// **************
            std::cout << "starting ICP" << std::endl;
            pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
            icp.setInputSource(temporary_frame.cloudptr);
            //icp.setInputCloud(cloudptr);
            icp.setInputTarget(savedFrame.cloudptr);
            pcl::PointCloud<pcl::PointXYZI> Final;
            icp.align(Final);
            std::cout << "has converged:" << icp.hasConverged() << " score: " <<
                         icp.getFitnessScore() << std::endl;
            std::cout << icp.getFinalTransformation() << std::endl;
            pcl::PointCloud<pcl::PointXYZI>::Ptr Finalptr = Final.makeShared();

            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> single_color(Finalptr, 0, 255, 0);
            viewer->updatePointCloud<pcl::PointXYZI>(Finalptr, single_color ,"frame1");
            viewer->updatePointCloud<pcl::PointXYZI>(savedFrame.cloudptr,"frame2");
            viewer->spin();
        }
        // "p"
        if(key == 112)
        {
            temporary_frame = getMeAFuckingFrame(cw);
            display(temporary_frame);
        }
        // "i"
        if(key == 112)
        {

        }
        // "q"
        if(key == 113)
        {
            break;
        }
        // "e"
        if(key == 101)
        {

        }
    }
    return 0;
}
