#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/conditional_removal.h>
//#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/filter.h>

/*
#include <pcl/surface/mls.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>
#include <pcl/point_types_conversion.h>
#include <pcl/surface/impl/mls.hpp> // MSL for PointXYZINormals
*/

#include "camerawrapper.h"
#include "frame.h"


#define FRAME_RECORD_SIZE 10

using namespace std;

// TODO move methods to helper class
// TODO get rid of code in main

float lastx = 0.0, lasty = 0.0, lastz = 0.0;
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

void statistical(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloudptr, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered)
{
    //    pcl::RadiusOutlierRemoval<pcl::PointXYZI> outrem;
    //    // build the filter
    //    outrem.setInputCloud(cloudptr);
    //    outrem.setRadiusSearch(0.15);
    //    outrem.setMinNeighborsInRadius (5);
    //    // apply filter
    //    outrem.filter (*cloud_filtered);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor;
    sor.setInputCloud (cloudptr);
    sor.setMeanK (30);
    sor.setStddevMulThresh (1.0);
    sor.filter (*cloud_filtered);
    std::cout << "filter removed " << (cloudptr->size() - cloud_filtered->size()) << " points" << std::endl;
}

/// create points by using the projection functions of librealsense
/// xyz are 3d coordinates in meters (e.g. 1.5 equals 1.5 meters)
/// rgb are the colors from the colorcam (neares point available - rounded)
/// a is the belief calculated in the creation of the frame
void fillCloudFromFrame(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, CameraWrapper cw, Frame image)
{
    // make sure it's empty
    cloud->clear();

    for(int y = 0; y < image.processedImageDepth.size().height; y++)
    {
        for (int x = 0; x < image.processedImageDepth.size().width; x++)
        {
            float distance = image.processedImageDepth.at<float>(y,x);
            if(distance != distance)
            {
                pcl::PointXYZRGBA point;
                point.x = std::numeric_limits<float>::quiet_NaN();
                point.y = std::numeric_limits<float>::quiet_NaN();
                point.z = std::numeric_limits<float>::quiet_NaN();
                point.r = std::numeric_limits<int>::quiet_NaN();
                point.g = std::numeric_limits<int>::quiet_NaN();
                point.b = std::numeric_limits<int>::quiet_NaN();
                point.a = std::numeric_limits<int>::quiet_NaN();
                cloud->push_back(point);
                continue;
            }
            pcl::PointXYZRGBA point;
            float depth[3] = {(float)x, (float)y, distance};
            float colorpx[2];
            cw.computePoints(depth, colorpx);
            point.x = -depth[0];
            point.y = -depth[1];
            point.z = depth[2];
            point.a = image.belief.at<uchar>(y,x);
            const int cx = (int)roundf(colorpx[0]), cy = (int)roundf(colorpx[1]);
            if(cx < 0 || cy < 0 || cx >= image.processedImageRGB.size().width || cy >= image.processedImageRGB.size().height)
            {
                point.b = 255;
                point.g = 255;
                point.r = 255;
            }else
            {
                point.b = image.processedImageRGB.at<cv::Vec3b>(cy,cx)[0];
                point.g = image.processedImageRGB.at<cv::Vec3b>(cy,cx)[1];
                point.r = image.processedImageRGB.at<cv::Vec3b>(cy,cx)[2];
            }
            cloud->push_back(point);
        }
    }
    cloud->is_dense = false;
    cloud->width = image.processedImageDepth.size().width;
    cloud->height = image.processedImageDepth.size().height;
}

/// remove uncertain points - filter by belief
/// also: passthrough is a dick (see: http://www.pcl-users.org/How-to-filter-based-on-color-using-PCL-td2791524.html )
/// condition removal sucks
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr filterBelief(int filterlimit, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr filtered (new pcl::PointCloud<pcl::PointXYZRGBA>);
    for(int y = 0; y < cloud->width; y++)
    {
        for (int x = 0; x < cloud->height; x++)
        {
            int position = y*cloud->height+x;
            if(cloud->at(position).a > filterlimit)
            {
                filtered->push_back(cloud->at(position));
            }
        }
    }
    return filtered;
}

int main()
{
    CameraWrapper cw(FRAME_RECORD_SIZE);

    Frame image = cw.record();
    cv::imshow("ir" , image.processedImageIR);
    cv::imshow("color" , image.processedImageRGB);

    cv::imshow("depth" , image.processedImageDepth);
    cv::imshow("belief", image.belief);

    cv::waitKey(0);

    /// convert a pointcloud
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
    fillCloudFromFrame(cloud, cw, image);

    std::cout << "ordered? " << cloud->isOrganized() << std::endl;

    /// filter points by belief
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr filtered = filterBelief(100, cloud);

    /// remove NaNs
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*filtered, *filtered, indices);
    std::cout << "NaNs removed " << indices.size() << std::endl;

    /// filter outliers
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_without_outliers (new pcl::PointCloud<pcl::PointXYZRGBA>);
    statistical(filtered,cloud_without_outliers);

    /// Display point cloud
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGBA> rgb(cloud_without_outliers);
    viewer->setBackgroundColor (0, 0, 0);
    viewer->initCameraParameters ();
    viewer->addCoordinateSystem(0.1, "Origin");
    viewer->addPointCloud<pcl::PointXYZRGBA>(cloud_without_outliers, rgb, "frame");
    viewer->spin();

    return 0;
}

/*
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


*/
