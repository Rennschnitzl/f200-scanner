#include "tracker.h"

Tracker::Tracker(double markersize, std::string boardfile)
{
    TheBoardConfigFile = boardfile;
    TheMarkerSize = markersize;

    TheBoardConfig.readFromFile(TheBoardConfigFile);

    ThresParam1 = 10.0;
    ThresParam2 = 13.0;

    debugmode = false;
}

/// uses rgb image and camera matrix to compute the markerposition
void Tracker::getTransformation(Frame &input)
{
    // fill with camera parameters saved in the frame
    TheCameraParameters.setParams(input.cameraMatrix_color, input.coefficients_color, cv::Size(1920,1080));

    // read from file
/*    std::string TheIntrinsicFile = "calibration.yml";
    TheCameraParameters.readFromXMLFile(TheIntrinsicFile);
    TheCameraParameters.resize(cv::Size(1920,1080)); */

    // init
    // DESIGN move this to initialization step with a fixed camera?
    TheBoardDetector.setParams(TheBoardConfig, TheCameraParameters, TheMarkerSize);
    TheBoardDetector.getMarkerDetector().setThresholdParams(ThresParam1, ThresParam2);
    TheBoardDetector.getMarkerDetector().setCornerRefinementMethod(aruco::MarkerDetector::HARRIS);
    TheBoardDetector.set_repj_err_thres(1.5);

    // run detection
    float probDetect = TheBoardDetector.detect(input.processedImageRGB);

    // show image with rgb input and detected markers
    if(debugmode){
        // draw detected markers
        cv::Mat TheInputImageCopy;
        input.processedImageRGB.copyTo(TheInputImageCopy);
        for (unsigned int i = 0; i < TheBoardDetector.getDetectedMarkers().size(); i++)
            TheBoardDetector.getDetectedMarkers()[i].draw(TheInputImageCopy, cv::Scalar(0, 0, 255), 1);

        // draw board axis
        if (TheCameraParameters.isValid()) {
            if (probDetect > 0.2) {
                aruco::CvDrawingUtils::draw3dAxis(TheInputImageCopy, TheBoardDetector.getDetectedBoard(), TheCameraParameters);
                aruco::CvDrawingUtils::draw3dCube(TheInputImageCopy, TheBoardDetector.getDetectedBoard(),TheCameraParameters);
                //aruco::CvDrawingUtils::draw3dBoardCube( TheInputImageCopy,TheBoardDetected,TheIntriscCameraMatrix,TheDistorsionCameraParams);
            }
        }
        cv::imshow("in", TheInputImageCopy);
        cv::imshow("thres", TheBoardDetector.getMarkerDetector().getThresholdedImage());
        cv::waitKey(1);
    }

    aruco::Board detBoard = TheBoardDetector.getDetectedBoard();
    input.transformMarker = createMatrixfromVectors(detBoard.Rvec, detBoard.Tvec);
    input.transformMarker*input.depth_to_color_transform.inverse();
    input.trackingprobability = probDetect;
}

double Tracker::getThresParam1() const
{
    return ThresParam1;
}

void Tracker::setThresParam1(double value)
{
    ThresParam1 = value;
}

double Tracker::getThresParam2() const
{
    return ThresParam2;
}

void Tracker::setThresParam2(double value)
{
    ThresParam2 = value;
}

/// creates Affine3f Matrix from rotation- and translationvectors computed by aruco
Eigen::Affine3f Tracker::createMatrixfromVectors(const cv::Mat &rvec, const cv::Mat &tvec)
{
    /// http://www.cplusplus.com/reference/iomanip/setprecision/
    std::cout << std::fixed << std::setprecision(2) << "rvec = ["
              << rvec.at<float>(0,0) << ", "
              << rvec.at<float>(1,0) << ", "
              << rvec.at<float>(2,0) << "] \t" << "tvec = ["
              << tvec.at<float>(0,0) << ", "
              << tvec.at<float>(1,0) << ", "
              << tvec.at<float>(2,0) << "]" << std::endl;

    /// http://pointclouds.org/documentation/tutorials/matrix_transform.php
    /// http://stackoverflow.com/questions/12933284/rodrigues-into-eulerangles-and-vice-versa
    Eigen::Affine3f mpose = Eigen::Affine3f::Identity();
    mpose.translation() << tvec.at<float>(0,0)*(-1), tvec.at<float>(1,0)*(-1), tvec.at<float>(2,0);
    float theta = sqrt(pow(rvec.at<float>(0,0),2) + pow(rvec.at<float>(1,0),2) + pow(rvec.at<float>(2,0),2));
    mpose.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f(rvec.at<float>(0,0)/theta*-1, rvec.at<float>(1,0)/theta*-1, rvec.at<float>(2,0)/theta)));

    //        /// http://stackoverflow.com/questions/25504397/eigen-combine-rotation-and-translation-into-one-matrix
    //        Eigen::Affine3f r = create_rotation_matrix(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0));
    //        Eigen::Affine3f markerpose(Eigen::Translation3f(Eigen::Vector3f((tvec.at<double>(0,0)*-2.4),(tvec.at<double>(1,0)*-2.4),tvec.at<double>(2,0)*2.4)));
    //        markerpose = markerpose*r;

    return mpose;
}

/// currently unused
Eigen::Affine3f Tracker::create_rotation_matrix(double ax, double ay, double az)
{
    Eigen::Affine3f rx =
            Eigen::Affine3f(Eigen::AngleAxisf(ax, Eigen::Vector3f(1, 0, 0)));
    Eigen::Affine3f ry =
            Eigen::Affine3f(Eigen::AngleAxisf(ay, Eigen::Vector3f(0, 1, 0)));
    Eigen::Affine3f rz =
            Eigen::Affine3f(Eigen::AngleAxisf(az, Eigen::Vector3f(0, 0, 1)));
    return rz * ry * rx;
}
