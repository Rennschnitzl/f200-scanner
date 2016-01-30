#include "tracking.h"

Tracker::Tracker(double markersize = 0.0325, std::string boardfile = "chessinfo_meter.yml")
{
    TheBoardConfigFile = boardfile;
    TheMarkerSize = markersize;

    TheBoardConfig.readFromFile(TheBoardConfigFile);

    ThresParam1 = 13.0;
    ThresParam2 = 8.0;
}

void Tracker::getTransformation(Frame &input)
{
    TheCameraParameters.CameraMatrix = input.cameraMatrix_color;
    TheCameraParameters.Distorsion = input.coefficients_color;
    TheCameraParameters.CamSize = input.processedImageRGB.size();

    // init
    // DESIGN move this to initialization step with a fixed camera?
    TheBoardDetector.setParams(TheBoardConfig, TheCameraParameters, TheMarkerSize);
    TheBoardDetector.getMarkerDetector().getThresholdParams(ThresParam1, ThresParam2);
    TheBoardDetector.getMarkerDetector().setCornerRefinementMethod(aruco::MarkerDetector::HARRIS);
    TheBoardDetector.set_repj_err_thres(1.5);

    // TODO actual tracking
    cv::Mat TheInputImage, TheInputImageCopy;
    input.processedImageRGB.copyTo(TheInputImage);
    float probDetect = TheBoardDetector.detect(input.processedImageRGB);

    // FIXME clean up this mess

    // draw detected markers
    input.processedImageRGB.copyTo(TheInputImageCopy);
    for (unsigned int i = 0; i < TheBoardDetector.getDetectedMarkers().size(); i++)
        TheBoardDetector.getDetectedMarkers()[i].draw(TheInputImageCopy, cv::Scalar(0, 0, 255), 1);

    // draw board axis
    if (TheCameraParameters.isValid()) {
        if (probDetect > 0.2) {
            aruco::CvDrawingUtils::draw3dAxis(TheInputImageCopy, TheBoardDetector.getDetectedBoard(), TheCameraParameters);
            // 		    CvDrawingUtils::draw3dCube(TheInputImageCopy, TheBoardDetector.getDetectedBoard(),TheCameraParameters);
            // draw3dBoardCube( TheInputImageCopy,TheBoardDetected,TheIntriscCameraMatrix,TheDistorsionCameraParams);
        }
    }

//    if (probDetect > 0.2){
//        aruco::Board detBoard = TheBoardDetector.getDetectedBoard();
////        std::cout << detBoard.Rvec << std::endl << detBoard.Tvec << std::endl;
//        Matrix = createMatrixfromVectors(detBoard.Rvec, detBoard.Tvec);

//    }else
//        Matrix = Eigen::Affine3f::Identity();

    cv::imshow("in", TheInputImageCopy);
    cv::imshow("thres", TheBoardDetector.getMarkerDetector().getThresholdedImage());
    cv::waitKey(0);//wait for key to be pressed

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
