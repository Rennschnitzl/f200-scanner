#ifndef TRACKING_H
#define TRACKING_H

#include <aruco/aruco.h>
#include <aruco/cvdrawingutils.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

#include "frame.h"
#include "helper.h"


#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <fstream>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class Tracker
{
public:
    Tracker(double markersize = 0.0325, std::string boardfile = "chessinfo_meter.yml");
    void getTransformation(Frame & input);

    double getThresParam1() const;
    void setThresParam1(double value);

    double getThresParam2() const;
    void setThresParam2(double value);

private:
    string TheBoardConfigFile;
    float TheMarkerSize;
    aruco::CameraParameters TheCameraParameters;
    aruco::BoardConfiguration TheBoardConfig;
    aruco::BoardDetector TheBoardDetector;
    double ThresParam1;
    double ThresParam2;
};

#endif // TRACKING_H
