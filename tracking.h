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
    Tracker();
    void getTransformation(Frame & input);
};

#endif // TRACKING_H
