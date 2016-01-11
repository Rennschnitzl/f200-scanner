#include <iostream>
#include "camerawrapper.h"
#include <opencv2/opencv.hpp>


using namespace std;

int main()
{

    int height = 480;
    int width = 640;

    CameraWrapper *cw = new CameraWrapper(0);

    std::vector<cv::Mat> irlist, depthlist;


    cw->recordStack(5,irlist, depthlist);

    cv::Mat depth_cv_8;
    cv::Mat depth_cv_rgb(height,width,CV_8UC3);

    cv::Mat ir_cv_rgb(height,width,CV_8UC3);
    depthlist[2].convertTo(depth_cv_8,CV_8U,1.0/256.0);
    cv::cvtColor(depth_cv_8,depth_cv_rgb,CV_GRAY2RGB);

    cv::cvtColor(irlist[2],ir_cv_rgb,CV_GRAY2RGB);


    cv::imshow("depth", depth_cv_rgb);
    cv::imshow("ir", ir_cv_rgb);
    cv::waitKey(0);

    delete cw;



    return 0;
}
