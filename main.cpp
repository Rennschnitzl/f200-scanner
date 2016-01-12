#include <iostream>
#include "camerawrapper.h"
#include "frame.h"
#include "converter.h"
#include <opencv2/opencv.hpp>

#define RECORD_STACK_SIZE 5

using namespace std;

int main()
{
    int height = 480;
    int width = 640;

    CameraWrapper cw(0);
    Frame frame1(width, height, cw.getCameraMatrix(),cw.getCoeffs());

    cw.recordStack(RECORD_STACK_SIZE,frame1.rawStackIR, frame1.rawStackDepth);

    Converter::analyseStack(frame1.rawStackDepth, frame1.belief, frame1.processedImageDepth);
    Converter::averageIR(frame1.rawStackIR, frame1.processedImageIR);



    cv::Mat depth_cv_8;
    cv::Mat depth_cv_rgb(height,width,CV_8UC3);
    
    frame1.processedImageDepth.convertTo(depth_cv_8,CV_8U,1.0/256.0);
	
    cv::cvtColor(depth_cv_8,depth_cv_rgb,CV_GRAY2RGB);

	cv::Mat ir_cv_rgb(height,width,CV_8UC3);
	
    cv::cvtColor(frame1.processedImageIR,ir_cv_rgb,CV_GRAY2RGB);

    cv::imshow("depth", depth_cv_rgb);
    cv::imshow("ir", ir_cv_rgb);
    cv::waitKey(0);

    return 0;
}
