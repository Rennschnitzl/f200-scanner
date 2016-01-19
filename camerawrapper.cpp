#include "camerawrapper.h"

CameraWrapper::CameraWrapper(int filtermode)
    : height(480), width(640), factor(31.25), offset(0.5),
      //depthcam(std::make_<CameraDriver>("/dev/video2", 0x49524e49))
    depthcam(new CameraDriver("/dev/video2", 0x49524e49))
	, slptm()
{
    loadCameraMatrix();
    slptm.tv_sec = 0;
    slptm.tv_nsec = 50000000;      //1000 ns = 1 us

    //CameraDriver *colorcam = new CameraDriver("/dev/video1", 0x56595559);
   
    //colorcam->startVideo();
    depthcam->startVideo();
    sleep(1);

    // set video settings to 11 patterns and raw values
    depthcam->setIvcamSetting(0);
    depthcam->setFilterSetting(filtermode);
    // TODO: find good laser power, max 16
    depthcam->setLaserPower(16);
}

CameraWrapper::~CameraWrapper()
{
    //colorcam->stopVideo();
    depthcam->stopVideo();
}

cv::Mat CameraWrapper::getCoeffs()
{
    return this->distCoeffs;
}

cv::Mat CameraWrapper::getCameraMatrix()
{
    return this->cameraMatrix;
}

void CameraWrapper::displayCameraProperties()
{
    double apertureWidth;
    double apertureHeight;
    double fovx;
    double fovy;
    double focalLength;
    cv::Point2d principalPoint;
    double aspectRatio;
    
	calibrationMatrixValues(cameraMatrix, cvSize(640,480), apertureWidth, apertureHeight, fovx, fovy, focalLength, principalPoint, aspectRatio);
    
	std::cout << "Camera-Matrix Data:" << "\nAperture (W/H): " << apertureWidth << " / " << apertureHeight << "\nFoV (x/y): "
              << fovx << " / " << fovy << "\nPrincipal point: " << principalPoint << "\nAspect ratio: " << aspectRatio << std::endl;
}

void CameraWrapper::recordStack(int frames, std::vector<cv::Mat> &irlist, std::vector<cv::Mat> &depthlist)
{
    clearBuffer();

    irlist.clear();
    depthlist.clear();

    std::vector<std::vector<u_int16_t> > depth;
    std::vector<std::vector<u_int8_t> > ir;
	
    for(int i=0; i<frames; i++)
    {
        nanosleep(&slptm, NULL);
        //colorcam->updateData(&rgbimage);
        while(!depthcam->updateDataIR(depth, ir))
        {
            std::cout << "error getting frame, retrying.." << std::endl;
        }
		
        irlist.push_back(convertIRtoCV(ir));
        depthlist.push_back(convertDepthtoCV(depth));
    }
}

void CameraWrapper::loadCameraMatrix()
{
    std::cout << "OpenCV version : " << CV_VERSION << std::endl;
    std::cout << "Loading camera matrix" << std::endl;
    // read settings from file
    cv::FileStorage fs2("calibration.yml", cv::FileStorage::READ);
    fs2["cameraMatrix"] >> cameraMatrix;
    fs2["distCoeffs"] >> distCoeffs;

    // Get some infos from the calibration matrix
    displayCameraProperties();
}

// TODO find easier way, should be a way to empty the buffer with commands
void CameraWrapper::clearBuffer()
{
    std::vector<std::vector<u_int16_t> > depth;
    std::vector<std::vector<u_int8_t> > ir;
    // clear buffer
    for(int i = 0; i<3; i++)
    {
        nanosleep(&slptm, NULL);
        //colorcam->updateData(&rgbimage);
        depthcam->updateDataIR(depth, ir);
    }
}

cv::Mat CameraWrapper::convertIRtoCV(std::vector<std::vector<u_int8_t> > ir)
{
    cv::Mat ir_cv(height,width, CV_8U);

    for(int j = 0 ; j < height ; j++){
        for(int i = 0 ; i < width ; i++){
            ir_cv.at<uchar>(j,i) = ir[j][i];
        }
    }
    return ir_cv;
}

cv::Mat CameraWrapper::convertDepthtoCV(std::vector<std::vector<u_int16_t> > depth)
{
    cv::Mat depth_cv(height,width, CV_16U);

    for(int j = 0 ; j < height ; j++){
        for(int i = 0 ; i < width ; i++){
            //depth = int(depth/this->factor + this->offset); // convert to mm
            u_int8_t high = (depth[j][i] >> 8) & 0xff;
            u_int8_t low = depth[j][i] & 0xff;
            cv::Vec2b depthpix_cv;
            depthpix_cv[0] = low;
            depthpix_cv[1] = high;
            depth_cv.at<cv::Vec2b>(j,i) = depthpix_cv;
        }
    }
    return depth_cv;
}
