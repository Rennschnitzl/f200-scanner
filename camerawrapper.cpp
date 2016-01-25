#include "camerawrapper.h"

//void CameraWrapper::loadCameraMatrix()
//{
//    std::cout << "OpenCV version : " << CV_VERSION << std::endl;
//    std::cout << "Loading camera matrix" << std::endl;
//    // read settings from file
//    cv::FileStorage fs2("calibration.yml", cv::FileStorage::READ);
//    fs2["cameraMatrix"] >> cameraMatrix;
//    fs2["distCoeffs"] >> distCoeffs;

//    // Get some infos from the calibration matrix
//    displayCameraProperties();
//}

CameraWrapper::CameraWrapper(int frames)
{
    // init camera
    e = 0;
    ctx = rs_create_context(RS_API_VERSION, &e);
    check_error();
    printf("There are %d connected RealSense devices.\n", rs_get_device_count(ctx, &e));
    check_error();
    if(rs_get_device_count(ctx, &e) == 0)
    {
        std::cout << "camera init broken" << std::cout;
    }
    else
    {
        dev = rs_get_device(ctx, 0, &e);
        check_error();
        printf("\nUsing device 0, an %s\n", rs_get_device_name(dev, &e));
        check_error();
        printf("    Serial number: %s\n", rs_get_device_serial(dev, &e));
        check_error();
        printf("    Firmware version: %s\n", rs_get_device_firmware_version(dev, &e));
        check_error();

        // set controls
        // TODO check control presets
        //    // set video settings to 11 patterns and raw values
        //    depthcam->setIvcamSetting(0);
        //    depthcam->setFilterSetting(filtermode);
        //    // TODO: find good laser power, max 16
        //    depthcam->setLaserPower(16);
        rs_enable_stream_preset(dev, RS_STREAM_DEPTH, RS_PRESET_BEST_QUALITY, &e);
        check_error();
        rs_enable_stream_preset(dev, RS_STREAM_COLOR, RS_PRESET_BEST_QUALITY, &e);
        check_error();
        rs_enable_stream_preset(dev, RS_STREAM_INFRARED, RS_PRESET_BEST_QUALITY, &e);
        check_error();
        rs_start_device(dev, &e);
        check_error();

        // get intrinsics and convert them to opencv mat
        // IR and DEPTH are the same for f200, so this is basically overhead
        rs_get_stream_intrinsics(dev, RS_STREAM_DEPTH, &depth_intrin, &e);
        check_error();
        rs_get_device_extrinsics(dev, RS_STREAM_DEPTH, RS_STREAM_COLOR, &depth_to_color, &e);
        check_error();
        rs_get_stream_intrinsics(dev, RS_STREAM_COLOR, &color_intrin, &e);
        check_error();
        rs_get_stream_intrinsics(dev, RS_STREAM_INFRARED, &ir_intrin, &e);
        check_error();
        depthScale = rs_get_device_depth_scale(dev, &e);
        check_error();
    }

    // set number of frames to record
    setStackSize(frames);
}

CameraWrapper::~CameraWrapper()
{
    // deactivate camera
    rs_stop_device(dev, &e);
    check_error();
    rs_disable_stream(dev,RS_STREAM_DEPTH, &e);
    check_error();
    rs_disable_stream(dev,RS_STREAM_COLOR, &e);
    check_error();
    rs_disable_stream(dev,RS_STREAM_INFRARED, &e);
    check_error();
}

Frame CameraWrapper::record()
{
    // clear buffer
    rs_wait_for_frames(dev, &e);
    check_error();

    std::vector<cv::Mat> depthstack;
    std::vector<cv::Mat> colorstack;
    std::vector<cv::Mat> irstack;

    // record a stack of frames
    for(int frame = 0; frame < framesToRecord; frame++)
    {
        // wait for frames
        rs_wait_for_frames(dev, &e);
        check_error();

        // get data
        const uint16_t * depth_image = (const uint16_t *)rs_get_frame_data(dev, RS_STREAM_DEPTH, &e);
        check_error();
        const uint8_t * color_image = (const uint8_t *)rs_get_frame_data(dev, RS_STREAM_COLOR, &e);
        check_error();
        const uint8_t * ir_image = (const uint8_t *)rs_get_frame_data(dev, RS_STREAM_INFRARED, &e);
        check_error();

        // convert depth to meters and float
        cv::Mat depthframe = cv::Mat(depth_intrin.height, depth_intrin.width, CV_32F);
        int dx, dy;
        for(dy=0; dy<depth_intrin.height; ++dy)
        {
            for(dx=0; dx<depth_intrin.width; ++dx)
            {
                /* Retrieve the 16-bit depth value and map it into a depth in meters */
                uint16_t depth_value = depth_image[dy * depth_intrin.width + dx];
                float depth_in_meters = depth_value * depthScale;
                depthframe.at<float>(dy, dx) = depth_in_meters;
            }
        }
        depthstack.push_back(depthframe);

        // color
        cv::Mat colorframeRGB = cv::Mat(color_intrin.height, color_intrin.width, CV_8UC3, cv::Scalar(0));
        memcpy(colorframeRGB.ptr(),color_image,color_intrin.height*color_intrin.width*3);
        // f200 gives RGB image, openCV uses BGR - so convert
        cv::Mat colorframeBGR;
        cv::cvtColor(colorframeRGB, colorframeBGR, CV_RGB2BGR);
        colorstack.push_back(colorframeBGR);

        // ir
        cv::Mat irframe = cv::Mat(ir_intrin.height, ir_intrin.width, CV_8UC1, cv::Scalar(0));
        memcpy(irframe.ptr(),ir_image,ir_intrin.height*ir_intrin.width);
        irstack.push_back(irframe);
    }


    // create Frame object
    Frame frame1;
    convertIntrinsicToOpenCV(depth_intrin, frame1.cameraMatrix_depth, frame1.coefficients_depth);
    convertIntrinsicToOpenCV(color_intrin, frame1.cameraMatrix_color, frame1.coefficients_color);
    convertIntrinsicToOpenCV(ir_intrin, frame1.cameraMatrix_ir, frame1.coefficients_ir);

    // call averager
    // TODO add interface


    // limit to 4 frames
    // average RGB images
    int framesToRecordRGB;
    if (framesToRecord > 4)
        framesToRecordRGB = 4;
    else
        framesToRecordRGB = framesToRecord;
    frame1.processedImageRGB = cv::Mat::zeros(color_intrin.height, color_intrin.width, CV_8UC3);
    for (int frame = 0; frame < framesToRecordRGB; frame++)
    {
        frame1.processedImageRGB = frame1.processedImageRGB + (1.0/framesToRecordRGB)*colorstack[frame];
    }

    // average IR
    frame1.processedImageIR = cv::Mat::zeros(ir_intrin.height, ir_intrin.width, CV_8UC1);
    for (int frame = 0; frame < framesToRecordRGB; frame++)
    {
        frame1.processedImageIR = frame1.processedImageIR + (1.0/framesToRecordRGB)*irstack[frame];
    }

    // fill averaged IR and depth

    // fill beliefmap

    return frame1;
}

void CameraWrapper::setStackSize(int frames)
{
    this->framesToRecord = frames;
}

void CameraWrapper::check_error()
{
    if(e)
    {
        printf("rs_error was raised when calling %s(%s):\n", rs_get_failed_function(e), rs_get_failed_args(e));
        printf("    %s\n", rs_get_error_message(e));
        exit(EXIT_FAILURE);
    }
}

void CameraWrapper::convertIntrinsicToOpenCV(const rs_intrinsics &in_intrinsics, cv::Mat &out_cammat, cv::Mat &out_coeffs)
{
    // TODO implement method
}

rs_intrinsics CameraWrapper::getIntrinsicsFromOpenCV(const cv::Mat &in_cammat, const cv::Mat &in_coeffs)
{
    // TODO implement method
}
