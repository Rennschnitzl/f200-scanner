#include "converter.h"

Converter::Converter()
{

}

std::string Converter::type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

void Converter::analyseStack(std::vector<cv::Mat> &stack, cv::Mat & believe, cv::Mat & result)
{
    cv::Mat belief(480, 640, CV_8U, cv::Scalar::all(0));
    cv::Mat processedImageDepth(480, 640, CV_16U, cv::Scalar::all(0));

    for (int i = 0; i < stack[0].rows; i++)
    {
        for (int j = 0; j < stack[0].cols; j++)
        {
            unsigned long tempresult_depth = 0;
            int depth_zeroes = 0;
            std::vector<int> var_values;
            for(int il = 0; il < stack.size(); il++)
            {
                tempresult_depth += (int)stack[il].at<ushort>(i,j);
                if((int)stack[il].at<ushort>(i,j) == 0){
                    depth_zeroes++;
                }else
                    var_values.push_back((int)stack[il].at<ushort>(i,j));
            }
            // calculate average

            double var = calculateVariance(var_values);

            // TODO: filter von unzuverlässigen daten. möglicherweise in neue methode auslagern
            if(depth_zeroes < stack.size()-1 && var < 1000.0)
            {
                processedImageDepth.at<ushort>(i,j) = tempresult_depth/(stack.size()-depth_zeroes);
                belief.at<uchar>(i,j) = (int)(255-((255/stack.size())*depth_zeroes));
                //std::cout << "zeroes: "<< depth_zeroes << " belief: " << (int)belief.at<uchar>(i,j) << std::endl;
            }
            else
            {
                processedImageDepth.at<ushort>(i,j) = 0;
                belief.at<uchar>(i,j) = 0;
            }
        }
    }
    believe = belief;
    result = processedImageDepth;
}

void Converter::averageIR(const std::vector<cv::Mat> & IRstack, cv::Mat & ir_avg)
{
    cv::Mat processedImageIR(480, 640, CV_8U, cv::Scalar::all(0));
    for (int i = 0; i < IRstack[0].rows; i++)
    {
        for (int j = 0; j < IRstack[0].cols; j++)
        {
            long tempresult_ir = 0;
            for(int il = 0; il < IRstack.size(); il++) {
                tempresult_ir += (int)IRstack[il].at<uchar>(i,j);
            }
            processedImageIR.at<uchar>(i,j) = tempresult_ir/IRstack.size();
        }
    }
    ir_avg = processedImageIR;
}

double Converter::calculateVariance(std::vector<int> var_values)
{
    if(var_values.size() != 0){
//        std::cout << "i do calcs" << std::endl;
//        std::cout << var_values[0] << std::endl;
        double mean = 0.0;
        int sum_av = 0;

        for ( int x=0; x < var_values.size(); x++)
        {
            sum_av += var_values[x];
        }

        mean =  (sum_av / var_values.size());

//        std::cout << mean << std::endl;

        //calculate variance
        double sum = 0.0;
        double temp =0.0;

        for ( int y =0; y < var_values.size(); y++)
        {
            temp = pow((var_values[y] - mean),2);
            sum += temp;
        }

        //std::cout << "i return" << sum/(var_values.size() -2) << std::endl;
        return sum/(var_values.size() -2);
    }
    else{
        return 1000.0;
    }
}

/// stolen from opencv contribution
void Converter::rescaleDepth(cv::InputArray in_in, int depth, cv::OutputArray out_out)
{
    cv::Mat in = in_in.getMat();
    CV_Assert(in.type() == CV_64FC1 || in.type() == CV_32FC1 || in.type() == CV_16UC1 || in.type() == CV_16SC1);
    CV_Assert(depth == CV_64FC1 || depth == CV_32FC1);

    int in_depth = in.depth();

    out_out.create(in.size(), depth);
    cv::Mat out = out_out.getMat();
    if (in_depth == CV_16U)
    {
        in.convertTo(out, depth, 1 / 1000.0); //convert to float so that it is in meters
        cv::Mat valid_mask = in == std::numeric_limits<ushort>::min(); // Should we do std::numeric_limits<ushort>::max() too ?
        out.setTo(std::numeric_limits<float>::quiet_NaN(), valid_mask); //set a$
    }
    if (in_depth == CV_16S)
    {
        in.convertTo(out, depth, 1 / 1000.0); //convert to float so tha$
        cv::Mat valid_mask = (in == std::numeric_limits<short>::min()) | (in == std::numeric_limits<short>::max()); // Should we do std::numeric_limits<ushort>::max() too ?
        out.setTo(std::numeric_limits<float>::quiet_NaN(), valid_mask); //set a$
    }
    if ((in_depth == CV_32F) || (in_depth == CV_64F))
        in.convertTo(out, depth);
}

/// stolen (and modified) from opencv contribution
void Converter::depthTo3d(const cv::Mat& in_depth, const cv::Mat& K, pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud, const cv::Mat& belief)
{
    pcl::PointXYZI newPoint;

    const double inv_fx = double(1) / K.at<double>(0, 0);
    const double inv_fy = double(1) / K.at<double>(1, 1);
    const double ox = K.at<double>(0, 2);
    const double oy = K.at<double>(1, 2);

    // Build z
    cv::Mat_<double> z_mat;
    if (z_mat.depth() == in_depth.depth())
        z_mat = in_depth;
    else
        rescaleDepth(in_depth, CV_64F, z_mat);

    // Mit precomputation
    // 0.0413511 seconds.
    // 0.0415454 seconds.

//    // Pre-compute some constants
//    cv::Mat_<double> x_cache(1, in_depth.cols), y_cache(in_depth.rows, 1);
//    double* x_cache_ptr = x_cache[0], *y_cache_ptr = y_cache[0];
//    for (int x = 0; x < in_depth.cols; ++x, ++x_cache_ptr)
//        *x_cache_ptr = (x - ox) * inv_fx;
//    for (int y = 0; y < in_depth.rows; ++y, ++y_cache_ptr)
//        *y_cache_ptr = (y - oy) * inv_fy;

//    y_cache_ptr = y_cache[0];


//    for (int y = 0; y < in_depth.rows; ++y, ++y_cache_ptr)
//    {
//        const double* x_cache_ptr_end = x_cache[0] + in_depth.cols;
//        const double* depth = z_mat[y];
//        for (x_cache_ptr = x_cache[0]; x_cache_ptr != x_cache_ptr_end; ++x_cache_ptr, ++depth)
//        {
//            double z = *depth;
//            if(isnan(z))
//            {
//                newPoint.z = std::numeric_limits<float>::quiet_NaN();
//                newPoint.x = std::numeric_limits<float>::quiet_NaN();
//                newPoint.y = std::numeric_limits<float>::quiet_NaN();
//                newPoint.intensity = std::numeric_limits<float>::quiet_NaN();
//                cloud->push_back(newPoint);
//            }else
//            {
//                newPoint.z = z;
//                newPoint.x = (*x_cache_ptr) * z * -1.0;
//                newPoint.y = (*y_cache_ptr) * z * -1.0;
//                //newPoint.intensity = (float)(int)belief.at<uchar>(y,x);
//                cloud->push_back(newPoint);
//            }

//        }
//    }


    // ohne precompute:
    // 0.0443601 seconds.
    // 0.0443742 seconds.

    cv::imshow("fack u", belief);

    for (int y = 0; y < in_depth.rows; ++y)
    {
        const double* depth = z_mat[y];
        for (int x = 0; x < in_depth.cols; ++x, ++depth)
        {
            double z = *depth;
            if(isnan(z) || z < 0.1) //
            {
//                newPoint.z = std::numeric_limits<float>::quiet_NaN();
//                newPoint.x = std::numeric_limits<float>::quiet_NaN();
//                newPoint.y = std::numeric_limits<float>::quiet_NaN();
//                newPoint.intensity = std::numeric_limits<float>::quiet_NaN();
//                cloud->push_back(newPoint);
            }else
            {
                double xmod = ((x - ox) * inv_fx);
                double ymod = ((y - oy) * inv_fy);
                newPoint.z = z;
                newPoint.x = xmod * z * -1.0;
                newPoint.y = ymod * z * -1.0;
                newPoint.intensity = (int)belief.at<uchar>(y,x);
                //std::cout << newPoint.intensity << " z: " << z << " mat " << (int)belief.at<uchar>(y,x) << std::endl;
                cloud->push_back(newPoint);
            }

        }
    }
    std::cout << "cloudsize: " << cloud->size() << std::endl;
}


void Converter::undistortDepth(cv::Mat &in_depth, const cv::Mat &camMatrix, const cv::Mat &coeffs, cv::Mat &belief)
{
    cv::Mat map1, map2, undistorted_depth, out_belief;
    cv::initUndistortRectifyMap(camMatrix,
                                coeffs,
                                cv::Mat(),
                                camMatrix,
// compute new scaling          cv::getOptimalNewCameraMatrix(cw.getCameraMatrix(),
//                                                              cw.getCoeffs(),
//                                                              ir_cv_rgb.size(),
//                                                              1,
//                                                              ir_cv_rgb.size(),
//                                                              0),
                                in_depth.size(),
                                CV_16SC2,
                                map1,
                                map2);
    cv::remap(in_depth, undistorted_depth, map1, map2, cv::INTER_NEAREST);

    // TODO: replace magic numbers with values from calibration matrix
    // correct depth
    // fancy überlegung von alex und peter
    for(int x = 0; x<640; x++)
    {
        for(int y = 0; y<480; y++)
        {
            int z = (int)undistorted_depth.at<ushort>(y,x);
            // calculate alpha, x axis
            double tanx = (0.44*fabs(x-328.001)) / 205;
            double atanx = atan(tanx); // (result in radians)
            // calculate side b, x axis
            double sideb = cos(atanx) * z;

            // calculate alpha, y axis
            tanx = (0.44*fabs(y-242.001)) / 205;
            atanx = atan(tanx); // (result in radians)
            // calculate side b, y axis
            sideb = cos(atanx) * sideb;

            undistorted_depth.at<ushort>(y,x) = sideb;

        }
    }

    // undistort belief-map
    cv::remap(belief, out_belief, map1, map2, cv::INTER_NEAREST);

    belief = out_belief;
    in_depth = undistorted_depth;
}
