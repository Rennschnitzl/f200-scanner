#include "converter.h"

Converter::Converter()
{

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
            if(depth_zeroes < stack.size()-0 && var < 100.0)
                processedImageDepth.at<ushort>(i,j) = tempresult_depth/(stack.size()-depth_zeroes);
            else
                processedImageDepth.at<ushort>(i,j) = 0;
            belief.at<uchar>(i,j) = (int)((240/stack.size())*depth_zeroes);
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

        return sum/(var_values.size() -2);
    }
    else{
        return 1000.0;
    }
}
