#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void GetGaussianKernel(Mat &gaus, const int size, const double sigma);

void My_Blur(Mat &src, Mat &dst, int ksize,double sigma = 1);
void My_PyrDown(const Mat& src,Mat &dst,const Size& dstsize = Size(),double sigma = 1);
void My_PyrUp(const Mat& src,Mat &ret,const Size& dstsize = Size(),double sigma = 1);

int main() {
    // namedWindow("Raw Image", WINDOW_NORMAL);
    // namedWindow("Convolved Image", WINDOW_NORMAL);
    // namedWindow("Downsampled Image", WINDOW_NORMAL);
    // namedWindow("Upsampled Image", WINDOW_NORMAL);

    Mat image = imread("/home/kangzhehao/practice/week2/task2/armor.png");
    if (image.empty()) {
        cerr << "Error: Image not found!" << endl;
        return -1;
    }

    resize(image,image,Size(image.cols/2,image.rows/2));
    imshow("Raw Image", image);

    Mat result;
    int kernelSize = 5;
    double blurSigma = 1;
    My_Blur(image, result, kernelSize, blurSigma);
    imshow("Convolved Image", result);

    Mat downsampled;
    Size downsampleSize(0, 0);
    double downsampleSigma = 1;
    My_PyrDown(image, downsampled, downsampleSize, downsampleSigma);
    imshow("Downsampled Image", downsampled);

    Mat upsampled;
    Size upsampleSize(0, 0); 
    double upsampleSigma = 10;
    My_PyrUp(downsampled, upsampled, upsampleSize, upsampleSigma);
    imshow("Upsampled Image", upsampled);

    waitKey(0);

    return 0;
}


void GetGaussianKernel(Mat &gaus, const int size, const double sigma) {
    Mat kernel(size, size, CV_64F);
    int center = size / 2;
    double sum = 0;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel.at<double>(i, j) = (1 / (2 * M_PI * sigma * sigma)) *
                                      exp(-((i - center) * (i - center) + (j - center) * (j - center)) / (2 * sigma * sigma));
            sum += kernel.at<double>(i, j);
        }
    }

    kernel /= sum;
    gaus = kernel; 
}


void My_Blur(Mat &src, Mat &dst, int ksize,double sigma ) {
    Mat kernel;
    GetGaussianKernel(kernel, ksize, sigma);
    int ConvolutionResult_row = src.rows - ksize + 1;
    int ConvolutionResult_col = src.cols - ksize + 1;

    dst = Mat::zeros(ConvolutionResult_row, ConvolutionResult_col, CV_8UC3);

    for (int i = 0; i < ConvolutionResult_row; i++) {
        for (int j = 0; j < ConvolutionResult_col; j++) {
            for (int k = 0; k < 3; k++) {
                double sum = 0;
                for (int m = 0; m < ksize; m++) {
                    for (int n = 0; n < ksize; n++) {
                        sum += (src.at<Vec3b>(i + m, j + n)[k] * kernel.at<double>(m, n));
                    }
                }
                dst.at<Vec3b>(i, j)[k] = static_cast<uchar>(sum);
            }
        }
    }
}

void My_PyrDown(const Mat& src,Mat &dst,const Size& dstsize ,double sigma  ){
    int dst_Rows  , dst_Cols;
    if(dstsize.empty()){
        dst_Rows = src.rows/2;
        dst_Cols = src.cols/2;
    }else {
        dst_Rows = dstsize.width;
        dst_Cols = dstsize.height;
    }
    dst = Mat(dst_Rows, dst_Cols, src.type());
    int scale = (int)src.rows/dst_Rows;

    Mat gaus;
    GetGaussianKernel(gaus,scale,sigma);

    for (int i = 0; i < dst_Rows; i++) {
        for (int j = 0; j < dst_Cols; j++) {
            int r = i * scale;
            int c = j * scale;

            for (int k = 0; k < 3; k++) {
                int sum = 0;
                for (int  m = 0; m < scale; m++)
                {
                    for (int n = 0; n < scale; n++)
                    {
                        sum += src.at<Vec3b>(r+m,c+n)[k] * gaus.at<double>(m,n);
                    }
                }
                dst.at<Vec3b>(i, j)[k] = sum;
            }
        }
    }
}

// void My_PyrUp(const Mat& src, Mat& ret, const Size& dstsize, double sigma) {
//     int srcRows = src.rows;
//     int srcCols = src.cols;

//     int dstRows, dstCols;
//     if (dstsize.empty()) {
//         dstRows = srcRows * 2;
//         dstCols = srcCols * 2;
//     } else {
//         dstRows = dstsize.width;
//         dstCols = dstsize.height;
//     }

//     Mat dst = Mat(dstRows, dstCols, src.type(),Scalar(0,0,0));

//     int scale = (int)dstRows/src.rows;

//    for (int i = 0; i < srcRows; i++) {
//         for (int j = 0; j < srcCols; j++) {
//             for (int k = 0; k < 3; k++) {
//                 dst.at<Vec3b>(i * scale, j * scale)[k] = src.at<Vec3b>(i, j)[k];
//             }
//         }
//     }

//     My_Blur(dst,ret,scale,sigma);
// }

void My_PyrUp(const Mat& src, Mat& ret, const Size& dstsize, double sigma) {
    int srcRows = src.rows;
    int srcCols = src.cols;
    
    int dstRows, dstCols;
    if (dstsize.empty()) {
        dstRows = srcRows * 2;
        dstCols = srcCols * 2;
    } else {
        dstRows = dstsize.width;
        dstCols = dstsize.height;
    }

    // 创建目标大小的图像
    Mat dst = Mat(dstRows, dstCols, src.type());

    int scale = (int)dstRows/src.rows;

    // 上采样，执行线性插值
    for (int i = 0; i < dstRows; ++i) {
        for (int j = 0; j < dstCols; ++j) {
            // 计算在源图像中的坐标位置
            float y = (float)i / scale;
            float x = (float)j / scale;

            // 找到最近的四个像素
            int x_low = floor(x);
            int y_low = floor(y);
            int x_high = x_low + 1;
            int y_high = y_low + 1;

            // 边界处理
            x_low = std::max(0, std::min(x_low, srcCols - 1));
            x_high = std::max(0, std::min(x_high, srcCols - 1));
            y_low = std::max(0, std::min(y_low, srcRows - 1));
            y_high = std::max(0, std::min(y_high, srcRows - 1));

            // 执行双线性插值
            for (int k = 0; k < 3; ++k) {
                float dx = x - x_low;
                float dy = y - y_low;

                float value = (1 - dx) * (1 - dy) * src.at<Vec3b>(y_low, x_low)[k] +
                              dx * (1 - dy) * src.at<Vec3b>(y_low, x_high)[k] +
                              (1 - dx) * dy * src.at<Vec3b>(y_high, x_low)[k] +
                              dx * dy * src.at<Vec3b>(y_high, x_high)[k];

                dst.at<Vec3b>(i, j)[k] = static_cast<uchar>(value);
            }
        }
    }

    // 对图像进行高斯模糊处理
    My_Blur(dst,ret,scale,sigma);
}
