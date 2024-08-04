#include <math.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "stdio.h"

using namespace std;
using namespace cv;

void GammaTransform(cv::Mat& image, cv::Mat& dist, double gamma = 1.5);
void My_GammaTransform(cv::Mat& image, cv::Mat& dist, double gamma = 1.5);

int main() {
    Mat image = imread("/home/kangzhehao/practice/week2/task2/armor.png");
    if (image.empty()) {
        cerr << "Error: Image not found!" << endl;
        return -1;
    }
    resize(image, image, Size(image.cols / 2, image.rows / 2));
    imshow("Raw Image", image);

    Mat dist;
    My_GammaTransform(image, dist, (1.f / 2.2));
    imshow("My_GammaTransform", dist);

    Mat dist1;
    GammaTransform(image, dist1, (1.f / 2.2));
    imshow("opencv_GammaTransform", dist1);

    waitKey(0);
    return 0;
}

void GammaTransform(cv::Mat& image, cv::Mat& dist, double gamma) {
    Mat imageGamma;
    // 灰度归一化
    image.convertTo(imageGamma, CV_64F, 1.0 / 255, 0);

    pow(imageGamma, gamma, dist);  // dist 要与imageGamma有相同的数据类型

    dist.convertTo(dist, CV_8U, 255, 0);
}

void My_GammaTransform(cv::Mat& image, cv::Mat& dist, double gamma) {
    dist = Mat(image.size(), image.type());

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < 3; k++) {
                dist.at<Vec3b>(i, j)[k] =
                    (uint8_t)(255 * std::pow((double)image.at<Vec3b>(i, j)[k] /
                                                 255.0f,
                                             gamma));
            }
        }
    }
}