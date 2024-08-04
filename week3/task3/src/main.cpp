#include "stdio.h"
#include<iostream> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace std;
using namespace cv;

#define CREATE_TRACKBAR(TrackbarValue)createTrackbar(#TrackbarValue, "video", nullptr, TrackbarValue*2, onThresholdSlide_##TrackbarValue);\
setTrackbarPos(#TrackbarValue, "video",TrackbarValue);

#define CALL_BACK_TRACKBAR(TrackbarValue)void onThresholdSlide_##TrackbarValue(int, void*) {\
    TrackbarValue = getTrackbarPos(#TrackbarValue, "video");\
    onThresholdSlide();\
}

// Global variables to hold the threshold and Gaussian blur size

vector<Point2f>dstPts ;
Mat image, channels[3], binary,frame;
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;


void Get_Pts() {
    image.copyTo(frame); // Update frame with the newly processed image

    cv::imshow("Original Image", frame);
    // 转换图像到HSV颜色空间
    cv::Mat hsvImage;
    cv::cvtColor(frame, hsvImage, cv::COLOR_BGR2HSV);

    // 定义绿色范围的阈值
    cv::Scalar lowerGreen = cv::Scalar(35, 50, 50); // 绿色的HSV最低值
    cv::Scalar upperGreen = cv::Scalar(90, 255, 255); // 绿色的HSV最高值

    // 根据阈值提取绿色部分
    cv::Mat greenMask;
    cv::inRange(hsvImage, lowerGreen, upperGreen, greenMask);

    // 通过与操作（bitwise AND）来获取绿色区域
    cv::Mat greenPart;
    cv::bitwise_and(frame, frame, greenPart, greenMask);

    cv::imshow("Green Parts", greenPart);

    
    split(greenPart,channels);
    binary = channels[1];

    imshow("after_threshold",binary);

    findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

    vector<vector<Point>>conPoly(contours.size());

    double peri0 = arcLength(contours[0], true);
    double peri1 = arcLength(contours[1], true);

    approxPolyDP(contours[0], conPoly[0], 0.02*peri0, true);
    approxPolyDP(contours[1], conPoly[1], 0.02*peri1, true);

        //获取矩形四个角点
    dstPts = { conPoly[0][0],conPoly[0][1],conPoly[0][2],conPoly[0][3],conPoly[1][0],conPoly[1][1],conPoly[1][2],conPoly[1][3] };

    for (int j = 0; j < dstPts.size(); j++)
    {
        putText(frame,std::to_string(j),dstPts[j],1,2,Scalar(0,0,255),2);
    } 

    for (int j = 0; j < dstPts.size(); j++) {
        circle(frame, dstPts[j], 5, Scalar(0, 0, 255), 2); // 在图像上绘制一个半径为 5 的红色圆
    }

    drawContours(frame,contours,-1,Scalar(0,0,255),2);

    imshow("video", frame);
}


int main() {
    namedWindow("Original Image",WINDOW_NORMAL);
    namedWindow("Green Parts",WINDOW_NORMAL);
    namedWindow("after_threshold",WINDOW_NORMAL);
    namedWindow("video",WINDOW_NORMAL);
    // 读取图像
    image = cv::imread("/home/kangzhehao/practice/week3/task3/1_whole.png");

    if (image.empty()) {
        std::cout << "Could not read the image" << std::endl;
        return -1;
    }

    //获得特征点的像素坐标
    Get_Pts(); 

    // 准备三维空间中的点
    std::vector<cv::Point3f> objectPoints = {{0,2,1},{0,2,0},{0,1,0},{0,1,1},{2,0,2},{1,0,2},{1,0,1},{2,0,1}};

    // 相机内参矩阵
    double fx=1900,fy=1900,cx=960,cy=540;
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    // 畸变系数
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

    // 用于存储输出的旋转和平移向量
    cv::Mat rvec, tvec;

    // 使用 solvePnP 函数进行 PnP 解算
    cv::solvePnP(objectPoints, dstPts, cameraMatrix, distCoeffs, rvec, tvec);

    // 打印结果
    std::cout << "Rotation Vector (r): "<<endl << rvec << std::endl;
    std::cout << "Translation Vector (t): " <<endl<< tvec << std::endl;

    cv::waitKey(0);

    return 0;
}

