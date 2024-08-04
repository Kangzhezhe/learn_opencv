//
// Created by lucas on 2020/10/13.
//
//
#include <iostream>
#include "opencv2/viz.hpp"
#include "opencv2/opencv.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
using namespace std;
using namespace cv;
using namespace Eigen;
void onThresholdSlide();
#define CREATE_TRACKBAR(TrackbarValue)createTrackbar(#TrackbarValue, "video", nullptr, 89, onThresholdSlide_##TrackbarValue);\
setTrackbarPos(#TrackbarValue, "video",TrackbarValue);
#define CALL_BACK_TRACKBAR(TrackbarValue)void onThresholdSlide_##TrackbarValue(int, void*) {\
    TrackbarValue = getTrackbarPos(#TrackbarValue, "video");\
    onThresholdSlide();\
}

void Get_Rotation(Eigen::Vector3d rpy_raw,Eigen::Matrix3d& R_AB);

viz::Viz3d myWindow("Coordinate Frame"); 
Eigen::Vector3d rpy_raw = Eigen::Vector3d(0,0,0);
int r=0,p=0,y=0;
int a=1,b=1,c=1;

CALL_BACK_TRACKBAR(r)
CALL_BACK_TRACKBAR(p)
CALL_BACK_TRACKBAR(y)
CALL_BACK_TRACKBAR(a)
CALL_BACK_TRACKBAR(b)
CALL_BACK_TRACKBAR(c)

int main(int argc, char **argv) {
    namedWindow("video", WINDOW_NORMAL);
    imshow("video",Mat(Scalar(1,1)));
    CREATE_TRACKBAR(r)
    CREATE_TRACKBAR(p)
    CREATE_TRACKBAR(y)
    CREATE_TRACKBAR(a)
    CREATE_TRACKBAR(b)
    CREATE_TRACKBAR(c)
    // 创建R\T
    onThresholdSlide();
    // ----------------------显示----------------------
    myWindow.spinOnce(1, true);
	while (!myWindow.wasStopped())
	{
        waitKey(1);
		myWindow.spinOnce(1, true);
	}
    return 0;
}

void Get_Rotation(Eigen::Vector3d rpy_raw,Eigen::Matrix3d& R_AB)
{
    rpy_raw = rpy_raw * M_PI / 180;
    R_AB = Eigen::AngleAxisd( Eigen::AngleAxisd(rpy_raw[0], Eigen::Vector3d::UnitX())* Eigen::AngleAxisd(rpy_raw[1], Eigen::Vector3d::UnitY()) *Eigen::AngleAxisd(rpy_raw[2], Eigen::Vector3d::UnitZ()) ) ;
}

void onThresholdSlide(){
    Eigen::Matrix3d R_AB;
    rpy_raw<<p,y,r;
    Get_Rotation(rpy_raw,R_AB);
    cv::Matx33d PoseR_0 ; // 旋转矩阵
    eigen2cv(R_AB,PoseR_0);
    cv::Vec3d PoseT_0 (a,b,c); // 平移向量
    // PoseR_0 = cv::Matx33d(0.0700133,	0.00325626,	0.997541,0.997535,	0.00439059,	-0.0700273,-0.00460782,	0.999985,	-0.00294083);
    // PoseT_0 = cv::Vec3d(-0.128633,0.000609044,0.122929) * 10;
    cv::Affine3d Transpose03(PoseR_0,PoseT_0); // 03相机变换矩阵
    // ----------------------设置坐标系----------------------
    myWindow.showWidget("Cam0",viz::WCoordinateSystem(),Transpose03);
}