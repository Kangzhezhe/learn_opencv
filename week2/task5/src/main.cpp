#include "stdio.h"
#include<iostream> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

#define CREATE_TRACKBAR(TrackbarValue)createTrackbar(#TrackbarValue, "video", nullptr, TrackbarValue*2, onThresholdSlide_##TrackbarValue);\
setTrackbarPos(#TrackbarValue, "video",TrackbarValue);

#define CALL_BACK_TRACKBAR(TrackbarValue)void onThresholdSlide_##TrackbarValue(int, void*) {\
    TrackbarValue = getTrackbarPos(#TrackbarValue, "video");\
    onThresholdSlide();\
}

// Global variables to hold the threshold and Gaussian blur size

int kThreshold = 79;
int kMaxThreshold = 100;

int kGaussianBlurSize = 5;
const int kMaxGaussianBlurSize = 15;

int h_w_rate=33;

int bound_width_min = 559;
int bound_width_max = 642;

Mat image, channels[3], binary, Gaussian,frame;
vector<vector<Point>> contours;
vector<Vec4i> hierarchy;
Rect boundRect;
RotatedRect box;
std::vector<cv::Point2f> boxPts;
int rec_index = 0;

float getDistance(Point pointO, Point pointA);
void onThresholdSlide(); 

CALL_BACK_TRACKBAR(kThreshold)
CALL_BACK_TRACKBAR(kMaxThreshold)
CALL_BACK_TRACKBAR(kGaussianBlurSize)
CALL_BACK_TRACKBAR(h_w_rate)
CALL_BACK_TRACKBAR(bound_width_min)
CALL_BACK_TRACKBAR(bound_width_max)

void onThresholdSlide() {
    image.copyTo(frame); // Update frame with the newly processed image

    split(frame, channels);
    imshow("1",channels[0]);
    threshold(channels[0], binary, kThreshold, kMaxThreshold, 0);

    // imshow("2",binary);
    // Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    // Mat closedImage;
    // morphologyEx(binary, closedImage, MORPH_CLOSE, element);

    imshow("3",binary);
    // GaussianBlur(binary, Gaussian, Size(2 * kGaussianBlurSize + 1, 2 * kGaussianBlurSize + 1), 0);

    // imshow("4",Gaussian);
    findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
    vector<vector<Point>> contoursFilterd;

    for (int i = 0; i < contours.size(); i++) {
        
        double area = contourArea(contours[i]);
        //11717
        if(area>10000 && area<12000){
            // putText(frame,std::to_string(area),contours[i].at(0),1,1,Scalar(0,0,255),2);
            // cout<<area<<endl;
            contoursFilterd.push_back(contours[i]);
            // boundRect = boundingRect(Mat(contours[i]));
            // box = minAreaRect(contours[i]);
            // box.points(boxPts);
            // // rectangle(frame, boundRect.tl(), boundRect.br(), Scalar(0, 255, 0), 2, 8, 0);
            // Point2f vertices[4];      //定义矩形的4个顶点
            // box.points(vertices);   //计算矩形的4个顶点
        }
    }

    vector<vector<Point>>conPoly(contoursFilterd.size());
	vector<Point>srcPts,dstPts ;

   double peri = arcLength(contoursFilterd[0], true);

    approxPolyDP(contoursFilterd[0], conPoly[0], 0.02*peri, true);
        //获取矩形四个角点
    srcPts = { conPoly[0][0],conPoly[0][1],conPoly[0][2],conPoly[0][3] };


    // for (int j = 0; j < conPoly[0].size(); j++)
    // {
    //     putText(frame,std::to_string(j),boxPts[j],1,2,Scalar(0,0,255),2);
    // } 
    
    int T_L = 1, T_R=0, B_R=3, B_L=2;

    // drawContours(frame,contoursFilterd,-1,Scalar(0,255,0),2);
    srcPts[B_R] += Point(10,-15); 


    double LeftHeight = getDistance(srcPts[T_L], srcPts[B_L]);
	double RightHeight = getDistance(srcPts[T_R], srcPts[B_R]);
	double MaxHeight = max(LeftHeight, RightHeight);

	double UpWidth = getDistance(srcPts[T_L], srcPts[T_R]);
	double DownWidth = getDistance(srcPts[B_L], srcPts[B_R]);
	double MaxWidth = max(UpWidth, DownWidth);

    Point2f SrcAffinePts[4] = { Point2f(srcPts[T_L]),Point2f(srcPts[T_R]) ,Point2f(srcPts[B_R]) ,Point2f(srcPts[B_L]) };
	Point2f DstAffinePts[4] = { Point2f(0,0),Point2f(MaxWidth,0),Point2f(MaxWidth,MaxHeight),Point2f(0,MaxHeight) };

    Mat img_warp;
    img_warp = Mat(Size(MaxWidth,MaxHeight),frame.type());
	Mat Affine = getPerspectiveTransform(SrcAffinePts,DstAffinePts);//根据对应点求取放射变换矩阵
    warpPerspective(frame,img_warp,Affine,img_warp.size());//进行放射变换

    imshow("video", frame);
    imshow("img_warp", img_warp);
}

void onMouse(int event,int x,int y,int flags,void* param)
{
	Mat* im = reinterpret_cast<Mat*>(param);
	switch(event)
	{
	case EVENT_LBUTTONDOWN:
		//左键按下显示像素值
		if (static_cast<int>(im->channels())==1)
		{
			//若为灰度图，显示鼠标点击的坐标以及灰度值
			cout<<"at("<<x<<","<<y<<")value is:"<<static_cast<int>(im->at<uchar>(x,y))<<endl;
		}
		else if (static_cast<int>(im->channels() == 3))
		{
			//若图像为彩色图像，则显示鼠标点击坐标以及对应的B, G, R值
			cout << "at (" << x << ", " << y << ")"
				<< "  B value is: " << static_cast<int>(im->at<Vec3b>(x, y)[0]) 
				<< "  G value is: " << static_cast<int>(im->at<Vec3b>(x, y)[1])
				<< "  R value is: " << static_cast<int>(im->at<Vec3b>(x, y)[2])
				<< endl;
		}
	}
}



int main() {
    namedWindow("video", WINDOW_NORMAL);
    namedWindow("img_warp", WINDOW_NORMAL);
    // namedWindow("1", WINDOW_NORMAL);
    // namedWindow("2", WINDOW_NORMAL);
    // namedWindow("3", WINDOW_NORMAL);
    // namedWindow("4", WINDOW_NORMAL);
    image = imread("/home/kangzhehao/practice/week2/task5/task5.jpg");

    // CREATE_TRACKBAR(kThreshold)
    // CREATE_TRACKBAR(kMaxThreshold)
    // CREATE_TRACKBAR(kGaussianBlurSize)
    // CREATE_TRACKBAR(h_w_rate)
    // CREATE_TRACKBAR(bound_width_min)
    // CREATE_TRACKBAR(bound_width_max)

    onThresholdSlide();

    setMouseCallback("video",onMouse,reinterpret_cast<void*>(&frame));//鼠标响应函数
    waitKey(0);
    return 0;
}

//两点间距离公式
float getDistance(Point pointO, Point pointA)
{
    float distance;
    distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);
    distance = sqrtf(distance);
	//屏幕分辨率为1366*768，一个像素约为0.214mm，所以这里乘以0.214转化为实际尺寸，当然分辨率不同，一个像素表示的实际长度也不同
    return distance;
}