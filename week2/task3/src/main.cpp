#include "stdio.h"
#include<iostream> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

// Global variables to hold the threshold and Gaussian blur size
int kThreshold = 152;
int kMaxThreshold = 164;

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
vector<Point2f> boxPts(4);
int rec_index = 0;


void onThresholdSlide(); 

void onThresholdSlide_KThreshod(int, void*) {
    kThreshold = getTrackbarPos("Threshold", "video");
    onThresholdSlide();
}
void onThresholdSlide_kMaxThreshold(int, void*){
    kMaxThreshold = getTrackbarPos("MaxThreshold", "video");
    onThresholdSlide();
}
void onThresholdSlide_kGauss(int, void*) {
    kGaussianBlurSize = getTrackbarPos("Gaussian Blur Size", "video");
    onThresholdSlide();
}
void onThresholdSlide_h_w_rate(int, void*) {
    h_w_rate = getTrackbarPos("h_w_rate", "video");
    onThresholdSlide();
}
void onThresholdSlide_bound_width_min(int, void*) {
    bound_width_min = getTrackbarPos("bound_width_min", "video");
    onThresholdSlide();
}
void onThresholdSlide_bound_width_max(int, void*) {
    bound_width_max = getTrackbarPos("bound_width_max", "video");
    onThresholdSlide();
}

void onThresholdSlide() {
    image.copyTo(frame); // Update frame with the newly processed image

    split(frame, channels);
    // imshow("1",channels[0]);
    threshold(channels[0], binary, kThreshold, kMaxThreshold, 0);

    // imshow("2",binary);
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat closedImage;
    morphologyEx(binary, closedImage, MORPH_CLOSE, element);

    // imshow("3",closedImage);
    GaussianBlur(closedImage, Gaussian, Size(2 * kGaussianBlurSize + 1, 2 * kGaussianBlurSize + 1), 0);

    // imshow("4",Gaussian);
    findContours(Gaussian, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

    for (int i = 0; i < contours.size(); i++) {
        boundRect = boundingRect(Mat(contours[i]));
        //h_w = 1/6 , width = 600
        if ((double)boundRect.height / boundRect.width >= (double)30/100  &&(double)boundRect.height / boundRect.width <= (double)h_w_rate/100  && boundRect.width > bound_width_min && boundRect.width < bound_width_max) {
            rectangle(frame, boundRect.tl(), boundRect.br(), Scalar(0, 255, 0), 2, 8, 0);
        }
    }
    imshow("video", frame);
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
    // namedWindow("1", WINDOW_NORMAL);
    // namedWindow("2", WINDOW_NORMAL);
    // namedWindow("3", WINDOW_NORMAL);
    // namedWindow("4", WINDOW_NORMAL);
    image = imread("/home/kangzhehao/practice/week2/task3/task3.png");

    createTrackbar("Threshold", "video", nullptr, kThreshold*2, onThresholdSlide_KThreshod);
    createTrackbar("MaxThreshold", "video", nullptr, kMaxThreshold*2, onThresholdSlide_kMaxThreshold);
    createTrackbar("Gaussian Blur Size", "video", nullptr, kMaxGaussianBlurSize, onThresholdSlide_kGauss);
    createTrackbar("h_w_rate", "video", nullptr, h_w_rate*2, onThresholdSlide_h_w_rate);
    createTrackbar("bound_width_min", "video", nullptr, bound_width_min*2, onThresholdSlide_bound_width_min);
    createTrackbar("bound_width_max", "video", nullptr, bound_width_max*2, onThresholdSlide_bound_width_max);

    setTrackbarPos("Threshold", "video",kThreshold);
    setTrackbarPos("MaxThreshold", "video",kMaxThreshold);
    setTrackbarPos("Gaussian Blur Size", "video",kGaussianBlurSize);
    setTrackbarPos("h_w_rate", "video",h_w_rate);
    setTrackbarPos("bound_width_min", "video",bound_width_min);
    setTrackbarPos("bound_width_max", "video",bound_width_max);

    onThresholdSlide();

    setMouseCallback("video",onMouse,reinterpret_cast<void*>(&frame));//鼠标响应函数
    waitKey(0);
    return 0;
}
