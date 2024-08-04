#include "iostream"
#include <algorithm>
#include <opencv2/opencv.hpp>  //头文件
using namespace std;
using namespace cv;
Mat RGB2HSV(Mat& srcImage);
int main(){
    Mat srcImg = imread("/home/kangzhehao/practice/week2/task1/armor.png");
    namedWindow("BGR",WINDOW_NORMAL);
    imshow("BGR",srcImg);//原始图像

    Mat hsvImg =  RGB2HSV(srcImg);
    namedWindow("MY_HSV",WINDOW_NORMAL);
    imshow("MY_HSV",hsvImg);//转换后的图像

    Mat opencv_hsvImage ;
    cvtColor(srcImg,opencv_hsvImage,COLOR_BGR2HSV);
    namedWindow("OPENCV_HSV",WINDOW_NORMAL);
    imshow("OPENCV_HSV",opencv_hsvImage);//调用opencv库BGR转HSV

    waitKey(0);

    return 0;
}

Mat RGB2HSV(Mat& srcImage){
    Mat hsvImage = Mat(srcImage.rows,srcImage.cols,CV_8UC3);
    for (int i = 0; i < srcImage.rows; i++)
    {
        for (int j  = 0; j < srcImage.cols; j++)
        {
            int B = srcImage.at<Vec3b>(i,j)[0];
            int G = srcImage.at<Vec3b>(i,j)[1];
            int R = srcImage.at<Vec3b>(i,j)[2];
            int Max = max({B,G,R});
            int Min = min({B,G,R});
            int H=0; int S=0; int V=Max;

            if (Max != Min) {
                if (R == Max) H = (G-B)*60/(Max-Min);
                if (G == Max) H = 120 + (B-R)*60/(Max-Min);
                if (B == Max) H = 240 + (R-G)*60/(Max-Min);
                if (H < 0) H = H + 360;
                S = ((Max-Min)*255/Max);
            }

            hsvImage.at<Vec3b>(i,j)[0] = H*180/360; // OpenCV  H: 0-180
            hsvImage.at<Vec3b>(i,j)[1] = S;
            hsvImage.at<Vec3b>(i,j)[2] = V;
        }
    }
    return hsvImage;
}
