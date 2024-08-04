#include <iostream>
#include "matplotlibcpp.h"
#include "KalmanFilter.hpp"
#include "Simulator.hpp"
#include "lowPassFilter.hpp"

namespace plt = matplotlibcpp;

int main() {

    double t=0;
    srand(114514);

    // 以一个静止滤波为例展示使用方法
    LowPassFilter<double, 2> lowPassFilter(0.1);

    // 滤波器初始化
    KalmanFilter<double, 2, 2> *kf;
    kf = new KalmanFilter<double, 2, 2>();

    // 仿真器初始化
    Simulator<double, 2> *simulator;
    simulator = new Simulator<double, 2>(Eigen::Vector2d(0, 0), 5,Eigen::Vector2d(0.01,0.01)); // 输入为起始点与方差

    // 设置状态转移矩阵
    kf->transition_matrix << 1, 0,
                            0, 1;
    // 设置测量矩阵
    kf->measurement_matrix << 1, 0,
                              0, 1;
    // 设置过程噪声协方差矩阵
    kf->process_noise_cov << 0.01, 0,
                            0, 0.01;
    // 设置测量噪声协方差矩阵
    kf->measurement_noise_cov << 5, 0,
                                0, 5;
    // 设置控制向量
    kf->control_vector << 0,
                         0;

    // 生成随机点
    Eigen::Vector2d measurement;

    // 存储绘图数据
    std::vector<double>  timesteps, measured_x, measured_y, kalman_x, kalman_y, lowpass_x, lowpass_y;

    while (t < 1000) {
        measurement = simulator->getMeasurement(t);

        // 使用低通滤波器对测量值进行滤波
        lowPassFilter.update(measurement);
        Eigen::Vector2d lp_estimate = lowPassFilter.estimate;

        // 预测
        kf->predict(measurement);
        // 更新
        kf->update();
        // 获取后验估计
        Eigen::Vector2d estimate = kf->posteriori_state_estimate;

        // 存储绘图数据
        timesteps.push_back(t);

        measured_x.push_back(measurement[0]);
        measured_y.push_back(measurement[1]);

        kalman_x.push_back(estimate[0]);
        kalman_y.push_back(estimate[1]);

        lowpass_x.push_back(lp_estimate[0]);
        lowpass_y.push_back(lp_estimate[1]);

        t++;
    }

    // 使用 Matplotlib-cpp 进行可视化
    plt::scatter(measured_x, measured_y, 2.0, {{"color", "blue"}, {"label", "Measured"}});  // 设置蓝色，标签为 "Measured"
    plt::scatter(lowpass_x, lowpass_y, 2.0, {{"color", "magenta"}, {"label", "Lowpass"}});  // 设置品红，标签为 "Lowpass"
    plt::scatter(kalman_x, kalman_y, 2.0, {{"color", "green"}, {"label", "Kalman"}});  // 设置绿色，标签为 "Kalman"

    plt::title("Comparison of Kalman Filter and Lowpass Filter");
    plt::xlabel("X");
    plt::ylabel("Y");

    plt::legend({{"loc", "upper left"}});

    plt::save("output_plot.png");

    plt::figure();
    plt::subplot(2, 1, 1);
    plt::plot(timesteps, measured_x, {{"color", "blue"}, {"label", "Measured X"}});
    plt::plot(timesteps, lowpass_x, {{"color", "magenta"}, {"label", "Lowpass X"}});
    plt::plot(timesteps, kalman_x, {{"color", "green"}, {"label", "Kalman X"}});
    plt::suptitle("X and Y axis Comparison");
    plt::ylabel("X");

    plt::subplot(2, 1, 2);
    plt::plot(timesteps, measured_y, {{"color", "blue"}, {"label", "Measured Y"}});
    plt::plot(timesteps, lowpass_y, {{"color", "magenta"}, {"label", "Lowpass Y"}});
    plt::plot(timesteps, kalman_y, {{"color", "green"}, {"label", "Kalman Y"}});
    plt::xlabel("Time");
    plt::ylabel("Y");

    plt::legend();
    plt::save("XY_t.png");

    return 0;
}


