#ifndef LOWPASSFILTER_H
#define LOWPASSFILTER_H

#include <eigen3/Eigen/Core>

template <typename T, int Size>
class LowPassFilter {
public:
    Eigen::Matrix<T, Size, 1> estimate; // 估计值

    LowPassFilter(T alpha) : alpha(alpha), initialized(false) {
        // 初始化其他成员变量
        estimate.setZero();
    }

    void update(const Eigen::Matrix<T, Size, 1>& measurement) {
        if (!initialized) {
            estimate = measurement;
            initialized = true;
        } else {
            estimate = alpha * measurement + (1.0 - alpha) * estimate;
        }
    }

private:
    T alpha; // 滤波系数
    bool initialized; // 是否已初始化
};

#endif

