cmake_minimum_required(VERSION 3.0)
project(KalmanFilter)

find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Python3 COMPONENTS Development NumPy REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR}
                    ${OpenCV_INCLUDE_DIRS}
                    ${Python3_INCLUDE_DIRS}
                    ${Python3_NumPy_INCLUDE_DIRS}  
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")

add_executable(KalmanFilter main.cpp)
target_link_libraries(KalmanFilter
                      ${EIGEN3_LIBRARIES}
                      ${OpenCV_LIBRARIES}
)

# No need to explicitly link Python3::NumPy, it should be included with Python3::Python
target_link_libraries(KalmanFilter Python3::Python)

