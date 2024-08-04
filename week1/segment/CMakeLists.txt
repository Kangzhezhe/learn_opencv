cmake_minimum_required(VERSION 2.8)
project(segmentation)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(   											
    ${PROJECT_SOURCE_DIR}/include
)
set(SRC_LIST src/main.cpp src/segment.cpp )

add_executable(segmentation ${SRC_LIST})
target_link_libraries(segmentation PRIVATE ${OpenCV_LIBS})