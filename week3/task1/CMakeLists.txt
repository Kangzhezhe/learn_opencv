cmake_minimum_required(VERSION 2.8)
project(test)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(   											
    ${PROJECT_SOURCE_DIR}/inc
)
set(SRC_LIST src/main.cpp)

add_executable(test ${SRC_LIST})
target_link_libraries(test PRIVATE ${OpenCV_LIBS})