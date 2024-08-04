cmake_minimum_required(VERSION 2.8)
project(task)
set(CMAKE_CXX_STANDARD 11)

# find_package()
# include_directories()
set(SRC_LIST main.cpp  )

add_executable(test ${SRC_LIST})
# target_link_libraries()