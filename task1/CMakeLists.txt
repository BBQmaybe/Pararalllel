cmake_minimum_required(VERSION 3.21)
project(task1)

set(CMAKE_CXX_STANDARD 17)

option(USE_DOUBLE "Using double" OFF)

if(USE_DOUBLE)
   add_definitions(-DUSE_DOUBLE)
endif()

add_executable(task1 1_1.cpp)