cmake_minimum_required(VERSION 3.21)
project(TED_C++)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug)
endif()


add_executable(X-TED_CPU main.cpp TED_C++.h TED.cpp dataset_preprocess.cpp parallel_computing.cpp TED_test.cpp)

