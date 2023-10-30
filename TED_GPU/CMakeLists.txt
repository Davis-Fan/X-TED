cmake_minimum_required(VERSION 3.21)
project(TED_GPU LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
find_package(CUDAToolkit)
set(EXTRA_LIBS pthread CUDA::nvToolsExt CUDA::cuda_driver)
set(CMAKE_CUDA_ARCHITECTURES "60;61;70")

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/utils)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set( CMAKE_CXX_FLAGS_DEBUG "-g -O0" )
set( CMAKE_CXX_FLAGS_RELEASE "-O3" )

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo -Xcompiler -ggdb -std=c++14 --extended-lambda")
else ()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG -Xcompiler -DNDEBUG -std=c++14 --extended-lambda")
endif ()

add_executable(TED_GPU main.cu TED_C++.h help.h swissport_test.cu dataset_preprocess.cu parallel_TED_18.cu TED.cu test_2.cu test_1.cu function.cu threshold_test.cu parallel_strategy.cu algo1.cu algo2.cu)

target_link_libraries(TED_GPU ${EXTRA_LIBS})

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set_target_properties(TED_GPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
