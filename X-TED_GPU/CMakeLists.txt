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
#    message("Debug mode")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -lineinfo -Xcompiler -ggdb -std=c++14 --extended-lambda")
else ()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DNDEBUG -Xcompiler -DNDEBUG -std=c++14 --extended-lambda")
endif ()

add_executable(X-TED_GPU main.cu TED_C++.h help.h test_TED.cu dataset_processing.cu parallel_X_TED.cu TED.cu function.cu)

target_link_libraries(X-TED_GPU ${EXTRA_LIBS})

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set_target_properties(X-TED_GPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
