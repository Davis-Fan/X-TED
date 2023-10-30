cmake_minimum_required(VERSION 3.21)
project(TED_C++)
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(CMAKE_CXX_STANDARD 14)

add_executable(TED_C++ main.cpp TED_C++.h TED.cpp test_1.cpp dataset_preprocess.cpp sentiment_test.cpp bolzano_test.cpp parallel_TED.cpp test_2.cpp parallel_TED_2.cpp parallel_TED_3.cpp parallel_TED_4.cpp parallel_TED_5.cpp parallel_TED_6.cpp parallel_TED_7.cpp parallel_TED_8.cpp parallel_TED_9.cpp parallel_TED_10.cpp parallel_TED_11.cpp parallel_TED_12.cpp parallel_TED_13.cpp parallel_TED_14.cpp parallel_TED_15.cpp test_3.cpp parallel_TED_16.cpp parallel_TED_17.cpp parallel_TED_18.cpp swissport_test.cpp python_test.cpp dblp_test.cpp)

