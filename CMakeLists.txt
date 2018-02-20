cmake_minimum_required(VERSION 3.0)
project(mpl)

#set(CMAKE_CXX_STANDARD 14)
#
#enable_language(C)
#enable_language(CXX)
#
#find_package(MPI REQUIRED)
#set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
#set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
#
#include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(examples)
add_subdirectory(test)
