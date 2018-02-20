cmake_minimum_required(VERSION 3.0)

project(mpl_tests)

enable_language(C)
enable_language(CXX)

find_package(MPI REQUIRED) 
set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

include_directories("../")

add_executable(send_recv send_recv.cc)
